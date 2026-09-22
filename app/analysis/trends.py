"""Movement and outliers, measured in SQL.

"Where are we lagging" has no answer inside a single number -- it needs a comparison.
Three are computed here, all without a model:

* **trend** -- the measure per month, so direction is visible;
* **movement** -- the most recent complete period against the one before it, per
  dimension value, so you see *which* segment moved rather than that the total did;
* **outlier rate** -- an outcome's rate within each segment against the overall rate,
  which is the closest thing the data holds to "this one is underperforming".

Segments with little volume are excluded rather than reported: a dealer with four
records can show a 100% failure rate and mean nothing, and a report that leads with
noise stops being read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.analysis import semantics
from app.tables import catalog, sql

MIN_SEGMENT_ROWS = 30          # below this, a rate is noise
MAX_DIMENSIONS = 5
MAX_PER_KIND = 6


@dataclass
class Signal:
    kind: str                  # trend | movement | outlier_rate
    severity: str
    table: str
    headline: str
    detail: str = ""
    evidence_sql: str = ""
    rows: list[list] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "severity": self.severity, "table": self.table,
                "headline": self.headline, "detail": self.detail,
                "evidence_sql": self.evidence_sql, "rows": self.rows,
                "columns": self.columns}


@dataclass
class TrendReport:
    table: str = ""
    signals: list[Signal] = field(default_factory=list)
    took: float = 0.0

    def as_dict(self) -> dict:
        return {"table": self.table, "took": self.took,
                "signals": [s.as_dict() for s in self.signals]}

    def summary(self) -> str:
        if not self.signals:
            return f"{self.table}: no clear movement or outliers found."
        lines = [f"{self.table}: {len(self.signals)} signal(s)."]
        for s in self.signals:
            lines.append(f"- [{s.kind}] {s.headline}")
        return "\n".join(lines)


def _rows(statement: str, limit: int = 60) -> tuple[list[str], list[list]]:
    res = sql.run(statement, limit=limit, timeout=config.TABLE_SQL_TIMEOUT)
    return (res.columns, [list(r) for r in res.rows]) if res.ok else ([], [])


def _pct(new: float, old: float) -> float | None:
    if not old:
        return None
    return 100.0 * (new - old) / abs(old)


# Ordered by how clearly each word means failure. "Pending" is last because it is
# usually a stage, not an outcome -- picking it over "Rejected" answers the wrong
# question.
#
# This list is only a fallback, for data nobody has described yet: a declared outcome
# in the semantic layer always wins. It cannot be complete -- every domain has its own
# word for a bad ending -- so a table with no declared outcomes is reported as such
# rather than quietly analysed without one.
BAD_WORDS = (
    "reject", "fail", "defect", "cancel", "error", "breach", "fault",
    "churn", "attrit", "terminat", "resign", "left", "exit", "lapsed",
    "default", "arrears", "unpaid", "overdue", "chargeback", "refund", "dispute",
    "decline", "lost", "abandon", "bounce", "void", "expire", "suspend", "block",
    "return", "escalat", "complain", "no-show", "noshow", "missed", "delay",
    "inactive", "closed", "pending")


def _pick(info: catalog.TableInfo, exclude: set[str] | None = None) -> dict[str, Any]:
    """What is worth comparing in this table, decided from the profile."""
    exclude = exclude or set()
    date_col = next((c.name for c in info.columns if c.role == "date"), "")
    # A column the diagnostics called corrupt is not a measure. Summing an hour-meter
    # column that reaches 6e14 produces a "+1278% movement" that is entirely the
    # artefact, and it crowds out every real signal.
    measures = [c for c in info.columns
                if c.role == "number" and c.name not in exclude]
    dims = [c for c in info.columns
            if c.role == "category" and 2 <= c.n_distinct <= 40]
    # Ranked by how much a breakdown by this column can tell you, not by how few
    # values it has. Sorting by raw cardinality made the choice unstable: cleaning
    # merged two spellings, a column dropped from 4 distinct values to 3, and the
    # most informative dimension was pushed out of the list entirely.
    dims.sort(key=lambda c: (abs(c.n_distinct - 8), c.n_distinct))
    # An outcome is a small set of states a record can be in, at least one of which
    # reads as a failure. Chosen by that test rather than by cardinality order -- the
    # status column is usually not the narrowest one.
    outcomes = [c for c in info.columns
                if c.role == "category" and 2 <= c.n_distinct <= 12
                and any(w in str(v).lower() for v in c.values for w in BAD_WORDS)]
    return {"date": date_col, "measures": measures[:2],
            "dims": dims[:MAX_DIMENSIONS], "outcomes": outcomes[:2]}


# --------------------------------------------------------------------------- checks
def _trend(table: str, date_col: str, measures: list) -> list[Signal]:
    if not date_col:
        return []
    q = catalog.quote
    parts = ", ".join(f"ROUND(SUM({q(m.name)}),2) AS {q(m.name)}" for m in measures)
    select = f"substr({q(date_col)},1,7) AS month, COUNT(*) AS records"
    if parts:
        select += ", " + parts
    stmt = (f"SELECT {select} FROM {q(table)} WHERE {q(date_col)} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 1")
    cols, rows = _rows(stmt)
    if len(rows) < 3:
        return []
    # The final month of an export is usually incomplete; comparing into it invents
    # a collapse that is really just a partial month.
    last, prev = rows[-2], rows[-3]
    change = _pct(last[1], prev[1])
    direction = "flat"
    if change is not None:
        direction = "up" if change > 0 else "down"
    head = (f"Records per month: {len(rows)} months from {rows[0][0]} to {rows[-1][0]}; "
            f"{last[0]} was {last[1]:,} ({direction}"
            + (f" {change:+.1f}% vs {prev[0]}" if change is not None else "") + ")")
    return [Signal(kind="trend", severity="low", table=table, headline=head,
                   detail=("The last month in the file is often partial, so the "
                           "comparison uses the last two complete months."),
                   evidence_sql=stmt, columns=cols, rows=rows)]


def _movement(table: str, date_col: str, dim, measure) -> list[Signal]:
    """Which segment moved between the last two complete periods."""
    if not date_col:
        return []
    q = catalog.quote
    months_cols, months = _rows(
        f"SELECT DISTINCT substr({q(date_col)},1,7) AS m FROM {q(table)} "
        f"WHERE {q(date_col)} IS NOT NULL ORDER BY 1", limit=400)
    if len(months) < 3:
        return []
    latest, before = months[-2][0], months[-3][0]
    value = f"SUM({q(measure.name)})" if measure else "COUNT(*)"
    label = measure.label if measure else "records"
    stmt = (
        f"SELECT {q(dim.name)} AS segment,"
        f" ROUND(SUM(CASE WHEN substr({q(date_col)},1,7)='{latest}' THEN "
        f"{'1' if not measure else q(measure.name)} ELSE 0 END),2) AS latest,"
        f" ROUND(SUM(CASE WHEN substr({q(date_col)},1,7)='{before}' THEN "
        f"{'1' if not measure else q(measure.name)} ELSE 0 END),2) AS previous"
        f" FROM {q(table)} WHERE {q(date_col)} IS NOT NULL GROUP BY 1")
    cols, rows = _rows(stmt)
    moved = []
    for seg, now, was in rows:
        now, was = float(now or 0), float(was or 0)
        if was < MIN_SEGMENT_ROWS and now < MIN_SEGMENT_ROWS:
            continue
        change = _pct(now, was)
        if change is None or abs(change) < 20:
            continue
        moved.append((abs(change), seg, now, was, change))
    if not moved:
        return []
    moved.sort(reverse=True)
    worst = moved[0]
    lines = [[seg, now, was, round(chg, 1)] for _, seg, now, was, chg in moved[:10]]
    return [Signal(
        kind="movement", severity="medium" if worst[0] >= 40 else "low", table=table,
        headline=(f"{label} by {dim.label}: {worst[1]} moved {worst[4]:+.0f}% "
                  f"from {before} to {latest} ({worst[3]:,.0f} to {worst[2]:,.0f})"),
        detail=f"{len(moved)} segment(s) moved more than 20% between those months.",
        evidence_sql=stmt,
        columns=["segment", f"{latest}", f"{before}", "change_%"], rows=lines)]


def _outlier_rate(table: str, dim, outcome, worst_value: str) -> list[Signal]:
    """Which segment has an unusually high rate of one outcome."""
    q = catalog.quote
    stmt = (
        f"SELECT {q(dim.name)} AS segment, COUNT(*) AS records,"
        f" SUM(CASE WHEN {q(outcome.name)} = '{worst_value.replace(chr(39), chr(39)*2)}'"
        f" THEN 1 ELSE 0 END) AS hits,"
        f" ROUND(100.0 * SUM(CASE WHEN {q(outcome.name)} = "
        f"'{worst_value.replace(chr(39), chr(39)*2)}' THEN 1 ELSE 0 END) / COUNT(*), 2)"
        f" AS rate_pct"
        f" FROM {q(table)} WHERE {q(dim.name)} IS NOT NULL"
        f" GROUP BY 1 HAVING records >= {MIN_SEGMENT_ROWS} ORDER BY rate_pct DESC")
    cols, rows = _rows(stmt)
    if len(rows) < 2:
        return []
    total = sum(int(r[1]) for r in rows)
    hits = sum(int(r[2]) for r in rows)
    overall = 100.0 * hits / total if total else 0.0
    top = rows[0]
    rate = float(top[3] or 0)
    # How surprising is this segment, given the overall rate and its own size?
    #
    # A raw excess count cannot answer that: twenty extra cases is decisive in a
    # segment of 130 and meaningless in one of 60,000. The normal approximation to
    # the binomial scales with the denominator, which is the whole point -- a small
    # segment has to be far more extreme before it counts.
    n_seg, hits_seg = int(top[1]), int(top[2])
    p = overall / 100.0
    expected = n_seg * p
    sd = (expected * (1.0 - p)) ** 0.5
    z = (hits_seg - expected) / sd if sd > 0 else 0.0
    excess = hits_seg - expected
    if overall <= 0 or rate < overall * 2 or z < 3.0 or excess < 5:
        return []
    return [Signal(
        kind="outlier_rate", severity="high" if rate > overall * 4 else "medium",
        table=table,
        headline=(f"{top[0]} has a {rate:.1f}% rate of '{worst_value}' in "
                  f"{outcome.label}, against {overall:.2f}% overall "
                  f"({int(top[2]):,} of {int(top[1]):,} records)"),
        detail=(f"About {excess:,.0f} more than the {expected:,.0f} expected at the "
                f"overall rate ({z:.1f} standard deviations). "
                f"Segments under {MIN_SEGMENT_ROWS} records are excluded, "
                f"because a small denominator produces a dramatic rate that means "
                f"nothing."),
        evidence_sql=stmt, columns=["segment", "records", worst_value, "rate_%"],
        rows=[[r[0], r[1], r[2], r[3]] for r in rows[:10]])]


def _worst_outcome_value(table: str, outcome) -> str:
    """The state that counts as failure.

    If the semantic layer declares it, that wins: somebody who knows the business
    said so. The word list is only a fallback for data nobody has described yet.
    """
    declared = semantics.bad_values(table, getattr(outcome, "name", ""))
    if declared:
        return declared[0]
    for word in BAD_WORDS:
        for value in outcome.values:
            if word in str(value).lower():
                return str(value)
    return ""


def bad_measures(report) -> set[str]:
    """Columns too corrupted to total up at all.

    Deliberately narrow. An hour-meter column reaching 154 million times its own
    99th percentile has a destroyed mean and cannot be summed meaningfully. Six
    rows at 212x, in two hundred thousand, move a total by a fraction of a percent
    -- excluding revenue over that would leave the business analysis with nothing
    to rank on but labour days, which is how a report ends up describing the wrong
    thing entirely. Those get a caveat, not a ban.
    """
    return {f.column for f in getattr(report, "findings", [])
            if f.kind == "outlier" and f.severity == "high"}


def caveat_measures(report) -> dict[str, str]:
    """Columns that are usable but need a health warning attached."""
    return {f.column: f.headline for f in getattr(report, "findings", [])
            if (f.kind == "outlier" and f.severity != "high") or f.kind == "sentinel"}


def scan_table(info: catalog.TableInfo, exclude: set[str] | None = None) -> TrendReport:
    started = time.perf_counter()
    report = TrendReport(table=info.name)
    picks = _pick(info, exclude)
    date_col, measures, dims = picks["date"], picks["measures"], picks["dims"]

    report.signals += _trend(info.name, date_col, measures)

    movement: list[Signal] = []
    for dim in dims:
        movement += _movement(info.name, date_col, dim, None)
        if measures:
            movement += _movement(info.name, date_col, dim, measures[0])
    movement.sort(key=lambda s: 0 if s.severity == "medium" else 1)
    report.signals += movement[:MAX_PER_KIND]

    rates: list[Signal] = []
    for outcome in picks["outcomes"]:
        worst = _worst_outcome_value(info.name, outcome)
        if not worst:
            continue
        for dim in dims:
            if dim.name == outcome.name:
                continue
            rates += _outlier_rate(info.name, dim, outcome, worst)
    rates.sort(key=lambda s: 0 if s.severity == "high" else 1)
    report.signals += rates[:MAX_PER_KIND]

    # No word list covers every domain. If nothing here looks like a failure state and
    # nobody has declared one, say so -- silence would read as "all is well".
    if not picks["outcomes"] and not any(s.kind == "outlier_rate" for s in report.signals):
        described = semantics.table(info.name)
        if not described or not described.outcomes:
            report.signals.append(Signal(
                kind="no_outcome", severity="low", table=info.name,
                headline=("No column was recognised as recording success or failure, "
                          "so no underperformance analysis was possible"),
                detail=("Declare which values mean good and bad for this table in the "
                        "Meaning tab, and segment-level failure rates become available.")))

    report.took = round(time.perf_counter() - started, 2)
    return report


def scan(table: str = "", exclude: set[str] | None = None) -> list[TrendReport]:
    infos = catalog.list_tables()
    if table:
        infos = [t for t in infos if t.name == table]
    else:
        covered = {m for t in infos if t.kind == "view" for m in t.members}
        infos = [t for t in infos if t.name not in covered]
    return [scan_table(t, exclude) for t in infos]
