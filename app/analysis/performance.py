"""What the business looks like: scale, leaders, laggards, concentration, momentum.

The diagnostics say what is broken. This says what is *happening* -- which is the
question people actually open the tool to ask, and the half that was missing.

Everything is computed against the metrics declared in the semantic layer, using the
``direction`` on each one.  That field is what turns a ranking into a judgement: the
top of a ``higher_is_better`` list is the leaders, the top of a ``lower_is_better``
list is the problem.  Without it a report can only say "here are some numbers, in
order", which is the difference between a table and an insight.

Every signal carries the rows and the SQL that produced them, so the same object can
be rendered as a chart, pinned to a dashboard, or re-checked by hand.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.analysis import semantics
from app.analysis.trends import Signal, _rows
from app.tables import catalog

MIN_SEGMENT_ROWS = 30
TOP_N = 8


@dataclass
class Measure:
    """A number worth ranking on, and which end of the ranking is good."""
    name: str
    sql: str
    direction: str = semantics.HIGHER
    label: str = ""

    @property
    def good_high(self) -> bool:
        return self.direction == semantics.HIGHER


@dataclass
class PerformanceReport:
    table: str = ""
    signals: list[Signal] = field(default_factory=list)
    took: float = 0.0

    def as_dict(self) -> dict:
        return {"table": self.table, "took": self.took,
                "signals": [s.as_dict() for s in self.signals]}

    def summary(self) -> str:
        if not self.signals:
            return f"{self.table}: no business signals computed."
        return "\n".join([f"{self.table}: {len(self.signals)} business signal(s)."]
                         + [f"- [{s.kind}] {s.headline}" for s in self.signals])


def _measures(info: catalog.TableInfo, exclude: set[str]) -> list[Measure]:
    """The metrics to rank on, declared ones first.

    A declared metric beats a detected column every time: somebody said what it
    means and which way is good, and that is exactly the knowledge no amount of
    profiling recovers.
    """
    out: list[Measure] = []
    described = semantics.table(info.name)
    if described:
        for m in described.metrics:
            # A metric built on a column the diagnostics called corrupt would rank
            # segments by an artefact.
            if any(bad in m.sql for bad in exclude):
                continue
            out.append(Measure(name=m.name, sql=m.sql, direction=m.direction,
                               label=m.means or m.name.replace("_", " ")))
    if not out:
        for col in info.columns:
            if col.role == "number" and col.name not in exclude:
                out.append(Measure(name=f"total_{col.name}",
                                   sql=f"SUM({catalog.quote(col.name)})",
                                   label=f"total {col.label}"))
    out.append(Measure(name="record_count", sql="COUNT(*)", label="records"))
    return out[:3]


def _dimensions(info: catalog.TableInfo) -> list[catalog.ColumnInfo]:
    dims = [c for c in info.columns
            if c.role == "category" and 2 <= c.n_distinct <= 60]
    # Same ordering as trends: a breakdown into eight-ish groups says more than one
    # into two, and the choice must not shift when a merge changes a count by one.
    dims.sort(key=lambda c: (abs(c.n_distinct - 8), c.n_distinct))
    return dims[:4]


def _headline(table: str, info: catalog.TableInfo, measure: Measure) -> list[Signal]:
    """The size and span of the business, before any breakdown."""
    q = catalog.quote
    date_col = next((c.name for c in info.columns if c.role == "date"), "")
    span = (f", MIN({q(date_col)}) AS first_seen, MAX({q(date_col)}) AS last_seen"
            if date_col else "")
    stmt = (f"SELECT COUNT(*) AS records, {measure.sql} AS {q(measure.name)}{span} "
            f"FROM {q(table)}")
    cols, rows = _rows(stmt)
    if not rows:
        return []
    r = rows[0]
    head = f"{int(r[0]):,} records"
    if r[1] is not None:
        head += f", {measure.label} {float(r[1]):,.0f}"
    if date_col and len(r) > 3 and r[2]:
        head += f", covering {str(r[2])[:10]} to {str(r[3])[:10]}"
    return [Signal(kind="headline", severity="low", table=table, headline=head,
                   detail="The overall scale of the data, before any breakdown.",
                   evidence_sql=stmt, columns=cols, rows=rows)]


def _ranking(table: str, dim, measure: Measure) -> list[Signal]:
    """Who is at each end, and by how much."""
    q = catalog.quote
    stmt = (f"SELECT {q(dim.name)} AS segment, COUNT(*) AS records, "
            f"{measure.sql} AS value FROM {q(table)} "
            f"WHERE {q(dim.name)} IS NOT NULL AND {q(dim.name)} <> '' "
            f"GROUP BY 1 HAVING records >= {MIN_SEGMENT_ROWS} "
            f"ORDER BY value {'DESC' if measure.good_high else 'ASC'}")
    cols, rows = _rows(stmt, limit=200)
    rows = [r for r in rows if r[2] is not None]
    if len(rows) < 2:
        return []

    best, worst = rows[0], rows[-1]
    total = sum(float(r[2]) for r in rows) or 1.0
    # A ranking where the best value is zero teaches nothing -- it usually means the
    # measure simply does not apply to that category (no labour days on a parts line).
    if float(best[2]) == 0.0 and not measure.good_high:
        return []

    # "Leads with the lowest cost" and "leads with the highest revenue" are both
    # correct, and stating it the wrong way round inverts the conclusion.
    share = f" ({100.0 * float(best[2]) / total:.0f}% of the total)" \
        if measure.good_high else ""
    ratio = (max(float(best[2]), float(worst[2]))
             / min(float(best[2]), float(worst[2]))) if float(min(
                 float(best[2]), float(worst[2]))) else 0.0

    signals = [Signal(
        kind="ranking", severity="low", table=table,
        headline=(f"{measure.label} by {dim.label}: best is {best[0]} at "
                  f"{float(best[2]):,.0f}{share}; worst is {worst[0]} at "
                  f"{float(worst[2]):,.0f}"),
        detail=("Higher is better for this measure." if measure.good_high
                else "LOWER is better for this measure, so the best performer is the "
                     "smallest number.")
        + (f" The gap between them is {ratio:,.0f}x." if ratio > 1.5 else "")
        + f" Segments under {MIN_SEGMENT_ROWS} records are excluded.",
        evidence_sql=stmt, columns=["segment", "records", measure.name],
        rows=[[r[0], r[1], r[2]] for r in rows[:TOP_N]])]

    # Concentration is a business fact in its own right: if most of the value sits
    # in three segments, that is both where the attention goes and where the risk is.
    if len(rows) >= 5 and measure.good_high:
        top5 = sum(float(r[2]) for r in rows[:5])
        share = 100.0 * top5 / total
        if share >= 50.0:
            signals.append(Signal(
                kind="concentration",
                severity="medium" if share >= 75.0 else "low", table=table,
                headline=(f"{measure.label} is concentrated: the top 5 of "
                          f"{len(rows)} {dim.label} account for {share:.0f}%"),
                detail=("Concentration is worth knowing both ways: it is where "
                        "effort pays back most, and where the exposure sits if one "
                        "of them stops."),
                evidence_sql=stmt, columns=["segment", "records", measure.name],
                rows=[[r[0], r[1], r[2]] for r in rows[:5]]))
    return signals


def _momentum(table: str, info: catalog.TableInfo, measure: Measure) -> list[Signal]:
    """Direction of travel on the measure that matters."""
    date_col = next((c.name for c in info.columns if c.role == "date"), "")
    if not date_col:
        return []
    q = catalog.quote
    stmt = (f"SELECT substr({q(date_col)},1,7) AS month, {measure.sql} AS value "
            f"FROM {q(table)} WHERE {q(date_col)} IS NOT NULL "
            f"GROUP BY 1 ORDER BY 1")
    cols, rows = _rows(stmt, limit=400)
    rows = [r for r in rows if r[1] is not None]
    if len(rows) < 3:
        return []
    # The last month of an export is usually partial, so the comparison uses the two
    # complete months before it -- otherwise every dataset looks like it collapsed.
    last, prev = rows[-2], rows[-3]
    change = (100.0 * (float(last[1]) - float(prev[1])) / abs(float(prev[1]))
              if float(prev[1]) else None)
    good = None if change is None else (change > 0) == measure.good_high
    verdict = "" if good is None else (" - moving the right way" if good
                                       else " - moving the wrong way")
    head = f"{measure.label} in {last[0]}: {float(last[1]):,.0f}"
    if change is not None:
        head += f", {change:+.1f}% on {prev[0]}{verdict}"
    return [Signal(kind="momentum",
                   severity="medium" if (good is False and abs(change or 0) > 15)
                   else "low",
                   table=table, headline=head,
                   detail=("Compared against the last two complete months; the final "
                           "month of an export is usually partial."),
                   evidence_sql=stmt, columns=["month", measure.name],
                   rows=[[r[0], r[1]] for r in rows])]


def scan_table(info: catalog.TableInfo,
               exclude: set[str] | None = None) -> PerformanceReport:
    started = time.perf_counter()
    report = PerformanceReport(table=info.name)
    exclude = exclude or set()
    measures = _measures(info, exclude)
    dims = _dimensions(info)
    if not measures:
        return report

    primary = measures[0]
    report.signals += _headline(info.name, info, primary)
    report.signals += _momentum(info.name, info, primary)
    for dim in dims[:3]:
        report.signals += _ranking(info.name, dim, primary)
    # A second measure pointing the other way often tells a different story: high
    # volume with high cost per job is not the same success as high volume alone.
    for measure in measures[1:2]:
        if dims:
            report.signals += _ranking(info.name, dims[0], measure)
    report.took = round(time.perf_counter() - started, 2)
    return report


def scan(table: str = "", exclude: set[str] | None = None) -> list[PerformanceReport]:
    infos = catalog.list_tables()
    if table:
        infos = [t for t in infos if t.name == table]
    else:
        covered = {m for t in infos if t.kind == "view" for m in t.members}
        infos = [t for t in infos if t.name not in covered]
    return [scan_table(t, exclude) for t in infos]
