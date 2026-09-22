"""Data-quality scanning, in SQL, with no model involved.

Everything here is a measurement.  A finding says what was counted and how, so it
can be checked -- which is the difference between a diagnostic and a guess.

The checks exist because real exports really do contain these defects: an hour-meter
reading of 600,200,200,200,300, a "mandays" of 4,938 (13.5 years on one job), a file
named for one quarter holding dates from three quarters away, and the same dealer
spelled two ways so every total splits in half.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from app import config
from app.tables import catalog, sql

# Values that mean "unknown" far more often than they mean themselves.
SENTINELS = (0, -1, 9, 99, 999, 9999, 99999, 999999)
SEVERITY = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Finding:
    kind: str                       # nulls | outlier | sentinel | duplicate | ...
    severity: str                   # high | medium | low
    table: str
    column: str
    headline: str                   # one line, already readable
    detail: str = ""
    evidence_sql: str = ""          # how it was measured, so it can be re-checked
    value: Any = None
    count: int = 0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "severity": self.severity, "table": self.table,
                "column": self.column, "headline": self.headline, "detail": self.detail,
                "evidence_sql": self.evidence_sql, "value": self.value,
                "count": self.count}


@dataclass
class Report:
    table: str = ""
    n_rows: int = 0
    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    took: float = 0.0

    def as_dict(self) -> dict:
        return {"table": self.table, "n_rows": self.n_rows, "took": self.took,
                "checked": self.checked,
                "findings": [f.as_dict() for f in self.findings]}

    def summary(self) -> str:
        """The compact text the agent is given as its starting evidence."""
        if not self.findings:
            return f"{self.table}: {self.n_rows:,} rows, no data-quality problems found."
        lines = [f"{self.table}: {self.n_rows:,} rows, "
                 f"{len(self.findings)} data-quality finding(s)."]
        for f in self.findings:
            lines.append(f"- [{f.severity}] {f.headline}")
        return "\n".join(lines)


def _q(statement: str) -> list[tuple]:
    res = sql.run(statement, limit=200, timeout=config.TABLE_SQL_TIMEOUT)
    return res.rows if res.ok else []


def _one(statement: str, default: Any = None) -> Any:
    rows = _q(statement)
    return rows[0][0] if rows and rows[0] else default


# --------------------------------------------------------------------------- checks
def _check_nulls(table: str, col: catalog.ColumnInfo, n_rows: int) -> list[Finding]:
    if not n_rows or col.n_filled >= n_rows:
        return []
    missing = n_rows - col.n_filled
    pct = 100.0 * missing / n_rows
    if pct < 1.0:
        sev = "low"
    elif pct < 25.0:
        sev = "medium"
    else:
        sev = "high"
    q = catalog.quote
    return [Finding(
        kind="nulls", severity=sev, table=table, column=col.name,
        headline=f"{col.label} is empty in {missing:,} rows ({pct:.1f}%)",
        detail=("Anything grouped by this column silently drops those rows, so totals "
                "by it will not add up to the overall total."),
        evidence_sql=f"SELECT COUNT(*) FROM {q(table)} "
                     f"WHERE {q(col.name)} IS NULL OR {q(col.name)} = ''",
        count=missing, value=round(pct, 2))]


def _check_constant(table: str, col: catalog.ColumnInfo, n_rows: int) -> list[Finding]:
    if n_rows < 10 or col.n_distinct != 1 or not col.n_filled:
        return []
    q = catalog.quote
    return [Finding(
        kind="constant", severity="low", table=table, column=col.name,
        headline=f"{col.label} holds a single value in every row",
        detail="It carries no information and cannot explain any difference.",
        evidence_sql=f"SELECT DISTINCT {q(col.name)} FROM {q(table)}",
        count=n_rows)]


def _check_numeric(table: str, col: catalog.ColumnInfo) -> list[Finding]:
    if col.role != "number":
        return []
    q = catalog.quote
    stats = _q(f"SELECT AVG({q(col.name)}), MIN({q(col.name)}), MAX({q(col.name)}), "
               f"COUNT(*) FROM {q(table)} WHERE {q(col.name)} IS NOT NULL")
    if not stats:
        return []
    avg, lo, hi, n = stats[0]
    if avg is None or n < 20:
        return []
    out: list[Finding] = []

    # Compared against the 99th percentile, not the mean. Two thirds of QUANTITY is
    # zero, which drags the mean to 0.6 and makes an ordinary value of 1,000 look
    # like a scandal; the percentile describes where the real data actually sits.
    p99 = _one(f"SELECT {q(col.name)} FROM {q(table)} WHERE {q(col.name)} IS NOT NULL "
               f"ORDER BY {q(col.name)} LIMIT 1 OFFSET {max(0, int(n * 0.99) - 1)}")
    if p99 is not None and hi is not None and p99 > 0 and hi > p99 * 100:
        cut = p99 * 100
        extreme = _one(f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} > {cut}", 0)
        if extreme:
            ratio = hi / p99
            out.append(Finding(
                kind="outlier", severity="high" if ratio > 10000 else "medium",
                table=table, column=col.name,
                headline=(f"{col.label} reaches {hi:,.0f}, {ratio:,.0f}x the 99th "
                          f"percentile ({p99:,.0f}) - {extreme:,} row(s)"),
                detail=("A value this far outside the range of the data is usually a "
                        "typo or a placeholder. It will distort any average, sum or "
                        "trend that includes it."),
                evidence_sql=f"SELECT {q(col.name)} FROM {q(table)} "
                             f"WHERE {q(col.name)} > {cut:.4f} "
                             f"ORDER BY {q(col.name)} DESC LIMIT 20",
                count=extreme, value=hi))

    if lo is not None and lo < 0:
        neg = _one(f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} < 0", 0)
        out.append(Finding(
            kind="negative", severity="medium", table=table, column=col.name,
            headline=f"{col.label} is negative in {neg:,} rows (min {lo:,.2f})",
            detail="Check whether these are genuine reversals or sign errors.",
            evidence_sql=f"SELECT * FROM {q(table)} WHERE {q(col.name)} < 0 LIMIT 20",
            count=neg, value=lo))

    # A single value repeating far more than its neighbours is usually a placeholder.
    top = _q(f"SELECT {q(col.name)}, COUNT(*) c FROM {q(table)} "
             f"WHERE {q(col.name)} IS NOT NULL GROUP BY 1 ORDER BY c DESC LIMIT 3")
    if top and n:
        value, count = top[0][0], top[0][1]
        share = 100.0 * count / n
        # > 2 rather than a larger number: the guard exists to skip a genuine 0/1
        # flag, not to ignore a five-value column that is two-thirds zero.
        if value in SENTINELS and share > 20.0 and col.n_distinct > 2:
            out.append(Finding(
                kind="sentinel", severity="medium", table=table, column=col.name,
                headline=f"{col.label} is {value} in {share:.0f}% of rows",
                detail=("A single repeated round number usually means 'not recorded' "
                        "rather than a real measurement. Averages over it will be wrong."),
                evidence_sql=f"SELECT {q(col.name)}, COUNT(*) FROM {q(table)} "
                             f"GROUP BY 1 ORDER BY 2 DESC LIMIT 5",
                count=count, value=value))
    return out


def _check_dates(table: str, col: catalog.ColumnInfo, source_col: str) -> list[Finding]:
    if col.role != "date" or not col.lo:
        return []
    import datetime as dt

    q = catalog.quote
    out: list[Finding] = []
    today = dt.date.today().isoformat()
    future = _one(f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} > '{today}'", 0)
    if future:
        out.append(Finding(
            kind="future_date", severity="high", table=table, column=col.name,
            headline=f"{col.label} is in the future for {future:,} rows (up to {col.hi})",
            detail=("Future dates break every trend and any 'last N days' filter. "
                    "Usually a typed year or a placeholder."),
            evidence_sql=f"SELECT {q(col.name)}, COUNT(*) FROM {q(table)} "
                         f"WHERE {q(col.name)} > '{today}' GROUP BY 1 "
                         f"ORDER BY 1 DESC LIMIT 20",
            count=future, value=col.hi))

    # A file named for one period holding another period's rows.
    if source_col:
        spread = _q(f"SELECT {q(source_col)}, MIN({q(col.name)}), MAX({q(col.name)}), "
                    f"COUNT(*) FROM {q(table)} WHERE {q(col.name)} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 1")
        for src, lo, hi, n in spread:
            months = _months_between(str(lo)[:7], str(hi)[:7])
            if months > 6:
                out.append(Finding(
                    kind="date_spread", severity="medium", table=table, column=col.name,
                    headline=(f"{src} covers {months} months of {col.label} "
                              f"({str(lo)[:10]} to {str(hi)[:10]})"),
                    detail=("A file named for one period containing far more than that "
                            "period means the export window and the file name disagree."),
                    evidence_sql=f"SELECT substr({q(col.name)},1,7) AS month, COUNT(*) "
                                 f"FROM {q(table)} WHERE {q(source_col)} = '{src}' "
                                 f"GROUP BY 1 ORDER BY 1",
                    count=n, value=f"{lo} .. {hi}"))
    return out


def _months_between(lo: str, hi: str) -> int:
    try:
        ly, lm = int(lo[:4]), int(lo[5:7])
        hy, hm = int(hi[:4]), int(hi[5:7])
        return (hy - ly) * 12 + (hm - lm)
    except (ValueError, IndexError):
        return 0


# Words that decorate a company name without identifying it.
_NOISE_TOKENS = {"pvt", "private", "ltd", "limited", "llp", "inc", "co", "company",
                 "and", "the", "of", "services", "service"}


def _tokens(text: str) -> frozenset[str]:
    """Identity of a label: its meaningful words, ignoring case and punctuation.

    Character similarity is the wrong test here. 'Bandhan Plus - PM' and
    'Bandhan Plus - CM' are 94% identical and mean completely different things,
    while 'Gangpur Sales & Services Pvt. Ltd.' and 'Gangpur Sales and Services'
    are the same dealer. Comparing word sets separates the two cases; a fuzzy
    ratio does not.
    """
    words = re.split(r"[^a-z0-9]+", str(text).lower().replace("&", " and "))
    return frozenset(w for w in words if w and w not in _NOISE_TOKENS)


def _check_near_duplicates(table: str, col: catalog.ColumnInfo) -> list[Finding]:
    """Two spellings of one name split every total that groups by it."""
    # Identifiers are excluded on purpose: '420320_8' and '420320_18' are different
    # branches, not a misspelling, and no textual test can tell otherwise.
    if col.role != "category" or not 2 <= col.n_distinct <= 400:
        return []
    q = catalog.quote
    rows = _q(f"SELECT {q(col.name)}, COUNT(*) FROM {q(table)} "
              f"WHERE {q(col.name)} IS NOT NULL AND {q(col.name)} <> '' "
              f"GROUP BY 1 ORDER BY 2 DESC LIMIT 200")
    values = [(str(v), c) for v, c in rows]
    seen: dict[frozenset[str], tuple[str, int]] = {}
    pairs: list[tuple[str, str, int, int]] = []
    for value, count in values:
        key = _tokens(value)
        if not key:
            continue
        if key in seen and seen[key][0] != value:
            other, other_count = seen[key]
            pairs.append((other, value, other_count, count))
        else:
            seen.setdefault(key, (value, count))
    if not pairs:
        return []
    shown = "; ".join(f"{a!r} ({ca:,}) vs {b!r} ({cb:,})" for a, b, ca, cb in pairs[:3])
    return [Finding(
        kind="near_duplicate", severity="high", table=table, column=col.name,
        headline=f"{col.label} has {len(pairs)} near-duplicate spelling(s)",
        detail=(f"Totals grouped by this column are split between them. {shown}"),
        evidence_sql=f"SELECT {q(col.name)}, COUNT(*) FROM {q(table)} "
                     f"GROUP BY 1 ORDER BY 2 DESC",
        count=len(pairs), value=shown[:400])]


def _check_duplicate_rows(table: str, cols: Sequence[catalog.ColumnInfo],
                          n_rows: int) -> list[Finding]:
    """Only meaningful for a column that is *meant* to be one row's key.

    A dealer id repeating across 200,000 service records is the data working
    correctly; flagging it as duplication is noise. A column only looks like a key
    when it is nearly unique to begin with.
    """
    q = catalog.quote
    ids = [c for c in cols
           if c.role == "id" and n_rows and c.n_distinct >= n_rows * 0.9][:1]
    if not ids:
        return []
    col = ids[0]
    dupes = _one(
        f"SELECT COUNT(*) FROM (SELECT {q(col.name)} FROM {q(table)} "
        f"WHERE {q(col.name)} IS NOT NULL GROUP BY 1 HAVING COUNT(*) > 1)", 0)
    if not dupes:
        return []
    worst = _q(f"SELECT {q(col.name)}, COUNT(*) c FROM {q(table)} GROUP BY 1 "
               f"ORDER BY c DESC LIMIT 3")
    detail = ", ".join(f"{v} appears {c:,}x" for v, c in worst)
    return [Finding(
        kind="duplicate_id", severity="medium", table=table, column=col.name,
        headline=f"{col.label} repeats: {dupes:,} value(s) appear more than once",
        detail=(f"If this is meant to identify one record, the table has duplicates and "
                f"every count is inflated. {detail}"),
        evidence_sql=f"SELECT {q(col.name)}, COUNT(*) c FROM {q(table)} "
                     f"GROUP BY 1 HAVING c > 1 ORDER BY c DESC LIMIT 20",
        count=dupes)]


# --------------------------------------------------------------------------- entry
def scan_table(info: catalog.TableInfo) -> Report:
    import time

    started = time.perf_counter()
    report = Report(table=info.name, n_rows=info.n_rows)
    source_col = catalog.SRC_COL if info.kind == "view" else ""
    checks = ("missing values", "constant columns", "numeric outliers",
              "placeholder values", "negative values", "future dates",
              "date coverage", "near-duplicate labels", "duplicate identifiers")
    report.checked = list(checks)

    for col in info.columns:
        if col.name.startswith("_"):
            continue
        report.findings += _check_nulls(info.name, col, info.n_rows)
        report.findings += _check_constant(info.name, col, info.n_rows)
        report.findings += _check_numeric(info.name, col)
        report.findings += _check_dates(info.name, col, source_col)
        report.findings += _check_near_duplicates(info.name, col)
    report.findings += _check_duplicate_rows(info.name, info.columns, info.n_rows)

    report.findings.sort(key=lambda f: (-SEVERITY.get(f.severity, 0), f.column))
    del report.findings[config.DIAG_MAX_FINDINGS:]
    report.took = round(time.perf_counter() - started, 2)
    return report


def scan(table: str = "") -> list[Report]:
    """Scan one table, or every table that is not already covered by a view."""
    infos = catalog.list_tables()
    if table:
        infos = [t for t in infos if t.name == table]
    else:
        covered = {m for t in infos if t.kind == "view" for m in t.members}
        infos = [t for t in infos if t.name not in covered]
    return [scan_table(t) for t in infos]
