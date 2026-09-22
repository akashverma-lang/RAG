"""Applying a dashboard-wide date range to a panel's saved SQL.

A filter has to bite *before* aggregation, so wrapping the panel query is no use --
`SELECT * FROM (SELECT COUNT(*) ...) WHERE date > x` filters the answer, not the
rows.  Instead each known table named in the query is replaced by a filtered
subquery aliased back to the same name, which leaves every column reference,
GROUP BY and alias in the original query working untouched:

    FROM sheet_1_all           ->  FROM (SELECT * FROM "sheet_1_all"
                                         WHERE "sr_close_date" >= '2025-10-01'
                                           AND "sr_close_date" <= '2025-12-31 23:59:59'
                                        ) AS "sheet_1_all"

Only identifiers that the catalogue knows *and* that have a date column are ever
touched, so a stray word in a string literal or an unknown table is left alone.
"""
from __future__ import annotations

import re
from typing import Iterable

from app.tables import catalog

# FROM/JOIN <name> optionally followed by [AS] <alias>. The alias must not be a
# keyword that legitimately follows a table name.
_NEXT_KEYWORDS = {
    "where", "group", "order", "limit", "having", "on", "using", "join", "inner",
    "left", "right", "full", "cross", "outer", "union", "except", "intersect",
    "window", "returning", "natural", "offset",
}
_SOURCE = re.compile(
    r'\b(?P<clause>from|join)\s+'
    r'(?P<quote>["\'`\[]?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)?'
    r'(?P<tail>\s+(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?',
    re.IGNORECASE)

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STRING = re.compile(r"'(?:[^']|'')*'")


def _sub_outside_strings(pattern: re.Pattern, repl, text: str) -> str:
    """Substitute only outside single-quoted literals.

    Without this, a value like 'from sheet_1_all is just text' gets rewritten as if
    it were a FROM clause, producing SQL that is both broken and surprising.
    """
    parts: list[str] = []
    pos = 0
    for m in _STRING.finditer(text):
        parts.append(pattern.sub(repl, text[pos:m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(pattern.sub(repl, text[pos:]))
    return "".join(parts)


def _valid(value: str) -> str:
    """Accept only a plain ISO date, so nothing can be smuggled into the SQL."""
    v = (value or "").strip()
    return v if _DATE_ONLY.match(v) else ""


def describe(date_from: str, date_to: str) -> str:
    lo, hi = _valid(date_from), _valid(date_to)
    if lo and hi:
        return f"{lo} to {hi}"
    if lo:
        return f"from {lo}"
    if hi:
        return f"up to {hi}"
    return ""


def apply_dates(statement: str, date_from: str = "", date_to: str = "",
                date_cols: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Return (sql, tables_filtered). Unchanged when there is nothing to apply."""
    lo, hi = _valid(date_from), _valid(date_to)
    if not lo and not hi:
        return statement, []

    cols = catalog.date_columns() if date_cols is None else date_cols
    if not cols:
        return statement, []

    touched: list[str] = []

    def clause(col: str) -> str:
        parts = []
        if lo:
            parts.append(f'{catalog.quote(col)} >= \'{lo}\'')
        if hi:
            # Dates are stored as ISO text, so the end of the day is a plain string
            # bound -- this keeps a whole-day range inclusive.
            parts.append(f'{catalog.quote(col)} <= \'{hi} 23:59:59\'')
        return " AND ".join(parts)

    def replace(m: re.Match) -> str:
        name = m.group("name")
        col = cols.get(name)
        if not col:
            return m.group(0)
        alias, tail = m.group("alias"), ""
        if alias and alias.lower() in _NEXT_KEYWORDS:
            # It was GROUP / WHERE / ORDER, not an alias. Put it back verbatim --
            # swallowing it turned "GROUP BY 1" into "BY 1".
            tail, alias = m.group("tail"), None
        touched.append(name)
        sub = (f'(SELECT * FROM {catalog.quote(name)} WHERE {clause(col)})'
               f' AS {catalog.quote(alias or name)}')
        return f'{m.group("clause")} {sub}{tail}'

    out = _sub_outside_strings(_SOURCE, replace, statement)
    return out, sorted(set(touched))


def bounds() -> dict[str, str]:
    """The overall min/max of every date column, to seed the picker sensibly."""
    lo, hi = "", ""
    for table in catalog.list_tables():
        for col in table.columns:
            if col.role == "date" and col.lo:
                lo = min(lo, col.lo[:10]) if lo else col.lo[:10]
                hi = max(hi, col.hi[:10]) if hi else col.hi[:10]
    return {"min": lo, "max": hi}


def filterable() -> Iterable[str]:
    return catalog.date_columns().keys()
