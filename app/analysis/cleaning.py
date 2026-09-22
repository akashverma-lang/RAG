"""Proposing fixes for the problems the diagnostics found, and applying them safely.

Reporting that a column is 82% empty is where most tools stop.  An analyst would go
further: trim the stray whitespace, fold the two spellings of one dealer into one,
null out the placeholder that is poisoning every average, and set aside the reading
of 600,200,200,200,300 so a total means something.

Three rules govern how that is done here:

* **Nothing is destroyed.**  Every fix is an expression in a SQL view named
  ``<table>_clean``.  The raw table is never modified, so a rule that turns out to be
  wrong is undone by deleting a line, not by re-importing 200,000 rows.
* **Nothing is applied silently.**  Each proposed rule says what it does, why, and
  how many rows it touches, measured before you accept it.
* **Nothing is guessed that can be counted.**  The proposals come from the profile
  and the diagnostics, not from a model's impression of the data.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app import config
from app.analysis import diagnose
from app.tables import catalog, sql

CLEAN_SUFFIX = "_clean"

# What each rule does, in the order it is applied to a column. Order matters: trim
# before matching values, or ' West ' never equals 'West'.
KIND_ORDER = ("trim", "blank_to_null", "map_value", "null_sentinel",
              "null_outlier", "null_future_date")


@dataclass
class Rule:
    table: str
    column: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    affected: int = 0
    enabled: bool = True

    @property
    def label(self) -> str:
        if self.kind == "trim":
            return "trim surrounding and repeated spaces"
        if self.kind == "blank_to_null":
            return "treat empty text as missing"
        if self.kind == "map_value":
            pairs = self.params.get("map", {})
            first = next(iter(pairs.items()), ("", ""))
            return (f"merge {len(pairs)} spelling(s) into one "
                    f"(e.g. {first[0]!r} -> {first[1]!r})")
        if self.kind == "null_sentinel":
            return f"treat {self.params.get('value')} as missing, not as a measurement"
        if self.kind == "null_outlier":
            return f"set aside values above {self.params.get('above'):,.0f}"
        if self.kind == "null_future_date":
            return "set aside dates in the future"
        return self.kind

    def as_dict(self) -> dict:
        return {"table": self.table, "column": self.column, "kind": self.kind,
                "params": self.params, "reason": self.reason,
                "affected": self.affected, "enabled": self.enabled,
                "label": self.label}


def path() -> Path:
    return Path(config.STORAGE_DIR) / "cleaning.yaml"


# --------------------------------------------------------------------------- storage
def load() -> list[Rule]:
    try:
        raw = yaml.safe_load(path().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    out: list[Rule] = []
    for entry in (raw.get("rules") or []):
        if not isinstance(entry, dict) or not entry.get("table"):
            continue
        out.append(Rule(
            table=str(entry["table"]), column=str(entry.get("column", "")),
            kind=str(entry.get("kind", "")), params=dict(entry.get("params") or {}),
            reason=str(entry.get("reason", "")),
            affected=int(entry.get("affected", 0) or 0),
            enabled=bool(entry.get("enabled", True))))
    return [r for r in out if r.kind in KIND_ORDER]


def save(rules: list[Rule]) -> None:
    body = {"rules": [{"table": r.table, "column": r.column, "kind": r.kind,
                       "params": r.params, "reason": r.reason,
                       "affected": r.affected, "enabled": r.enabled}
                      for r in rules]}
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        "# Fixes applied as a SQL view. The raw table is never modified:\n"
        "# delete a rule here and the next rebuild drops it.\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True,
                         default_flow_style=False), encoding="utf-8")


def save_text(text: str) -> None:
    yaml.safe_load(text)
    file = path()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


def raw_text() -> str:
    try:
        return path().read_text(encoding="utf-8")
    except OSError:
        return ""


# --------------------------------------------------------------------------- propose
def _count(statement: str) -> int:
    res = sql.run(statement, limit=1)
    return int(res.rows[0][0]) if res.ok and res.rows and res.rows[0] else 0


def _propose_text(table: str, col: catalog.ColumnInfo) -> list[Rule]:
    q = catalog.quote
    out: list[Rule] = []
    untidy = _count(f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} IS NOT NULL "
                    f"AND {q(col.name)} <> TRIM({q(col.name)})")
    if untidy:
        out.append(Rule(table, col.name, "trim", reason=(
            f"{untidy:,} values have surrounding whitespace, so they group as "
            f"separate categories from the same value without it"), affected=untidy))
    blanks = _count(f"SELECT COUNT(*) FROM {q(table)} "
                    f"WHERE TRIM(COALESCE({q(col.name)}, '')) = '' "
                    f"AND {q(col.name)} IS NOT NULL")
    if blanks:
        out.append(Rule(table, col.name, "blank_to_null", reason=(
            f"{blanks:,} values are empty text rather than missing, which COUNT() "
            f"treats as present"), affected=blanks))
    return out


def _propose_duplicates(table: str, col: catalog.ColumnInfo) -> list[Rule]:
    """Fold spellings that are the same words into whichever is most common."""
    if col.role != "category" or not 2 <= col.n_distinct <= 400:
        return []
    q = catalog.quote
    res = sql.run(f"SELECT {q(col.name)}, COUNT(*) c FROM {q(table)} "
                  f"WHERE {q(col.name)} IS NOT NULL AND {q(col.name)} <> '' "
                  f"GROUP BY 1 ORDER BY c DESC", limit=400)
    if not res.ok:
        return []
    groups: dict[frozenset[str], list[tuple[str, int]]] = {}
    for value, count in res.rows:
        groups.setdefault(diagnose._tokens(str(value)), []).append((str(value), count))
    mapping: dict[str, str] = {}
    moved = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = members[0][0]              # the most common spelling wins
        for value, count in members[1:]:
            mapping[value] = canonical
            moved += count
    if not mapping:
        return []
    return [Rule(table, col.name, "map_value", params={"map": mapping}, reason=(
        f"{len(mapping)} spelling(s) covering {moved:,} rows are the same value "
        f"written differently; totals grouped by this column are split between them"),
        affected=moved)]


def _propose_numeric(table: str, col: catalog.ColumnInfo,
                     findings: list[diagnose.Finding]) -> list[Rule]:
    q = catalog.quote
    out: list[Rule] = []
    for f in findings:
        if f.column != col.name:
            continue
        if f.kind == "sentinel":
            out.append(Rule(table, col.name, "null_sentinel",
                            params={"value": f.value}, reason=(
                f"{f.count:,} rows hold the placeholder {f.value}; counted as a real "
                f"measurement it drags every average towards it"), affected=f.count))
        elif f.kind == "outlier":
            above = _outlier_cut(table, col)
            if above:
                n = _count(f"SELECT COUNT(*) FROM {q(table)} "
                           f"WHERE {q(col.name)} > {above}")
                out.append(Rule(table, col.name, "null_outlier",
                                params={"above": above}, reason=(
                    f"{n:,} value(s) sit far outside the rest of the column "
                    f"(max {f.value:,.0f}); they distort any total or average"),
                    affected=n))
    return out


def _outlier_cut(table: str, col: catalog.ColumnInfo) -> float:
    """A threshold well clear of the real data: 100x the 99th percentile."""
    q = catalog.quote
    n = _count(f"SELECT COUNT({q(col.name)}) FROM {q(table)}")
    if n < 100:
        return 0.0
    res = sql.run(f"SELECT {q(col.name)} FROM {q(table)} "
                  f"WHERE {q(col.name)} IS NOT NULL ORDER BY {q(col.name)} "
                  f"LIMIT 1 OFFSET {int(n * 0.99) - 1}", limit=1)
    p99 = float(res.rows[0][0]) if res.ok and res.rows and res.rows[0][0] else 0.0
    return round(p99 * 100.0, 4) if p99 > 0 else 0.0


def _propose_dates(table: str, col: catalog.ColumnInfo) -> list[Rule]:
    if col.role != "date":
        return []
    q = catalog.quote
    today = dt.date.today().isoformat()
    n = _count(f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} > '{today}'")
    if not n:
        return []
    return [Rule(table, col.name, "null_future_date", params={"after": today}, reason=(
        f"{n:,} rows are dated after today, which breaks every trend and any "
        f"'last N days' filter"), affected=n)]


def propose(table: str) -> list[Rule]:
    """Everything worth fixing in this table, measured. Nothing is applied."""
    info = next((t for t in catalog.list_tables() if t.name == table), None)
    if info is None:
        return []
    findings = diagnose.scan_table(info).findings
    rules: list[Rule] = []
    for col in info.columns:
        if col.name.startswith("_"):
            continue
        if col.type == "TEXT":
            rules += _propose_text(table, col)
            rules += _propose_duplicates(table, col)
            rules += _propose_dates(table, col)
        else:
            rules += _propose_numeric(table, col, findings)
    return [r for r in rules if r.affected > 0]


# --------------------------------------------------------------------------- apply
def _expression(column: str, rules: list[Rule]) -> str:
    """Fold this column's rules into one SQL expression, innermost first."""
    q = catalog.quote
    expr = q(column)
    for kind in KIND_ORDER:
        for rule in [r for r in rules if r.kind == kind and r.enabled]:
            if kind == "trim":
                # Collapse runs of spaces as well: 'Electro Controls -   Pune' and
                # 'Electro Controls - Pune' are one dealer.
                expr = (f"TRIM(REPLACE(REPLACE(REPLACE({expr}, '  ', ' '), "
                        f"'  ', ' '), '  ', ' '))")
            elif kind == "blank_to_null":
                expr = f"NULLIF({expr}, '')"
            elif kind == "map_value":
                cases = "".join(
                    f" WHEN {_lit(k)} THEN {_lit(v)}"
                    for k, v in (rule.params.get("map") or {}).items())
                if cases:
                    expr = f"CASE {expr}{cases} ELSE {expr} END"
            elif kind == "null_sentinel":
                expr = f"NULLIF({expr}, {_num(rule.params.get('value'))})"
            elif kind == "null_outlier":
                expr = (f"CASE WHEN {expr} > {_num(rule.params.get('above'))} "
                        f"THEN NULL ELSE {expr} END")
            elif kind == "null_future_date":
                expr = (f"CASE WHEN {expr} > {_lit(str(rule.params.get('after')))} "
                        f"THEN NULL ELSE {expr} END")
    return expr


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _num(value: Any) -> str:
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return "0"


def view_name(table: str) -> str:
    return f"{table}{CLEAN_SUFFIX}"


def build(table: str, rules: list[Rule] | None = None) -> dict:
    """Create or replace the cleaned view for one table."""
    info = next((t for t in catalog.list_tables() if t.name == table), None)
    if info is None:
        raise ValueError(f"no such table: {table}")
    rules = [r for r in (rules if rules is not None else load())
             if r.table == table and r.enabled]
    name = view_name(table)
    if not rules:
        drop(table)
        return {"view": "", "rules": 0, "columns": 0}

    q = catalog.quote
    by_column: dict[str, list[Rule]] = {}
    for r in rules:
        by_column.setdefault(r.column, []).append(r)

    select: list[str] = []
    for col in info.columns:
        expr = (_expression(col.name, by_column[col.name])
                if col.name in by_column else q(col.name))
        select.append(f"{expr} AS {q(col.name)}")
    for extra in (catalog.ROW_COL, catalog.SRC_COL):
        if any(c.name == extra for c in info.columns):
            continue
        select.insert(0, q(extra))

    db = catalog.connect()
    with catalog._lock:                                         # noqa: SLF001
        db.execute(f"DROP VIEW IF EXISTS {q(name)}")
        db.execute(f"CREATE VIEW {q(name)} AS SELECT {', '.join(select)} "
                   f"FROM {q(table)}")
        db.commit()

    # Registered in the catalogue, and profiled against the cleaned values -- so the
    # schema the model sees describes the data it will actually be querying. An
    # enum listing a value the cleaning just merged away would send it looking for
    # rows that no longer exist under that spelling.
    from app.tables import importer

    fixed = ", ".join(sorted({r.column for r in rules}))
    described = catalog.TableInfo(
        name=name, kind="view", source_file=info.source_file, sheet=info.sheet,
        n_rows=info.n_rows, n_cols=info.n_cols, fingerprint=info.fingerprint,
        members=[table],
        note=(f"cleaned version of {table}: {len(rules)} fix(es) applied to "
              f"{fixed}. Same rows, corrected values. Prefer this one."))
    det = [importer.detect.Column(ord=c.ord, label=c.label, name=c.name,
                                  type=c.type, role=c.role) for c in info.columns]
    described.columns = importer.profile(name, det)
    catalog.record_table(described)
    return {"view": name, "rules": len(rules), "columns": len(by_column)}


def drop(table: str) -> None:
    catalog.drop_table(view_name(table))


def build_all(rules: list[Rule] | None = None) -> list[dict]:
    rules = load() if rules is None else rules
    tables = sorted({r.table for r in rules if r.enabled})
    return [build(t, rules) for t in tables]


def cleaned_tables() -> set[str]:
    """Tables that currently have a cleaned view."""
    db = catalog.connect()
    with catalog._lock:                                         # noqa: SLF001
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name LIKE ?",
            (f"%{CLEAN_SUFFIX}",)).fetchall()
    return {r["name"][:-len(CLEAN_SUFFIX)] for r in rows}


def impact(table: str) -> dict:
    """What actually changed, measured against the raw table."""
    name = view_name(table)
    if table not in cleaned_tables():
        return {}
    started = time.perf_counter()
    info = next((t for t in catalog.list_tables() if t.name == table), None)
    if info is None:
        return {}
    q = catalog.quote
    # _row is the row number inside its own sheet, so it repeats across the members of
    # a union view. Joining on it alone pairs every row with its namesake in the other
    # file and reports the entire table as changed.
    has_source = _count(f"SELECT COUNT(*) FROM pragma_table_info('{table}') "
                        f"WHERE name = '{catalog.SRC_COL}'") > 0
    on = f"a.{q(catalog.ROW_COL)} = b.{q(catalog.ROW_COL)}"
    if has_source:
        on += f" AND a.{q(catalog.SRC_COL)} IS b.{q(catalog.SRC_COL)}"

    out: list[dict] = []
    for col in info.columns:
        if col.name.startswith("_"):
            continue
        changed = _count(
            f"SELECT COUNT(*) FROM {q(table)} a JOIN {q(name)} b ON {on} "
            f"WHERE COALESCE(CAST(a.{q(col.name)} AS TEXT), '@') <> "
            f"COALESCE(CAST(b.{q(col.name)} AS TEXT), '@')")
        if changed:
            before = _count(f"SELECT COUNT(DISTINCT {q(col.name)}) FROM {q(table)}")
            after = _count(f"SELECT COUNT(DISTINCT {q(col.name)}) FROM {q(name)}")
            out.append({"column": col.name, "label": col.label, "changed": changed,
                        "distinct_before": before, "distinct_after": after})
    return {"view": name, "columns": out,
            "took": round(time.perf_counter() - started, 2)}
