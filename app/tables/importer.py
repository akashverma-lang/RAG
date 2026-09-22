"""Loading a tabular sheet into SQLite, then describing it accurately.

Rows are streamed and inserted in batches, so a 100k-row workbook costs seconds and
a bounded amount of memory -- versus hours of embedding for a result that still
could not answer "how many" or "what is the total".

After the load, every column is profiled against the *whole* table rather than the
200-row detection sample.  That is what makes generated SQL correct: the model gets
told that claim_status is one of {'Approved', 'Rejected', ...} instead of guessing.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from app import config
from app.tables import catalog, detect
from app.tables.catalog import ROW_COL, SRC_COL, ColumnInfo, TableInfo, quote

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


# --------------------------------------------------------------------------- coercion
def _coerce(value: Any, col: detect.Column) -> Any:
    """Turn one spreadsheet cell into the value SQLite should store."""
    text = detect.as_text(value)
    if not text:
        return None
    if col.role == "date":
        return detect.parse_date(text) or text
    if col.type == "INTEGER":
        n = detect.parse_number(text)
        return int(n) if n is not None else None
    if col.type == "REAL":
        return detect.parse_number(text)
    return text


# --------------------------------------------------------------------------- load
def _create_table(db: sqlite3.Connection, name: str, cols: Sequence[detect.Column]) -> None:
    defs = ", ".join(f"{quote(c.name)} {c.type}" for c in cols)
    db.execute(f"DROP TABLE IF EXISTS {quote(name)}")
    db.execute(
        f"CREATE TABLE {quote(name)} ("
        f"{quote(ROW_COL)} INTEGER, {quote(SRC_COL)} TEXT, {defs})"
    )


def _index_table(db: sqlite3.Connection, name: str, cols: Sequence[detect.Column]) -> None:
    """Index the columns people actually filter and group by."""
    for c in cols:
        if c.role in {"category", "date", "id"}:
            idx = f"idx_{name}_{c.name}"[:60]
            db.execute(
                f"CREATE INDEX IF NOT EXISTS {quote(idx)} ON {quote(name)} ({quote(c.name)})")


def load_sheet(path: Path, probe: detect.SheetProbe, table: str,
               on_progress: Progress = _noop) -> int:
    """Stream one sheet into a fresh SQLite table. Returns the row count."""
    db = catalog.connect()
    cols = probe.columns
    n_cols = len(cols)
    placeholders = ",".join("?" * (n_cols + 2))
    insert = f"INSERT INTO {quote(table)} VALUES({placeholders})"
    source = path.name

    with catalog._lock:                                     # noqa: SLF001 - same module family
        _create_table(db, table, cols)
        batch: list[tuple] = []
        total = 0
        for i, raw in enumerate(detect.iter_rows(path, probe.sheet), start=1):
            if i <= probe.header_row:
                continue
            if not any(detect.as_text(v) for v in raw):
                continue
            row = [raw[j] if j < len(raw) else None for j in range(n_cols)]
            batch.append((i, source, *(_coerce(v, c) for v, c in zip(row, cols))))
            if len(batch) >= config.TABLE_INSERT_BATCH:
                db.executemany(insert, batch)
                total += len(batch)
                batch.clear()
                on_progress(f"{table}: {total:,} rows")
        if batch:
            db.executemany(insert, batch)
            total += len(batch)
        _index_table(db, table, cols)
        db.commit()
    return total


# --------------------------------------------------------------------------- profile
def profile(table: str, cols: Sequence[detect.Column]) -> list[ColumnInfo]:
    """Measure every column across the whole table so the schema prompt is truthful."""
    db = catalog.connect()
    parts: list[str] = ["COUNT(*) AS n_all"]
    for i, c in enumerate(cols):
        q = quote(c.name)
        parts += [f"COUNT({q}) AS f{i}", f"COUNT(DISTINCT {q}) AS d{i}",
                  f"MIN({q}) AS lo{i}", f"MAX({q}) AS hi{i}"]
    with catalog._lock:                                     # noqa: SLF001
        agg = db.execute(f"SELECT {', '.join(parts)} FROM {quote(table)}").fetchone()

    out: list[ColumnInfo] = []
    for i, c in enumerate(cols):
        info = ColumnInfo(ord=c.ord, name=c.name, label=c.label, type=c.type, role=c.role,
                          n_filled=agg[f"f{i}"] or 0, n_distinct=agg[f"d{i}"] or 0)
        lo, hi = agg[f"lo{i}"], agg[f"hi{i}"]

        # The 200-row sample can call a column categorical that has thousands of
        # distinct values across the full sheet -- fix that now we can count properly.
        if info.role == "category" and info.n_distinct > config.TABLE_ENUM_MAX:
            info.role = "text"
        elif info.role == "text" and 0 < info.n_distinct <= config.TABLE_ENUM_MAX:
            info.role = "category"

        if info.role in {"number", "date"}:
            info.lo, info.hi = ("" if lo is None else str(lo)), ("" if hi is None else str(hi))
        elif info.role == "category" and info.n_distinct:
            with catalog._lock:                             # noqa: SLF001
                info.values = [str(r[0]) for r in db.execute(
                    f"SELECT DISTINCT {quote(c.name)} FROM {quote(table)} "
                    f"WHERE {quote(c.name)} IS NOT NULL AND {quote(c.name)} <> '' "
                    f"ORDER BY 1 LIMIT ?", (config.TABLE_ENUM_MAX,))]
        else:                                               # id / free text -> one sample
            with catalog._lock:                             # noqa: SLF001
                row = db.execute(
                    f"SELECT {quote(c.name)} FROM {quote(table)} "
                    f"WHERE {quote(c.name)} IS NOT NULL AND {quote(c.name)} <> '' LIMIT 1"
                ).fetchone()
            info.lo = "" if not row or row[0] is None else str(row[0])
        out.append(info)
    return out


# --------------------------------------------------------------------------- views
def _common_prefix(names: Sequence[str]) -> str:
    if not names:
        return ""
    first, last = min(names), max(names)
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i].rstrip("_")


def rebuild_union_views(on_progress: Progress = _noop) -> list[str]:
    """Expose one queryable view per group of tables that share a schema.

    Two quarterly exports with identical headers are one dataset as far as a
    question is concerned; without the view the model has to remember to UNION them
    and usually answers about a single quarter instead.
    """
    db = catalog.connect()
    with catalog._lock:                                     # noqa: SLF001
        for r in db.execute("SELECT name FROM _catalog_tables WHERE kind='view'").fetchall():
            db.execute(f"DROP VIEW IF EXISTS {quote(r['name'])}")
        db.execute("DELETE FROM _catalog_columns WHERE table_name IN "
                   "(SELECT name FROM _catalog_tables WHERE kind='view')")
        db.execute("DELETE FROM _catalog_tables WHERE kind='view'")
        db.commit()
    # Dropped straight out of the tables rather than through drop_table, so the
    # cached catalogue still lists views that no longer exist until it is told.
    catalog._invalidate()                                   # noqa: SLF001
    with catalog._lock:                                     # noqa: SLF001
        groups: dict[str, list[sqlite3.Row]] = {}
        for r in db.execute(
                "SELECT * FROM _catalog_tables WHERE kind='table' ORDER BY name").fetchall():
            groups.setdefault(r["fingerprint"], []).append(r)

    made: list[str] = []
    for fp, rows in groups.items():
        if len(rows) < 2:
            continue
        members = [r["name"] for r in rows]
        base = _common_prefix(members) or "combined"
        view = f"{base}_all"[:60]
        if view in members:
            view = f"{base}_union"[:60]

        cols = catalog.list_tables(with_columns=True)
        by_name = {t.name: t for t in cols}
        template = by_name.get(members[0])
        if not template or not template.columns:
            continue
        col_names = [ROW_COL, SRC_COL] + [c.name for c in template.columns]
        select_list = ", ".join(quote(n) for n in col_names)
        body = " UNION ALL ".join(
            f"SELECT {select_list} FROM {quote(m)}" for m in members)
        with catalog._lock:                                 # noqa: SLF001
            db.execute(f"DROP VIEW IF EXISTS {quote(view)}")
            db.execute(f"CREATE VIEW {quote(view)} AS {body}")
            db.commit()

        det_cols = [detect.Column(ord=c.ord, label=c.label, name=c.name,
                                  type=c.type, role=c.role) for c in template.columns]
        info = TableInfo(
            name=view, kind="view", source_file=", ".join(r["source_file"] for r in rows),
            sheet="", n_rows=sum(r["n_rows"] for r in rows), n_cols=len(template.columns),
            fingerprint=fp, members=members,
            note=(f"all {len(members)} sheets with this schema combined; "
                  f"{SRC_COL} says which file a row came from. "
                  f"Query this for anything spanning more than one file."),
        )
        info.columns = profile(view, det_cols)
        catalog.record_table(info)
        made.append(view)
        on_progress(f"view {view} over {len(members)} tables")
    return made


# --------------------------------------------------------------------------- entry
def import_file(path: Path, probe: detect.FileProbe | None = None,
                on_progress: Progress = _noop) -> dict:
    """Import every tabular sheet of one spreadsheet. Returns a small summary.

    The caller can pass a probe it already computed -- the indexer detects on its
    worker pool, and re-opening an 11 MB workbook just to re-detect is wasteful.
    """
    started = time.time()
    if probe is None:
        probe = detect.probe_file(path)
    if not probe.ok:
        reasons = "; ".join(f"{r.sheet or '?'}: {r.reason}" for r in probe.rejected)
        return {"ok": False, "tables": [], "rows": 0, "reason": reasons or "not tabular"}

    # The catalogue is keyed on the resolved path, which is what the indexer looks a
    # file up by.  Recording the unresolved one instead meant a second import never
    # found the first, so the old tables were left behind next to the new ones.
    source = str(path.resolve())
    for name in catalog.tables_for_source(source):          # replace a previous import
        catalog.drop_table(name)

    single = len(probe.sheets) == 1
    made: list[str] = []
    rows_total = 0
    for sheet in probe.sheets:
        table = detect.table_name_for(path, sheet.sheet, single)
        on_progress(f"loading {table}")
        n = load_sheet(path, sheet, table, on_progress)
        info = TableInfo(
            name=table, kind="table", source_file=source, sheet=sheet.sheet,
            n_rows=n, n_cols=len(sheet.columns), fingerprint=sheet.fingerprint,
        )
        info.columns = profile(table, sheet.columns)
        catalog.record_table(info)
        made.append(table)
        rows_total += n
        on_progress(f"{table}: {n:,} rows loaded")

    return {"ok": True, "tables": made, "rows": rows_total,
            "skipped": [f"{r.sheet}: {r.reason}" for r in probe.rejected],
            "took": round(time.time() - started, 2)}


def is_candidate(path: Path) -> bool:
    # .xls is absent on purpose: openpyxl cannot read the legacy format, so those
    # files keep the text path rather than failing an import.
    return config.TABLES_ENABLED and path.suffix.lower() in {
        ".xlsx", ".xlsm", ".csv", ".tsv"}
