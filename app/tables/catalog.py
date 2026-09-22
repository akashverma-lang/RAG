"""The table catalogue: what exists in tables.db, and how it is described to the LLM.

Text-to-SQL lives or dies on the schema description.  A bare column list produces
queries that filter on 'west' when the data says 'ZO_West (CS)', so every column
carries its real value vocabulary: the full list for low-cardinality columns, the
range for numbers and dates, a couple of samples otherwise.

Tables live in their own database (storage/tables.db) so the vector index and its
FTS triggers are never touched by an import, and so queries can open the file
strictly read-only.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app import config

_CATALOG_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS _catalog_tables (
    name        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'table',   -- table | view
    source_path TEXT NOT NULL DEFAULT '',
    source_file TEXT NOT NULL DEFAULT '',
    sheet       TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    n_rows      INTEGER NOT NULL DEFAULT 0,
    n_cols      INTEGER NOT NULL DEFAULT 0,
    members     TEXT NOT NULL DEFAULT '',        -- JSON list, for union views
    note        TEXT NOT NULL DEFAULT '',
    imported_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS _catalog_columns (
    table_name TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    name       TEXT NOT NULL,
    label      TEXT NOT NULL,
    type       TEXT NOT NULL,
    role       TEXT NOT NULL,
    n_filled   INTEGER NOT NULL DEFAULT 0,
    n_distinct INTEGER NOT NULL DEFAULT 0,
    lo         TEXT NOT NULL DEFAULT '',
    hi         TEXT NOT NULL DEFAULT '',
    values_json TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (table_name, name)
);
"""

ROW_COL = "_row"            # original 1-based row number in the sheet
SRC_COL = "_source_file"    # which workbook a row came from, for union views


@dataclass
class ColumnInfo:
    ord: int
    name: str
    label: str
    type: str
    role: str
    n_filled: int = 0
    n_distinct: int = 0
    lo: str = ""
    hi: str = ""
    values: list[str] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    kind: str
    source_file: str
    sheet: str
    n_rows: int
    n_cols: int
    fingerprint: str = ""
    members: list[str] = field(default_factory=list)
    note: str = ""
    columns: list[ColumnInfo] = field(default_factory=list)


_lock = threading.RLock()
_db: sqlite3.Connection | None = None

# Reading the catalogue is a query per table plus a JSON parse per column, and a
# single chat request reads it three times -- to render the schema, to name the
# tables a query touched, and to tell an identifier from a measure. None of that
# changes between reads, so it is cached and thrown away whenever a write bumps the
# version. Callers get their own objects, so nobody can edit the cache by accident.
_version = 0
_cache: dict = {}


def _invalidate() -> None:
    global _version
    _version += 1
    _cache.clear()


def connect() -> sqlite3.Connection:
    """The single writable connection, guarded by _lock (see store.Store)."""
    global _db
    with _lock:
        if _db is None:
            Path(config.TABLES_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            _db = sqlite3.connect(config.TABLES_DB_PATH, check_same_thread=False, timeout=30.0)
            _db.row_factory = sqlite3.Row
            _db.executescript(_CATALOG_SCHEMA)
            _db.commit()
        return _db


def connect_readonly() -> sqlite3.Connection:
    """A fresh read-only connection -- generated SQL never gets a writable handle."""
    path = Path(config.TABLES_DB_PATH)
    if not path.exists():
        raise FileNotFoundError("no tables have been imported yet")
    uri = f"file:{path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10.0)
    con.row_factory = sqlite3.Row
    return con


def quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


# --------------------------------------------------------------------------- writes
def record_table(info: TableInfo) -> None:
    db = connect()
    with _lock:
        db.execute(
            """INSERT INTO _catalog_tables
                   (name, kind, source_path, source_file, sheet, fingerprint,
                    n_rows, n_cols, members, note, imported_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                   kind=excluded.kind, source_path=excluded.source_path,
                   source_file=excluded.source_file, sheet=excluded.sheet,
                   fingerprint=excluded.fingerprint, n_rows=excluded.n_rows,
                   n_cols=excluded.n_cols, members=excluded.members,
                   note=excluded.note, imported_at=excluded.imported_at""",
            (info.name, info.kind, info.source_file, Path(info.source_file).name or
             info.source_file, info.sheet, info.fingerprint, info.n_rows, info.n_cols,
             json.dumps(info.members), info.note, time.time()),
        )
        db.execute("DELETE FROM _catalog_columns WHERE table_name=?", (info.name,))
        db.executemany(
            """INSERT INTO _catalog_columns
                   (table_name, ord, name, label, type, role, n_filled,
                    n_distinct, lo, hi, values_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            [(info.name, c.ord, c.name, c.label, c.type, c.role, c.n_filled,
              c.n_distinct, c.lo, c.hi, json.dumps(c.values)) for c in info.columns],
        )
        db.commit()
    _invalidate()


def drop_table(name: str) -> None:
    db = connect()
    with _lock:
        row = db.execute("SELECT kind FROM _catalog_tables WHERE name=?", (name,)).fetchone()
        kind = row["kind"] if row else "table"
        db.execute(f"DROP {'VIEW' if kind == 'view' else 'TABLE'} IF EXISTS {quote(name)}")
        db.execute("DELETE FROM _catalog_tables WHERE name=?", (name,))
        db.execute("DELETE FROM _catalog_columns WHERE table_name=?", (name,))
        db.commit()
    _invalidate()


def tables_for_source(source_path: str) -> list[str]:
    db = connect()
    with _lock:
        return [r["name"] for r in db.execute(
            "SELECT name FROM _catalog_tables WHERE source_path=? AND kind='table'",
            (source_path,))]


def reset() -> None:
    """Drop every imported table -- part of 'rebuild from scratch'."""
    db = connect()
    with _lock:
        names = [(r["name"], r["kind"]) for r in
                 db.execute("SELECT name, kind FROM _catalog_tables")]
        for name, kind in names:
            db.execute(f"DROP {'VIEW' if kind == 'view' else 'TABLE'} IF EXISTS {quote(name)}")
        db.execute("DELETE FROM _catalog_tables")
        db.execute("DELETE FROM _catalog_columns")
        db.commit()
        db.execute("VACUUM")
    _invalidate()


# --------------------------------------------------------------------------- reads
def _row_to_table(r: sqlite3.Row) -> TableInfo:
    return TableInfo(
        name=r["name"], kind=r["kind"], source_file=r["source_file"], sheet=r["sheet"],
        n_rows=r["n_rows"], n_cols=r["n_cols"], fingerprint=r["fingerprint"],
        members=json.loads(r["members"] or "[]"), note=r["note"],
    )


def _copy(infos: list[TableInfo]) -> list[TableInfo]:
    """Fresh objects off the cached ones, so a caller editing one cannot poison it."""
    return [replace(t, members=list(t.members),
                    columns=[replace(c, values=list(c.values)) for c in t.columns])
            for t in infos]


def list_tables(with_columns: bool = True) -> list[TableInfo]:
    key = ("tables", with_columns)
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return _copy(hit)
    db = connect()
    with _lock:
        rows = db.execute(
            "SELECT * FROM _catalog_tables ORDER BY kind DESC, name").fetchall()
        out = [_row_to_table(r) for r in rows]
        if with_columns:
            for t in out:
                t.columns = [
                    ColumnInfo(
                        ord=c["ord"], name=c["name"], label=c["label"], type=c["type"],
                        role=c["role"], n_filled=c["n_filled"], n_distinct=c["n_distinct"],
                        lo=c["lo"], hi=c["hi"], values=json.loads(c["values_json"] or "[]"),
                    )
                    for c in db.execute(
                        "SELECT * FROM _catalog_columns WHERE table_name=? ORDER BY ord",
                        (t.name,))
                ]
        _cache[key] = out
    return _copy(out)


def has_tables() -> bool:
    if not config.TABLES_ENABLED or not Path(config.TABLES_DB_PATH).exists():
        return False
    try:
        db = connect()
        with _lock:
            row = db.execute("SELECT COUNT(*) n FROM _catalog_tables").fetchone()
        return bool(row and row["n"])
    except sqlite3.Error:
        return False


def column_roles() -> dict[str, str]:
    """Column name -> role, so a caller can tell an identifier from a measure."""
    out: dict[str, str] = {}
    try:
        for table in list_tables():
            for col in table.columns:
                out.setdefault(col.name, col.role)
    except sqlite3.Error:
        pass
    return out


def date_columns() -> dict[str, str]:
    """Table name -> its first date column, for dashboard-wide date filtering."""
    out: dict[str, str] = {}
    for table in list_tables():
        for col in table.columns:
            if col.role == "date":
                out[table.name] = col.name
                break
    return out


def stats() -> dict:
    infos = list_tables(with_columns=False)
    real = [t for t in infos if t.kind == "table"]
    return {
        "tables": len(real),
        "views": len([t for t in infos if t.kind == "view"]),
        "rows": sum(t.n_rows for t in real),
        "names": [t.name for t in infos],
    }


# --------------------------------------------------------------------------- prompt
_HINT_BUDGET = 520      # chars of value vocabulary per column


def _column_line(c: ColumnInfo) -> str:
    bits = [f"{c.name} {c.type}"]
    if c.label.lower().replace(" ", "_") != c.name:
        bits.append(f'-- "{c.label}"')
    hint = ""
    if c.role == "date" and c.lo:
        hint = f"date, {c.lo} .. {c.hi}"
    elif c.role == "number" and c.lo:
        hint = f"number, {c.lo} .. {c.hi}"
    elif c.values:
        shown: list[str] = []
        used = 0
        for v in c.values[:config.TABLE_ENUM_MAX]:
            item = repr(v)
            if used + len(item) > _HINT_BUDGET and shown:
                break
            shown.append(item)
            used += len(item) + 2
        more = "" if len(shown) >= c.n_distinct else f", ... ({c.n_distinct} distinct)"
        hint = f"one of: {', '.join(shown)}{more}"
    elif c.role == "id":
        hint = f"identifier, {c.n_distinct} distinct" + (f", e.g. {c.lo}" if c.lo else "")
    elif c.role == "text":
        hint = f"free text, {c.n_distinct} distinct"
        if c.lo:
            hint += f", e.g. {c.lo[:70]!r}"
    if hint:
        bits.append(f"[{hint}]")
    return "  " + " ".join(bits)


def render_schema(tables: list[TableInfo] | None = None) -> str:
    """The schema block handed to the model before it writes SQL.

    A union view and its members have identical columns, so spelling all of them out
    triples the prompt for no gain -- and on a free cloud tier that is the difference
    between answering and hitting a token limit.  The view is described in full; its
    members get one line each saying they share its columns.
    """
    if tables is None:
        with _lock:
            cached = _cache.get("schema")
        if cached is not None:
            return cached
    infos = tables if tables is not None else list_tables()
    if not infos:
        if tables is None:
            with _lock:
                _cache["schema"] = ""
        return ""
    covered: dict[str, str] = {}
    for t in infos:
        if t.kind == "view":
            for m in t.members:
                covered[m] = t.name
    # A cleaned view supersedes what it cleans. Spelling both out in full doubles the
    # prompt, and worse, offers the model a table whose values the cleaning has just
    # corrected -- it would filter on a spelling that no longer exists there.
    for t in infos:
        if t.name.endswith("_clean") and t.members:
            covered[t.members[0]] = t.name
    infos = ([t for t in infos if t.name.endswith("_clean")]
             + [t for t in infos if not t.name.endswith("_clean")])

    out: list[str] = []
    for t in infos:
        if t.kind == "view":
            head = f"VIEW {t.name}  -- {t.note or 'union of ' + ', '.join(t.members)}"
        else:
            head = f"TABLE {t.name}  -- from {t.source_file}"
            if t.sheet and t.sheet not in t.source_file:
                head += f" [sheet {t.sheet}]"
        out.append(f"{head}  ({t.n_rows:,} rows)")
        if t.name in covered:
            why = ("the uncorrected values" if covered[t.name].endswith("_clean")
                   else "this one file on its own")
            out.append(f"  same columns as {covered[t.name]}; query that instead "
                       f"unless you specifically need {why}.")
        else:
            out += [_column_line(c) for c in t.columns]
        out.append("")
    rendered = "\n".join(out).strip()
    if tables is None:
        with _lock:
            _cache["schema"] = rendered
    return rendered
