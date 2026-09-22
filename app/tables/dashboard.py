"""Saved panels, and running them.

A panel is a question you already asked and trusted: a title, the SQL that answered
it, and how to draw the result.  Running a dashboard is just running that SQL again,
which at these sizes costs milliseconds -- so panels are always live, never a cached
snapshot that quietly goes stale.

Dashboards live in their own database.  A full re-index calls catalog.reset(), which
drops every table in tables.db; a dashboard describes the data rather than deriving
from it, so it has to survive that.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import config
from app.tables import catalog, charts, filters, sql

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS dashboards (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS panels (
    id           INTEGER PRIMARY KEY,
    dashboard_id INTEGER NOT NULL REFERENCES dashboards(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    sql          TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT '',    -- '' = decide from the result shape
    width        TEXT NOT NULL DEFAULT 'half',
    position     INTEGER NOT NULL DEFAULT 0,
    question     TEXT NOT NULL DEFAULT '',
    created      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_panels_dash ON panels(dashboard_id, position);
"""

_lock = threading.RLock()
_db: sqlite3.Connection | None = None


def connect() -> sqlite3.Connection:
    global _db
    with _lock:
        if _db is None:
            Path(config.DASHBOARDS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            _db = sqlite3.connect(config.DASHBOARDS_DB_PATH,
                                  check_same_thread=False, timeout=30.0)
            _db.row_factory = sqlite3.Row
            _db.executescript(_SCHEMA)
            _migrate(_db)
            _db.commit()
        return _db


def _migrate(db: sqlite3.Connection) -> None:
    """Add columns to a dashboards.db written by an earlier version."""
    have = {r["name"] for r in db.execute("PRAGMA table_info(dashboards)")}
    for column, decl in (("filter_from", "TEXT NOT NULL DEFAULT ''"),
                         ("filter_to", "TEXT NOT NULL DEFAULT ''")):
        if column not in have:
            db.execute(f"ALTER TABLE dashboards ADD COLUMN {column} {decl}")


@dataclass
class Panel:
    id: int
    dashboard_id: int
    title: str
    sql: str
    kind: str = ""
    width: str = "half"
    position: int = 0
    question: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "dashboard_id": self.dashboard_id, "title": self.title,
                "sql": self.sql, "kind": self.kind, "width": self.width,
                "position": self.position, "question": self.question}


@dataclass
class Dashboard:
    id: int
    name: str
    filter_from: str = ""
    filter_to: str = ""
    panels: list[Panel] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "filter_from": self.filter_from,
                "filter_to": self.filter_to, "panels": [p.as_dict() for p in self.panels]}


# --------------------------------------------------------------------------- crud
def _row_to_panel(r: sqlite3.Row) -> Panel:
    return Panel(id=r["id"], dashboard_id=r["dashboard_id"], title=r["title"],
                 sql=r["sql"], kind=r["kind"], width=r["width"],
                 position=r["position"], question=r["question"])


def list_dashboards() -> list[dict]:
    db = connect()
    with _lock:
        rows = db.execute(
            "SELECT d.id, d.name, d.updated, COUNT(p.id) AS n_panels "
            "FROM dashboards d LEFT JOIN panels p ON p.dashboard_id = d.id "
            "GROUP BY d.id ORDER BY d.created").fetchall()
    return [{"id": r["id"], "name": r["name"], "n_panels": r["n_panels"],
             "updated": r["updated"]} for r in rows]


def create_dashboard(name: str) -> int:
    db = connect()
    now = time.time()
    with _lock:
        cur = db.execute("INSERT INTO dashboards(name, created, updated) VALUES(?,?,?)",
                         (name.strip()[:80] or "Dashboard", now, now))
        db.commit()
        return int(cur.lastrowid)


def default_dashboard() -> int:
    """The dashboard a pinned panel goes to when none was chosen."""
    existing = list_dashboards()
    return existing[0]["id"] if existing else create_dashboard("My dashboard")


def rename_dashboard(dash_id: int, name: str) -> None:
    db = connect()
    with _lock:
        db.execute("UPDATE dashboards SET name=?, updated=? WHERE id=?",
                   (name.strip()[:80] or "Dashboard", time.time(), dash_id))
        db.commit()


def set_filters(dash_id: int, date_from: str = "", date_to: str = "") -> None:
    """The date range applied to every panel of this dashboard."""
    db = connect()
    with _lock:
        db.execute("UPDATE dashboards SET filter_from=?, filter_to=?, updated=? WHERE id=?",
                   (date_from or "", date_to or "", time.time(), dash_id))
        db.commit()


def delete_dashboard(dash_id: int) -> None:
    db = connect()
    with _lock:
        db.execute("DELETE FROM panels WHERE dashboard_id=?", (dash_id,))
        db.execute("DELETE FROM dashboards WHERE id=?", (dash_id,))
        db.commit()


def get_dashboard(dash_id: int) -> Dashboard | None:
    db = connect()
    with _lock:
        row = db.execute("SELECT * FROM dashboards WHERE id=?", (dash_id,)).fetchone()
        if not row:
            return None
        panels = [_row_to_panel(r) for r in db.execute(
            "SELECT * FROM panels WHERE dashboard_id=? ORDER BY position, id", (dash_id,))]
    return Dashboard(id=row["id"], name=row["name"],
                     filter_from=row["filter_from"] or "", filter_to=row["filter_to"] or "",
                     panels=panels)


def add_panel(dash_id: int, title: str, statement: str, kind: str = "",
              width: str = "half", question: str = "") -> int:
    """Store a panel. The SQL is validated now *and* again every time it runs."""
    sql.validate(statement)                       # raises UnsafeSQL on anything but a SELECT
    db = connect()
    with _lock:
        n = db.execute("SELECT COUNT(*) c FROM panels WHERE dashboard_id=?",
                       (dash_id,)).fetchone()["c"]
        if n >= config.DASHBOARD_MAX_PANELS:
            raise ValueError(f"a dashboard holds at most {config.DASHBOARD_MAX_PANELS} panels")
        pos = db.execute("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM panels "
                         "WHERE dashboard_id=?", (dash_id,)).fetchone()["p"]
        cur = db.execute(
            """INSERT INTO panels(dashboard_id, title, sql, kind, width, position,
                                  question, created)
               VALUES(?,?,?,?,?,?,?,?)""",
            (dash_id, title.strip()[:120] or "Untitled", statement.strip(),
             kind if kind in charts.KINDS else "", width, pos, question.strip()[:400],
             time.time()))
        db.execute("UPDATE dashboards SET updated=? WHERE id=?", (time.time(), dash_id))
        db.commit()
        return int(cur.lastrowid)


def update_panel(panel_id: int, **fields: Any) -> None:
    allowed = {"title", "kind", "width", "position"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return
    if "kind" in sets and sets["kind"] not in charts.KINDS:
        sets["kind"] = ""
    db = connect()
    with _lock:
        db.execute(f"UPDATE panels SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?",
                   (*sets.values(), panel_id))
        db.commit()


def delete_panel(panel_id: int) -> None:
    db = connect()
    with _lock:
        db.execute("DELETE FROM panels WHERE id=?", (panel_id,))
        db.commit()


# --------------------------------------------------------------------------- running
def _coverage_note(statement: str) -> str:
    """Warn when a panel reads one file although a combined view exists.

    A dashboard that silently covers a single quarter is worse than one that says
    it does, because nothing on screen looks wrong.
    """
    try:
        infos = catalog.list_tables(with_columns=False)
    except Exception:                                           # noqa: BLE001
        return ""
    low = statement.lower()
    views = {v.name: v for v in infos if v.kind == "view"}
    for view in views.values():
        members = [m for m in view.members if m.lower() in low]
        if members and view.name.lower() not in low:
            return (f"reads {', '.join(members)} only; "
                    f"{view.name} would cover all {len(view.members)} files")
    return ""


def run_panel(panel: Panel, date_from: str = "", date_to: str = "",
              date_cols: dict[str, str] | None = None) -> dict:
    """Execute one panel and prepare it for drawing.

    The dashboard's date range is injected into the saved SQL rather than stored
    with it, so clearing the range restores exactly the query that was pinned.
    """
    statement, filtered = filters.apply_dates(panel.sql, date_from, date_to, date_cols)
    result = sql.run(statement, limit=config.PANEL_MAX_ROWS,
                     timeout=config.PANEL_SQL_TIMEOUT)
    out = panel.as_dict()
    out["took"] = result.took
    out["filtered"] = filtered
    if not result.ok:
        out["error"] = result.error
        out["chart"] = charts.Chart(kind="table", notes=[result.error]).as_dict()
        out["options"] = ["table"]
        return out

    chart = charts.infer(result.columns, result.rows,
                         catalog.column_roles(), prefer=panel.kind)
    if result.truncated:
        chart.notes.append(
            f"query matched more than {config.PANEL_MAX_ROWS} rows; showing the first")
    note = _coverage_note(panel.sql)
    if note:
        chart.notes.append(note)
    if date_from or date_to:
        span = filters.describe(date_from, date_to)
        chart.notes.append(f"date filter {span} applied" if filtered
                           else f"date filter {span} does not apply to this panel")
    out["error"] = ""
    out["chart"] = chart.as_dict()
    out["options"] = charts.options_for(chart)
    out["columns"] = result.columns
    return out


def run_dashboard(dash_id: int) -> dict | None:
    dash = get_dashboard(dash_id)
    if dash is None:
        return None
    started = time.time()
    # Resolved once for the whole board rather than per panel.
    date_cols = catalog.date_columns()
    panels = [run_panel(p, dash.filter_from, dash.filter_to, date_cols)
              for p in dash.panels]
    return {"id": dash.id, "name": dash.name, "panels": panels,
            "filter_from": dash.filter_from, "filter_to": dash.filter_to,
            "bounds": filters.bounds(), "took": round(time.time() - started, 3)}
