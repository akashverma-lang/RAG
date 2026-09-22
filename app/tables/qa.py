"""Answering a question from the tables: plan, run, and package the result.

The output deliberately looks like the vector path's output -- a context string plus
numbered sources -- so the chat endpoint can hand either one to the same model.  The
difference is what the context contains: not a sample of rows that happened to embed
near the question, but the exact result of a query over every row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from app import config
from app.tables import catalog, charts, sql


@dataclass
class TableAnswer:
    used: bool = False              # did the tables actually answer it
    mode: str = "semantic"          # sql | semantic
    sql: str = ""
    why: str = ""                   # why the model declined, when mode == semantic
    unanswerable: str = ""          # parts of the question the tables could not cover
    missing: str = ""               # a column the question needed that does not exist
    error: str = ""
    context: str = ""
    sources: list[dict] = field(default_factory=list)
    n_rows: int = 0
    truncated: bool = False
    took: float = 0.0
    query_took: float = 0.0


def _tables_in(statement: str, names: Sequence[str]) -> list[str]:
    low = statement.lower()
    return [n for n in names if n.lower() in low]


def answer(question: str, history: Sequence[dict] | None = None,
           model: str | None = None) -> TableAnswer:
    """Try to answer from SQL. A falsy .used means the caller should search documents."""
    started = time.time()
    out = TableAnswer()
    if not config.TABLES_ENABLED or not catalog.has_tables():
        return out
    # Caught here rather than at the provider: a greeting should cost nothing and
    # must never come back holding the previous question's chart.
    if sql.is_small_talk(question):
        out.why = "greeting, not a question about the data"
        return out

    try:
        plan, result = sql.plan_and_run(question, history, model)
    except Exception as exc:                                    # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
        out.took = round(time.time() - started, 2)
        return out

    out.mode, out.sql, out.why = plan.mode, plan.sql, plan.why
    out.unanswerable = plan.unanswerable
    out.missing = plan.missing
    if plan.mode != "sql" or result is None:
        out.took = round(time.time() - started, 2)
        return out
    if not result.ok:
        out.error = result.error
        out.took = round(time.time() - started, 2)
        return out

    infos = {t.name: t for t in catalog.list_tables()}
    used_tables = _tables_in(result.sql, list(infos))
    scanned = sum(infos[n].n_rows for n in used_tables if n in infos and
                  infos[n].kind == "table") or \
        sum(infos[n].n_rows for n in used_tables if n in infos)
    where = ", ".join(used_tables) or "the imported tables"

    table_md = sql.to_markdown(result)
    if not result.rows:
        body = "The query ran successfully and matched no rows."
    else:
        body = f"{table_md}\n\n{sql.describe(result)}"

    # Naming the identifier columns explicitly is the only reliable way to stop the
    # model "tidying" them: left to itself it renders sd_id 419968 as "419,968",
    # which reads as a different record.
    id_cols = sorted({c.name for n in used_tables for c in infos[n].columns
                      if c.role == "id" and c.name in result.columns})
    note = (f"\nIdentifier columns (reproduce exactly, never add separators or round): "
            f"{', '.join(id_cols)}." if id_cols else "")

    # A half-answered question has to say which half. Otherwise the user reads a
    # confident total and assumes the rest of what they asked was covered too.
    limits = (f"\n\nNOT ANSWERABLE from this data, tell the user plainly: "
              f"{plan.unanswerable}") if plan.unanswerable else ""
    out.context = (
        f"[1] SQL query over {where}"
        + (f" ({scanned:,} rows scanned)" if scanned else "")
        + f"\nQuery: {result.sql}{note}\n\n{body}{limits}"
    )
    # A readable default name, so pinning does not start with an empty box.
    preview_chart = charts.infer(result.columns, result.rows, catalog.column_roles())
    out.sources = [{
        "n": 1,
        "suggested_title": charts.suggest_title(preview_chart, question),
        "file": where,
        "rel_path": where,
        "path": "",
        "loc": f"{result.n_rows} row(s)",
        "heading": "SQL result",
        "score": 1.0,
        "kind": "sql",
        "sql": result.sql,
        "columns": result.columns,
        "rows": [list(r) for r in result.rows[:50]],
        "truncated": result.truncated,
        "scanned": scanned,
        "preview": table_md[:600],
        "unanswerable": plan.unanswerable,
    }]
    out.used = True
    out.n_rows = result.n_rows
    out.truncated = result.truncated
    out.query_took = result.took
    out.took = round(time.time() - started, 2)
    return out
