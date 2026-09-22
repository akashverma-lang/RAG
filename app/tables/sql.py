"""Turning a question into SQL, and running that SQL safely.

The model is given the schema and asked for one of two answers: a SELECT, or an
admission that the question is not about the tables at all.  That second option is
the router -- it costs nothing extra and keeps prose questions on the vector path.

Nothing here trusts the model's output.  A generated statement is parsed, checked
to be a single read-only SELECT, and executed on a connection opened read-only,
under a wall-clock limit, with the row count capped as it is fetched.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app import config
from app.core import llm
from app.core.intent import is_follow_up, is_small_talk
from app.analysis import semantics
from app.tables import catalog

# Anything that could write, attach a file, or reach outside the query.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|"
    r"reindex|analyze|trigger|commit|rollback|begin|savepoint|load_extension)\b",
    re.IGNORECASE)
# ...except where the word is a call rather than a statement.  SQLite's scalar
# functions include replace(), and rejecting `SELECT replace(city,'  ',' ')` sends a
# perfectly safe query back to the user as "unsafe".
_FUNCTION_FORM = re.compile(r"\b(replace|trim|likelihood)\s*\(", re.IGNORECASE)
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_FENCE = re.compile(r"^```(?:json|sql)?\s*|\s*```$", re.IGNORECASE)

SQL_SYSTEM = """You answer questions about the user's spreadsheets by writing SQLite SQL.

{schema}

Reply with ONE JSON object and nothing else:
  {{"mode": "sql", "sql": "SELECT ...", "unanswerable": "..."}}
        - the tables can answer the question, or ANY PART of it. Name the parts they
          cannot answer in "unanswerable"; omit it when they answer everything.
  {{"mode": "semantic", "why": "...", "missing": "..."}}
        - NOTHING in the question can be answered from these tables: it is about \
document prose, needs meaning-based search over free text, or the data simply does \
not hold any of it.
        - Set "missing" ONLY when the question really was about this tabular data and \
a needed column is absent. Name the column that would be required and the closest one \
that exists: "no column records when a claim was raised, only sr_close_date". Leave \
"missing" out entirely when the question was never about the tables (a question about \
a report's wording is not a missing column).

Rules for the SQL:
- SQLite dialect. Exactly ONE statement. SELECT or WITH only. Never modify anything.
- Use only the tables and columns shown above, spelled exactly as given.
- Filter with the exact literal values listed for a column - they are the real
  spelling in the data. If the user's wording is close but not identical, map it to
  the listed value (for example "west" -> 'ZO_West (CS)'). For loose matching use
  LIKE with % wildcards.
- Dates are ISO text, so plain string comparison works: col >= '2025-10-01'.
  Use substr(col,1,7) for a month and substr(col,1,4) for a year.
- Prefer a view whose name ends in _all when the question is not about one single file.
- If the question asks how many / total / average / highest, return the AGGREGATE,
  not the rows. Never make the user count rows themselves.
- When listing rows, select only the columns that matter and add ORDER BY and LIMIT.
- Never SUM or AVG a column described as an identifier.
- Always alias aggregates to a readable name, e.g. COUNT(*) AS rows.

Answering only part of it:
- A question with two halves is normal. If one half is computable and the other is
  not, still return "sql" for the half that is, and say plainly in "unanswerable"
  what is missing and why - naming the column that would be needed.
  Example: asked for total value AND average resolution time, when the table records
  a close date but no open date, return the totals and set "unanswerable" to
  "average resolution time: the table has sr_close_date but no open date, so a
  duration cannot be computed".
- "semantic" is only for a question where NOTHING is computable. Declining a
  question you could half-answer leaves the user with nothing at all, which is the
  worse outcome.

When NOT to write SQL:
- A greeting, thanks, or any message with no question in it is "semantic". Never
  answer one by repeating an earlier query.
- Earlier turns are context for a follow-up only. If the new message does not
  actually ask about the data, do not reuse the previous question's query."""

RETRY_SYSTEM = """That SQL failed. Fix it and reply with the same JSON format.

{schema}

The query you sent:
{sql}

SQLite reported:
{error}

Reply with ONE JSON object: {{"mode": "sql", "sql": "..."}} or {{"mode": "semantic", \
"why": "..."}} if the tables cannot answer this after all."""


@dataclass
class Plan:
    mode: str = "semantic"       # sql | semantic
    sql: str = ""
    why: str = ""
    unanswerable: str = ""       # parts of the question the tables cannot cover
    missing: str = ""            # the column that would have been needed, and is not there
    raw: str = ""


@dataclass
class Result:
    ok: bool = False
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str = ""
    took: float = 0.0

    @property
    def n_rows(self) -> int:
        return len(self.rows)


class UnsafeSQL(ValueError):
    pass


# --------------------------------------------------------------------------- validation
def validate(sql: str) -> str:
    """Return a single safe read-only statement, or raise UnsafeSQL."""
    text = (sql or "").strip()
    text = _FENCE.sub("", text).strip()
    if not text:
        raise UnsafeSQL("empty query")

    bare = _COMMENT.sub(" ", text).strip().rstrip(";").strip()
    if not bare:
        raise UnsafeSQL("query is only comments")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeSQL("only SELECT queries are allowed")

    # Every remaining check runs on the statement with string literals blanked out.
    # 'Pending for Submission' must not trip the keyword guard, and a value like
    # 'a;b' is one statement and not two -- checking the raw text refused both, and
    # the user saw a correct query rejected for no reason they could see.
    without_strings = re.sub(r"'(?:[^']|'')*'", "''", bare)
    if ";" in without_strings:
        raise UnsafeSQL("only one statement is allowed")
    hit = _FORBIDDEN.search(_FUNCTION_FORM.sub("(", without_strings))
    if hit:
        raise UnsafeSQL(f"'{hit.group(0)}' is not allowed in a query")
    return bare


def run(sql: str, limit: int | None = None,
        timeout: float | None = None) -> Result:
    """Execute a validated SELECT read-only, capped by rows and wall clock."""
    limit = limit or config.TABLE_MAX_ROWS
    limit = max(1, min(int(limit), config.TABLE_ROW_CEILING))
    timeout = config.TABLE_SQL_TIMEOUT if timeout is None else timeout
    res = Result(sql=sql)
    started = time.perf_counter()
    try:
        safe = validate(sql)
        res.sql = safe
    except UnsafeSQL as exc:
        res.error = str(exc)
        return res

    con = None
    watchdog = None
    try:
        con = catalog.connect_readonly()
        # A timer thread calling interrupt() costs nothing while the query runs.  The
        # obvious alternative, set_progress_handler, fires a Python callback every few
        # thousand VM steps and measured 10x slower on a full scan of this data.
        watchdog = threading.Timer(timeout, con.interrupt)
        watchdog.daemon = True
        watchdog.start()
        cur = con.execute(safe)
        res.columns = [d[0] for d in (cur.description or [])]
        fetched = cur.fetchmany(limit + 1)
        res.truncated = len(fetched) > limit
        res.rows = [tuple(r) for r in fetched[:limit]]
        res.ok = True
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        res.error = ("query timed out - try narrowing it"
                     if "interrupt" in msg.lower() else msg)
    except sqlite3.Error as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    except FileNotFoundError as exc:
        res.error = str(exc)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if con is not None:
            con.close()
        res.took = round(time.perf_counter() - started, 3)
    return res


# --------------------------------------------------------------------------- planning
def _parse_plan(text: str) -> Plan:
    raw = (text or "").strip()
    body = _FENCE.sub("", raw).strip()
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(body[start:end + 1])
            mode = str(data.get("mode", "")).lower().strip()
            if mode in {"sql", "semantic"}:
                return Plan(mode=mode, sql=str(data.get("sql", "")).strip(),
                            why=str(data.get("why", "")).strip(),
                            unanswerable=str(data.get("unanswerable", "")).strip(),
                            missing=str(data.get("missing", "")).strip(),
                            raw=raw)
        except (json.JSONDecodeError, AttributeError):
            pass
    # A model that ignored the format but produced a query is still useful.
    if re.match(r"^\s*(select|with)\b", body, re.IGNORECASE):
        return Plan(mode="sql", sql=body, raw=raw)
    return Plan(mode="semantic", why="could not parse a plan", raw=raw)


def _history_messages(history: Sequence[dict] | None) -> list[dict]:
    out: list[dict] = []
    for turn in (history or [])[-4:]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role == "user" and content:
            out.append({"role": "user", "content": content[:600]})
    return out


def plan(question: str, history: Sequence[dict] | None = None,
         model: str | None = None, schema: str | None = None) -> Plan:
    """Ask the model for SQL, or for permission to fall back to vector search.

    This runs before a single token can stream, so it is deliberately impatient: if
    the provider is slow or unreachable, searching documents is a far better outcome
    than leaving the user in front of a spinner.
    """
    schema = catalog.render_schema() if schema is None else schema
    if not schema:
        return Plan(mode="semantic", why="no tables imported")
    if is_small_talk(question):
        return Plan(mode="semantic", why="greeting, not a question about the data")
    meaning = semantics.render(question=question)
    system = SQL_SYSTEM.format(schema=schema)
    if meaning:
        # Metric definitions travel with the schema so "claim value" means the same
        # expression every time somebody asks for it.
        system += f"\n\nWhat this data means (use these definitions):\n{meaning}"
    messages = [{"role": "system", "content": system}]
    # History only for a question that actually depends on it. A standalone question
    # carrying the previous turn invites the model to answer that one again.
    if is_follow_up(question):
        messages += _history_messages(history)
    messages.append({"role": "user", "content": question})
    try:
        raw = llm.complete(messages, model=model, temperature=0.0,
                           timeout=config.TABLE_PLAN_TIMEOUT)
    except Exception as exc:                                    # noqa: BLE001
        return Plan(mode="semantic", why=f"could not reach the model ({exc})")
    return _parse_plan(raw)


def plan_and_run(question: str, history: Sequence[dict] | None = None,
                 model: str | None = None) -> tuple[Plan, Result | None]:
    """Plan, execute, and re-prompt once if SQLite rejects the query."""
    schema = catalog.render_schema()
    p = plan(question, history, model, schema)
    if p.mode != "sql" or not p.sql:
        return p, None

    result = run(p.sql)
    attempts = 0
    while not result.ok and attempts < config.TABLE_SQL_RETRIES:
        attempts += 1
        messages = [{"role": "system", "content": RETRY_SYSTEM.format(
            schema=schema, sql=result.sql or p.sql, error=result.error)}]
        messages.append({"role": "user", "content": question})
        try:
            raw = llm.complete(messages, model=model, temperature=0.0,
                               timeout=config.TABLE_PLAN_TIMEOUT)
        except Exception:                                       # noqa: BLE001
            return Plan(mode="semantic", why="model unreachable during retry"), result
        retry = _parse_plan(raw)
        if retry.mode != "sql" or not retry.sql:
            return retry, result
        p = retry
        result = run(retry.sql)
    return p, result


# --------------------------------------------------------------------------- rendering
def to_markdown(result: Result, max_col: int = 90) -> str:
    """Render rows as a markdown table -- what the answering model actually reads."""
    if not result.columns:
        return ""
    head = "| " + " | ".join(result.columns) + " |"
    rule = "| " + " | ".join("---" for _ in result.columns) + " |"
    lines = [head, rule]
    for row in result.rows:
        cells = []
        for v in row:
            s = "" if v is None else str(v)
            s = s.replace("|", "\\|").replace("\n", " ")
            cells.append(s[:max_col] + ("..." if len(s) > max_col else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe(result: Result) -> str:
    """One line of provenance so the model knows how complete the rows are."""
    if result.truncated:
        return (f"{result.n_rows} rows shown (the query returned more; "
                f"only the first {result.n_rows} are included).")
    if result.n_rows == 1 and len(result.columns) == 1:
        return "Single value computed over the whole table."
    return f"{result.n_rows} row(s), the complete result of the query."
