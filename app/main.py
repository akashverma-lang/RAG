"""FastAPI server: one chat page + a small JSON/SSE API."""
from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
import sys
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app import setup as first_run
from app.core import embed, intent, llm, router
from app.core.store import get_store
from app.ingestion import ocr
from app.ingestion.indexer import get_indexer
from app.analysis import agent as analyst
from app.analysis import cleaning
from app.analysis import gaps as gap_finder
from app.analysis import semantics
from app.retrieval import decompose
from app.retrieval import locate
from app.retrieval import search as retrieval
from app.tables import catalog as table_catalog
from app.tables import dashboard as dashboards
from app.tables import qa as table_qa
from app.tables import sql as table_sql

FRONTEND = config.ROOT / "frontend"

# The two halves of answering a question -- planning SQL and searching documents --
# do not depend on each other, so they run together rather than one after the other.
# A pool, not a thread per request, because creating one costs more than the work
# when the answer turns out to be a greeting.
_side_work = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def warm() -> None:
        try:
            print(f"[startup] embedder ready: {embed.warmup()}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            print(f"[startup] embedder not ready: {exc}", flush=True)
        # Off the request path on purpose: it is a no-op when the index is already
        # in step, and a one-off build for a database that predates it.
        try:
            n = get_store().rebuild_name_index()
            if n:
                print(f"[startup] file-name index built for {n:,} files", flush=True)
        except Exception as exc:                                # noqa: BLE001
            print(f"[startup] file-name index skipped: {exc}", flush=True)

    def warm_providers() -> None:
        # Opens the connection to each provider before the page asks for the model
        # list, so the first /api/health is already warm rather than paying TLS
        # setup for three endpoints while the user waits.
        try:
            llm.refresh_providers(force=True)
            print(f"[startup] providers ready: "
                  f"{[p['key'] for p in llm.health()['providers'] if p['online']]}",
                  flush=True)
        except Exception as exc:                                # noqa: BLE001
            print(f"[startup] provider probe failed: {exc}", flush=True)

    threading.Thread(target=warm, name="warmup", daemon=True).start()
    threading.Thread(target=warm_providers, name="providers", daemon=True).start()
    print(f"[startup] data dir : {config.DATA_DIR}", flush=True)
    print(f"[startup] storage  : {config.STORAGE_DIR}", flush=True)
    print(f"[startup] open     : http://{config.HOST}:{config.PORT}", flush=True)
    yield


app = FastAPI(title="Local RAG", version="1.0.0", docs_url="/api/docs", lifespan=lifespan)
# Not "*": the page is served from this very server, so it needs no cross-origin
# access at all. Allowing every origin meant any site open in the same browser could
# call this API and read the user's documents out of it.
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- models
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=200)
    rerank: bool | None = None
    model: str | None = Field(default=None, max_length=200)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    tables: bool | None = None      # False forces document search only
    files: list[str] = Field(default_factory=list, max_length=2000)   # rel_paths
    deep: bool | None = None        # None = follow config.DEEP_ENABLED


class AnalyzeRequest(BaseModel):
    question: str = Field(default="", max_length=2000)
    table: str = Field(default="", max_length=200)
    # Several files at once: a gap only becomes interesting when the same measure is
    # compared across everything that reports it.
    tables: list[str] = Field(default_factory=list, max_length=50)
    model: str | None = Field(default=None, max_length=200)


class OpportunityRequest(BaseModel):
    tables: list[str] = Field(default_factory=list, max_length=50)
    model: str | None = Field(default=None, max_length=200)


class CleaningRequest(BaseModel):
    table: str = ""
    rules: list[dict] | None = None      # omit to use what is already saved


class SemanticsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200000)


class SqlRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)
    limit: int | None = Field(default=None, ge=1, le=config.TABLE_ROW_CEILING)


class DashboardRequest(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    filter_from: str | None = None
    filter_to: str | None = None


class PanelRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    sql: str = Field(min_length=1, max_length=20000)
    kind: str = Field(default="", max_length=40)
    width: str = Field(default="half", max_length=20)
    question: str = Field(default="", max_length=4000)
    dashboard_id: int | None = Field(default=None, ge=1)


class PanelPatch(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    kind: str | None = Field(default=None, max_length=40)
    width: str | None = Field(default=None, max_length=20)
    position: int | None = Field(default=None, ge=0, le=10000)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=200)
    rerank: bool | None = None


class IndexRequest(BaseModel):
    rebuild: bool = False


class SetupRequest(BaseModel):
    data_dir: str = Field(default="", max_length=4000)
    storage_dir: str = Field(default="", max_length=4000)
    groq_key: str = Field(default="", max_length=400)
    gemini_key: str = Field(default="", max_length=400)
    model: str = Field(default="", max_length=200)


class BrowseRequest(BaseModel):
    title: str = Field(default="Choose a folder", max_length=120)


# --------------------------------------------------------------------------- page
@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    """The chat page, read from disk on every request.

    Sent with no-store because the file is edited in place and the server does not
    restart to pick it up. Without the header the browser applies its own heuristic
    caching to a 200 with no validators, and a plain reload can keep showing the
    previous version of the page indefinitely, which looks exactly like a change
    that did not take.
    """
    page = FRONTEND / "index.html"
    if not page.exists():
        raise HTTPException(500, "frontend/index.html is missing")
    return HTMLResponse(
        page.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


# --------------------------------------------------------------------------- setup
@app.get("/api/setup")
def setup_state() -> dict:
    """What the first-run screen shows: where things point, and what is missing."""
    return first_run.state()


@app.post("/api/setup/browse")
def setup_browse(req: BrowseRequest) -> dict:
    """Open a native folder chooser on this machine.

    The browser cannot give a server a filesystem path, and this server is on the
    same machine as the browser, so it opens the dialog itself. Typing a path still
    works, which is the only option when the app is reached from another machine.
    """
    try:
        return {"ok": True, "path": first_run.pick_folder(req.title)}
    except Exception as exc:                                    # noqa: BLE001
        return {"ok": False, "path": "", "error": str(exc)}


@app.post("/api/setup")
def setup_save(req: SetupRequest) -> dict:
    """Validate and save the settings, then say whether a restart is needed.

    Paths and keys are read once at import into module constants that the whole app
    closes over, so they cannot be swapped underneath a running process safely. The
    honest thing is to save, report it, and have the caller restart.
    """
    values = {
        "DATA_DIR": req.data_dir.strip().strip('"'),
        "STORAGE_DIR": req.storage_dir.strip().strip('"'),
        "GROQ_API_KEY": req.groq_key.strip(),
        "GEMINI_API_KEY": req.gemini_key.strip(),
        "LLM_MODEL": req.model.strip(),
    }
    # A blank key means "leave the saved one alone", so that a masked field the user
    # did not touch does not wipe a working key.
    saved = first_run.read_env()
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if not values[key]:
            values[key] = saved.get(key, getattr(config, key, "") or "")
    if not values["STORAGE_DIR"]:
        values["STORAGE_DIR"] = str(config.STORAGE_DIR)

    problems = first_run.validate(values)
    if problems:
        return {"ok": False, "problems": problems}

    path = first_run.write_env(values)
    return {"ok": True, "saved_to": str(path), "restart_required": True}


@app.post("/api/setup/restart")
def setup_restart() -> dict:
    """Restart so the new settings take effect.

    The process re-executes itself in the background and the page polls until the
    port answers again. Done from a thread so this request can return first --
    replacing the process while it is still writing a response would look to the
    browser like a crash.
    """
    def go() -> None:
        time.sleep(0.4)
        try:
            if getattr(sys, "frozen", False):
                # argv[0] is already the executable in a packaged build, so passing
                # sys.argv whole would hand it its own path twice as an argument.
                os.execv(sys.executable, [sys.executable] + sys.argv[1:])
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as exc:                                # noqa: BLE001
            # Nothing else can be done from in here; exiting non-zero lets a
            # supervising launcher start it again, and the page says what happened.
            print(f"[setup] could not restart automatically: {exc}", flush=True)
            os._exit(3)

    threading.Thread(target=go, name="restart", daemon=True).start()
    return {"ok": True}


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204, content="")


# --------------------------------------------------------------------------- info
@app.get("/api/health")
def health() -> dict:
    store = get_store()
    return {
        "llm": llm.health(),
        "index": store.stats(),
        "ocr": ocr.info(),
        "data_dir": str(config.DATA_DIR),
        "data_dir_exists": config.DATA_DIR.exists(),
        "embed_model": config.EMBED_MODEL,
        "embed_backend": config.EMBED_BACKEND,
        "rerank": config.RERANK_ENABLED,
        "top_k": config.TOP_K,
        "indexing": get_indexer().running,
        "ann": store.ann_stats(),
        "tables": table_catalog.stats() if config.TABLES_ENABLED else {"tables": 0, "rows": 0},
        "analysis": {"enabled": config.ANALYSIS_ENABLED, **router.describe()},
        "setup": first_run.state(),
    }


@app.get("/api/files")
def files(q: str = "", limit: int = Query(300, le=2000)) -> dict:
    return {"files": get_store().list_files(limit=limit, query=q)}


@app.get("/api/open")
def open_file(path: str) -> FileResponse:
    """Serve an indexed source file, for the links under an answer's citations.

    Two gates, because one was not enough. Staying inside DATA_DIR stops the obvious
    traversal, but it still left every other file in that folder readable through the
    port -- the tax return sitting next to the documents, never indexed, never
    mentioned. So the file must also be one the index actually knows about.
    """
    target = Path(path).resolve()
    root = config.DATA_DIR.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(403, "file is outside DATA_DIR")
    if config.OPEN_INDEXED_ONLY and not get_store().is_indexed(str(target)):
        raise HTTPException(403, "that file is not in the index")
    if not target.is_file():
        raise HTTPException(404, "file not found")
    media, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media or "application/octet-stream",
                        filename=target.name)


# --------------------------------------------------------------------------- indexing
@app.post("/api/index")
def start_index(req: IndexRequest) -> dict:
    ix = get_indexer()
    if not ix.start(rebuild=req.rebuild):
        return {"started": False, "reason": "an index run is already in progress"}
    return {"started": True}


@app.get("/api/index/status")
def index_status() -> dict:
    return get_indexer().progress.snapshot()


@app.post("/api/index/cancel")
def cancel_index() -> dict:
    get_indexer().cancel()
    return {"cancelled": True}


# --------------------------------------------------------------------------- search
@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    hits = retrieval.search(req.question, top_k=req.top_k, use_rerank=req.rerank)
    return {
        "hits": [
            {k: v for k, v in h.items() if k != "text"} | {"preview": h["text"][:400]}
            for h in hits
        ]
    }


# --------------------------------------------------------------------------- tables
@app.get("/api/tables")
def tables() -> dict:
    """Everything that was loaded into SQL, with the schema the model is shown."""
    if not config.TABLES_ENABLED:
        return {"enabled": False, "tables": [], "schema": ""}
    infos = table_catalog.list_tables()
    return {
        "enabled": True,
        "stats": table_catalog.stats(),
        "schema": table_catalog.render_schema(infos),
        "tables": [
            {
                "name": t.name, "kind": t.kind, "source_file": Path(t.source_file).name,
                "sheet": t.sheet, "n_rows": t.n_rows, "n_cols": t.n_cols,
                "members": t.members, "note": t.note,
                "columns": [
                    {"name": c.name, "label": c.label, "type": c.type, "role": c.role,
                     "n_distinct": c.n_distinct, "lo": c.lo, "hi": c.hi,
                     "values": c.values[:12]}
                    for c in t.columns
                ],
            }
            for t in infos
        ],
    }


@app.post("/api/sql")
def run_sql(req: SqlRequest) -> dict:
    """Run a read-only SELECT yourself -- useful for checking a generated query."""
    if not config.TABLES_ENABLED:
        raise HTTPException(400, "tables are disabled")
    result = table_sql.run(req.sql, limit=req.limit)
    return {
        "ok": result.ok, "sql": result.sql, "error": result.error,
        "columns": result.columns, "rows": [list(r) for r in result.rows],
        "truncated": result.truncated, "took": result.took,
    }


# --------------------------------------------------------------------------- dashboards
@app.get("/api/dashboards")
def list_dashboards() -> dict:
    return {"dashboards": dashboards.list_dashboards()}


@app.post("/api/dashboards")
def create_dashboard(req: DashboardRequest) -> dict:
    return {"id": dashboards.create_dashboard(req.name or "Dashboard")}


@app.patch("/api/dashboards/{dash_id}")
def update_dashboard(dash_id: int, req: DashboardRequest) -> dict:
    """Rename, or set the date range applied to every panel."""
    if req.name is not None:
        dashboards.rename_dashboard(dash_id, req.name)
    if req.filter_from is not None or req.filter_to is not None:
        dashboards.set_filters(dash_id, req.filter_from or "", req.filter_to or "")
    return {"ok": True}


@app.delete("/api/dashboards/{dash_id}")
def delete_dashboard(dash_id: int) -> dict:
    dashboards.delete_dashboard(dash_id)
    return {"ok": True}


@app.get("/api/dashboards/{dash_id}")
def run_dashboard(dash_id: int) -> dict:
    """Run every panel and return the data ready to draw. Panels are always live."""
    out = dashboards.run_dashboard(dash_id)
    if out is None:
        raise HTTPException(404, "no such dashboard")
    return out


@app.post("/api/panels")
def add_panel(req: PanelRequest) -> dict:
    dash_id = req.dashboard_id or dashboards.default_dashboard()
    try:
        panel_id = dashboards.add_panel(
            dash_id, req.title, req.sql, kind=req.kind, width=req.width,
            question=req.question)
    except table_sql.UnsafeSQL as exc:
        raise HTTPException(400, f"refused: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": panel_id, "dashboard_id": dash_id}


@app.patch("/api/panels/{panel_id}")
def patch_panel(panel_id: int, req: PanelPatch) -> dict:
    dashboards.update_panel(panel_id, **req.model_dump(exclude_none=True))
    return {"ok": True}


@app.delete("/api/panels/{panel_id}")
def remove_panel(panel_id: int) -> dict:
    dashboards.delete_panel(panel_id)
    return {"ok": True}


# --------------------------------------------------------------------------- semantics
@app.get("/api/semantics")
def get_semantics() -> dict:
    """What the columns mean and which direction is good. Plain YAML, editable."""
    data = semantics.load(force=True)
    return {
        **semantics.status(),
        "exists": semantics.exists(),
        "path": str(semantics.path()),
        "text": semantics.raw_text(),
        "error": data.get("error", ""),
        "domain": data.get("domain", ""),
        "glossary": [{"term": g.term, "means": g.means, "also": g.also}
                     for g in data["glossary"]],
        "tables": [
            {"name": t.name, "description": t.description, "grain": t.grain,
             "outcomes": [{"column": o.column, "good": o.good, "bad": o.bad}
                          for o in t.outcomes],
             "metrics": [{"name": m.name, "sql": m.sql, "direction": m.direction,
                          "means": m.means} for m in t.metrics]}
            for t in data["tables"].values()],
    }


@app.put("/api/semantics")
def put_semantics(req: SemanticsRequest) -> dict:
    try:
        semantics.save(req.text)
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not save: {exc}") from exc
    return {"ok": True}


@app.post("/api/semantics/draft")
def draft_semantics() -> dict:
    """A first version written from the catalogue. One model call; edit afterwards."""
    try:
        return {"text": semantics.draft()}
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not draft: {exc}") from exc


@app.post("/api/semantics/update")
def update_semantics() -> dict:
    """Describe only the tables that are new, keeping every existing line.

    This is the path for "I added another spreadsheet": a full re-draft would throw
    away the corrections that make the file worth having.
    """
    try:
        out = semantics.update()
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not update: {exc}") from exc
    return out


@app.post("/api/semantics/document-terms")
def document_terms() -> dict:
    """Glossary entries read out of the indexed documents themselves.

    Spreadsheet values give the codes in the data; only the documents give the
    vocabulary of the documents, and those are the terms that make a question miss.
    """
    try:
        addition = semantics.draft_document_terms()
        merged = semantics.merge(semantics.raw_text(), addition)             if semantics.exists() else addition
        return {"text": merged, "changed": merged != semantics.raw_text()}
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not read terms: {exc}") from exc


# --------------------------------------------------------------------------- cleaning
@app.get("/api/cleaning")
def get_cleaning(table: str = "") -> dict:
    """Saved fixes, and which tables currently have a cleaned view."""
    rules = [r for r in cleaning.load() if not table or r.table == table]
    return {"rules": [r.as_dict() for r in rules],
            "cleaned": sorted(cleaning.cleaned_tables()),
            "path": str(cleaning.path()), "text": cleaning.raw_text()}


@app.post("/api/cleaning/propose")
def propose_cleaning(req: CleaningRequest) -> dict:
    """Everything worth fixing in a table, measured. Nothing is applied."""
    table = req.table or next(
        (t.name for t in table_catalog.list_tables(with_columns=False)
         if t.kind == "view" and not t.name.endswith(cleaning.CLEAN_SUFFIX)), "")
    if not table:
        raise HTTPException(400, "no table to examine")
    try:
        rules = cleaning.propose(table)
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not examine {table}: {exc}") from exc
    return {"table": table, "rules": [r.as_dict() for r in rules]}


@app.post("/api/cleaning/apply")
def apply_cleaning(req: CleaningRequest) -> dict:
    """Save the rules and rebuild the cleaned view. The raw table is untouched."""
    try:
        if req.rules is not None:
            rules = [cleaning.Rule(
                table=str(r.get("table", "")), column=str(r.get("column", "")),
                kind=str(r.get("kind", "")), params=dict(r.get("params") or {}),
                reason=str(r.get("reason", "")),
                affected=int(r.get("affected", 0) or 0),
                enabled=bool(r.get("enabled", True))) for r in req.rules]
            keep = [r for r in cleaning.load()
                    if r.table not in {x.table for x in rules}]
            cleaning.save(keep + rules)
        built = cleaning.build_all()
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(400, f"could not apply: {exc}") from exc
    return {"built": built, "cleaned": sorted(cleaning.cleaned_tables())}


@app.get("/api/cleaning/impact")
def cleaning_impact(table: str) -> dict:
    """What actually changed, measured against the raw table."""
    return cleaning.impact(table) or {"columns": []}


# --------------------------------------------------------------------------- analysis
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> StreamingResponse:
    """Investigate the data: measure, probe, then report. Streams its own steps."""
    def generate() -> Iterator[str]:
        if not config.ANALYSIS_ENABLED:
            yield _sse({"type": "error", "text": "analysis is disabled"})
            return
        if not table_catalog.has_tables():
            yield _sse({"type": "error", "text":
                        "No spreadsheets have been loaded as tables yet."})
            return
        try:
            scope = req.table or (req.tables[0] if req.tables else "")
            for event in analyst.run(req.question, scope, req.model):
                yield _sse(event)
        except Exception as exc:                                # noqa: BLE001
            yield _sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.post("/api/opportunities")
def opportunities(req: OpportunityRequest) -> StreamingResponse:
    """Measured gaps across the chosen tables, then a plan for each.

    Streamed in two stages on purpose. Finding the gaps is arithmetic over the user's
    rows and takes a few seconds; writing the plans is a model call. The gaps go out
    the moment they exist, so the page has something real on it -- with the prize
    already computed -- while the plans are still being written.
    """
    def generate() -> Iterator[str]:
        started = time.time()
        try:
            if not config.TABLES_ENABLED or not table_catalog.has_tables():
                yield _sse({"type": "error", "text":
                            "No spreadsheet data is imported, so there is nothing to "
                            "compare. Index a folder containing spreadsheets first."})
                return

            yield _sse({"type": "status", "text": "measuring every segment"})
            found = gap_finder.find(req.tables or None)
            if not found:
                yield _sse({"type": "error", "text":
                            "No measurable gap was found. That happens when the data "
                            "has no comparable operating units, or when every segment "
                            "is already within 8% of the best quarter."})
                return

            yield _sse({"type": "gaps",
                        "gaps": [g.as_dict() for g in found],
                        "took": round(time.time() - started, 2)})

            yield _sse({"type": "status",
                        "text": f"working out how to close {len(found)} gaps"})
            plans = gap_finder.plan(found, req.model)
            yield _sse({"type": "plans", "plans": plans})
            yield _sse({"type": "done", "took": round(time.time() - started, 2),
                        "n_gaps": len(found), "n_plans": len(plans)})
        except Exception as exc:                                # noqa: BLE001
            yield _sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/api/diagnostics")
def diagnostics(table: str = "") -> dict:
    """The deterministic scan on its own -- no model calls, no waiting."""
    from app.analysis import diagnose, trends

    diags = diagnose.scan(table)
    excluded: set[str] = set()
    for d in diags:
        excluded |= trends.bad_measures(d)
    return {"diagnostics": [d.as_dict() for d in diags],
            "trends": [t.as_dict() for t in trends.scan(table, exclude=excluded)],
            "excluded": sorted(excluded)}


# --------------------------------------------------------------------------- chat
def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _answer(messages: list[dict], req: "ChatRequest", notices: list[str]):
    """Stream the answer, moving to another model rather than failing outright.

    The chosen model used to be the only one tried, so a free tier answering "rate
    limit reached" at the wrong moment turned a working question into an error
    message. The user is told which model actually replied -- being moved silently
    is its own kind of wrong answer.
    """
    def moved(failed: str, nxt: str, exc: Exception) -> None:
        notices.append(f"{failed} could not answer ({exc}). Using {nxt} instead.")

    return router.stream(messages, role="answer", model=req.model,
                         temperature=req.temperature, fallback=True,
                         on_failover=moved)


def _drain(notices: list[str]) -> Iterator[str]:
    while notices:
        yield _sse({"type": "notice", "text": notices.pop(0)})


def _log_if_failed(fut: Future) -> None:
    """A search whose answer came from SQL instead is never read. Say so if it broke."""
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        print(f"[chat] background search failed: {type(exc).__name__}: {exc}", flush=True)


def _data_note(table: "table_qa.TableAnswer | None", context: str) -> str:
    """Why the spreadsheets could not answer, when that is worth saying."""
    if not table:
        return ""
    if table.missing:
        # A named missing column means the question really was about the data, so
        # say so even when document search then found passages that look relevant.
        return (table.why or f"there is no {table.missing} in the data").strip()
    if not context and table.why:
        return table.why.strip()
    return ""


def _scope_has_tables(rel_paths: list[str]) -> bool:
    """Is any chosen file one that was loaded as a SQL table?"""
    if not rel_paths:
        return True
    try:
        sources = {Path(t.source_file).name for t in table_catalog.list_tables(
            with_columns=False) if t.source_file}
    except Exception:                                           # noqa: BLE001
        return True
    return any(Path(p).name in sources for p in rel_paths)


def _index_summary() -> str:
    """A plain description of what is searchable, for answering 'what can you do'."""
    lines: list[str] = []
    stats = get_store().stats()
    if stats["chunks"]:
        exts = ", ".join(f"{e['n']} {e['ext'] or 'file'}" for e in stats["by_ext"][:6])
        lines.append(f"{stats['files']} documents ({exts}) searchable by meaning.")
    if config.TABLES_ENABLED:
        infos = table_catalog.list_tables()
        # A union view already stands for its members; listing both just repeats
        # the same columns back at the model.
        covered = {m for t in infos if t.kind == "view" for m in t.members}
        for t in infos:
            if t.name in covered:
                continue
            cols = ", ".join(c.label for c in t.columns[:10])
            lines.append(f"Spreadsheet table '{t.name}' with {t.n_rows:,} rows. "
                         f"Columns include: {cols}.")
    return "\n".join(lines) or "Nothing has been indexed yet."


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def generate() -> Iterator[str]:
        started = time.time()
        notices: list[str] = []
        try:
            # A greeting is not a search. Sent down either path it produced a
            # confident answer about whatever happened to rank highest, so it is
            # answered directly, from what is indexed rather than from the index.
            if intent.is_small_talk(req.question):
                yield _sse({"type": "sources", "sources": [], "took": 0.0,
                            "n_hits": 0, "route": "chat"})
                messages = llm.build_greeting_messages(
                    req.question, _index_summary(), req.history)
                for piece in _answer(messages, req, notices):
                    yield _sse({"type": "token", "text": piece})
                yield _sse({"type": "done", "took": round(time.time() - started, 2),
                            "route": "chat"})
                return

            # "Where is that file?" is answered from the index's own file table, not
            # from the contents of any document. Neither of the paths below can do
            # it: SQL reads inside spreadsheets and retrieval reads inside documents,
            # while the answer here is the row that says where the file came from.
            #
            # The reply is assembled in code rather than written by the model. A path
            # is the one answer where a plausible invention is worse than none: it
            # sends someone to a folder that does not exist, and it does not look
            # wrong until they get there.
            if locate.is_location_question(req.question):
                yield _sse({"type": "status", "text": "looking through your folders"})
                hits = locate.find(req.question)
                if hits:
                    text = locate.render(req.question, hits)
                    yield _sse({"type": "sources",
                                "sources": locate.sources_for(hits),
                                "took": round(time.time() - started, 2),
                                "n_hits": len(hits), "route": "files"})
                    yield _sse({"type": "status", "text": ""})
                    # Chunked rather than sent whole, so the page renders it the way
                    # it renders every other answer.
                    for i in range(0, len(text), 120):
                        yield _sse({"type": "token", "text": text[i:i + 120]})
                    yield _sse({"type": "done",
                                "took": round(time.time() - started, 2),
                                "route": "files"})
                    return
                # Nothing matched. If a file name was typed out, say so plainly and
                # stop: searching the documents would answer a question about where a
                # file is with a passage from inside a different one.
                typed = locate.named_in(req.question)
                if typed:
                    name = typed[0]
                    yield _sse({"type": "sources", "sources": [], "took":
                                round(time.time() - started, 2), "n_hits": 0,
                                "route": "files"})
                    msg = (f"No indexed file is named **{name}**.\n\n"
                           f"It may not be inside `{config.DATA_DIR}`, or it may not "
                           f"have been indexed yet. Manage \u2192 Files lists "
                           f"everything that was read, and Index files picks up "
                           f"anything new.")
                    for i in range(0, len(msg), 120):
                        yield _sse({"type": "token", "text": msg[i:i + 120]})
                    yield _sse({"type": "done", "took":
                                round(time.time() - started, 2), "route": "files"})
                    return
                yield _sse({"type": "notice", "text":
                            "No indexed file matched that. Searching the documents "
                            "instead."})

            # Structured questions ("how many", "total by region") go to SQL, which
            # reads every row.  The vector path can only ever see top_k chunks, so it
            # would answer those from a handful of rows and sound just as certain.
            # A scope naming only documents means the spreadsheets were not chosen,
            # so the SQL path is skipped rather than answering about something the
            # user just excluded.
            scoped_tables = _scope_has_tables(req.files) if req.files else True
            table: table_qa.TableAnswer | None = None
            use_tables = (req.tables is not False and scoped_tables
                          and table_catalog.has_tables())

            deep = config.DEEP_ENABLED if req.deep is None else req.deep

            # Both paths start now.  Writing the SQL is a model round trip and so is
            # rewriting the query for retrieval; run in sequence the user waits for
            # the sum of them, and the second one only begins once the first has
            # already decided it cannot help.  Whichever path loses is simply
            # discarded -- the wasted work is local CPU, not the user's time.
            #
            # In deep mode the speculative search is still worth starting, but as
            # raw hits rather than a finished answer: the deep path needs the
            # original wording as one part among several, and running it here means
            # it happens *while* the planning call is in flight instead of after it.
            searching: Future | None = None
            deep_base: Future | None = None
            try:
                if deep:
                    deep_base = _side_work.submit(
                        retrieval.search, req.question,
                        top_k=config.DEEP_PER_PART_K, use_rerank=req.rerank,
                        history=req.history, files=req.files or None)
                    deep_base.add_done_callback(_log_if_failed)
                else:
                    searching = _side_work.submit(
                        retrieval.retrieve, req.question, top_k=req.top_k,
                        use_rerank=req.rerank, history=req.history,
                        files=req.files or None)
                    searching.add_done_callback(_log_if_failed)
            except RuntimeError:                # pool shut down: fall back to inline
                searching = deep_base = None
            if use_tables:
                yield _sse({"type": "status", "text": "checking your spreadsheets"})
                table = table_qa.answer(req.question, req.history, req.model)

            if table and table.used:
                yield _sse({"type": "sources", "sources": table.sources,
                            "took": round(time.time() - started, 2),
                            "n_hits": table.n_rows, "route": "sql", "sql": table.sql})
                yield _sse({"type": "status", "text": "thinking"})
                messages = llm.build_table_messages(req.question, table.context,
                                                    req.history)
                first = True
                for piece in _answer(messages, req, notices):
                    if first:
                        yield from _drain(notices)
                        yield _sse({"type": "status", "text": ""})
                        first = False
                    yield _sse({"type": "token", "text": piece})
                yield _sse({"type": "done", "took": round(time.time() - started, 2),
                            "route": "sql"})
                return

            if table and table.error:
                yield _sse({"type": "notice", "text":
                            f"SQL query failed ({table.error}); searching documents instead."})

            # Deep mode: ask what the question is actually made of, then search each
            # part on its own. Every way this can go wrong -- the planning call
            # fails, the question does not split, the parts match nothing -- leaves
            # `parts` empty and falls through to the ordinary single search below,
            # so the worst case costs one call and still answers.
            parts: list[str] = []
            if deep:
                yield _sse({"type": "status", "text": "working out the parts"})
                parts = decompose.plan(req.question, model=req.model,
                                       history=req.history)

            context = ""
            sources: list[dict] = []
            hits: list[dict] = []
            coverage: list[dict] = []
            if parts:
                yield _sse({"type": "status",
                            "text": f"searching {len(parts)} parts separately"})
                yield _sse({"type": "plan", "parts": parts})
                try:
                    base = None
                    if deep_base is not None:
                        try:
                            base = deep_base.result()
                        except Exception:       # noqa: BLE001
                            base = None         # let it be searched again below
                    context, sources, hits, coverage = decompose.retrieve(
                        req.question, parts, top_k=req.top_k, use_rerank=req.rerank,
                        history=req.history, files=req.files or None,
                        base_hits=base)
                except Exception as exc:                        # noqa: BLE001
                    print(f"[deep] falling back to one search: {exc}", flush=True)
                    parts = []
                if parts and not context:
                    parts = []          # nothing found part-wise; let the whole
                                        # question try, it may still match something

            if not parts:
                yield _sse({"type": "status", "text": "searching your documents"})
                if deep_base is not None:
                    # Deep planning declined to split. The speculative search of the
                    # whole question has been running the whole time and is exactly
                    # what the plain path would have produced.
                    try:
                        base = deep_base.result()
                        passages = retrieval.expand_with_neighbors(base)
                        context, sources = retrieval.build_context(passages)
                        hits = base
                    except Exception as exc:    # noqa: BLE001
                        print(f"[deep] base search failed: {exc}", flush=True)
                        context, sources, hits = retrieval.retrieve(
                            req.question, top_k=req.top_k, use_rerank=req.rerank,
                            history=req.history, files=req.files or None)
                elif searching is None:
                    context, sources, hits = retrieval.retrieve(
                        req.question, top_k=req.top_k, use_rerank=req.rerank,
                        history=req.history, files=req.files or None)
                else:
                    context, sources, hits = searching.result()
                    searching = None

            took = round(time.time() - started, 2)
            yield _sse({"type": "sources", "sources": sources, "took": took,
                        "n_hits": len(hits), "route": "vector",
                        "parts": parts, "coverage": coverage})
            if not context:
                stats = get_store().stats()
                if not stats["chunks"]:
                    hint = ("The index is empty. Click Index files to scan "
                            f"{config.DATA_DIR}.")
                    if table_catalog.has_tables():
                        hint = ("No document text matched, and the spreadsheet tables "
                                "could not answer this either. Try naming a column or "
                                "value from the sheet.")
                    yield _sse({"type": "error", "text": hint})
                    return
            yield _sse({"type": "status", "text": "thinking"})
            messages = llm.build_messages(
                req.question, context, req.history,
                knowledge=semantics.render(question=req.question)[:1500],
                # "missing" is the signal, "why" is the sentence: the first says the
                # question really was about the tables and a column is absent, the
                # second explains it in words worth showing. A question about a
                # report's wording sets no "missing", so a good document answer is
                # never prefaced with an irrelevant note about SQL.
                data_note=_data_note(table, context),
                plan=parts)
            first = True
            for piece in _answer(messages, req, notices):
                if first:
                    yield from _drain(notices)
                    yield _sse({"type": "status", "text": ""})
                    first = False
                yield _sse({"type": "token", "text": piece})
            yield _sse({"type": "done", "took": round(time.time() - started, 2),
                        "route": "vector"})
        except Exception as exc:                                # noqa: BLE001
            yield _sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT,
                reload=False, log_level="info")


if __name__ == "__main__":
    run()
