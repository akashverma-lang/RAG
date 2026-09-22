"""Central configuration. Everything is driven by the .env file at the project root."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Where the program's own files are. A PyInstaller build unpacks them to a
# temporary folder and points sys._MEIPASS at it, so __file__ is not where the
# bundled frontend actually lives.
FROZEN = bool(getattr(sys, "frozen", False))
ROOT = (Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        if FROZEN else Path(__file__).resolve().parent.parent)

# Settings come from the user's own profile first, and only then from a .env in the
# source tree. A packaged build is often installed somewhere the user cannot write,
# so the setup screen saves next to their data instead -- see app/setup.py. Reading
# the profile copy first means a saved setting always beats a stale checkout value.
def _settings_file() -> Path:
    override = os.getenv("RAG_CONFIG_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
    elif os.name == "nt":
        base = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")) / "LocalRAG"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "LocalRAG"
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "local-rag"
    return base / "settings.env"


SETTINGS_FILE = _settings_file()
if SETTINGS_FILE.exists():
    load_dotenv(SETTINGS_FILE)
load_dotenv(ROOT / ".env")          # never overrides what was just loaded


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _list(key: str, default: str) -> list[str]:
    raw = os.getenv(key, "").strip() or default
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


# --------------------------------------------------------------------------- paths
# Defaults that work on a machine that has never run this before. Both are only
# used until the setup screen writes real ones.
_DEFAULT_DATA = str(Path.home() / "Documents")
_DEFAULT_STORAGE = str(SETTINGS_FILE.parent / "index")
DATA_DIR = Path(os.getenv("DATA_DIR", _DEFAULT_DATA).strip().strip('"')).expanduser()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", _DEFAULT_STORAGE).strip().strip('"')).expanduser()
DB_PATH = STORAGE_DIR / "index.db"
VEC_PATH = STORAGE_DIR / "vectors.f32"
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", str(STORAGE_DIR / "models")).strip()).expanduser()

# --------------------------------------------------------------------------- server
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = _int("PORT", 8000)
# Which browser origins may call the API. The page is served from this same server,
# so it needs nothing beyond localhost; "*" let any site a user happened to have
# open read their documents through this port. Set CORS_ORIGINS to widen it only if
# you are deliberately serving a front end from somewhere else.
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    f"http://localhost:{PORT},http://127.0.0.1:{PORT}").split(",") if o.strip()]
# Citation links open the source file. Only files that are actually in the index can
# be opened, so the endpoint cannot be used to read the rest of the folder.
OPEN_INDEXED_ONLY = _bool("OPEN_INDEXED_ONLY", True)
# How long the file and chunk counts may be reused before being recounted. The page
# polls health every 15 seconds and the counts only move when an index run ends, so
# recounting per poll is three full table passes for a number that has not changed --
# 173ms at 2M chunks, and seconds at a hundred million.
STATS_TTL = _float("STATS_TTL", 30.0)

# --------------------------------------------------------------------------- ingestion
INCLUDE_EXT = _list(
    "INCLUDE_EXT",
    ".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xlsm,.xls,.csv,.tsv,.txt,.md,.markdown,.rst,.log,"
    ".json,.jsonl,.yaml,.yml,.xml,.html,.htm,.eml,.msg,.epub,.rtf,.py,.js,.ts,.java,.sql,.ini,.cfg",
)
IGNORE_DIRS = set(_list("IGNORE_DIRS", ".git,node_modules,__pycache__,.venv,venv,$recycle.bin,system volume information,.idea,.vscode"))
MAX_FILE_MB = _float("MAX_FILE_MB", 200.0)
INGEST_WORKERS = _int("INGEST_WORKERS", max(4, (os.cpu_count() or 4)))
# Re-indexing a file appends new vectors and abandons the old ones, so the vector
# file grows with every run. When this much of it is dead, an index run rewrites it
# -- every search multiplies the whole matrix, so dead rows are paid for on every
# question. 0 disables the rewrite.
VECTOR_COMPACT_RATIO = _float("VECTOR_COMPACT_RATIO", 0.25)

# --------------------------------------------------------------------------- ocr
# Reads text out of pictures: standalone image files, scanned PDF pages, and images
# embedded inside Word / PowerPoint / Excel / PDF documents.
OCR_ENABLED = _bool("OCR_ENABLED", True)
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").strip().lower()   # auto|rapidocr|tesseract|off
OCR_MIN_CHARS = _int("OCR_MIN_CHARS", 60)      # PDF page with less text than this -> page OCR
OCR_IMAGE_MIN_PX = _int("OCR_IMAGE_MIN_PX", 90)     # ignore icons/bullets below this size
OCR_MAX_IMAGES_PER_FILE = _int("OCR_MAX_IMAGES_PER_FILE", 60)
OCR_MIN_TEXT_LEN = _int("OCR_MIN_TEXT_LEN", 8)      # shorter results are treated as noise
OCR_MAX_PIXELS = _int("OCR_MAX_PIXELS", 4_000_000)  # downscale anything bigger
OCR_WORKERS = _int("OCR_WORKERS", 2)                # concurrent OCR engines (RAM/CPU cap)
OCR_DPI = _int("OCR_DPI", 200)                      # render resolution for scanned pages
OCR_LANG = os.getenv("OCR_LANG", "eng").strip()     # Tesseract only
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
if OCR_ENABLED:                                    # only walk images when OCR can read them
    INCLUDE_EXT += [e for e in IMAGE_EXT if e not in INCLUDE_EXT]

# --------------------------------------------------------------------------- chunking
CHUNK_CHARS = _int("CHUNK_CHARS", 1400)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 60)

# --------------------------------------------------------------------------- embeddings
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").strip().lower()   # auto|fastembed|sentence_transformers
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()
EMBED_BATCH = _int("EMBED_BATCH", 64)
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

# --------------------------------------------------------------------------- retrieval
# On CPU-only machines the LLM spends most of its time reading the context, so
# these two settings dominate response latency. Raise them if you have a GPU.
TOP_K = _int("TOP_K", 5)                 # chunks handed to the LLM
CANDIDATES_DENSE = _int("CANDIDATES_DENSE", 40)
CANDIDATES_BM25 = _int("CANDIDATES_BM25", 40)
RRF_K = _int("RRF_K", 60)
RERANK_ENABLED = _bool("RERANK_ENABLED", True)
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2").strip()
# How many fused candidates the cross-encoder rescores. This is the biggest local
# cost in answering a question -- roughly 0.1s per candidate on a CPU -- and the
# main quality/latency dial: fewer is faster, but a passage ranked 20th by fusion
# can no longer be promoted to first. 0 keeps the old rule of thumb (4x TOP_K).
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 0)
# ...and a ceiling on what that rule of thumb may produce. The depth was derived as
# 4x top_k with nothing above it, so raising top_k in the UI quietly multiplied the
# slowest stage of the whole pipeline: top_k=50 asks for 200 candidates, and at the
# measured 155ms each that is 31 seconds of reranking for one question; top_k=200
# is over two minutes.
#
# Capping costs nothing measurable. Reranking 8 real questions at depth 32 returned
# exactly the same top 5 as depth 40, and only below 24 did the results start to
# move (16 held 89% of them, 8 held 75%). The knee is well under this ceiling.
RERANK_MAX_CANDIDATES = _int("RERANK_MAX_CANDIDATES", 48)
NEIGHBOR_WINDOW = _int("NEIGHBOR_WINDOW", 0)   # also pull N chunks either side of a hit

# --- approximate search -----------------------------------------------------
# Exact search multiplies the whole vector matrix by the query. That is the right
# answer while the matrix is small -- it is one BLAS call and it is exact -- but its
# cost is the size of the corpus, and at 20M documents the matrix is 143 GB and a
# single query would read all of it.
#
# Past ANN_MIN_VECTORS an IVF-PQ index takes over (see core/ann.py): vectors are
# clustered so a query only visits a few cells, and each is compressed from 1536
# bytes to 64, which is what lets the whole thing stay in RAM. The float32 vectors
# remain the source of truth and anything the index does not cover yet is still
# searched exactly, so results are never stale.
ANN_ENABLED = _bool("ANN_ENABLED", True)
# Below this, exact search is both faster and exact, so there is nothing to gain.
ANN_MIN_VECTORS = _int("ANN_MIN_VECTORS", 200_000)
ANN_CELLS = _int("ANN_CELLS", 0)            # 0 = about one cell per 48 vectors
# Cells visited per query: the accuracy/latency dial, and a floor.
ANN_NPROBE = _int("ANN_NPROBE", 64)
# ...and the reason it is only a floor. Accuracy tracks the *fraction* of the corpus
# a query looks at, not the number of cells it opens, and the cell count grows with
# the corpus -- so a fixed nprobe quietly scans less and less as the index grows.
# Measured: 64 probes covered 8% of a 60k corpus and returned recall 0.98, but only
# 1.1% of a 2M corpus, where it returned 0.74. Holding the fraction instead holds
# the accuracy.
#
# The honest trade this makes: a constant fraction means query cost grows with the
# corpus. Constant cost is available -- pin ANN_SCAN_FRACTION to 0 and use the
# nprobe floor alone -- but recall then falls away as the corpus grows. There is no
# setting that holds both, and this one prefers the answer staying right.
ANN_SCAN_FRACTION = _float("ANN_SCAN_FRACTION", 0.02)
# Candidates the approximate pass hands to the exact re-score. Larger costs a few
# hundred more 1.5 KB reads and buys back most of what quantisation loses.
ANN_SHORTLIST = _int("ANN_SHORTLIST", 512)
# Rebuild the index when this many vector rows have been appended since it was
# built. Those rows are searched exactly in the meantime, which is correct but gets
# slower the longer the tail grows.
ANN_REBUILD_AFTER = _int("ANN_REBUILD_AFTER", 100_000)
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 6000)
MIN_SCORE = _float("MIN_SCORE", 0.0)

# --- query expansion --------------------------------------------------------
# A question rarely uses the words the documents use. Each variant is searched
# separately and the ranked lists are fused, so a passage that only matches one
# phrasing still surfaces. Reranking stays against the original wording.
# A short question is only treated as a follow-up when it actually means something
# related to the last one. Measured with the indexing embedder: genuine follow-ups
# scored 0.519 and above against the previous question, changes of subject 0.489 and
# below, so anything in between separates them.
FOLLOW_UP_SIMILARITY = _float("FOLLOW_UP_SIMILARITY", 0.50)

EXPAND_QUERIES = _int("EXPAND_QUERIES", 3)        # 0 disables expansion entirely
EXPAND_WITH_MODEL = _bool("EXPAND_WITH_MODEL", True)
EXPAND_TIMEOUT = _float("EXPAND_TIMEOUT", 12.0)

# Deep mode. Expansion asks the same question in other words; this asks the
# *different* questions a compound question is made of, searches each on its own,
# and interleaves the shortlists so every part contributes evidence. One search can
# only have one winner, so "compare A and B" otherwise fills top_k with whichever of
# A and B the corpus covers better.
#
# It costs a planning call plus one retrieval per part. That is the point: it buys
# recall on questions a single ranked list cannot serve.
DEEP_ENABLED = _bool("DEEP_ENABLED", True)        # the default when a request says nothing
DEEP_MAX_PARTS = _int("DEEP_MAX_PARTS", 4)        # below 2, splitting means nothing
DEEP_PER_PART_K = _int("DEEP_PER_PART_K", 4)      # passages taken from each part
DEEP_TIMEOUT = _float("DEEP_TIMEOUT", 20.0)       # the planning call only
DEEP_WORKERS = _int("DEEP_WORKERS", 4)            # parts searched at once

# How much of the answer may come from the model's own knowledge:
#   grounded - documents only, refuse otherwise (the original behaviour)
#   blended  - may add general knowledge, but must label which is which
ANSWER_MODE = os.getenv("ANSWER_MODE", "blended").strip().lower()

# Whether to invite the model to draw. The page renders a subset of mermaid's
# flowchart syntax itself -- no library, no network -- so this only decides whether
# the model is told the option exists.
DIAGRAMS_ENABLED = _bool("DIAGRAMS_ENABLED", True)

# --------------------------------------------------------------------------- llm
# Several providers can be active at once; the UI lists every model they offer and
# a question can be sent to any of them. LLM_MODEL is just the default selection.
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").strip().lower()   # default provider
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()   # blank: pick the first model offered

# --- Groq: free tier, OpenAI-compatible, serves open-source models ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")

# --- Gemini: free tier, and it speaks the OpenAI protocol on this endpoint, so it
# needs no client of its own. ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai").strip().rstrip("/")

# --- any other OpenAI-compatible endpoint (LM Studio, OpenRouter, Cerebras, vLLM) ---
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_LABEL = os.getenv("OPENAI_LABEL", "Custom").strip()

MODEL_LIST_TTL = _float("MODEL_LIST_TTL", 1800.0)  # cloud /models cache: spares the daily quota
NUM_CTX = _int("NUM_CTX", 4096)
TEMPERATURE = _float("TEMPERATURE", 0.2)
MAX_TOKENS = _int("MAX_TOKENS", 1024)
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m").strip()   # keep the model resident in RAM
HISTORY_TURNS = _int("HISTORY_TURNS", 6)
# Total budget for an answer: generous, because a local model on CPU is slow.
REQUEST_TIMEOUT = _float("REQUEST_TIMEOUT", 600.0)
# But a stalled connection must never hold the request for that long. CONNECT_TIMEOUT
# bounds reaching the server and READ_TIMEOUT bounds the gap *between* streamed
# chunks, so a dead socket fails in seconds while a slow-but-alive answer still runs
# to REQUEST_TIMEOUT.
CONNECT_TIMEOUT = _float("CONNECT_TIMEOUT", 10.0)
READ_TIMEOUT = _float("READ_TIMEOUT", 90.0)

# --------------------------------------------------------------------------- tables
# Spreadsheets that are genuinely tabular are loaded into a real SQLite table and
# queried with generated SQL instead of being embedded.  Top-k retrieval can only
# ever show the model a handful of rows, so it cannot count, sum or rank; SQL can,
# exactly, and importing 100k rows takes seconds rather than hours of embedding.
TABLES_ENABLED = _bool("TABLES_ENABLED", True)
TABLES_DB_PATH = STORAGE_DIR / "tables.db"
TABLE_MIN_ROWS = _int("TABLE_MIN_ROWS", 5)        # fewer rows -> not worth a table
TABLE_MIN_COLS = _int("TABLE_MIN_COLS", 2)
TABLE_PROBE_ROWS = _int("TABLE_PROBE_ROWS", 200)  # rows read to detect shape + types
TABLE_ENUM_MAX = _int("TABLE_ENUM_MAX", 40)       # distinct values listed in the schema prompt
TABLE_MAX_ROWS = _int("TABLE_MAX_ROWS", 200)      # rows a query may hand back to the LLM
# The ceiling, as opposed to the default above. A request may ask for more rows than
# TABLE_MAX_ROWS -- the dashboard export legitimately does -- but not for an
# unbounded number: the result set is materialised in memory and serialised to JSON,
# so "give me everything" is a way to exhaust the process rather than a query.
TABLE_ROW_CEILING = _int("TABLE_ROW_CEILING", 50_000)
TABLE_SQL_TIMEOUT = _float("TABLE_SQL_TIMEOUT", 20.0)
TABLE_SQL_RETRIES = _int("TABLE_SQL_RETRIES", 1)  # re-prompts after a SQL error
# Writing a query is a small, structured reply, so it gets a much tighter budget than
# an answer.  It runs before anything can stream, so every second of it is a second
# the user stares at a spinner -- if it overruns, fall back to searching documents.
TABLE_PLAN_TIMEOUT = _float("TABLE_PLAN_TIMEOUT", 25.0)
TABLE_INSERT_BATCH = _int("TABLE_INSERT_BATCH", 5000)

# --------------------------------------------------------------------------- dashboards
# Panels are saved questions: a title, the SQL, and how to draw it.  They live in
# their own database because a full re-index resets tables.db, and a dashboard must
# survive that -- it describes the data, it is not derived from it.
DASHBOARDS_DB_PATH = STORAGE_DIR / "dashboards.db"
DASHBOARD_MAX_PANELS = _int("DASHBOARD_MAX_PANELS", 24)
PANEL_MAX_ROWS = _int("PANEL_MAX_ROWS", 500)      # rows one panel query may return
PANEL_SQL_TIMEOUT = _float("PANEL_SQL_TIMEOUT", 15.0)
# A bar chart of 29,206 engine serial numbers is an unreadable smear that also
# freezes the browser, so a categorical axis is capped and the rest reported.
CHART_MAX_CATEGORIES = _int("CHART_MAX_CATEGORIES", 15)
CHART_MAX_POINTS = _int("CHART_MAX_POINTS", 400)  # points on a line before downsampling

# --------------------------------------------------------------------------- analysis
# The analyst agent: many small structured calls to plan and query, then one long
# call to write the report.  Those are different jobs, so they get different models.
#
# AGENT_MODEL must be FAST above all -- a 20-step loop multiplies its latency by 20.
# Gemini's newer "thinking" Flash models spend 30-40s and their whole token budget on
# hidden reasoning (measured: finish_reason "length" with zero content tokens), so
# flash-lite is the deliberate choice, not a downgrade.
# The single model call that turns measured gaps into an action plan.
GAP_PLAN_TIMEOUT = _float("GAP_PLAN_TIMEOUT", 90.0)
ANALYSIS_ENABLED = _bool("ANALYSIS_ENABLED", True)
AGENT_MODEL = os.getenv("AGENT_MODEL", "custom::gemini-flash-lite-latest").strip()
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "groq::openai/gpt-oss-120b").strip()
AGENT_MAX_STEPS = _int("AGENT_MAX_STEPS", 18)      # tool calls before it must conclude
AGENT_MAX_SECONDS = _float("AGENT_MAX_SECONDS", 240.0)
AGENT_ROWS_PER_STEP = _int("AGENT_ROWS_PER_STEP", 40)   # rows one probe may return
# A step is one small JSON object, but a long "thought" plus a long query can
# still overrun the default answer budget and truncate the JSON mid-string.
AGENT_MAX_TOKENS = _int("AGENT_MAX_TOKENS", 1500)
# Free tiers are rate limited per minute, and a loop fires calls back to back.
PROVIDER_RPM = _int("PROVIDER_RPM", 12)            # per provider, client-side pacing
PROVIDER_RETRIES = _int("PROVIDER_RETRIES", 2)     # backoff attempts before failover
# Deterministic scans that need no model at all.
DIAG_OUTLIER_SIGMA = _float("DIAG_OUTLIER_SIGMA", 6.0)
DIAG_MAX_FINDINGS = _int("DIAG_MAX_FINDINGS", 60)
DIAG_FUZZY_RATIO = _float("DIAG_FUZZY_RATIO", 0.90)     # near-duplicate category labels

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
