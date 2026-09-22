#!/usr/bin/env python
"""Local RAG - single entry point.

    python rag.py doctor     check that everything is installed and reachable
    python rag.py index      read DATA_DIR into the index (add --rebuild to start over)
    python rag.py serve      start the chat UI
    python rag.py test       run the built-in self-test on sample documents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "[ ok ]", "[warn]", "[FAIL]"


# --------------------------------------------------------------------------- doctor
def cmd_doctor(_args: argparse.Namespace) -> int:
    from app import config

    problems = 0
    print("Local RAG - environment check\n" + "-" * 58)
    print(f"{OK} python {sys.version.split()[0]}")

    # --- python packages ---
    required = {
        "fastapi": "web server", "uvicorn": "web server", "httpx": "http client",
        "numpy": "vector maths", "fastembed": "embeddings",
        "pymupdf": "PDF", "docx": "Word", "pptx": "PowerPoint", "openpyxl": "Excel",
    }
    optional = {
        "rapidocr_onnxruntime": "OCR for images (pip install rapidocr-onnxruntime)",
        "bs4": "better HTML parsing (pip install beautifulsoup4)",
        "xlrd": "legacy .xls (pip install xlrd)",
        "extract_msg": "Outlook .msg (pip install extract-msg)",
        "pytesseract": "alternative OCR engine",
    }
    import importlib.util as iu

    for module, why in required.items():
        if iu.find_spec(module):
            print(f"{OK} {module:<22} {why}")
        else:
            print(f"{BAD} {module:<22} MISSING - run: pip install -r requirements.txt")
            problems += 1
    for module, why in optional.items():
        mark = OK if iu.find_spec(module) else WARN
        print(f"{mark} {module:<22} {why}")

    # --- folders ---
    print("-" * 58)
    if config.DATA_DIR.exists():
        n = sum(1 for _ in config.DATA_DIR.rglob("*") if _.is_file())
        print(f"{OK} DATA_DIR   {config.DATA_DIR}  ({n} files)")
        if n == 0:
            print(f"{WARN} the folder is empty - put your documents there")
    else:
        print(f"{BAD} DATA_DIR   {config.DATA_DIR}  DOES NOT EXIST")
        print(f"       fix: edit {ROOT / '.env'} and set DATA_DIR to your documents folder")
        problems += 1
    print(f"{OK} STORAGE    {config.STORAGE_DIR}")

    # --- ocr ---
    from app.ingestion import ocr

    if not config.OCR_ENABLED:
        print(f"{WARN} OCR        disabled (set OCR_ENABLED=true in .env)")
    elif ocr.available():
        print(f"{OK} OCR        engine: {ocr.engine_name()}")
    else:
        print(f"{WARN} OCR        no engine - run: pip install rapidocr-onnxruntime")

    # --- llm ---
    from app.core import llm

    health = llm.health()
    if health["model_ready"]:
        print(f"{OK} LLM        default: {health['model']}")
    elif health["online"]:
        print(f"{WARN} LLM        {health['hint']}")
    else:
        print(f"{BAD} LLM        {health['hint']}")
        print("       fix: open the app and add a free Groq or Gemini API key,")
        print("            or set GROQ_API_KEY / GEMINI_API_KEY in .env")
        problems += 1

    for status in health["providers"]:
        where = "local" if status["local"] else "cloud"
        if status["online"]:
            print(f"{OK} provider   {status['label']} ({where}): "
                  f"{status['n_models']} chat models")
        else:
            mark = BAD if status["local"] else WARN
            print(f"{mark} provider   {status['label']} ({where}): "
                  f"{status['error'] or 'unreachable'}")
    for model in health["models"]:
        print(f"              {model['ref']}")

    # --- index ---
    from app.core.store import get_store

    stats = get_store().stats()
    if stats["chunks"]:
        print(f"{OK} INDEX      {stats['files']} files, {stats['chunks']} chunks "
              f"({stats['failed']} failed)")
    else:
        print(f"{WARN} INDEX      empty - run: python rag.py index")

    print("-" * 58)
    print("All good - run: python rag.py serve" if not problems
          else f"{problems} problem(s) above must be fixed first")
    return 1 if problems else 0


# --------------------------------------------------------------------------- index
def cmd_index(args: argparse.Namespace) -> int:
    from app.ingestion.indexer import index_once

    summary = index_once(rebuild=args.rebuild, show_progress=not args.quiet)
    print(f"{summary['phase']}: +{summary['added']} added, ~{summary['updated']} updated, "
          f"={summary['skipped']} unchanged, -{summary['removed']} removed, "
          f"!{summary['failed']} failed, {summary['chunks']} chunks in {summary['elapsed']}s")
    if summary["failed"]:
        print("failed files are listed in the UI under Manage -> Files")
    if summary.get("error"):
        print("ERROR:", summary["error"])
        return 1
    return 0


# --------------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from app import config

    host = args.host or config.HOST
    port = args.port or config.PORT
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    # The application object rather than "app.main:app": uvicorn resolves an import
    # string by module name, which a packaged build cannot always satisfy.
    from app.main import app as application

    uvicorn.run(application, host=host, port=port, log_level="info")
    return 0


# --------------------------------------------------------------------------- test
def cmd_test(args: argparse.Namespace) -> int:
    # Each suite runs in its own process. The document self-test points DATA_DIR and
    # STORAGE_DIR at a temp folder via os.environ before app.config is imported, so
    # anything that imports config first would silently send it at the real data dir.
    import subprocess

    print("=" * 60 + "\ntable path\n" + "=" * 60, flush=True)
    failed = subprocess.call([sys.executable, str(ROOT / "tests" / "test_tables.py")])

    print("\n" + "=" * 60 + "\nanalysis\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_analysis.py")]) or failed

    # Its own process: it points DATA_DIR at a temp folder before app.config loads.
    print("\n" + "=" * 60 + "\nincremental indexing\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_indexing.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nscale: the approximate index\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_scale.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nscale: what must not grow with the corpus\n" + "=" * 60,
          flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_scale_paths.py")],
        cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\ndeep mode: splitting a question\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_deep.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nfirst-run setup\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_setup.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nconversational context\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_context.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nfinding a file on disk\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_locate.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nbusiness gaps\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_gaps.py")], cwd=str(ROOT)) or failed

    print("\n" + "=" * 60 + "\nsecurity\n" + "=" * 60, flush=True)
    failed = subprocess.call(
        [sys.executable, str(ROOT / "tests" / "test_security.py")], cwd=str(ROOT)) or failed

    # The chat page. Skipped rather than failed when node is absent, because the
    # Python side is perfectly usable without it.
    print("\n" + "=" * 60 + "\nchat page\n" + "=" * 60, flush=True)
    import shutil as _shutil
    if _shutil.which("node"):
        for script in ("check_frontend.js", "check_frontend_boot.js",
                       "check_frontend_send.js", "check_frontend_canvas.js",
                       "check_frontend_diagram.js", "check_frontend_xss.js",
                       "check_frontend_dashboard.js"):
            failed = subprocess.call(
                ["node", str(ROOT / "tests" / script)], cwd=str(ROOT)) or failed
    else:
        print("node is not installed; skipping the chat page checks")

    print("\n" + "=" * 60 + "\ndocument path\n" + "=" * 60, flush=True)
    cmd = [sys.executable, "-c",
           "import sys; from tests.run_selftest import main; "
           f"sys.exit(main(keep={bool(args.keep)}))"]
    return subprocess.call(cmd, cwd=str(ROOT)) or failed


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag", description="Local RAG - chat with your own documents, offline.")
    subs = parser.add_subparsers(dest="command")

    subs.add_parser("doctor", help="check the installation and report what is missing")

    p_index = subs.add_parser("index", help="read DATA_DIR into the search index")
    p_index.add_argument("--rebuild", action="store_true",
                         help="delete the index and read every file again")
    p_index.add_argument("--quiet", action="store_true",
                         help="no progress bar, just the closing summary")

    p_serve = subs.add_parser("serve", help="start the chat web UI")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.add_argument("--open", action="store_true", help="open a browser window")

    p_test = subs.add_parser("test", help="run the self-test on generated sample files")
    p_test.add_argument("--keep", action="store_true", help="keep the temporary test data")

    args = parser.parse_args()
    handlers = {"doctor": cmd_doctor, "index": cmd_index,
                "serve": cmd_serve, "test": cmd_test}
    if args.command not in handlers:
        parser.print_help()
        return 0
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
