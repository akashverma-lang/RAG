"""The entry point of the packaged application.

Double-clicking an icon should start the server, open the browser at it, and stay out
of the way. Everything it needs to say goes in the console window behind, which most
people will never look at -- so the important half of the messaging lives in the page
instead.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    # Under PyInstaller the working directory is wherever the icon was clicked from,
    # which is not where the bundle is. Anything resolved relative to "." would miss.
    os.chdir(Path(sys.executable).parent)

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def free_port(preferred: int) -> int:
    """The preferred port, or the next free one.

    A second copy of the app, or anything else already on 8000, should not produce a
    stack trace on startup -- it should quietly move over.
    """
    for port in [preferred] + list(range(preferred + 1, preferred + 40)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def open_when_ready(url: str, timeout: float = 90.0) -> None:
    """Open the browser once the port answers, not before.

    Opening immediately shows a connection error while the embedder is still
    loading, which reads as "the app is broken" on the one run where the user has no
    reason to think otherwise.
    """
    host, _, port = url.rsplit("/", 1)[-1].partition(":")
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, int(port))) == 0:
                break
        time.sleep(0.3)
    try:
        webbrowser.open(url)
    except Exception:                                           # noqa: BLE001
        pass


def main() -> int:
    from app import config

    port = free_port(config.PORT)
    url = f"http://{config.HOST}:{port}"

    print("=" * 62)
    print("  Local RAG - your documents, searchable, on this machine")
    print("=" * 62)
    print(f"  settings : {config.SETTINGS_FILE}")
    print(f"  documents: {config.DATA_DIR}")
    print(f"  index    : {config.STORAGE_DIR}")
    print(f"  open     : {url}")
    print("\n  Leave this window open. Close it to stop the app.\n", flush=True)

    threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    import uvicorn

    from app.main import app as application

    try:
        uvicorn.run(application, host=config.HOST, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    # Needed before anything spawns a process in a frozen build, or each child
    # re-runs the launcher and the app starts over and over.
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
