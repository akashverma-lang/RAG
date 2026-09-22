"""First-run configuration: where the documents are, and which model answers.

Someone who installs this gets a folder of files and no idea what a .env is. The four
things they must supply -- the folder to read, where to keep the index, and at least
one free API key -- are asked for in the browser on first run and written to a
settings file for them.

The file lives beside the user's own data, not beside the program. A packaged build
usually ends up somewhere the user cannot write to, so a .env next to the executable
would fail to save on exactly the machines this exists for.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Keys the setup screen owns. Everything else in the file is left untouched when it
# saves, so hand-tuned settings survive a trip through the form.
FIELDS = ("DATA_DIR", "STORAGE_DIR", "GROQ_API_KEY", "GEMINI_API_KEY", "LLM_MODEL")
SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY")

_LINE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$")


def config_dir() -> Path:
    """Where the settings file belongs on this machine.

    Under the user's own profile rather than the install directory: Program Files is
    not writable, and a portable copy on a USB stick should not scribble on itself
    either.
    """
    override = os.getenv("RAG_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "LocalRAG"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LocalRAG"
    return Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "local-rag"


def env_path() -> Path:
    return config_dir() / "settings.env"


def dev_env_path() -> Path:
    """The .env in the source tree, used when running from a checkout."""
    return Path(__file__).resolve().parent.parent / ".env"


def active_env_path() -> Path:
    """Whichever file is actually in force: the user's if it exists, else the dev one."""
    user = env_path()
    if user.exists():
        return user
    dev = dev_env_path()
    return dev if dev.exists() else user


def read_env(path: Path | None = None) -> dict[str, str]:
    path = path or active_env_path()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def write_env(values: dict[str, str]) -> Path:
    """Merge these keys into the settings file, leaving every other line alone."""
    path = env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Seed from whatever is currently in force, so the first save from a source
    # checkout carries the existing tuning across rather than starting bare.
    existing_lines: list[str] = []
    source = active_env_path()
    if source.exists():
        existing_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()

    remaining = dict(values)
    out: list[str] = []
    for raw in existing_lines:
        m = _LINE.match(raw) if not raw.lstrip().startswith("#") else None
        if m and m.group(1) in remaining:
            key = m.group(1)
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(raw)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# written by the setup screen")
        out += [f"{k}={v}" for k, v in remaining.items()]

    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    # The file holds API keys. On POSIX make it owner-only; Windows inherits the
    # profile directory's permissions, which are already per-user.
    try:
        if not sys.platform.startswith("win"):
            path.chmod(0o600)
    except OSError:
        pass
    return path


def mask(value: str) -> str:
    """Enough of a key to recognise it, never enough to use it."""
    v = (value or "").strip()
    if not v:
        return ""
    return f"{v[:4]}…{v[-4:]} ({len(v)} chars)" if len(v) > 12 else "•" * len(v)


def state() -> dict:
    """What the setup screen needs to draw itself."""
    from app import config

    env = read_env()
    data_dir = Path(config.DATA_DIR)
    has_key = bool(config.GROQ_API_KEY or config.GEMINI_API_KEY
                   or config.OPENAI_API_KEY)
    return {
        "configured": bool(data_dir.exists() and has_key),
        "reasons": [r for r in (
            None if data_dir.exists() else "the documents folder does not exist",
            None if has_key else "no API key has been added yet",
        ) if r],
        "settings_file": str(env_path()),
        "data_dir": str(config.DATA_DIR),
        "data_dir_exists": data_dir.exists(),
        "storage_dir": str(config.STORAGE_DIR),
        "groq_key": mask(env.get("GROQ_API_KEY", config.GROQ_API_KEY)),
        "gemini_key": mask(env.get("GEMINI_API_KEY", config.GEMINI_API_KEY)),
        "has_groq": bool(config.GROQ_API_KEY),
        "has_gemini": bool(config.GEMINI_API_KEY),
        "model": config.LLM_MODEL,
    }


def validate(values: dict[str, str]) -> list[str]:
    """Everything wrong with these settings, in words the user can act on."""
    problems: list[str] = []

    data = (values.get("DATA_DIR") or "").strip()
    if not data:
        problems.append("Choose the folder holding your documents.")
    else:
        p = Path(data).expanduser()
        if not p.exists():
            problems.append(f"That documents folder does not exist: {p}")
        elif not p.is_dir():
            problems.append(f"That is a file, not a folder: {p}")

    store = (values.get("STORAGE_DIR") or "").strip()
    if store:
        p = Path(store).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"Cannot write to the index folder: {exc}")

    if not (values.get("GROQ_API_KEY") or "").strip() and \
       not (values.get("GEMINI_API_KEY") or "").strip():
        problems.append("Add a Groq or a Gemini API key. Both have a free tier.")
    return problems


def pick_folder(title: str = "Choose a folder") -> str:
    """A native folder chooser.

    The browser cannot hand a server a filesystem path, and this server runs on the
    same machine as the browser, so it opens the dialog itself. Typing a path stays
    available for anyone running this headless or over a remote session, where no
    dialog can appear.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:                                    # noqa: BLE001
        raise RuntimeError(f"no folder chooser on this machine ({exc})") from exc

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)      # else it opens behind the browser
        chosen = filedialog.askdirectory(title=title, mustexist=True)
    finally:
        try:
            root.destroy()
        except Exception:                                       # noqa: BLE001
            pass
    return str(Path(chosen)) if chosen else ""
