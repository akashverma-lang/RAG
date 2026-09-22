"""Plain text, Markdown, source code, JSON/YAML, HTML and XML."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import HEADING_RE, clean, html_to_text, read_text_file


def extract_text(path: Path) -> list[dict]:
    text = clean(read_text_file(path))
    if not text:
        return []

    if path.suffix.lower() in {".md", ".markdown", ".rst"}:
        blocks: list[dict] = []
        heading, buf = "", []
        for line in text.split("\n"):
            match = HEADING_RE.match(line)
            if match:
                if buf:
                    blocks.append({"text": clean("\n".join(buf)), "loc": "",
                                   "heading": heading})
                    buf = []
                heading = match.group(2).strip()
            buf.append(line)
        if buf:
            blocks.append({"text": clean("\n".join(buf)), "loc": "", "heading": heading})
        return [b for b in blocks if b["text"]]

    return [{"text": text, "loc": "", "heading": ""}]


def extract_json(path: Path) -> list[dict]:
    raw = read_text_file(path)
    if path.suffix.lower() == ".jsonl":
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.dumps(json.loads(line), ensure_ascii=False))
            except Exception:                                 # noqa: BLE001
                lines.append(line)
        return [{"text": clean("\n".join(lines)), "loc": "", "heading": path.stem}]
    try:
        pretty = json.dumps(json.loads(raw), indent=1, ensure_ascii=False, default=str)
    except Exception:                                         # noqa: BLE001
        pretty = raw
    return [{"text": clean(pretty), "loc": "", "heading": path.stem}]


def extract_yaml(path: Path) -> list[dict]:
    return [{"text": clean(read_text_file(path)), "loc": "", "heading": path.stem}]


def extract_html(path: Path) -> list[dict]:
    text = html_to_text(read_text_file(path))
    return [{"text": text, "loc": "", "heading": path.stem}] if text else []


def extract_xml(path: Path) -> list[dict]:
    text = clean(re.sub(r"<[^>]+>", " ", read_text_file(path)))
    return [{"text": text, "loc": "", "heading": path.stem}] if text else []


__all__ = ["extract_text", "extract_json", "extract_yaml", "extract_html", "extract_xml"]
