"""Helpers shared by every extractor.

Extractors all return the same shape - a list of *blocks*::

    {"text": "...", "loc": "p. 3", "heading": "Revenue"}

``loc`` is the human-readable pointer used in citations (page / slide / sheet /
row range) and ``heading`` is the nearest structural title, which the chunker
prepends to the embedded text so every chunk carries its own context.
"""
from __future__ import annotations

import csv
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

csv.field_size_limit(10_000_000)

WS_RE = re.compile(r"[ \t\x0b\f\r]+")
NL_RE = re.compile(r"\n{3,}")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
ROWS_PER_BLOCK = 40

# media folders inside the OOXML zip containers (docx / pptx / xlsx)
OOXML_MEDIA = ("word/media/", "ppt/media/", "xl/media/", "media/")


class UnsupportedFile(Exception):
    """Raised when a file cannot be read - surfaced in the UI's Files tab."""


def clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return NL_RE.sub("\n\n", text).strip()


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- html
class _Stripper(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data)


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        return clean(soup.get_text("\n"))
    except Exception:                                         # noqa: BLE001
        parser = _Stripper()
        try:
            parser.feed(html)
        except Exception:                                     # noqa: BLE001
            return clean(re.sub(r"<[^>]+>", " ", html))
        return clean(" ".join(parser.parts))


# --------------------------------------------------------------------------- tables
def table_to_text(rows: list[list[str]]) -> str:
    """Render a table as one "col: value | col: value" line per row."""
    rows = [[clean(c) for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if body and any(h for h in header):
        lines = []
        for r in body:
            pairs = [f"{h}: {v}" for h, v in zip(header, r) if v]
            lines.append(" | ".join(pairs) if pairs else " | ".join(r))
        return "\n".join(lines)
    return "\n".join(" | ".join(r) for r in rows)


def rows_block(rows: list[list[str]], sheet: str, start_row: int,
               header: list[str]) -> dict:
    """Turn a batch of spreadsheet rows into one block with a row-range citation."""
    lines = []
    for r in rows:
        if header:
            line = " | ".join(f"{h}: {v}" for h, v in zip(header, r) if str(v).strip())
        else:
            line = " | ".join(str(v) for v in r if str(v).strip())
        if line:
            lines.append(line)
    span = f"rows {start_row}-{start_row + len(rows) - 1}"
    return {"text": "\n".join(lines),
            "loc": f"{sheet} {span}" if sheet else span,
            "heading": sheet}


# --------------------------------------------------------------------------- ooxml
def iter_ooxml_media(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for every picture stored inside an OOXML container.

    Word, Excel and PowerPoint files are zip archives; anything a user pasted in
    as a picture lives under one of the media folders.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                lowered = name.lower()
                if not lowered.startswith(OOXML_MEDIA):
                    continue
                if lowered.endswith((".emf", ".wmf", ".svg", ".bin")):
                    continue                                   # vector art, not OCR-able
                try:
                    yield name, zf.read(name)
                except Exception:                              # noqa: BLE001
                    continue
    except (zipfile.BadZipFile, OSError):
        return


def ocr_media_blocks(path: Path, label: str = "image") -> list[dict]:
    """OCR every embedded picture of an OOXML file that isn't tied to a location."""
    from app.ingestion import ocr

    if not ocr.available():
        return []
    budget = ocr.ImageBudget()
    blocks: list[dict] = []
    for name, data in iter_ooxml_media(path):
        text = budget.read(data)
        if text:
            blocks.append({"text": f"[{label}: {Path(name).name}] {text}",
                           "loc": f"{label} {Path(name).name}", "heading": ""})
    return blocks
