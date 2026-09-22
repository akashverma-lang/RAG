"""Structure-aware chunking.

Blocks coming out of the extractors already respect natural boundaries (a PDF page,
a slide, 40 spreadsheet rows).  Here we pack small blocks together and split big
ones on paragraph -> sentence boundaries, keeping a character overlap so a fact
that straddles a boundary still lands whole in at least one chunk.
"""
from __future__ import annotations

import re

from app import config

PARA_RE = re.compile(r"\n\s*\n")
SENT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[\"'])")
LINE_RE = re.compile(r"\n")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split a too-long paragraph, preferring sentence then line then hard cuts."""
    pieces = [p for p in SENT_RE.split(text) if p.strip()]
    if len(pieces) == 1:
        pieces = [p for p in LINE_RE.split(text) if p.strip()]
    if len(pieces) == 1 and len(text) > size:
        pieces = [text[i:i + size] for i in range(0, len(text), max(1, size - overlap))]
        return pieces

    out: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > size:
            out.append(buf.strip())
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + " " + piece).strip() if tail else piece
        else:
            buf = f"{buf} {piece}".strip()
    if buf.strip():
        out.append(buf.strip())
    return out


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out: list[str] = []
    buf = ""
    for para in PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.extend(_hard_split(para, size, overlap))
            continue
        if buf and len(buf) + len(para) + 2 > size:
            out.append(buf.strip())
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{para}" if tail else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_blocks(
    blocks: list[dict],
    *,
    size: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[dict]:
    size = size or config.CHUNK_CHARS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    min_chars = min_chars if min_chars is not None else config.MIN_CHUNK_CHARS

    # 1. pack consecutive small blocks that share a heading into one chunk
    packed: list[dict] = []
    for b in blocks:
        text = b["text"].strip()
        if not text:
            continue
        if packed:
            prev = packed[-1]
            same_ctx = prev["heading"] == b["heading"]
            if same_ctx and len(prev["text"]) + len(text) + 2 <= size:
                prev["text"] = f"{prev['text']}\n\n{text}"
                if b["loc"] and b["loc"] not in prev["loc"]:
                    prev["loc"] = f"{prev['loc']}, {b['loc']}" if prev["loc"] else b["loc"]
                continue
        packed.append({"text": text, "loc": b.get("loc", ""), "heading": b.get("heading", "")})

    # 2. split anything still oversized
    chunks: list[dict] = []
    for b in packed:
        for part in split_text(b["text"], size, overlap):
            part = part.strip()
            if len(part) < min_chars and chunks and chunks[-1]["heading"] == b["heading"]:
                chunks[-1]["text"] += "\n" + part      # glue scraps onto the previous chunk
                continue
            if len(part) < min_chars:
                continue
            chunks.append({"text": part, "loc": b["loc"], "heading": b["heading"]})
    return chunks


def embed_text(chunk: dict, file_name: str, rel_path: str) -> str:
    """What we actually embed: the chunk plus a compact breadcrumb for context."""
    bits = [f"file: {file_name}"]
    folder = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    if folder:
        bits.append(f"folder: {folder}")
    if chunk.get("heading"):
        bits.append(f"section: {chunk['heading']}")
    if chunk.get("loc"):
        bits.append(f"at: {chunk['loc']}")
    return " | ".join(bits) + "\n" + chunk["text"]
