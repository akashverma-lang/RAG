"""PDF, Word and RTF extraction.

Both PDF and Word get OCR treatment: a PDF page with (almost) no text layer is
rendered and read as an image, pictures embedded in an otherwise-textual page are
read individually, and pictures inside a .docx are read in the position they
appear so their text lands next to the surrounding paragraphs.
"""
from __future__ import annotations

import re
from pathlib import Path

from app import config
from app.ingestion import ocr

from .common import UnsupportedFile, clean, read_text_file, table_to_text


# --------------------------------------------------------------------------- pdf
def extract_pdf(path: Path) -> list[dict]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # PyMuPDF < 1.24.3

    blocks: list[dict] = []
    budget = ocr.ImageBudget() if ocr.available() else None

    with fitz.open(path) as doc:
        toc: dict[int, str] = {}
        try:
            for _level, title, page_no in doc.get_toc() or []:
                toc.setdefault(page_no, title)
        except Exception:                                     # noqa: BLE001
            pass

        heading = ""
        for number, page in enumerate(doc, start=1):
            heading = toc.get(number, heading)
            text = clean(page.get_text("text"))
            was_scanned = False

            # a page with no text layer is a scan -> render it and read the picture
            if budget is not None and len(text) < config.OCR_MIN_CHARS:
                scanned = _ocr_page(page, budget)
                if scanned:
                    text = clean(f"{text}\n{scanned}") if text else scanned
                    was_scanned = True

            if text:
                blocks.append({"text": text, "loc": f"p. {number}", "heading": heading})

            # Charts, screenshots and diagrams sitting on an otherwise textual page.
            # Skipped when the page was scanned, since the picture *is* the page and
            # reading it again would duplicate the text we already have.
            if budget is not None and not was_scanned:
                for caption in _ocr_page_images(doc, page, budget):
                    blocks.append({"text": caption, "loc": f"p. {number} image",
                                   "heading": heading})
    return blocks


def _ocr_page(page, budget: ocr.ImageBudget) -> str:
    """Render a whole page to a bitmap and OCR it (scanned documents)."""
    try:
        pixmap = page.get_pixmap(dpi=config.OCR_DPI)
        return budget.read(pixmap.tobytes("png"))
    except Exception:                                         # noqa: BLE001
        return ""


def _ocr_page_images(doc, page, budget: ocr.ImageBudget) -> list[str]:
    """OCR each raster image embedded in a page."""
    out: list[str] = []
    try:
        images = page.get_images(full=True)
    except Exception:                                         # noqa: BLE001
        return out
    for entry in images:
        xref = entry[0]
        try:
            data = doc.extract_image(xref).get("image")
        except Exception:                                     # noqa: BLE001
            continue
        text = budget.read(data) if data else ""
        if text:
            out.append(f"[image] {text}")
    return out


# --------------------------------------------------------------------------- word
def extract_docx(path: Path) -> list[dict]:
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    budget = ocr.ImageBudget() if ocr.available() else None
    blocks: list[dict] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = clean("\n".join(buf))
            if text:
                blocks.append({"text": text, "loc": "", "heading": heading})
            buf.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para = Paragraph(child, document)
            if budget is not None:
                for caption in _docx_images(document, child, qn, budget):
                    buf.append(caption)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            if style.lower().startswith(("heading", "title", "subtitle")):
                flush()
                heading = text
                buf.append(f"## {text}")
            else:
                buf.append(text)

        elif tag == "tbl":
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            text = table_to_text(rows)
            if text:
                flush()
                blocks.append({"text": text, "loc": "table", "heading": heading})
    flush()

    for section in document.sections:      # headers/footers often carry the doc title
        for part in (section.header, section.footer):
            try:
                text = clean("\n".join(p.text for p in part.paragraphs))
            except Exception:                                 # noqa: BLE001
                text = ""
            if len(text) > 30:
                blocks.append({"text": text, "loc": "header/footer", "heading": ""})
    return blocks


def _docx_images(document, paragraph_el, qn, budget: ocr.ImageBudget) -> list[str]:
    """OCR pictures anchored in one paragraph, keeping them in reading order."""
    out: list[str] = []
    try:
        blips = paragraph_el.findall(".//" + qn("a:blip"))
    except Exception:                                         # noqa: BLE001
        return out
    for blip in blips:
        rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if not rid:
            continue
        try:
            part = document.part.related_parts[rid]
            text = budget.read(part.blob)
        except Exception:                                     # noqa: BLE001
            continue
        if text:
            out.append(f"[image] {text}")
    return out


# --------------------------------------------------------------------------- rtf
def extract_rtf(path: Path) -> list[dict]:
    raw = read_text_file(path)
    raw = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), raw)
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
    text = clean(raw.replace("{", " ").replace("}", " "))
    return [{"text": text, "loc": "", "heading": path.stem}] if text else []


__all__ = ["extract_pdf", "extract_docx", "extract_rtf", "UnsupportedFile"]
