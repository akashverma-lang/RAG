"""Registry mapping a file extension to the function that reads it.

Adding a new format means writing one ``extract_x(path) -> list[block]`` function
in the right module and adding one line to ``EXTRACTORS`` below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from app import config

from .common import UnsupportedFile, clean
from .documents import extract_docx, extract_pdf, extract_rtf
from .images import extract_image
from .mail import extract_eml, extract_epub, extract_msg
from .presentations import extract_pptx
from .spreadsheets import extract_csv, extract_xls, extract_xlsx
from .textual import extract_html, extract_json, extract_text, extract_xml, extract_yaml

Extractor = Callable[[Path], list[dict]]

# Plain-text-ish formats that only need decoding.
TEXT_LIKE = (
    ".txt", ".md", ".markdown", ".rst", ".log", ".ini", ".cfg", ".conf", ".env",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".go", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql", ".r", ".m", ".tex", ".css",
)

EXTRACTORS: dict[str, Extractor] = {
    # documents
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".docm": extract_docx,
    ".rtf": extract_rtf,
    # presentations
    ".pptx": extract_pptx,
    ".pptm": extract_pptx,
    # spreadsheets
    ".xlsx": extract_xlsx,
    ".xlsm": extract_xlsx,
    ".xls": extract_xls,
    ".csv": extract_csv,
    ".tsv": extract_csv,
    # structured text
    ".json": extract_json,
    ".jsonl": extract_json,
    ".yaml": extract_yaml,
    ".yml": extract_yaml,
    ".html": extract_html,
    ".htm": extract_html,
    ".xml": extract_xml,
    # mail & books
    ".eml": extract_eml,
    ".msg": extract_msg,
    ".epub": extract_epub,
}

for _ext in TEXT_LIKE:
    EXTRACTORS.setdefault(_ext, extract_text)
for _ext in config.IMAGE_EXT:
    EXTRACTORS.setdefault(_ext, extract_image)


def _legacy(target_ext: str) -> Extractor:
    def run(path: Path) -> list[dict]:
        from .legacy import via_office_com

        return via_office_com(path, target_ext)

    return run


EXTRACTORS[".doc"] = _legacy(".docx")
EXTRACTORS[".ppt"] = _legacy(".pptx")


def extractor_for(ext: str) -> Extractor:
    fn = EXTRACTORS.get(ext.lower())
    if fn is None:
        raise UnsupportedFile(f"no extractor for {ext}")
    return fn


def supported(ext: str) -> bool:
    return ext.lower() in EXTRACTORS


def extract(path: Path) -> list[dict]:
    """Read one file into normalised blocks. Raises UnsupportedFile on failure."""
    blocks = extractor_for(path.suffix)(path) or []
    out: list[dict] = []
    for block in blocks:
        text = clean(block.get("text", ""))
        if text:
            out.append({"text": text,
                        "loc": block.get("loc", ""),
                        "heading": block.get("heading", "")})
    return out


__all__ = ["extract", "supported", "extractor_for", "EXTRACTORS", "UnsupportedFile"]
