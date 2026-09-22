"""Standalone image files - photos, screenshots, scans. Pure OCR."""
from __future__ import annotations

from pathlib import Path

from app.ingestion import ocr

from .common import UnsupportedFile


def extract_image(path: Path) -> list[dict]:
    if not ocr.available():
        raise UnsupportedFile(
            "OCR is not available - install it with: pip install rapidocr-onnxruntime"
        )
    text = ocr.image_text(path.read_bytes())
    if not text:
        return []
    return [{"text": text, "loc": "image", "heading": path.stem}]


__all__ = ["extract_image"]
