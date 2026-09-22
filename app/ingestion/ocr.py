"""OCR for images - standalone image files, scanned PDF pages, and pictures that
sit *inside* Word, PowerPoint, Excel and PDF documents.

Two engines are supported, picked automatically:

* **RapidOCR** (default) - pure ``pip install``, ONNX based, no system installer,
  ships its own models, works fully offline.
* **Tesseract** - used if RapidOCR is absent and the Tesseract binary is present.
  Better for unusual languages, but needs a separate installer.

OCR is far slower than text parsing, so this module is defensive about what it
runs on: tiny icons, decorative images and duplicates are skipped, results that
look like noise are dropped, and the number of concurrent engines is capped so a
big PowerPoint deck cannot swamp the machine.
"""
from __future__ import annotations

import hashlib
import io
import queue
import threading
from typing import Any

from app import config

_LOCK = threading.Lock()
_ENGINE_NAME: str | None = None          # None = not probed yet, "" = unavailable
_POOL: queue.Queue | None = None
_WARNED = False


# --------------------------------------------------------------------------- engine
def _probe() -> str:
    """Decide which engine to use. Returns "rapidocr", "tesseract" or ""."""
    want = config.OCR_ENGINE
    if not config.OCR_ENABLED or want == "off":
        return ""

    if want in {"auto", "rapidocr"}:
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401

            return "rapidocr"
        except Exception:                                     # noqa: BLE001
            if want == "rapidocr":
                return ""

    if want in {"auto", "tesseract"}:
        try:
            import pytesseract

            if config.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            pytesseract.get_tesseract_version()
            return "tesseract"
        except Exception:                                     # noqa: BLE001
            return ""
    return ""


def engine_name() -> str:
    global _ENGINE_NAME, _POOL
    if _ENGINE_NAME is not None:
        return _ENGINE_NAME
    with _LOCK:
        if _ENGINE_NAME is None:
            _ENGINE_NAME = _probe()
            _POOL = queue.Queue()
            if _ENGINE_NAME:
                print(f"[ocr] engine: {_ENGINE_NAME}", flush=True)
            elif config.OCR_ENABLED:
                print("[ocr] no engine available - install with: "
                      "pip install rapidocr-onnxruntime", flush=True)
    return _ENGINE_NAME


def available() -> bool:
    return bool(engine_name())


def _new_engine() -> Any:
    if _ENGINE_NAME == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()
    import pytesseract

    return pytesseract


class _Borrowed:
    """Borrow an engine from a bounded pool so OCR cannot spawn one per thread."""

    def __init__(self) -> None:
        self.engine: Any = None

    def __enter__(self) -> Any:
        assert _POOL is not None
        try:
            self.engine = _POOL.get_nowait()
        except queue.Empty:
            with _LOCK:
                # created lazily; the queue caps how many ever exist at once
                if getattr(_Borrowed, "_count", 0) < config.OCR_WORKERS:
                    _Borrowed._count = getattr(_Borrowed, "_count", 0) + 1
                    try:
                        self.engine = _new_engine()
                    except Exception:              # noqa: BLE001
                        # Give the slot back. Counting a failed construction as a
                        # live engine permanently shrinks the pool, and once every
                        # slot has failed this way OCR blocks forever on get().
                        _Borrowed._count -= 1
                        raise
            if self.engine is None:
                self.engine = _POOL.get()          # wait for one to come back
        return self.engine

    def __exit__(self, *_exc: Any) -> None:
        if self.engine is not None and _POOL is not None:
            _POOL.put(self.engine)


# --------------------------------------------------------------------------- helpers
def _prepare(data: bytes) -> Any | None:
    """Load, sanity-check and (if huge) downscale an image. Returns a PIL image."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:                                         # noqa: BLE001
        return None
    w, h = img.size
    if w < config.OCR_IMAGE_MIN_PX or h < config.OCR_IMAGE_MIN_PX:
        return None                                            # icon, bullet, spacer
    if w * h > config.OCR_MAX_PIXELS:
        scale = (config.OCR_MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _layout_text(result: list) -> str:
    """Rebuild reading order from the detected boxes, one output line per visual line.

    RapidOCR returns each detected box separately.  Joining them with spaces turns
    an invoice into one long run-on string, which models read badly; keeping the
    line breaks preserves the "label: value" structure of forms and tables.
    """
    entries = []
    for item in result:
        if len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        if not text:
            continue
        try:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
        except (TypeError, IndexError):
            entries.append((0.0, 0.0, 10.0, text))
            continue
        entries.append((min(ys), min(xs), max(ys) - min(ys), text))
    if not entries:
        return ""

    entries.sort(key=lambda e: (e[0], e[1]))
    heights = sorted(e[2] for e in entries)
    tolerance = max(4.0, heights[len(heights) // 2] * 0.6)

    lines: list[list[tuple[float, str]]] = []
    baseline = None
    for top, left, _height, text in entries:
        if baseline is None or top - baseline > tolerance:
            lines.append([])
            baseline = top
        lines[-1].append((left, text))
    return "\n".join(" ".join(t for _x, t in sorted(line)) for line in lines)


def _clean_result(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line).strip()
    if len(text) < config.OCR_MIN_TEXT_LEN:
        return ""
    stripped = text.replace("\n", "")
    letters = sum(c.isalnum() for c in stripped)
    if letters < len(stripped) * 0.4:        # mostly punctuation => recognition noise
        return ""
    return text


def image_text(data: bytes) -> str:
    """OCR raw image bytes. Returns "" when there is no engine or no readable text."""
    global _WARNED
    if not available():
        return ""
    img = _prepare(data)
    if img is None:
        return ""
    try:
        with _Borrowed() as engine:
            if _ENGINE_NAME == "rapidocr":
                import numpy as np

                out = engine(np.asarray(img))
                result = out[0] if isinstance(out, tuple) else out
                if not result:
                    return ""
                text = _layout_text(result)
            else:
                text = engine.image_to_string(img, lang=config.OCR_LANG)
    except Exception as exc:                                  # noqa: BLE001
        if not _WARNED:
            print(f"[ocr] failed: {type(exc).__name__}: {exc}", flush=True)
            _WARNED = True
        return ""
    return _clean_result(text)


class ImageBudget:
    """Per-file guard: skips duplicate images and caps how many get OCR'd."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = config.OCR_MAX_IMAGES_PER_FILE if limit is None else limit
        self.seen: set[str] = set()
        self.used = 0

    def read(self, data: bytes) -> str:
        """OCR one image unless it is a duplicate or the budget is spent."""
        if not data or self.used >= self.limit:
            return ""
        digest = hashlib.sha1(data).hexdigest()
        if digest in self.seen:
            return ""                                          # repeated logo/watermark
        self.seen.add(digest)
        text = image_text(data)
        if text:
            self.used += 1
        return text


def info() -> dict:
    return {
        "enabled": config.OCR_ENABLED,
        "engine": engine_name() or None,
        "workers": config.OCR_WORKERS,
        "max_images_per_file": config.OCR_MAX_IMAGES_PER_FILE,
    }
