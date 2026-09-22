"""Legacy binary Office formats (.doc, .xls, .ppt).

These predate the OOXML zip containers and have no usable pure-Python reader.  If
Microsoft Office and pywin32 are installed we convert to the modern format in a
temp file and reuse the normal extractor; otherwise the file is reported as failed
with a message telling the user to re-save it.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .common import UnsupportedFile

# target extension -> (COM ProgID, SaveAs format id)
_APPS = {
    ".docx": ("Word.Application", 16),
    ".xlsx": ("Excel.Application", 51),
    ".pptx": ("PowerPoint.Application", 24),
}


def via_office_com(path: Path, target_ext: str) -> list[dict]:
    try:
        import win32com.client
    except ImportError as exc:
        raise UnsupportedFile(
            f"legacy {path.suffix} needs Microsoft Office + pywin32 "
            f"(pip install pywin32), or re-save the file as {target_ext}"
        ) from exc

    prog_id, file_format = _APPS[target_ext]
    out = Path(tempfile.gettempdir()) / f"ragconv_{os.getpid()}_{path.stem}{target_ext}"
    app = None
    try:
        app = win32com.client.Dispatch(prog_id)
        try:
            app.Visible = False
        except Exception:                                     # noqa: BLE001
            pass

        if prog_id.startswith("Word"):
            doc = app.Documents.Open(str(path), ReadOnly=True)
            doc.SaveAs2(str(out), FileFormat=file_format)
            doc.Close(False)
        elif prog_id.startswith("Excel"):
            app.DisplayAlerts = False
            book = app.Workbooks.Open(str(path), ReadOnly=True)
            book.SaveAs(str(out), FileFormat=file_format)
            book.Close(False)
        else:
            deck = app.Presentations.Open(str(path), WithWindow=False)
            deck.SaveAs(str(out), FileFormat=file_format)
            deck.Close()
    except Exception as exc:                                  # noqa: BLE001
        raise UnsupportedFile(f"could not convert legacy file: {exc}") from exc
    finally:
        try:
            if app is not None:
                app.Quit()
        except Exception:                                     # noqa: BLE001
            pass

    from . import extractor_for

    try:
        return extractor_for(target_ext)(out)
    finally:
        out.unlink(missing_ok=True)


__all__ = ["via_office_com"]
