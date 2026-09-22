"""Email (.eml, Outlook .msg) and ePub."""
from __future__ import annotations

import zipfile
from pathlib import Path

from .common import UnsupportedFile, clean, html_to_text


def extract_eml(path: Path) -> list[dict]:
    import email
    from email import policy

    message = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    head = "\n".join(
        f"{key}: {message.get(key)}"
        for key in ("From", "To", "Cc", "Date", "Subject") if message.get(key)
    )

    body = ""
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            if part.get_filename():
                continue
            if ctype == "text/plain":
                body += str(part.get_content())
            elif ctype == "text/html" and not body:
                body += html_to_text(str(part.get_content()))
    else:
        content = message.get_content()
        body = str(content) if message.get_content_type() == "text/plain" \
            else html_to_text(str(content))

    names = [p.get_filename() for p in message.walk() if p.get_filename()]
    if names:
        head += "\nAttachments: " + ", ".join(names)

    return [{"text": clean(f"{head}\n\n{body}"), "loc": "email",
             "heading": clean(str(message.get("Subject") or path.stem))}]


def extract_msg(path: Path) -> list[dict]:
    try:
        import extract_msg
    except ImportError as exc:
        raise UnsupportedFile("install extract-msg to index Outlook .msg files") from exc

    message = extract_msg.Message(str(path))
    head = "\n".join(
        f"{label}: {value}" for label, value in (
            ("From", message.sender), ("To", message.to), ("Cc", message.cc),
            ("Date", message.date), ("Subject", message.subject),
        ) if value
    )
    body = message.body or ""
    if not body and getattr(message, "htmlBody", None):
        body = html_to_text(message.htmlBody.decode("utf-8", "replace"))
    subject = clean(str(message.subject or path.stem))
    message.close()
    return [{"text": clean(f"{head}\n\n{body}"), "loc": "email", "heading": subject}]


def extract_epub(path: Path) -> list[dict]:
    blocks: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist()
                 if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in sorted(names):
            try:
                text = html_to_text(archive.read(name).decode("utf-8", "replace"))
            except Exception:                                 # noqa: BLE001
                continue
            if len(text) > 40:
                blocks.append({"text": text, "loc": Path(name).stem,
                               "heading": Path(name).stem})
    return blocks


__all__ = ["extract_eml", "extract_msg", "extract_epub"]
