"""Answering "where is that file?" -- by name, by half a name, or by what it is about.

Three ways a person refers to a file they cannot find, in descending order of how
much they remember:

1. the exact name, sometimes with the extension, sometimes not;
2. a fragment or a mangled version of it -- "the kirloskar one", "rams study";
3. nothing about the name at all, only what is inside it -- "the report about gas
   engine reliability".

Each is tried in that order and the first that produces a confident match wins, so
the cheap string comparison is not made to compete with the expensive semantic
search, and the semantic search is not skipped just because the name happened to
contain a common word.

The answer is assembled here rather than written by the model, deliberately. A file
path is the one kind of answer where a plausible-sounding invention is worse than no
answer at all: it sends someone to a folder that does not exist and it is not
obviously wrong until they get there. Everything below is read out of the index.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from app import config

# The phrasings that mean "tell me where", as opposed to "tell me what it says".
_ASKS_WHERE = re.compile(
    r"\b(where(\s+(is|are|was|were|can\s+i\s+find|do\s+i\s+find|did\s+you\s+find|"
    r"it\s+is|they\s+are))?|where'?s|locat(e|ed|ion)|which\s+(folder|directory|drive|path)|"
    r"what\s+(folder|directory|drive|path)|full\s+path|file\s*path|"
    r"find\s+(the\s+)?(file|document)|show\s+me\s+(the\s+)?(path|location|folder))\b",
    re.IGNORECASE)

# Words that are in the question because it is a question, not because they are part
# of a file name. Stripped before anything is matched against a name.
_NOISE = {
    "where", "wheres", "is", "are", "was", "were", "the", "a", "an", "of", "for",
    "my", "our", "your", "this", "that", "these", "those", "it", "its", "in", "on",
    "at", "to", "from", "please", "can", "you", "tell", "me", "show", "find",
    "locate", "located", "location", "folder", "directory", "drive", "path",
    "full", "file", "files", "document", "documents", "doc", "docs", "stored",
    "saved", "kept", "exactly", "which", "what", "and", "give", "i", "do", "does",
    "holds", "hold", "holding", "keeps", "contains", "containing", "sits", "sitting",
    "lives", "live", "stored", "placed", "put", "there", "here",
}

# A word that names a kind of file rather than a file. Each one narrows the search
# by extension instead of diluting the name match, which is the opposite of what it
# was doing: "sales spreadsheet" scored half a match on sales_2025.xlsx because
# "spreadsheet" is not in the name, and lost to a PDF.
_SHEET = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}
_DOC = {".docx", ".doc", ".rtf", ".odt"}
_SLIDES = {".pptx", ".ppt"}
_TEXT = {".txt", ".md", ".markdown", ".rst", ".log"}
_MAIL = {".eml", ".msg"}
_TYPE_WORDS: dict[str, set[str]] = {
    "spreadsheet": _SHEET, "spreadsheets": _SHEET, "workbook": _SHEET,
    "excel": _SHEET, "sheet": _SHEET, "sheets": _SHEET, "xlsx": _SHEET,
    "xls": _SHEET, "csv": {".csv"}, "tsv": {".tsv"},
    "pdf": {".pdf"}, "word": _DOC, "docx": _DOC,
    "presentation": _SLIDES, "slides": _SLIDES, "deck": _SLIDES, "powerpoint": _SLIDES,
    "pptx": _SLIDES, "text": _TEXT, "txt": _TEXT, "markdown": _TEXT, "notes": _TEXT,
    "email": _MAIL, "mail": _MAIL, "eml": _MAIL,
    "image": set(config.IMAGE_EXT), "photo": set(config.IMAGE_EXT),
    "picture": set(config.IMAGE_EXT), "scan": {".pdf"} | set(config.IMAGE_EXT),
}

# An acronym appears in no single word of a file name, so the index cannot find it.
# That one case falls back to a scan, bounded so it stays a lookup rather than a
# crawl: past this many files an acronym simply has to be typed more fully.
ACRONYM_SCAN_LIMIT = 20000

_WORD = re.compile(r"[A-Za-z0-9]+")
# A name typed into the question: "open 260621 Kirloskar.pdf", "the report.docx".
_NAMED = re.compile(r"[\w][\w \-.()&+]{1,120}\.(?:" + "|".join(
    e.lstrip(".") for e in sorted({e.lstrip(".") for e in config.INCLUDE_EXT})) + r")\b",
    re.IGNORECASE)


@dataclass
class Found:
    """One candidate file and why it was chosen."""
    path: str
    rel_path: str
    name: str
    size: int
    n_chunks: int
    score: float
    how: str                      # "name" | "partial" | "content"
    why: str = ""
    snippet: str = ""
    segments: list[str] = field(default_factory=list)


def is_location_question(text: str) -> bool:
    """Is this asking where something lives, rather than what it says?"""
    q = (text or "").strip()
    if not q or len(q) > 300:
        return False
    if not _ASKS_WHERE.search(q):
        return False
    # "Where does the report say the deadline is" is a content question wearing a
    # location question's clothes. A second verb about saying or stating gives it
    # away.
    if re.search(r"\b(say|says|said|state[sd]?|mention(s|ed)?|explain(s|ed)?)\b", q, re.I):
        return False
    return True


def _trim_lead(text: str) -> str:
    """Drop the question's own words from the front of a matched file name.

    Real file names contain spaces ("Process Quality Index Kagal.xlsx"), so the
    pattern has to allow them -- which means it also swallows the words in front of
    the name. Stripping stops at the first word that is not part of asking, so noise
    words *inside* a name ("... to SB Energy_Prepared for OpenAI ...") survive.
    """
    words = text.split()
    while words and words[0].lower().strip(".,:;") in _NOISE:
        words.pop(0)
    return " ".join(words)


def named_in(question: str) -> list[str]:
    """File names typed out in full, extension and all."""
    out = []
    for m in _NAMED.finditer(question or ""):
        name = _trim_lead(m.group(0).strip())
        if name and "." in name:
            out.append(name)
    return out


def _tokens(text: str) -> list[str]:
    """The words that could plausibly be part of a file's name."""
    return [w.lower() for w in _WORD.findall(text or "")
            if w.lower() not in _NOISE and w.lower() not in _TYPE_WORDS]


def wanted_exts(question: str) -> set[str]:
    """Extensions implied by the kind of file the question asked for."""
    out: set[str] = set()
    for w in (x.lower() for x in _WORD.findall(question or "")):
        if w in _TYPE_WORDS:
            out |= _TYPE_WORDS[w]
    return out


def _stem(name: str) -> str:
    return os.path.splitext(name)[0].lower()


def _initials(stem: str) -> str:
    """"Process Quality Index Kagal" -> "pqik".

    People refer to their own files by the acronym far more often than by the name
    printed on them, and no amount of token matching finds "PQI" inside "Process
    Quality Index" -- the letters are there but the words are not.
    """
    return "".join(w[0] for w in _WORD.findall(stem) if w)[:12].lower()


def segments_for(path: str) -> list[str]:
    """The path broken into the steps someone would actually click through.

    Relative to DATA_DIR where possible, because the folders above it are not part of
    the user's mental model of where their documents are -- they chose the data
    folder once and think in terms of what is inside it.
    """
    try:
        rel = os.path.relpath(path, str(config.DATA_DIR))
    except ValueError:                       # different drive: no relative path exists
        rel = None
    root = str(config.DATA_DIR)
    drive, _tail = os.path.splitdrive(root)
    parts: list[str] = []
    if drive:
        parts.append(f"Local Disk ({drive})")
    else:
        parts.append(root.split(os.sep)[0] or os.sep)
    # The data folder's own name, then whatever is under it.
    base = os.path.basename(root.rstrip("\\/")) or root
    parts.append(base)
    if rel and not rel.startswith(".."):
        parts += [p for p in re.split(r"[\\/]+", rel) if p]
    else:
        parts += [p for p in re.split(r"[\\/]+", path) if p and p != drive]
    return parts


def _by_name(question: str, rows: Sequence[dict]) -> list[Found]:
    """Exact and near-exact name matches, which need no model and cannot be wrong."""
    typed = named_in(question)
    qt = _tokens(question)
    want = wanted_exts(question)
    out: list[Found] = []

    for r in rows:
        name = r["name"]
        low = name.lower()
        stem = _stem(name)
        score, how, why = 0.0, "partial", ""

        for t in typed:
            tl = t.lower().strip()
            if tl == low:
                score, how, why = 1.0, "name", "it is exactly the name you gave"
                break
            if tl in low or low in tl:
                score, how, why = max(score, 0.93), "name", f"the name matches “{t}”"

        if score < 0.93 and qt:
            # An acronym the user typed against the initials of the name. Required to
            # be at least three letters: two-letter initials collide constantly.
            initials = _initials(stem)
            for w in qt:
                if (len(w) >= 3 and w.isalpha()
                        and (w == initials or initials.startswith(w))):
                    if 0.9 > score:
                        score, how = 0.9, "name"
                        why = f"“{w.upper()}” are the initials of “{os.path.splitext(name)[0]}”"
                    break
            st = set(_tokens(stem))
            if st:
                covered = sum(1 for w in qt if w in st) / len(qt)
                shared = sum(1 for w in st if w in qt) / len(st)
                ratio = SequenceMatcher(None, " ".join(qt), stem).ratio()
                # Covering the words the user typed matters most; the other two stop
                # a long file name from winning just by containing everything.
                blended = covered * 0.6 + shared * 0.2 + ratio * 0.2
                if covered >= 0.5 and blended > score:
                    score = min(blended, 0.92)
                    how = "name" if covered >= 0.99 else "partial"
                    hit = [w for w in qt if w in st]
                    why = "the name contains " + ", ".join(f"“{w}”" for w in hit[:4])

        # The question asked for a spreadsheet and this is a PDF. Not impossible --
        # people misremember formats -- but it should lose to an actual spreadsheet.
        if score > 0 and want and (r.get("ext") or "").lower() not in want:
            score *= 0.45

        if score > 0:
            out.append(Found(path=r["path"], rel_path=r["rel_path"], name=name,
                             size=r.get("size") or 0, n_chunks=r.get("n_chunks") or 0,
                             score=round(score, 3), how=how, why=why,
                             segments=segments_for(r["path"])))
    out.sort(key=lambda f: -f.score)
    return out


def _by_content(question: str, rows: Sequence[dict], limit: int) -> list[Found]:
    """Which file is this about? Answered with the retrieval that already exists."""
    from app.retrieval import search as search_mod

    # The location words would otherwise be embedded along with the subject and drag
    # the query toward any passage that happens to discuss folders.
    subject = " ".join(_tokens(question)) or question
    try:
        hits = search_mod.search(subject, top_k=max(8, limit * 3))
    except Exception as exc:                                    # noqa: BLE001
        print(f"[locate] content search failed: {exc}", flush=True)
        return []

    want = wanted_exts(question)
    by_path: dict[str, dict] = {r["path"]: r for r in rows}
    best: dict[str, Found] = {}
    for rank, h in enumerate(hits):
        row = by_path.get(h.get("path"))
        if not row or h["path"] in best:
            continue
        if want and (row.get("ext") or "").lower() not in want:
            continue          # the kind of file was stated; respect it
        best[h["path"]] = Found(
            path=row["path"], rel_path=row["rel_path"], name=row["name"],
            size=row.get("size") or 0, n_chunks=row.get("n_chunks") or 0,
            # Ranked, not scored: cross-encoder scores are not comparable to the
            # string ratios above, so position is the honest signal.
            score=round(max(0.35, 0.75 - rank * 0.06), 3),
            how="content", why="its contents match what you described",
            snippet=(h.get("text") or "")[:220].replace("\n", " ").strip(),
            segments=segments_for(row["path"]))
    return sorted(best.values(), key=lambda f: -f.score)[:limit]


def find(question: str, limit: int = 4) -> list[Found]:
    """Candidate files for a location question, most likely first.

    The shortlist comes from the name index rather than from the whole files table.
    The first version loaded every row and scored it in Python, which measured 2.7
    seconds at 200,000 files and grows linearly -- ten million would be minutes, for
    a question that should be instant. The scorer is unchanged; it now runs over a
    few hundred candidates instead of the corpus.
    """
    from app.core.store import get_store

    store = get_store()
    # The words that could be part of a name, plus any name typed out in full. An
    # acronym is searched as itself: the index tokenises on _ - . so "PQI" matches a
    # file called PQI_report, and the Python scorer catches "Process Quality Index".
    terms = _tokens(question) + [w for n in named_in(question)
                                 for w in _WORD.findall(os.path.splitext(n)[0])]
    rows = [r for r in store.search_files_by_name(terms) if r.get("path")]

    # An acronym matches no token in the index, because the initials appear in no
    # single word. That case still needs the whole list, but only when the indexed
    # lookup found nothing, and only up to a bounded number of rows.
    if not rows and terms:
        rows = [r for r in store.files_for_lookup(limit=ACRONYM_SCAN_LIMIT)
                if r.get("path")]
    if not rows:
        return []

    named = _by_name(question, rows)
    # One confident name match is an answer, not a shortlist: offering alternatives
    # next to an exact hit only makes the reader doubt it.
    if named and named[0].score >= 0.93:
        top = [f for f in named if f.score >= 0.93]
        return top[:limit] if len(top) > 1 else top[:1]
    strong = [f for f in named if f.score >= 0.55]
    if strong:
        return strong[:limit]
    # An explicit file name that matched nothing is an answer in itself. Falling
    # through to "what is this about" would hand back a confident path to a file
    # they did not ask for.
    if named_in(question):
        return []
    return _by_content(question, rows, limit) or [f for f in named if f.score >= 0.4][:limit]


def _mermaid(segments: Sequence[str]) -> str:
    """The path as a flowchart, one node per folder you would open."""
    safe = []
    for s in segments:
        # Brackets and pipes are the flowchart syntax's own punctuation; a file named
        # "report[final].pdf" would otherwise close the label early.
        safe.append(re.sub(r"\s+", " ", str(s).replace("[", "(").replace("]", ")")
                           .replace("|", "/").replace('"', "'")).strip()[:44])
    lines = ["flowchart TD"]
    for i, label in enumerate(safe):
        node = f"P{i}"
        shape = f"(({label}))" if i == len(safe) - 1 else f"[{label}]"
        lines.append(f"    {node}{shape}" if i == 0 else
                     f"    P{i - 1} --> {node}{shape}")
    return "\n".join(lines)


def _human_size(n: int) -> str:
    step = 1024.0
    val = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if val < step or unit == "GB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= step
    return f"{val:.1f} GB"


def render(question: str, found: Sequence[Found]) -> str:
    """The answer itself: where it is, how sure, and the route to it."""
    if not found:
        return ""
    top = found[0]
    folder = os.path.dirname(top.path)

    if top.how == "name" and top.score >= 0.93:
        opener = f"**{top.name}** is in:"
    elif top.how == "content":
        opener = (f"You did not name a file, so this is the closest match by what is "
                  f"inside it — **{top.name}**:")
    else:
        opener = f"The closest match is **{top.name}**:"

    out = [opener, "", f"`{top.path}`", ""]
    out.append(f"- Folder: `{folder}`")
    out.append(f"- File name: `{top.name}`")
    out.append(f"- Size: {_human_size(top.size)}"
               + (f" · {top.n_chunks} indexed chunks" if top.n_chunks else ""))
    if top.why:
        out.append(f"- Matched because {top.why}")
    if top.snippet:
        out.append(f"- From inside it: “{top.snippet}…”")

    out += ["", "```mermaid", _mermaid(top.segments), "```"]

    others = [f for f in found[1:] if f.score >= 0.4][:3]
    if others:
        out += ["", "**Other files it could be**", ""]
        for f in others:
            out.append(f"- `{f.rel_path}` — {f.why or 'similar name'}")
    return "\n".join(out)


def sources_for(found: Sequence[Found]) -> list[dict]:
    """The matches as source cards, so each one is clickable and openable."""
    return [
        {"n": i, "file": f.name, "rel_path": f.rel_path, "path": f.path,
         "loc": os.path.dirname(f.rel_path) or ".", "heading": "",
         "score": f.score, "preview": f.snippet or f.path}
        for i, f in enumerate(found, start=1)
    ]
