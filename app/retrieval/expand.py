"""Asking the question several ways, because the documents use different words.

Someone asks about "machine breakdowns"; the document only ever says "corrective
maintenance" or "CM". Dense retrieval is supposed to bridge that, but a small
embedding model over a small corpus often does not, and BM25 certainly will not --
there is no shared token to match on.

So the question is rewritten before searching, from two sources:

* the **glossary** in the semantic layer, which is free and exact -- it already knows
  that BD means breakdown and PM means preventive maintenance, because those are the
  values in the data;
* the **fast model**, asked for a few alternative phrasings including the words a
  document would plausibly use.

Each variant is searched separately and the ranked lists are fused, so a passage that
only matches one phrasing still surfaces. Reranking then happens against the user's
*original* wording, because that is the question that actually has to be answered.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from app import config
from app.analysis import semantics

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
# A fence the model put on its own line, or after a "Here you go:" preamble, is not
# at the ends of the string and so survives _FENCE.
_FENCE_ANY = re.compile(r"^\s*```(?:json|javascript)?\s*$", re.IGNORECASE | re.MULTILINE)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.){2,300})"')
_LIST_JUNK = re.compile(r"^[\s\[\]{},:*\-]*$")

EXPAND_SYSTEM = """Rewrite a search query so it matches documents that use different
words for the same thing.

{glossary}

Reply with ONLY a JSON array of {n} short search queries, best first:
["...", "..."]

Rules:
- Each one must mean the same as the original question. Do not answer it, do not
  broaden it into a different question.
- Vary the vocabulary: use the formal or technical term where the user used a plain
  one, and the plain word where they used jargon. Expand abbreviations, and also
  include the abbreviation itself.
- Keep them short, like search queries, not sentences.
- If a term appears in the list above, use its official wording in at least one."""


def _usable(text: str) -> bool:
    """A search query, as opposed to a bracket the model was still in the middle of.

    The fence stripper above catches the common shapes, but it only knows the tags it
    was told about; a leftover ```JSON5 line has letters in it and would otherwise
    read as a perfectly good phrasing of the question.
    """
    text = text.strip()
    if "`" in text:
        return False
    return (len(text) > 2 and not _LIST_JUNK.match(text)
            and any(ch.isalnum() for ch in text))


def _parse_list(text: str) -> list[str]:
    body = _FENCE_ANY.sub("", (text or "").strip())
    body = _FENCE.sub("", body.strip()).strip()
    start, end = body.find("["), body.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(body[start:end + 1])
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # The reply is usually well formed but cut off, because a short max_tokens ends
    # it mid-array. Left to the line fallback that produced '[' and 'json' as search
    # queries -- each one then embedded and fused into the ranking, diluting the
    # phrasings that were real. The complete quoted items are still there, so take
    # those and drop the half-written one.
    if start >= 0:
        salvaged = [m.group(1).strip() for m in _QUOTED.finditer(body[start:])]
        salvaged = [x for x in salvaged if _usable(x)]
        if salvaged:
            return salvaged[:6]

    # A model that ignored the format but wrote one query per line is still useful.
    lines = [re.sub(r'^\s*[-*\d.")\]]+\s*', "", ln).strip().strip('",').strip(' "')
             for ln in body.splitlines() if ln.strip()]
    return [ln for ln in lines if _usable(ln)][:6]


def _dedupe(question: str, candidates: Sequence[str], limit: int) -> list[str]:
    seen = {question.strip().lower()}
    out: list[str] = []
    for text in candidates:
        key = text.strip().lower()
        if not key or key in seen or len(text) > 300:
            continue
        seen.add(key)
        out.append(text.strip())
        if len(out) >= limit:
            break
    return out


def variants(question: str, model: str | None = None,
             use_model: bool | None = None) -> list[str]:
    """Alternative wordings of the question. Never includes the original."""
    limit = config.EXPAND_QUERIES
    if limit <= 0:
        return []

    # The glossary half is free and never wrong, so it goes first and is kept even
    # when the model call fails or is switched off.
    found = semantics.synonyms(question)

    use_model = config.EXPAND_WITH_MODEL if use_model is None else use_model
    if use_model and len(found) < limit:
        entries = semantics.relevant_glossary(question) or semantics.load()["glossary"][:14]
        glossary = ("Terms used in this data:\n" + "\n".join(
            f"  {g.term} = {g.means}" + (f" (also: {', '.join(g.also)})" if g.also else "")
            for g in entries)) if entries else ""
        try:
            from app.core import router

            text, _used = router.complete(
                [{"role": "system",
                  "content": EXPAND_SYSTEM.format(glossary=glossary, n=limit)},
                 {"role": "user", "content": question}],
                role="agent", model=model, temperature=0.3,
                timeout=config.EXPAND_TIMEOUT, max_tokens=300)
            found += _parse_list(text)
        except Exception:                                       # noqa: BLE001
            pass                # expansion is an optimisation; never fail the search
    return _dedupe(question, found, limit)
