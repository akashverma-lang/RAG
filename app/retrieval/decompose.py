"""Splitting a question into the parts it is actually made of, and searching for each.

One search can only return one ranked list, and a ranked list has one winner. Ask
"compare the failure rate of the R550 and the SL90" and every retrieval signal --
BM25, dense, the reranker -- scores passages against the whole sentence at once.
Whichever engine is better represented in the corpus dominates the shortlist, and
top_k fills up with it. The answer then reads as if the other engine barely exists,
which is not a model failure: the passages about it were never retrieved.

Query expansion does not fix this. It rewrites the question into other *wordings* of
the same question, so every variant still carries both engines and still returns the
same lopsided list.

So the question is split into its parts, each part is searched on its own, and the
shortlists are interleaved rather than merged by score. Interleaving is the whole
point: a global re-sort would simply restore the original imbalance, because the
dominant part's passages really do score higher. Round-robin guarantees that every
part contributes evidence to the context, in proportion to the number of parts
rather than in proportion to how well the corpus covers them.

Cost: one planning call, plus one retrieval per part (each of which already makes its
own expansion call). That is the trade this module exists to make.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from app import config

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_FENCE_ANY = re.compile(r"^\s*```(?:json|javascript)?\s*$", re.IGNORECASE | re.MULTILINE)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.){4,300})"')
_JUNK = re.compile(r"^[\s\[\]{},:*\-]*$")

PLAN_SYSTEM = """Split a question into the separate lookups needed to answer it.

Reply with ONLY a JSON array of self-contained questions, in reading order:
["...", "..."]

Split when the question asks about more than one thing:
- two or more subjects being compared
- several attributes wanted about one subject
- a question whose answer depends on first establishing something else

Do NOT split when one search would answer it. Reply with [] for those.

Rules:
- At most {n} parts. Fewer is better.
- Each part must stand alone: repeat the subject by name instead of writing "it",
  "they" or "the former". A part is going to be used as a search query on its own,
  with none of the other parts for context.
- Never invent a sub-question about something the user did not mention.
- Do not answer anything. Only split."""


def _parse_list(text: str) -> list[str]:
    """Same tolerance as query expansion: models fence, prefix and truncate JSON."""
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
    # Truncated mid-array: the complete quoted items are still usable.
    if start >= 0:
        salvaged = [m.group(1).strip() for m in _QUOTED.finditer(body[start:])]
        if salvaged:
            return salvaged
    lines = [re.sub(r'^\s*[-*\d.")\]]+\s*', "", ln).strip().strip('",').strip(' "')
             for ln in body.splitlines() if ln.strip()]
    return [ln for ln in lines if ln and not _JUNK.match(ln)]


def _usable(part: str, question: str) -> bool:
    """A real sub-question, not a fragment, a fence, or the question echoed back."""
    text = part.strip()
    if len(text) < 8 or len(text) > 300 or "`" in text:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    # A "split" that reproduces the original buys an extra search for nothing.
    return text.strip(" ?.").lower() != question.strip(" ?.").lower()


def plan(question: str, model: str | None = None,
         history: Sequence[dict] | None = None,
         max_parts: int | None = None) -> list[str]:
    """The sub-questions this question breaks into, or [] to search it whole.

    Never raises: an unsplit question is the current behaviour, which is a perfectly
    good answer path, so every failure here degrades to it.
    """
    limit = config.DEEP_MAX_PARTS if max_parts is None else max_parts
    if limit < 2 or not question.strip():
        return []

    # A follow-up ("and the SL90?") is unsplittable on its own but means something
    # with the previous turn attached. Only the last user turn is worth the tokens.
    asked = question
    prev = [m.get("content") or "" for m in (history or [])
            if m.get("role") == "user" and (m.get("content") or "").strip()]
    if prev and len(question.split()) <= 12:
        asked = f"Earlier question: {prev[-1]}\nNow asked: {question}"

    try:
        from app.core import router

        text, _used = router.complete(
            [{"role": "system", "content": PLAN_SYSTEM.format(n=limit)},
             {"role": "user", "content": asked}],
            role="agent", model=model, temperature=0.0,
            timeout=config.DEEP_TIMEOUT, max_tokens=400)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[deep] planning skipped: {exc}", flush=True)
        return []

    seen: set[str] = set()
    out: list[str] = []
    for part in _parse_list(text):
        if not _usable(part, question):
            continue
        key = part.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part.strip())
        if len(out) >= limit:
            break
    # One part is not a decomposition, it is a paraphrase, and searching it instead
    # of the user's own wording is strictly worse.
    return out if len(out) >= 2 else []


def retrieve(question: str, parts: Sequence[str], *,
             top_k: int | None = None,
             use_rerank: bool | None = None,
             history: Sequence[dict] | None = None,
             files: Sequence[str] | None = None,
             per_part: int | None = None,
             base_hits: list[dict] | None = None) -> tuple[str, list[dict], list[dict], list[dict]]:
    """Search every part plus the original, then interleave the shortlists.

    Returns ``(context, sources, hits, coverage)``. ``coverage`` reports how many
    passages each part actually contributed, which is the only honest way to tell
    the difference between "this part has no evidence" and "this part lost the
    ranking to a louder one".
    """
    from app.retrieval import search as search_mod

    per_part = per_part or config.DEEP_PER_PART_K
    # The user's own wording leads: it is the only query guaranteed to mean the whole
    # question, and the reranker was tuned against it.
    queries = [question] + [p for p in parts if p.strip()]

    def one(q: str) -> list[dict]:
        try:
            return search_mod.search(q, top_k=per_part, use_rerank=use_rerank,
                                     history=history, files=files)
        except Exception as exc:                                # noqa: BLE001
            print(f"[deep] part failed ({q[:60]!r}): {exc}", flush=True)
            return []

    # The caller may already have searched the original wording, concurrently with
    # the planning call that produced these parts. Re-running it would throw away
    # work that is already paid for.
    todo = queries if base_hits is None else queries[1:]

    # Every part is an independent network round trip (expansion) plus local CPU, so
    # they run together; the wall clock is one part, not the sum of them.
    workers = max(1, min(len(todo), config.DEEP_WORKERS)) if todo else 1
    if todo:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deep") as pool:
            done = list(pool.map(one, todo))
    else:
        done = []
    per_query = done if base_hits is None else [base_hits[:per_part]] + done

    # Round robin, best-of-each-part first. Sorting by score here would undo the
    # entire exercise: the dominant part's passages genuinely score higher, which is
    # why it crowded the others out of a single search in the first place.
    chosen: list[dict] = []
    seen: set[int] = set()
    coverage = [{"query": q, "found": len(hits), "used": 0}
                for q, hits in zip(queries, per_query)]
    for rank in range(per_part):
        for i, hits in enumerate(per_query):
            if rank >= len(hits):
                continue
            hit = hits[rank]
            cid = hit.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            chosen.append(hit)
            coverage[i]["used"] += 1

    if not chosen:
        return "", [], [], coverage

    passages = search_mod.expand_with_neighbors(chosen)
    # expand_with_neighbors re-sorts by score, which would re-impose exactly the
    # ranking the interleave just removed. Restore the interleaved order.
    #
    # A passage is not a hit: neighbouring chunks are merged into runs, so it carries
    # (file_id, ords) and no chunk id at all. It is placed by the best-placed hit
    # inside it, which is also what keeps a merged run beside the hit it grew from.
    rank_of = {(h["file_id"], h["ord"]): i for i, h in enumerate(chosen)}
    far = len(rank_of) + 1

    def placement(p: dict) -> int:
        return min((rank_of.get((p["file_id"], o), far) for o in p.get("ords", ())),
                   default=far)

    passages.sort(key=lambda p: (placement(p), -p["score"]))

    context, sources = search_mod.build_context(passages)
    return context, sources, chosen, coverage
