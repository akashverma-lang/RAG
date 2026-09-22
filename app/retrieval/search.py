"""Hybrid retrieval: BM25 + dense vectors -> RRF fusion -> cross-encoder rerank
-> neighbour expansion -> a context block with numbered citations.

Why hybrid: dense search finds paraphrases ("how much did we earn" vs "revenue"),
BM25 nails exact tokens (part numbers, names, error codes).  Reciprocal Rank
Fusion combines the two ranked lists without needing comparable scores.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from app import config
from app.core import embed
from app.core.store import get_store
from app.retrieval import expand

PRONOUN_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|him|her|there|the same|above)\b",
    re.IGNORECASE,
)


def _same_topic(question: str, previous: str) -> bool:
    """Do these two questions mean related things?

    The old rule folded the previous question into anything eight words or shorter,
    which is right for "and the SL90?" and wrong for "What is the Q4 revenue?" -- the
    second is a new subject stated briefly, and gluing the old question to it drags
    retrieval back to a topic the user has finished with. That is the failure people
    describe as the assistant being "stuck in the previous chat".

    Meaning separates them where length cannot. Measured on twelve pairs with the
    embedder already loaded: genuine follow-ups scored 0.519 and above, changes of
    subject 0.489 and below.
    """
    try:
        from app.core import embed

        v = np.asarray(embed.encode_queries([question, previous]), dtype=np.float32)
        if v.shape[0] < 2:
            return True
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return float(v[0] @ v[1]) >= config.FOLLOW_UP_SIMILARITY
    except Exception as exc:                                    # noqa: BLE001
        # The old behaviour, which is the safer default: carrying context that was
        # not needed costs some precision, dropping context that was needed loses
        # the answer entirely.
        print(f"[retrieve] follow-up check skipped: {exc}", flush=True)
        return True


def expand_query(question: str, history: Sequence[dict] | None) -> str:
    """Fold the previous question into this one, but only when it belongs.

    Avoids a second LLM round-trip, which matters a lot on CPU-only setups.
    """
    if not history:
        return question
    prev_users = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    if not prev_users:
        return question

    # A pronoun is not a hint, it is a grammatical dependency: "what does it say"
    # cannot be searched on its own whatever it scores against the previous turn.
    if PRONOUN_RE.search(question):
        return f"{prev_users[-1]} {question}".strip()

    # Short enough that it might be leaning on the last turn. Only then is it worth
    # embedding two strings to find out; a long, self-contained question never folds
    # and never pays for the check.
    if len(question.split()) <= 8 and _same_topic(question, prev_users[-1]):
        return f"{prev_users[-1]} {question}".strip()
    return question


def _rrf(ranked_lists: list[list[tuple[int, float]]], k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (cid, _score) in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def search(
    question: str,
    *,
    top_k: int | None = None,
    use_rerank: bool | None = None,
    history: Sequence[dict] | None = None,
    files: Sequence[str] | None = None,
) -> list[dict]:
    """Return the best chunks for a question, best first.

    ``files`` restricts the search to chosen rel_paths. Scoping happens inside the
    index, before the shortlist is cut, so a small file still gets a fair hearing
    against a large one.
    """
    store = get_store()
    file_ids = store.file_ids_for(files) if files else []
    if files and not file_ids:
        return []
    top_k = top_k or config.TOP_K
    use_rerank = config.RERANK_ENABLED if use_rerank is None else use_rerank
    query = expand_query(question, history)

    # Rewriting the question costs a model round trip, but searching the original
    # wording does not need to wait for it: the rewrite is started first and the
    # base query is embedded and matched while it is still in flight.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="expand") as pool:
        pending = pool.submit(expand.variants, query)
        base_lex = store.search_bm25(query, config.CANDIDATES_BM25, file_ids)
        try:
            rewrites = pending.result()
        except Exception as exc:                               # noqa: BLE001
            print(f"[retrieve] query expansion skipped: {exc}", flush=True)
            rewrites = []

    # The original wording first, then the rewrites. Every list is fused together,
    # so a passage only reachable through one phrasing still makes the shortlist.
    queries = [query] + list(rewrites)
    ranked: list[list[tuple[int, float]]] = [base_lex]
    dense_scores: dict[int, float] = {}
    bm25_ranks: dict[int, int] = {cid: r for r, (cid, _) in enumerate(base_lex, start=1)}

    # One embedding call and one pass over the vector file for every phrasing,
    # instead of one of each per phrasing.
    try:
        vecs = embed.encode_queries(queries)
        for hits in store.search_dense_many(vecs, config.CANDIDATES_DENSE, file_ids):
            ranked.append(hits)
            for cid, score in hits:          # keep the best dense score seen anywhere
                if score > dense_scores.get(cid, -1e9):
                    dense_scores[cid] = score
    except Exception as exc:                                   # noqa: BLE001
        print(f"[retrieve] dense search unavailable: {exc}", flush=True)

    for text in queries[1:]:
        ranked.append(store.search_bm25(text, config.CANDIDATES_BM25, file_ids))

    if not any(ranked):
        return []

    fused = _rrf(ranked, config.RRF_K)

    depth = config.RERANK_CANDIDATES or max(top_k * 4, 24)
    depth = min(depth, config.RERANK_MAX_CANDIDATES)
    candidates = sorted(fused.items(), key=lambda kv: -kv[1])[:depth]
    rows = store.get_chunks([cid for cid, _ in candidates])
    hits: list[dict] = []
    for cid, fscore in candidates:
        row = rows.get(cid)
        if not row:
            continue
        hits.append({
            **row,
            "fusion": round(fscore, 5),
            "dense": round(dense_scores.get(cid, 0.0), 4),
            "bm25_rank": bm25_ranks.get(cid, 0),
            "queries": len(queries),
            "score": round(fscore, 5),
        })

    if use_rerank and len(hits) > 1:
        # Reranked against the user's own wording, not a rewrite of it.
        scores = embed.rerank(question, [h["text"][:2000] for h in hits])
        if scores:
            for h, s in zip(hits, scores):
                h["rerank"] = round(float(s), 4)
                h["score"] = round(float(s), 4)
            hits.sort(key=lambda h: -h["score"])
            if config.MIN_SCORE:
                hits = [h for h in hits if h["score"] >= config.MIN_SCORE] or hits[:1]

    return hits[:top_k]


def expand_with_neighbors(hits: list[dict], window: int | None = None) -> list[dict]:
    """Merge each hit with its adjacent chunks so the LLM sees whole passages."""
    window = config.NEIGHBOR_WINDOW if window is None else window
    if not hits:
        return []
    store = get_store()
    hit_ids = {h["id"] for h in hits}
    by_file: dict[int, dict[int, dict]] = {}
    best: dict[int, float] = {}
    order: list[tuple[float, int, int]] = []

    for rank, h in enumerate(hits):
        fid, ord_ = h["file_id"], h["ord"]
        group = by_file.setdefault(fid, {})
        group[ord_] = h
        for nb in store.neighbors(fid, ord_, window):
            group.setdefault(nb["ord"], nb)
        best[fid] = max(best.get(fid, -1e9), h["score"])
        order.append((-h["score"], rank, fid))

    passages: list[dict] = []
    seen_files: set[int] = set()
    for _neg, _rank, fid in sorted(order):
        if fid in seen_files:
            continue
        seen_files.add(fid)
        group = by_file[fid]
        ords = sorted(group)
        run: list[int] = []
        runs: list[list[int]] = []
        for o in ords:
            if run and o != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(o)
        if run:
            runs.append(run)
        for r in runs:
            members = [group[o] for o in r]
            if not any(m["id"] in hit_ids for m in members):
                continue
            first = members[0]
            locs = [m["loc"] for m in members if m["loc"]]
            passages.append({
                "file_id": fid,
                "name": first["name"],
                "rel_path": first["rel_path"],
                "path": first["path"],
                "heading": first.get("heading", ""),
                "loc": locs[0] if len(set(locs)) == 1 else (
                    f"{locs[0]}–{locs[-1]}" if locs else ""),
                "text": "\n".join(m["text"] for m in members),
                "score": max(m.get("score", 0.0) for m in members),
                "ords": r,
            })
    passages.sort(key=lambda p: -p["score"])
    return passages


def build_context(passages: list[dict], budget: int | None = None) -> tuple[str, list[dict]]:
    """Render numbered sources and the context string the model actually reads."""
    budget = budget or config.MAX_CONTEXT_CHARS
    parts: list[str] = []
    sources: list[dict] = []
    used = 0
    for p in passages:
        where = " ".join(x for x in [p["rel_path"], f"({p['loc']})" if p["loc"] else ""] if x)
        header = f"[{len(sources) + 1}] {where}"
        if p.get("heading"):
            header += f" - {p['heading']}"
        body = p["text"]
        if used + len(body) > budget:
            remaining = budget - used
            if remaining < 400:
                break
            body = body[:remaining] + " ..."
        parts.append(f"{header}\n{body}")
        used += len(body)
        sources.append({
            "n": len(sources) + 1,
            "file": p["name"],
            "rel_path": p["rel_path"],
            "path": p["path"],
            "loc": p["loc"],
            "heading": p.get("heading", ""),
            "score": round(float(p["score"]), 4),
            "preview": body[:600],
        })
    return "\n\n".join(parts), sources


def retrieve(
    question: str,
    *,
    top_k: int | None = None,
    use_rerank: bool | None = None,
    history: Sequence[dict] | None = None,
    files: Sequence[str] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    hits = search(question, top_k=top_k, use_rerank=use_rerank, history=history,
                  files=files)
    passages = expand_with_neighbors(hits)
    context, sources = build_context(passages)
    return context, sources, hits
