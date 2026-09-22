"""Deep mode: splitting a question, and interleaving the parts' shortlists.

The model call is stubbed. What is worth testing here is not whether a model can
split a sentence -- that is the model's job and it changes with the model -- but the
machinery around it: that a malformed reply degrades to the ordinary single search
instead of failing the question, and that the interleave actually protects the
quieter part, which is the entire reason this module exists.

Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = "  pass  ", "  FAIL  "
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print((PASS if ok else FAIL) + name + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def _with_reply(monkey_reply):
    """Run decompose.plan against a stubbed model reply."""
    from app.core import router
    from app.retrieval import decompose

    real = router.complete
    router.complete = lambda *a, **k: (monkey_reply, "stub")
    try:
        return decompose.plan("compare the R550 and the SL90 failure rates")
    finally:
        router.complete = real


def main() -> int:
    from app import config
    from app.retrieval import decompose

    print("-" * 60)
    print("planning: what the model says vs what we act on")
    print("-" * 60)

    parts = _with_reply('["what is the R550 failure rate", "what is the SL90 failure rate"]')
    check("a clean JSON split is used", parts == [
        "what is the R550 failure rate", "what is the SL90 failure rate"], str(parts))

    parts = _with_reply('```json\n["first question here", "second question here"]\n```')
    check("a fenced reply is unwrapped", len(parts) == 2, str(parts))

    # Models truncate mid-array when max_tokens runs out. The complete items are
    # still good; the half-written one must not become a search query.
    parts = _with_reply('["a complete sub question", "another complete one", "trunca')
    check("a truncated array keeps only whole items",
          parts == ["a complete sub question", "another complete one"], str(parts))

    check("an empty split means do not split", _with_reply("[]") == [])
    check("prose instead of JSON does not split", _with_reply("I cannot do that") == [])

    # One part is a paraphrase, not a decomposition: searching it *instead of* the
    # user's own wording is strictly worse than not splitting at all.
    check("a single part is rejected", _with_reply('["only one thing"]') == [])

    # The question echoed back adds a search that returns what the plain search
    # already returns.
    parts = _with_reply('["compare the R550 and the SL90 failure rates", "the SL90 rate"]')
    check("the question echoed back is dropped",
          parts == [], str(parts))

    def boom(*_a, **_k):
        raise RuntimeError("provider down")
    from app.core import router
    real = router.complete
    router.complete = boom
    try:
        check("a dead provider degrades to one search",
              decompose.plan("compare a and b") == [])
    finally:
        router.complete = real

    check("max_parts below 2 disables splitting",
          decompose.plan("compare a and b", max_parts=1) == [])

    print("\n" + "-" * 60)
    print("interleaving: the quiet part keeps its evidence")
    print("-" * 60)

    # A corpus where one part dominates: "loud" scores far above "quiet" everywhere.
    # A single ranked list would fill top_k with loud and never mention quiet, which
    # is the failure this module is built to prevent.
    from app.retrieval import search as search_mod

    loud = [{"id": i, "file_id": 1, "ord": i, "score": 0.9, "name": "a.pdf",
             "rel_path": "a.pdf", "path": "a.pdf", "loc": f"p{i}", "heading": "",
             "text": f"loud passage {i}"} for i in range(1, 5)]
    quiet = [{"id": 100 + i, "file_id": 2, "ord": i, "score": 0.1, "name": "b.pdf",
              "rel_path": "b.pdf", "path": "b.pdf", "loc": f"p{i}", "heading": "",
              "text": f"quiet passage {i}"} for i in range(1, 5)]

    calls: list[str] = []

    def fake_search(q, **_k):
        calls.append(q)
        return quiet if "quiet" in q else loud

    real_search = search_mod.search
    real_neighbors = search_mod.expand_with_neighbors
    search_mod.search = fake_search
    # Neighbour expansion needs the store; the ordering it hands back is what is
    # under test, so it is replaced by the identity plus the score re-sort it does.
    search_mod.expand_with_neighbors = lambda hits, window=None: sorted(
        [{**h, "ords": [h["ord"]]} for h in hits], key=lambda p: -p["score"])
    try:
        ctx, sources, hits, coverage = decompose.retrieve(
            "loud thing and quiet thing",
            ["tell me about the loud thing", "tell me about the quiet thing"],
            per_part=3)
    finally:
        search_mod.search = real_search
        search_mod.expand_with_neighbors = real_neighbors

    check("every part was searched, plus the original wording",
          len(calls) == 3, f"{len(calls)} searches")

    files = [s["rel_path"] for s in sources]
    check("the low-scoring part still reaches the context",
          "b.pdf" in files, str(files))

    quiet_used = next(c["used"] for c in coverage if "quiet" in c["query"])
    check("the quiet part contributed passages", quiet_used > 0, f"used={quiet_used}")

    # The interleave must survive expand_with_neighbors, which re-sorts by score and
    # would otherwise put every loud passage back in front.
    check("the interleave is not undone by the score re-sort",
          files[0] != files[1] if len(files) > 1 else False,
          " -> ".join(files[:4]))

    check("coverage is reported per part", len(coverage) == 3 and
          all("query" in c and "found" in c and "used" in c for c in coverage))

    # A part that finds nothing must not cost the others their slots.
    search_mod.search = lambda q, **_k: [] if "quiet" in q else loud
    search_mod.expand_with_neighbors = lambda hits, window=None: sorted(
        [{**h, "ords": [h["ord"]]} for h in hits], key=lambda p: -p["score"])
    try:
        ctx2, sources2, _h, cov2 = decompose.retrieve(
            "loud thing and quiet thing",
            ["tell me about the loud thing", "tell me about the quiet thing"],
            per_part=3)
    finally:
        search_mod.search = real_search
        search_mod.expand_with_neighbors = real_neighbors
    check("a part with no evidence does not empty the answer",
          bool(ctx2) and len(sources2) > 0, f"{len(sources2)} sources")

    # The caller searches the original wording while the planning call is in flight.
    # Those hits must be used, not thrown away and searched again.
    calls.clear()
    search_mod.search = fake_search
    search_mod.expand_with_neighbors = lambda hits, window=None: sorted(
        [{**h, "ords": [h["ord"]]} for h in hits], key=lambda p: -p["score"])
    try:
        _c, s3, _h, cov3 = decompose.retrieve(
            "loud thing and quiet thing",
            ["tell me about the loud thing", "tell me about the quiet thing"],
            per_part=3, base_hits=loud)
    finally:
        search_mod.search = real_search
        search_mod.expand_with_neighbors = real_neighbors
    check("pre-searched hits are reused, not re-run",
          len(calls) == 2, f"{len(calls)} searches for 2 parts")
    # The reused hits are trimmed to per_part like any other part's, so `found`
    # reports what was taken, not what the earlier search happened to return.
    check("the reused hits still count as the first part",
          len(cov3) == 3 and cov3[0]["used"] > 0
          and cov3[0]["query"] == "loud thing and quiet thing",
          f"coverage[0]={cov3[0] if cov3 else None}")
    check("reusing them does not lose the quiet part",
          any(x["rel_path"] == "b.pdf" for x in s3),
          str([x["rel_path"] for x in s3]))

    print("\n" + "-" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all deep-mode tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
