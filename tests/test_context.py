"""Deciding when the previous question still applies.

A short question used to have the previous one glued onto it automatically -- anything
eight words or fewer. That is right for "and the valve-cover gaskets?" and wrong for
"What is the Q4 revenue?", where it drags retrieval back to a subject the user has
finished with. It is the behaviour people describe as the assistant being stuck in the
earlier conversation.

Length cannot tell those apart; meaning can. Measured with the indexing embedder over
twelve pairs, genuine follow-ups scored 0.519 and above against the previous question
and changes of subject 0.489 and below, so the threshold sits between them.

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


PREV = "What is the P90 failure time for the O2 sensors?"
HISTORY = [{"role": "user", "content": PREV}]

# (question, should the previous question be folded in, why)
CASES = [
    ("And the valve-cover gaskets?", True, "elliptical follow-up"),
    ("What mitigations were recommended?", True, "same subject, stated shortly"),
    ("What about maintainability?", True, "same subject"),
    ("What is the Q4 revenue?", False, "a new subject, stated shortly"),
    ("How many claims in 2025?", False, "a new subject"),
    ("Where is my sales spreadsheet?", False, "a new subject"),
    ("List the plant names", False, "a new subject"),
    ("What does it say about mitigation?", True, "a pronoun depends on the last turn"),
    ("Compare that with the reliability target", True, "a demonstrative depends on it"),
    ("Explain the complete redundancy architecture proposed for the whole campus",
     False, "long and self-contained"),
]


def main() -> int:
    from app import config
    from app.retrieval import search

    print("-" * 66)
    print("folding the previous question in, or leaving it out")
    print("-" * 66)
    for question, should_fold, why in CASES:
        out = search.expand_query(question, HISTORY)
        folded = out != question
        check(f"{'carries' if should_fold else 'stands alone':<13} {why}",
              folded == should_fold,
              f"{question[:46]}")

    print("\n" + "-" * 66)
    print("the edges")
    print("-" * 66)
    check("no history means nothing to fold",
          search.expand_query("and the gaskets?", []) == "and the gaskets?")
    check("history with no user turn is ignored",
          search.expand_query("and the gaskets?",
                              [{"role": "assistant", "content": "x"}])
          == "and the gaskets?")
    folded = search.expand_query("And the valve-cover gaskets?", HISTORY)
    check("the folded query keeps the new question",
          "valve-cover gaskets" in folded, folded[:70])
    check("...and adds the previous one", PREV in folded, folded[:70])

    # The check costs an embedding call, so it must only run where it can change the
    # outcome: a long question never folds and must not pay for it.
    calls = {"n": 0}
    real = search._same_topic
    search._same_topic = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), True)[1]
    def delta(fn):
        before = calls["n"]
        fn()
        return calls["n"] - before

    try:
        # Nine words: past the point where folding is even considered. (An earlier
        # version of this line was exactly eight and so legitimately did pay.)
        n = delta(lambda: search.expand_query(
            "Explain the complete redundancy architecture proposed for the whole campus",
            HISTORY))
        check("a long question does not pay for the check", n == 0, f"{n} calls")
        n = delta(lambda: search.expand_query("What does it say about that?", HISTORY))
        check("a pronoun does not pay for it either", n == 0, f"{n} calls")
        n = delta(lambda: search.expand_query("What about maintainability?", HISTORY))
        check("a short question without a pronoun does", n == 1, f"{n} calls")
    finally:
        search._same_topic = real

    # An embedder that will not load must not lose context that was needed: carrying
    # it when it was not is a loss of precision, dropping it is a loss of the answer.
    real_same = search._same_topic
    import app.core.embed as embed_mod
    real_enc = embed_mod.encode_queries

    def boom(*a, **k):
        raise RuntimeError("embedder unavailable")
    embed_mod.encode_queries = boom
    try:
        out = search.expand_query("What about maintainability?", HISTORY)
        check("a broken embedder falls back to carrying context", out != "What about maintainability?")
    finally:
        embed_mod.encode_queries = real_enc
        search._same_topic = real_same

    check("the threshold is configurable",
          0.0 < config.FOLLOW_UP_SIMILARITY < 1.0, str(config.FOLLOW_UP_SIMILARITY))

    print("\n" + "-" * 66)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all context tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
