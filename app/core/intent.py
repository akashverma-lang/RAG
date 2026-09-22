"""What kind of message is this?

Two cheap, local checks that decide how much machinery a message deserves.  Both
exist because of a specific wrong answer:

* A greeting used to be sent to the SQL planner along with the previous turn, and
  came back holding the *previous question's* query -- you said "Hi" and got a chart.
  Sent to the document search instead, it answered with whatever passages happened
  to rank highest, which is worse: confident, sourced, and about nothing you asked.
* A standalone question used to carry the whole conversation into the planner, which
  invites the model to answer the earlier question again.  Only a genuine follow-up
  needs that context.
"""
from __future__ import annotations

import re

_SMALL_TALK = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|sup|namaste|hola|salaam|thanks?|thank you|thx|ty|"
    r"ok(ay)?|k|cool|nice|great|awesome|perfect|good|good (morning|afternoon|evening|night)|"
    r"bye|goodbye|see you|see ya|how are you|who are you|what can you do|what do you do|"
    r"help|test|testing)"
    r"(\s+(there|all|again|mate|buddy|team|guys|everyone|bro|sir|man))?"
    r"\b[\s!.?,]*$", re.IGNORECASE)

# A follow-up leans on the previous turn; a fresh question stands alone.
_ANAPHORIC = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|there|the same|above|instead|"
    r"as well|too|also|which one|what about|how about)\b", re.IGNORECASE)


def is_small_talk(text: str) -> bool:
    """A greeting or pleasantry with no question in it."""
    return bool(_SMALL_TALK.match((text or "").strip()))


def is_follow_up(question: str) -> bool:
    """Does answering this need the previous turn?"""
    q = (question or "").strip()
    if not q or is_small_talk(q):
        return False
    # Four words or fewer is a fragment ("and West?", "by month"). Anything longer
    # that does not point at an earlier turn is treated as a fresh question, so it is
    # answered on its own terms rather than in the shadow of the last one.
    return bool(_ANAPHORIC.search(q)) or len(q.split()) <= 4
