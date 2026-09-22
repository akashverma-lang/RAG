"""Choosing a model per job, pacing it, and failing over when a free tier says no.

Two different jobs, two different models:

* **agent** -- many small structured calls (write a query, decide the next probe).
  Latency multiplies by the step count, so this must be fast above all.  Gemini's
  newer "thinking" Flash models measured 30-42s per call and returned
  ``finish_reason: length`` with *zero* content tokens, the whole budget spent on
  hidden reasoning; flash-lite answers the same prompt in ~2s.
* **answer** -- one long call that writes prose for a human to read.  Quality and
  streaming matter, latency much less.

Free tiers are rate limited per minute and a loop fires calls back to back, so every
call goes through a client-side token bucket, and a provider that refuses is skipped
in favour of the next one rather than failing the whole analysis.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Iterator, Sequence

from app import config
from app.core import llm


class _Bucket:
    """One provider's requests-per-minute allowance, shared across threads."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)
        return delay


_buckets: dict[str, _Bucket] = {}
_buckets_lock = threading.Lock()


def _bucket(provider_key: str) -> _Bucket | None:
    """The allowance for a provider, or None where there is nothing to ration.

    The bucket exists because a free cloud tier counts requests per minute.  A model
    running on this machine counts nothing, so pacing it to 12 a minute bought no
    safety at all and simply sat the user in front of a spinner for five seconds
    before every second question.
    """
    try:
        provider = llm.providers().get(provider_key)
    except Exception:                              # noqa: BLE001
        provider = None
    if provider is not None and provider.local:
        return None
    with _buckets_lock:
        if provider_key not in _buckets:
            _buckets[provider_key] = _Bucket(config.PROVIDER_RPM)
        return _buckets[provider_key]


def _pace(provider_key: str) -> None:
    bucket = _bucket(provider_key)
    if bucket is not None:
        bucket.wait()


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "quota" in text or "exhaust" in text


def chain(role: str, model: str | None = None, fallback: bool = False) -> list[str]:
    """Models to try in order for this job, best first.

    A model named by the caller is normally used on its own: an analysis step that
    silently ran somewhere else would make its result unattributable. Answering a
    question is the opposite case -- ``fallback=True`` keeps the chosen model first
    but lets the rest of the chain catch it, so a free tier saying "rate limit" at
    the wrong moment costs the user a different model rather than their answer.
    """
    if model and not fallback:
        first = [model]
    elif model:
        first = [model, config.ANSWER_MODEL, config.AGENT_MODEL]
    elif role == "agent":
        first = [config.AGENT_MODEL, config.ANSWER_MODEL]
    else:
        first = [config.ANSWER_MODEL, config.AGENT_MODEL]
    known = llm.providers()
    out: list[str] = []
    for ref in first + [config.LLM_MODEL]:
        ref = (ref or "").strip()
        if not ref or ref in out:
            continue
        key = ref.split(llm.SEP)[0] if llm.SEP in ref else ""
        if key and key not in known:
            continue                      # provider not configured; skip quietly
        out.append(ref)
    return out or [config.LLM_MODEL]


def complete(messages: Sequence[dict], role: str = "agent", model: str | None = None,
             temperature: float = 0.0, timeout: float | None = None,
             max_tokens: int | None = None) -> tuple[str, str]:
    """Run a non-streaming call. Returns (text, model_actually_used).

    Raises the last error only when every model in the chain failed.
    """
    last: Exception | None = None
    for ref in chain(role, model):
        provider_key = ref.split(llm.SEP)[0] if llm.SEP in ref else "default"
        for attempt in range(config.PROVIDER_RETRIES + 1):
            _pace(provider_key)
            try:
                text = llm.complete(list(messages), model=ref,
                                    temperature=temperature, timeout=timeout,
                                    max_tokens=max_tokens)
                if text.strip():
                    return text, ref
                last = RuntimeError(f"{ref} returned an empty response")
                break                      # empty is not worth retrying on this model
            except Exception as exc:       # noqa: BLE001
                last = exc
                if _is_rate_limit(exc) and attempt < config.PROVIDER_RETRIES:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break                      # move to the next model in the chain
    raise last or RuntimeError("no model available")


def stream(messages: Sequence[dict], role: str = "answer", model: str | None = None,
           temperature: float | None = None, fallback: bool = False,
           max_tokens: int | None = None,
           on_failover: "Callable[[str, str, Exception], None] | None" = None,
           ) -> Iterator[str]:
    """Stream a call, falling over to the next model if the first refuses.

    Failover only happens before the first token: once text is flowing the caller
    has already shown it, and restarting would duplicate the answer.

    ``on_failover`` is called with (failed ref, next ref, error) before a retry, so
    the caller can tell the user their answer is coming from somewhere else. Being
    moved to another model without being told is its own kind of wrong answer.
    """
    last: Exception | None = None
    refs = chain(role, model, fallback=fallback)
    for i, ref in enumerate(refs):
        provider_key = ref.split(llm.SEP)[0] if llm.SEP in ref else "default"
        _pace(provider_key)
        started = False
        try:
            for piece in llm.stream_chat(list(messages), model=ref,
                                         temperature=temperature,
                                         max_tokens=max_tokens):
                started = True
                yield piece
            return
        except Exception as exc:           # noqa: BLE001
            last = exc
            if started:
                raise
            if on_failover and i + 1 < len(refs):
                try:
                    on_failover(ref, refs[i + 1], exc)
                except Exception:          # noqa: BLE001
                    pass
    raise last or RuntimeError("no model available")


def describe() -> dict:
    return {"agent": config.AGENT_MODEL, "answer": config.ANSWER_MODEL,
            "agent_chain": chain("agent"), "answer_chain": chain("answer"),
            "rpm": config.PROVIDER_RPM}
