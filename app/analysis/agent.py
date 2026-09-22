"""The analyst loop: look at the evidence, probe further, then write it up.

The loop is deliberately small.  It is handed the deterministic findings first --
data-quality defects, month-over-month movement, segment outlier rates -- so it
starts from measurements rather than from a blank page.  Its own job is to *follow
up*: check whether a spike is one dealer or all of them, whether a gap is seasonal,
whether two findings are the same underlying thing.

Every claim it ends up making is therefore attached to a query that produced it, and
those queries go through the same read-only guard as everything else.

Two models, because these are two different jobs.  Each step is a small structured
decision where latency multiplies by the step count, so it goes to the fast model;
the final write-up is one long piece of prose, so it goes to the better one.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from app import config
from app.core import router
from app.analysis import diagnose, performance, semantics, trends
from app.tables import catalog, sql

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

STEP_SYSTEM = """You are a data analyst investigating a dataset by running SQL.

{schema}
{meaning}

Evidence already measured for you (you did not have to run these):
{evidence}

Your job is to FOLLOW UP on that evidence and answer the user's request. Each turn,
reply with ONE JSON object and nothing else:

  {{"thought": "<at most 15 words>", "action": "sql", "sql": "SELECT ..."}}
  {{"thought": "<at most 15 words>", "action": "finish"}}

Keep "thought" to a short phrase. A long one pushes the query out of the reply and
the whole step is wasted.

Rules:
- SQLite. One statement. SELECT or WITH only. Never modify anything.
- Use the exact literal values listed for a column; they are the real spelling.
- Aggregate. Never ask for raw rows unless you need to see examples of something.
- Each query must test something the evidence does NOT already tell you. Do not
  re-run a query whose answer is already above.
- Prefer questions that separate causes: is a spike one segment or all of them? one
  month or the whole period? does it survive excluding the corrupted column?
- Anything a query returns is a fact. Anything else is a guess, and you must not
  report guesses as findings.
- You have at most {steps} queries. Finish as soon as you can support an answer;
  finishing early is better than padding.
- If the data cannot answer the request, finish and say so."""

REPORT_SYSTEM = """You are writing the findings of a data investigation for the person
who owns the data. Everything below was measured; nothing was estimated.

You are NOT writing a data-quality audit. Somebody opening this wants to know how the
business is doing, where the value is, and what to act on. Data problems matter only
where they change a conclusion, and they go at the end.

Write in markdown, in this order:

- **Open with the business picture** in two or three sentences: the scale, the
  direction of travel, and the single most important thing you found. Lead with a
  figure, not with a caveat.
- "## Where it is working" - the segments, products or periods that are performing.
  Say who, by how much, and what share of the total they represent. Concentration is
  itself a finding: "the top 5 of 34 dealers carry 71% of the value" tells someone
  where their business actually is.
- "## Where it is not" - the underperformers, with the size of the gap. Be specific
  about which segment and by what multiple. If a measure is one where LOWER is
  better, say so explicitly, or the reader will read the ranking backwards.
- "## What this means" - two to four conclusions a manager could act on, each tied to
  a figure above. Not generic advice: "ZO_ISC produces 1% of the value" is a
  conclusion, "improve dealer performance" is not.
- "## Caveats" - LAST, and short. Only the data problems that would actually change
  one of the conclusions above. Name the column and what it prevents. Do not list
  every missing field; an audit of null counts belongs nowhere near the top.

Rules:
- Quote real figures with thousands separators. Never invent one, never recompute one,
  never round away the point of a number.
- Never reformat an identifier or code; reproduce it exactly.
- Where a share or a multiple makes a number meaningful, give it: "400,099,572 of
  439,240,380 (91%)" says something that "400,099,572" does not.
- The data shows WHAT, not WHY. Never invent a cause for a movement.
- If the evidence genuinely does not support a conclusion, say so in one line rather
  than padding.
- No preamble, no "as an AI", no restating these instructions."""


@dataclass
class Step:
    n: int
    thought: str = ""
    sql: str = ""
    ok: bool = False
    error: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    took: float = 0.0

    def as_dict(self) -> dict:
        return {"n": self.n, "thought": self.thought, "sql": self.sql, "ok": self.ok,
                "error": self.error, "columns": self.columns, "rows": self.rows[:12],
                "took": self.took}

    def observation(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "no rows matched"
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join("" if v is None else str(v)[:60] for v in r)
                         for r in self.rows[:12])
        more = "" if len(self.rows) <= 12 else f"\n({len(self.rows)} rows, first 12)"
        return f"{head}\n{body}{more}"


def _parse(text: str) -> dict:
    body = _FENCE.sub("", (text or "").strip()).strip()
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(body[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    if re.match(r"^\s*(select|with)\b", body, re.IGNORECASE):
        return {"action": "sql", "sql": body}
    return {"action": "unparsed", "thought": "could not read the reply"}


def gather_evidence(table: str = "") -> dict:
    """The deterministic half. Runs first, costs no model calls."""
    diags = diagnose.scan(table)
    excluded: set[str] = set()
    caveats: dict[str, str] = {}
    for d in diags:
        excluded |= trends.bad_measures(d)
        caveats.update(trends.caveat_measures(d))
    return {"diagnostics": diags,
            "performance": performance.scan(table, exclude=excluded),
            "trends": trends.scan(table, exclude=excluded),
            "excluded_measures": sorted(excluded), "caveats": caveats}


def _evidence_text(evidence: dict) -> str:
    """Business first, then movement, then the caveats.

    Order matters more than it looks: the model writes in roughly the order it
    reads, and evidence that opens with fourteen data-quality defects produces a
    report about data quality rather than about the business.
    """
    parts = ["=== HOW THE BUSINESS IS DOING ==="]
    parts += [p.summary() for p in evidence.get("performance", [])]
    parts.append("=== WHAT CHANGED ===")
    parts += [t.summary() for t in evidence["trends"]]
    parts.append("=== CAVEATS ON THE NUMBERS ABOVE ===")
    parts += [d.summary() for d in evidence["diagnostics"]]
    if evidence["excluded_measures"]:
        parts.append("Columns too corrupted to total at all: "
                     + ", ".join(evidence["excluded_measures"]))
    if evidence.get("caveats"):
        parts.append("Usable but imperfect: "
                     + "; ".join(f"{k} ({v})" for k, v in evidence["caveats"].items()))
    return "\n\n".join(p for p in parts if p) or "No automated findings."


def run(question: str = "", table: str = "",
        model: str | None = None) -> Iterator[dict]:
    """Run the investigation, yielding events as they happen.

    Yields dicts: {type: status|evidence|step|report_token|done|error, ...}
    """
    started = time.time()
    request = question.strip() or (
        "Review this dataset. Find what is unreliable, what changed recently, and "
        "where performance is worst. Be specific about which segments and by how much.")

    yield {"type": "status", "text": "measuring data quality and trends"}
    try:
        evidence = gather_evidence(table)
    except Exception as exc:                                    # noqa: BLE001
        yield {"type": "error", "text": f"could not scan the data: {exc}"}
        return

    yield {"type": "evidence",
           "diagnostics": [d.as_dict() for d in evidence["diagnostics"]],
           "trends": [t.as_dict() for t in evidence["trends"]],
           "excluded": evidence["excluded_measures"],
           "took": round(time.time() - started, 1)}

    schema = catalog.render_schema()
    # What the columns mean and which direction is good. Without this, "lagging" has
    # no definition and the model has to invent one.
    meaning = semantics.render(table, request)
    if meaning:
        meaning = f"\nWhat this data means:\n{meaning}\n"
    system = STEP_SYSTEM.format(schema=schema, meaning=meaning,
                                evidence=_evidence_text(evidence),
                                steps=config.AGENT_MAX_STEPS)
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": request}]
    steps: list[Step] = []
    unparsed = 0

    for n in range(1, config.AGENT_MAX_STEPS + 1):
        if time.time() - started > config.AGENT_MAX_SECONDS:
            yield {"type": "status", "text": "time budget reached; writing up"}
            break
        yield {"type": "status", "text": f"investigating ({n}/{config.AGENT_MAX_STEPS})"}
        try:
            raw, used = router.complete(messages, role="agent", model=model,
                                        timeout=config.TABLE_PLAN_TIMEOUT,
                                        max_tokens=config.AGENT_MAX_TOKENS)
        except Exception as exc:                                # noqa: BLE001
            yield {"type": "status", "text": f"model unavailable ({exc}); writing up"}
            break

        plan = _parse(raw)
        if plan.get("action") == "finish":
            # A deliberate "finish" is a success, not a failed query.
            steps.append(Step(n=n, thought=str(plan.get("thought", ""))[:400], ok=True))
            yield {"type": "step", "step": steps[-1].as_dict(), "final": True}
            break
        if not plan.get("sql"):
            # An unparseable reply is usually a truncated one. Losing the whole
            # investigation to a single malformed step is far too brittle, so the
            # step is retried with a reminder rather than ending the loop.
            unparsed += 1
            if unparsed >= 3:
                yield {"type": "status", "text": "model kept replying unusably; writing up"}
                break
            # The reply itself has to go in too. Appending only the correction left
            # two user turns back to back, which some providers reject outright --
            # so the retry meant to rescue the step could end the run instead.
            messages.append({"role": "assistant", "content": raw[:1500]})
            messages.append({"role": "user", "content":
                             "That reply could not be read. Send ONE JSON object, with "
                             "a thought of at most 15 words, and nothing else."})
            continue

        step = Step(n=n, thought=str(plan.get("thought", ""))[:400],
                    sql=str(plan["sql"]).strip())
        result = sql.run(step.sql, limit=config.AGENT_ROWS_PER_STEP,
                         timeout=config.TABLE_SQL_TIMEOUT)
        step.ok, step.error, step.took = result.ok, result.error, result.took
        step.columns = result.columns
        step.rows = [list(r) for r in result.rows]
        steps.append(step)
        yield {"type": "step", "step": step.as_dict(), "model": used}

        messages.append({"role": "assistant", "content": raw[:1500]})
        messages.append({"role": "user",
                         "content": f"Result of query {n}:\n{step.observation()}"})

    # ---- write it up on the better model
    yield {"type": "status", "text": "writing the report"}
    transcript = "\n\n".join(
        f"Query {s.n}: {s.thought}\n{' '.join(s.sql.split())}\n{s.observation()}"
        for s in steps if s.sql)
    report_input = (
        f"Request: {request}\n\n"
        # The writer needs the same definitions the investigator had, or it will
        # describe a rising rejection rate as though nobody minds.
        + (f"=== What this data means ==={meaning}\n\n" if meaning else "")
        + f"=== Measured findings ===\n{_evidence_text(evidence)}\n\n"
        + f"=== Follow-up queries and their results ===\n{transcript or 'none'}")
    messages = [{"role": "system", "content": REPORT_SYSTEM},
                {"role": "user", "content": report_input}]
    text: list[str] = []
    try:
        for piece in router.stream(messages, role="answer", temperature=0.2):
            text.append(piece)
            yield {"type": "report_token", "text": piece}
    except Exception as exc:                                    # noqa: BLE001
        yield {"type": "error", "text": f"could not write the report: {exc}"}
        return

    yield {"type": "done", "took": round(time.time() - started, 1),
           "steps": len(steps), "report": "".join(text)}
