"""Where the business is leaving something on the table, and what it is worth.

A *gap* here is not an opinion. It is the distance between one segment and another
segment **in the same dataset** that is already doing better, multiplied by the
volume of the one behind. That definition does two things the usual "AI recommends"
output cannot:

* it makes the prize arithmetic rather than rhetoric -- the number comes from the
  user's own rows, so it can be checked, and the SQL that produced it ships with it;
* it makes the target demonstrably reachable, because a peer is already there. The
  benchmark is never a round number someone liked the sound of; it is a plant, a
  line, a region that hit it last quarter with the same process.

The model is asked to write the plan, never the number. That split is the whole
design: a plausible-sounding "£2M opportunity" invented by a language model is worse
than no answer, because it is indistinguishable from a real one until someone acts
on it.

Sizing method: segments are ranked on a measure, the benchmark is the volume-weighted
**top-quartile** performer, and each segment below it contributes
``(benchmark - its value) x its share of volume``. Segments under MIN_ROWS are
excluded -- a two-record segment "underperforming" is noise, not a gap.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from app import config
from app.analysis import semantics, trends
from app.tables import catalog

MIN_ROWS = 30                 # a segment smaller than this is noise, not a gap
MIN_SEGMENTS = 4              # fewer than this and "top quartile" means nothing
MIN_RELATIVE_GAP = 0.08       # below 8% behind the benchmark is not worth a slide
MAX_GAPS = 12


@dataclass
class Gap:
    """One measurable shortfall, sized in the units the business already uses."""
    id: str
    table: str
    kind: str                       # benchmark | decline | concentration | coverage
    title: str
    measure: str
    dimension: str = ""
    unit: str = ""
    benchmark: float = 0.0
    benchmark_holder: str = ""
    laggards: list[dict] = field(default_factory=list)
    prize: float = 0.0
    prize_basis: str = ""
    records_affected: int = 0
    severity: str = "medium"
    confidence: str = "measured"
    evidence_sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    good_high: bool = True

    def as_dict(self) -> dict:
        return {
            "id": self.id, "table": self.table, "kind": self.kind,
            "title": self.title, "measure": self.measure, "dimension": self.dimension,
            "unit": self.unit, "benchmark": self.benchmark,
            "benchmark_holder": self.benchmark_holder, "laggards": self.laggards,
            "prize": self.prize, "prize_basis": self.prize_basis,
            "records_affected": self.records_affected, "severity": self.severity,
            "confidence": self.confidence, "evidence_sql": self.evidence_sql,
            "columns": self.columns, "rows": self.rows, "good_high": self.good_high,
        }


def _slug(*parts: str) -> str:
    raw = "-".join(str(p) for p in parts if p)
    return re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:60] or "gap"


def _quartile(values: Sequence[float], good_high: bool) -> float:
    """The top-quartile value: what a quarter of the segments already manage."""
    ordered = sorted(values, reverse=good_high)
    if not ordered:
        return 0.0
    idx = max(0, min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.25)) - 1))
    return float(ordered[idx])


def _rate_sql(measure_sql: str) -> str:
    """The measure expressed per record.

    Comparing totals across segments measures how big they are, not how well they
    do: a zone with a tenth of the claims will always have a tenth of the value, and
    calling it "100% behind" is arithmetic dressed up as a finding. Per record, a
    small zone with good numbers ranks where it deserves to.

    An average is already per record and is left alone.
    """
    if re.search(r"\bAVG\s*\(", measure_sql, re.IGNORECASE):
        return measure_sql
    return f"CAST({measure_sql} AS REAL) / NULLIF(COUNT(*), 0)"


def _segments(table: str, dim: str, measure_sql: str, measure_name: str,
              good_high: bool) -> tuple[str, list[list]]:
    """Every segment on one dimension: volume, total, and the per-record rate."""
    q = catalog.quote
    rate = _rate_sql(measure_sql)
    stmt = (f"SELECT {q(dim)} AS segment, COUNT(*) AS records, "
            f"{rate} AS per_record, {measure_sql} AS total FROM {q(table)} "
            f"WHERE {q(dim)} IS NOT NULL AND {q(dim)} <> '' "
            f"GROUP BY 1 HAVING records >= {MIN_ROWS} "
            f"ORDER BY per_record {'DESC' if good_high else 'ASC'}")
    _cols, rows = trends._rows(stmt, limit=400)
    clean = [[r[0], int(r[1] or 0), float(r[2])]
             for r in rows if r[2] is not None and str(r[0]).strip()]
    return stmt, clean


# An operating unit: somewhere that does the same job as its siblings, so that one
# doing it better is a lesson the others can learn.
_PEER_NAME = re.compile(
    r"(^|_)(zone|region|area|territory|district|cluster|branch|plant|line|shop|"
    r"site|depot|warehouse|store|office|centre|center|location|team|crew|dealer|"
    r"distributor|vendor|supplier|sd|division|unit|facility|factory|workshop)(_|$)",
    re.IGNORECASE)

# A column that describes the record's own state or kind rather than who handled it.
_NOT_PEER_NAME = re.compile(
    r"(^|_)(status|state|stage|phase|type|kind|class|category|flag|result|outcome|"
    r"disposition|reason|mode|priority|severity|level|grade|currency|source|"
    r"channel|method|label|tag)(_|$)",
    re.IGNORECASE)

# Values that betray a lifecycle even when the column is not named like one.
_STATE_VALUE = re.compile(
    r"^(approved|rejected|pending|open|closed|complete[d]?|cancel+ed|draft|"
    r"submitted|active|inactive|new|in progress|on hold|done|failed|passed|"
    r"yes|no|true|false|y|n|0|1)\b",
    re.IGNORECASE)


def _peer_kind(col) -> str:
    """Is this dimension a set of peers, a lifecycle, or merely a guess?

    Benchmarking is only meaningful across units doing the same job. Run against
    claim_status it produced "bring Approved up to Pending for Submission", which is
    not underperformance at all -- those are different stages of the same claim, and
    acting on it would mean nothing.
    """
    name = col.name or ""
    if _NOT_PEER_NAME.search(name):
        return "not-peer"
    values = [str(v) for v in (getattr(col, "values", None) or [])]
    if values and sum(bool(_STATE_VALUE.match(v.strip())) for v in values) >= max(
            2, len(values) // 2):
        return "not-peer"
    if _PEER_NAME.search(name):
        return "peer"
    return "maybe"


def _is_identifier(info, measure) -> bool:
    """A numeric column whose values are all different is a serial number.

    Summing one produces "total SR_NO", ranks plants by how many rows they happen to
    have, and reads as a finding. The catalogue types most of these as ids already;
    this catches the ones it typed as numbers.
    """
    for col in info.columns:
        if col.role != "number" or catalog.quote(col.name) not in measure.sql:
            continue
        rows = getattr(info, "n_rows", 0) or 0
        if rows >= MIN_ROWS and col.n_distinct >= rows * 0.95:
            return True
        if re.search(r"(^|_)(id|no|num|number|sr|serial|code|ref)(_|$)",
                     col.name, re.IGNORECASE):
            return True
    return False


def _benchmark_gap(table: str, dim, measure, info) -> Gap | None:
    """One dimension, one measure: who is behind the quarter that is already ahead."""
    stmt, rows = _segments(table, dim.name, measure.sql, measure.name,
                           measure.good_high)
    if len(rows) < MIN_SEGMENTS:
        return None

    values = [r[2] for r in rows]
    bench = _quartile(values, measure.good_high)
    if bench == 0:
        return None

    holder = next((r[0] for r in rows if (r[2] >= bench if measure.good_high
                                          else r[2] <= bench)), "")

    # "Behind" depends on which way is good. A maintenance cost of 40 beats one of
    # 90; a throughput of 40 does not beat one of 90. Getting this backwards turns
    # the best performer into the problem.
    laggards: list[dict] = []
    prize = 0.0
    for seg, recs, val in rows:
        behind = (bench - val) if measure.good_high else (val - bench)
        if behind <= 0:
            continue
        rel = behind / abs(bench) if bench else 0.0
        if rel < MIN_RELATIVE_GAP:
            continue
        # Both sides of the comparison are per record, so the shortfall is per
        # record too, and multiplying by the volume behind it gives the total in the
        # measure's own units.
        gain = behind * recs
        laggards.append({
            "segment": str(seg), "records": recs, "value": round(val, 4),
            "behind": round(behind, 4), "behind_pct": round(rel * 100, 1),
            "gain": round(gain, 2),
        })
        prize += gain

    if not laggards:
        return None
    laggards.sort(key=lambda d: -d["gain"])

    total_now = sum(abs(r[2]) for r in rows) or 1.0
    lift = 100.0 * prize / total_now
    severity = "high" if lift >= 15 else "medium" if lift >= 5 else "low"

    unit = measure.label or measure.name
    return Gap(
        id=_slug(table, dim.name, measure.name),
        table=table, kind="benchmark",
        title=(f"{len(laggards)} {dim.label} behind the best quarter on "
               f"{measure.label or measure.name}"),
        measure=measure.label or measure.name, dimension=dim.label or dim.name,
        unit=unit, benchmark=round(bench, 4), benchmark_holder=str(holder),
        laggards=laggards[:10], prize=round(prize, 2),
        prize_basis=(
            f"Every {dim.label} below the top-quartile value of {bench:,.2f} "
            f"brought up to it, weighted by how many records sit in each. "
            f"{holder} already achieves this, on the same data."),
        records_affected=sum(d["records"] for d in laggards),
        severity=severity, evidence_sql=stmt,
        columns=["segment", "records", f"{measure.name} per record"],
        rows=[[r[0], r[1], round(r[2], 4)] for r in rows[:12]],
        good_high=measure.good_high,
    )


def _coverage_gaps(table: str, info) -> list[Gap]:
    """Columns too empty to steer by. A different kind of gap: blocked, not behind."""
    q = catalog.quote
    out: list[Gap] = []
    stmt = f"SELECT COUNT(*) FROM {q(table)}"
    _c, rows = trends._rows(stmt, limit=1)
    total = int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else 0
    if total < MIN_ROWS:
        return out

    for col in info.columns:
        if col.role not in {"number", "date", "category"}:
            continue
        s = (f"SELECT COUNT(*) FROM {q(table)} WHERE {q(col.name)} IS NULL "
             f"OR TRIM(CAST({q(col.name)} AS TEXT)) = ''")
        _c2, r2 = trends._rows(s, limit=1)
        missing = int(r2[0][0]) if r2 and r2[0] and r2[0][0] is not None else 0
        pct = 100.0 * missing / total if total else 0.0
        if pct < 20.0 or pct >= 100.0:
            continue
        out.append(Gap(
            id=_slug(table, col.name, "coverage"), table=table, kind="coverage",
            title=f"{col.label or col.name} is missing on {pct:.0f}% of records",
            measure=col.label or col.name, unit="records",
            prize=float(missing), records_affected=missing,
            prize_basis=(
                f"{missing:,} of {total:,} rows have no {col.label or col.name}. "
                f"Every question about it is answered from the {100 - pct:.0f}% that "
                f"do, which silently excludes the rest."),
            severity="high" if pct >= 50 else "medium",
            confidence="measured", evidence_sql=s,
            columns=["missing", "total"], rows=[[missing, total]]))
    return out[:3]


def find(tables: Sequence[str] | None = None, exclude: set[str] | None = None,
         limit: int = MAX_GAPS) -> list[Gap]:
    """Every measurable gap across the chosen tables, biggest prize first."""
    from app.analysis import performance

    # The diagnostics already know which columns are corrupt -- an hour meter reading
    # of 8.9 billion, a saving field with impossible values. Ranking on one produces a
    # confident finding about an artefact, so when the caller has not run them, run
    # them here rather than benchmarking on rubbish.
    if exclude is None:
        exclude = set()
        try:
            from app.analysis import diagnose
            for report in diagnose.scan(""):
                exclude |= trends.bad_measures(report)
        except Exception as exc:                                # noqa: BLE001
            print(f"[gaps] diagnostics unavailable: {exc}", flush=True)
    infos = [t for t in catalog.list_tables()
             if not tables or t.name in set(tables)]
    found: list[Gap] = []

    for info in infos:
        try:
            measures = performance._measures(info, exclude)
            all_dims = performance._dimensions(info)
        except Exception as exc:                                # noqa: BLE001
            print(f"[gaps] {info.name}: cannot profile ({exc})", flush=True)
            continue

        kinds = {d.name: _peer_kind(d) for d in all_dims}
        peers = [d for d in all_dims if kinds[d.name] == "peer"]
        maybes = [d for d in all_dims if kinds[d.name] == "maybe"]
        # Real operating units if the table has them; otherwise the unclassified
        # ones, flagged as a weaker comparison.
        dims = peers or maybes
        inferred = not peers

        for measure in measures:
            # record_count ranks segments by how big they are, which is a fact about
            # the data rather than a shortfall in the business.
            if measure.name == "record_count":
                continue
            if _is_identifier(info, measure):
                continue
            for dim in dims:
                try:
                    gap = _benchmark_gap(info.name, dim, measure, info)
                except Exception as exc:                        # noqa: BLE001
                    print(f"[gaps] {info.name}.{dim.name}: {exc}", flush=True)
                    continue
                if gap:
                    if inferred:
                        gap.confidence = "inferred"
                        gap.prize_basis += (
                            f" Note: {gap.dimension} was not recognised as a set of "
                            f"comparable operating units, so check that these "
                            f"segments really do the same job before acting.")
                    found.append(gap)
        try:
            found += _coverage_gaps(info.name, info)
        except Exception as exc:                                # noqa: BLE001
            print(f"[gaps] {info.name} coverage: {exc}", flush=True)

    # Rank by how much of the measure is in play, not by raw prize: a prize of 900
    # on a measure totalling 1,000 matters more than 5,000 on one totalling 900,000.
    def weight(g: Gap) -> float:
        rank = {"high": 3, "medium": 2, "low": 1}.get(g.severity, 1)
        return rank * 1e9 + g.prize

    found.sort(key=weight, reverse=True)

    # One gap per (table, dimension) -- three measures on the same breakdown are the
    # same conversation, and a deck of near-duplicates reads as padding.
    seen: set[tuple[str, str]] = set()
    unique: list[Gap] = []
    for g in found:
        key = (g.table, g.dimension or g.measure)
        if key in seen:
            continue
        seen.add(key)
        unique.append(g)
    return unique[:limit]


def brief(gaps: Sequence[Gap]) -> str:
    """The measured facts, laid out for the model to write a plan against."""
    if not gaps:
        return "No measurable gaps were found in the selected data."
    out: list[str] = []
    for i, g in enumerate(gaps, start=1):
        out.append(f"GAP {i} (id={g.id}, severity={g.severity})")
        out.append(f"  what: {g.title}")
        out.append(f"  table: {g.table}")
        if g.kind == "benchmark":
            direction = "higher is better" if g.good_high else "LOWER is better"
            out.append(f"  measure: {g.measure} ({direction})")
            out.append(f"  benchmark (per record) already achieved by "
                       f"{g.benchmark_holder}: {g.benchmark:,.2f}")
            out.append(f"  behind: " + "; ".join(
                f"{d['segment']} at {d['value']:,.2f} ({d['behind_pct']:.0f}% behind, "
                f"{d['records']:,} records)" for d in g.laggards[:5]))
            out.append(f"  prize if all reach the benchmark: {g.prize:,.2f} {g.unit}")
        else:
            out.append(f"  scale: {g.prize:,.0f} records affected")
        out.append(f"  how that was computed: {g.prize_basis}")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- plan
PLAN_SYSTEM = """You are an operations analyst. Measured gaps are given to you. For
each one, write the plan to close it.

Reply with ONLY a JSON object keyed by gap id:
{"<id>": {"why": "...", "actions": ["...", "...", "..."], "owner": "...",
          "horizon": "...", "verify": "...", "risk": "..."}}

Rules:
- NEVER state a number that is not in the measured facts. The prize has been computed
  from the data; do not restate, round, recompute or embellish it.
- "why": the two or three most likely operational reasons this gap exists, phrased as
  things to check, not as conclusions. You have the numbers, not the context.
- "actions": exactly three concrete steps, each one a thing a named person could
  start next week. No "leverage synergies", no "implement best practices". Say what
  is done, to what, by whom.
- The benchmark is already achieved by a peer in the same business, so the first
  action is almost always to go and look at how they do it.
- "owner": the role that would own this, e.g. "Zone service manager".
- "horizon": realistic time to see movement, e.g. "one quarter".
- "verify": the single measurement that proves it worked - reuse the measure named
  in the facts.
- "risk": the most likely way this effort produces nothing, stated plainly.
- Keep every field under 40 words. Write for someone who has ten minutes."""


def plan(found: Sequence[Gap], model: str | None = None) -> dict[str, dict]:
    """Ask the model how to close each gap. Never for how big it is.

    Returns {gap_id: plan}. A gap with no plan is still shown: the measurement is the
    part that had to be right, and the reader can act on it without the prose.
    """
    if not found:
        return {}
    import json as _json

    from app.core import router

    try:
        text, _used = router.complete(
            [{"role": "system", "content": PLAN_SYSTEM},
             {"role": "user", "content": brief(found)}],
            role="answer", model=model, temperature=0.2,
            timeout=config.GAP_PLAN_TIMEOUT, max_tokens=2200)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[gaps] planning failed: {exc}", flush=True)
        return {}

    body = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip(),
                  flags=re.MULTILINE).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = _json.loads(body[start:end + 1])
    except Exception:                                           # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}

    known = {g.id for g in found}
    out: dict[str, dict] = {}
    for key, value in data.items():
        if key not in known or not isinstance(value, dict):
            continue
        actions = value.get("actions") or []
        if isinstance(actions, str):
            actions = [actions]
        out[key] = {
            "why": str(value.get("why", ""))[:400],
            "actions": [str(a)[:300] for a in actions][:4],
            "owner": str(value.get("owner", ""))[:120],
            "horizon": str(value.get("horizon", ""))[:120],
            "verify": str(value.get("verify", ""))[:300],
            "risk": str(value.get("risk", ""))[:300],
        }
    return out
