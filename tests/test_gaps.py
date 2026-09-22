"""Finding business gaps, and refusing to invent them.

Every check here exists because the first version got it wrong against real data:

* it compared **totals** across segments, so a zone with a tenth of the claims was
  "100% behind" for being small. Segments are compared per record now.
* it summed a serial-number column and reported "total SR_NO", ranking plants by how
  many rows they happened to have.
* it benchmarked across **claim_status**, producing "bring Approved up to Pending for
  Submission" -- those are stages of one claim, not peers, and the advice meant
  nothing.
* it multiplied an average of 8.9 billion (a corrupt hour-meter column) by 128,000
  records and reported a prize of 4.9 quadrillion.

The SQL is stubbed so these test the reasoning, not the database.

Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
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


@dataclass
class Col:
    name: str
    role: str = "category"
    n_distinct: int = 5
    label: str = ""
    values: list = field(default_factory=list)


@dataclass
class Info:
    name: str = "t"
    n_rows: int = 1000
    columns: list = field(default_factory=list)


@dataclass
class M:
    name: str
    sql: str
    direction: str = "higher"
    label: str = ""

    @property
    def good_high(self) -> bool:
        return self.direction == "higher"


def main() -> int:
    from app.analysis import gaps

    print("-" * 64)
    print("which columns are peers, and which are stages of one thing")
    print("-" * 64)
    cases = [
        (Col("zone_name", values=["ZO_East", "ZO_North"]), "peer", "a geography"),
        (Col("plant_name", values=["Kagal", "Nashik"]), "peer", "a site"),
        (Col("branch_id", values=["B1", "B2"]), "peer", "a branch"),
        (Col("dealer_name", values=["A Ltd", "B Ltd"]), "peer", "a dealer"),
        (Col("claim_status", values=["Approved", "Rejected", "Pending"]),
         "not-peer", "a lifecycle, by name"),
        (Col("claim_type", values=["AMC - BD", "AMC - CM"]), "not-peer", "a kind"),
        (Col("outcome", values=["Passed", "Failed"]), "not-peer", "a result"),
        (Col("stage_flag", values=["x", "y"]), "not-peer", "a flag"),
        # Named innocuously, but the values give it away.
        (Col("disposition_code", values=["Open", "Closed", "Cancelled"]),
         "not-peer", "a lifecycle, by its values"),
        (Col("workstream", values=["Alpha", "Beta"]), "maybe", "unrecognised"),
    ]
    for col, want, why in cases:
        got = gaps._peer_kind(col)
        check(f"{why}: {col.name}", got == want, f"{got} (wanted {want})")

    print("\n" + "-" * 64)
    print("a serial number is not a measure")
    print("-" * 64)
    # Every value distinct across the whole table: a sequence, not a quantity.
    serial = Info(columns=[Col("sr", role="number", n_distinct=1000)], n_rows=1000)
    check("a fully unique numeric column is rejected",
          gaps._is_identifier(serial, M("total_sr", 'SUM("sr")')))
    named = Info(columns=[Col("claim_no", role="number", n_distinct=40)], n_rows=1000)
    check("an id-shaped name is rejected",
          gaps._is_identifier(named, M("total_claim_no", 'SUM("claim_no")')))
    real = Info(columns=[Col("index", role="number", n_distinct=33)], n_rows=1000)
    check("a real measure is kept",
          not gaps._is_identifier(real, M("total_index", 'SUM("index")')))

    print("\n" + "-" * 64)
    print("segments are compared per record, not by size")
    print("-" * 64)
    check("a SUM is divided by the row count",
          "NULLIF(COUNT(*), 0)" in gaps._rate_sql('SUM("x")'), gaps._rate_sql('SUM("x")'))
    check("an AVG is already per record",
          gaps._rate_sql('AVG("x")') == 'AVG("x")')

    # Two segments with identical per-record performance and very different volume.
    # Comparing totals calls the small one 90% behind; comparing rates calls them
    # equal, which they are.
    rows = [["Big", 900, 10.0], ["Small", 100, 10.0], ["Weak", 500, 5.0],
            ["Mid", 400, 8.0]]
    captured = {}

    def fake_segments(table, dim, measure_sql, measure_name, good_high):
        captured["called"] = True
        return "SELECT ...", [list(r) for r in rows]

    real_segments = gaps._segments
    gaps._segments = fake_segments
    try:
        info = Info(columns=[Col("zone", values=["a", "b"])])
        gap = gaps._benchmark_gap("t", Col("zone", label="zone"),
                                  M("value", 'SUM("v")', label="value"), info)
    finally:
        gaps._segments = real_segments

    check("a gap is produced", gap is not None)
    if gap:
        behind = {d["segment"] for d in gap.laggards}
        check("equal performers are not called behind",
              "Small" not in behind and "Big" not in behind, str(sorted(behind)))
        check("the genuinely weak segment is flagged", "Weak" in behind)
        # Benchmark is the top-quartile value (10.0); Weak is 5 behind over 500 rows.
        weak = next(d for d in gap.laggards if d["segment"] == "Weak")
        check("the prize is the shortfall times the volume behind it",
              abs(weak["gain"] - 2500.0) < 1.0, f"{weak['gain']}")
        check("the benchmark is a value a peer actually reached",
              abs(gap.benchmark - 10.0) < 1e-6, str(gap.benchmark))
        check("the peer holding it is named", gap.benchmark_holder in {"Big", "Small"},
              gap.benchmark_holder)
        check("the query that produced it is kept", bool(gap.evidence_sql))
        check("the arithmetic is explained", "top-quartile" in gap.prize_basis)

    print("\n" + "-" * 64)
    print("lower-is-better measures are not inverted")
    print("-" * 64)
    cost_rows = [["Cheap", 300, 2.0], ["Ok", 300, 3.0], ["Dear", 300, 9.0],
                 ["Worst", 300, 12.0]]
    gaps._segments = lambda *a, **k: ("SELECT ...", [list(r) for r in cost_rows])
    try:
        gap = gaps._benchmark_gap(
            "t", Col("zone", label="zone"),
            M("cost", 'SUM("c")', direction="lower", label="cost per job"),
            Info(columns=[Col("zone")]))
    finally:
        gaps._segments = real_segments
    if gap:
        behind = {d["segment"] for d in gap.laggards}
        check("the cheapest is the benchmark, not the problem",
              "Cheap" not in behind and gap.benchmark == 2.0, str(gap.benchmark))
        check("the expensive ones are the gap",
              {"Dear", "Worst"} <= behind, str(sorted(behind)))
    else:
        check("a lower-is-better gap is produced", False)

    print("\n" + "-" * 64)
    print("the model writes the plan, never the number")
    print("-" * 64)
    from app.core import router

    g = gaps.Gap(id="g1", table="t", kind="benchmark", title="x", measure="m",
                 prize=1234.0, benchmark=10.0, benchmark_holder="Big",
                 laggards=[{"segment": "Weak", "records": 5, "value": 5.0,
                            "behind": 5.0, "behind_pct": 50.0, "gain": 25.0}])
    real_complete = router.complete
    router.complete = lambda *a, **k: (
        '{"g1": {"why": "w", "actions": ["a", "b", "c"], "owner": "o",'
        ' "horizon": "h", "verify": "v", "risk": "r"}}', "stub")
    try:
        out = gaps.plan([g])
    finally:
        router.complete = real_complete
    check("a plan is parsed", out.get("g1", {}).get("owner") == "o", str(out))
    check("three actions survive", len(out.get("g1", {}).get("actions", [])) == 3)

    # A plan for a gap that was never measured is dropped: it would appear on screen
    # with no number behind it.
    router.complete = lambda *a, **k: ('{"made_up": {"actions": ["x"]}}', "stub")
    try:
        out = gaps.plan([g])
    finally:
        router.complete = real_complete
    check("a plan for an unknown gap is dropped", out == {}, str(out))

    def boom(*a, **k):
        raise RuntimeError("provider down")
    router.complete = boom
    try:
        out = gaps.plan([g])
    finally:
        router.complete = real_complete
    check("a dead provider still leaves the measurements", out == {})

    check("the prompt forbids inventing numbers",
          "NEVER state a number" in gaps.PLAN_SYSTEM)
    check("the brief carries the computed prize", "1,234" in gaps.brief([g]))

    print("\n" + "-" * 64)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all gap tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
