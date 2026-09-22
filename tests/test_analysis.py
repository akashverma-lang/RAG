"""Tests for the analyst: diagnostics, trends and routing.

Only the deterministic halves are tested here -- they are the ones that must never
be wrong, because the agent treats their output as established fact.  Runs against a
throwaway database and makes no model calls.
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                          # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="rag_analysis_test_"))
config.TABLES_DB_PATH = _TMP / "tables.db"
config.DASHBOARDS_DB_PATH = _TMP / "dashboards.db"

from app.analysis import diagnose, trends                       # noqa: E402
from app.tables import catalog, importer                        # noqa: E402

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"  got={got!r} want={want!r}"))
    if not ok:
        _failures.append(label)


def check_true(label: str, got) -> None:
    check(label, bool(got), True)


def build_fixture() -> catalog.TableInfo:
    """A table with one of every defect the scanner is meant to catch."""
    random.seed(11)
    header = ["Ticket ID", "Closed Date", "Region", "Dealer", "Status",
              "Hours", "Amount", "Units", "Note"]
    rows = []
    regions = ["North", "South", "East"]
    for i in range(400):
        month = 9 + (i % 4)                       # 2025-09 .. 2025-12
        day = (i % 27) + 1
        # South rejects far more often than anyone else -- the outlier the rate
        # check is supposed to surface.
        region = regions[i % 3]
        if region == "South":
            status = "Rejected" if i % 4 == 0 else "Approved"
        else:
            status = "Rejected" if i % 60 == 0 else "Approved"
        dealer = "Acme Power Pvt. Ltd." if i % 2 else "Acme Power & Co"
        rows.append([
            f"TK-{10000 + i}", f"2025-{month:02d}-{day:02d} 10:00:00", region, dealer,
            status,
            9_999_999_999 if i == 5 else round(random.uniform(1, 900), 1),   # outlier
            round(random.uniform(50, 5000), 2),
            0 if i % 3 else random.randint(1, 4),                            # sentinel 0
            "" if i % 5 == 0 else "note text",                               # nulls
        ])
    rows.append([f"TK-{99999}", "2031-01-01 09:00:00", "North", "Acme Power & Co",
                 "Approved", 5.0, 100.0, 1, "future"])                       # future date

    path = _TMP / "tickets.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    importer.import_file(path)
    return [t for t in catalog.list_tables() if t.name == "tickets"][0]


# --------------------------------------------------------------------------- tests
def test_diagnostics(info: catalog.TableInfo) -> None:
    print("data-quality scan")
    report = diagnose.scan_table(info)
    kinds = {f.kind for f in report.findings}
    by_col = {(f.kind, f.column) for f in report.findings}

    check_true("outlier found", ("outlier", "hours") in by_col)
    check_true("future date found", ("future_date", "closed_date") in by_col)
    check_true("missing values found", ("nulls", "note") in by_col)
    check_true("sentinel found", ("sentinel", "units") in by_col)
    check_true("near-duplicate dealer found", ("near_duplicate", "dealer") in by_col)

    # Precision: things that are NOT defects must not be reported.
    check_true("clean amount column is not flagged", ("outlier", "amount") not in by_col)
    check_true("region is not called a duplicate",
               ("near_duplicate", "region") not in by_col)
    # A repeating foreign key is the data working, not duplication.
    check_true("a non-unique id column is not called duplicated",
               ("duplicate_id", "dealer") not in by_col)

    outlier = next(f for f in report.findings
                   if f.kind == "outlier" and f.column == "hours")
    check("the outlier is high severity", outlier.severity, "high")
    check_true("it carries SQL to re-check it", outlier.evidence_sql.lower().startswith("select"))
    check_true("the summary mentions the table", info.name in report.summary())


def test_no_false_positives_on_meaningful_labels() -> None:
    """Labels that differ by a real word are different things, not typos."""
    print("\nnear-duplicate precision")
    from app.analysis.diagnose import _tokens

    same = [("Acme Power & Co", "Acme Power and Co"),
            ("Gangpur Sales & Services Pvt. Ltd.", "Gangpur Sales and Services"),
            ("Electro Controls -   Coimbatore", "Electro Controls - Coimbatore")]
    for a, b in same:
        check(f"same: {a[:26]!r}", _tokens(a) == _tokens(b), True)

    # These bit us on the real data: 94% character-identical, entirely different.
    different = [("Bandhan Plus - PM", "Bandhan Plus - CM"),
                 ("Submitted to ZSM", "Submitted to ASM"),
                 ("420320_8", "420320_18"),
                 ("AMC - BD", "AMC - CM")]
    for a, b in different:
        check(f"different: {a!r} vs {b!r}", _tokens(a) == _tokens(b), False)


def test_trends(info: catalog.TableInfo) -> None:
    print("\nmovement and outlier rates")
    report = diagnose.scan_table(info)
    excluded = trends.bad_measures(report)
    check_true("the corrupt column is excluded from measures", "hours" in excluded)

    signals = trends.scan_table(info, exclude=excluded)
    kinds = [s.kind for s in signals.signals]
    check_true("a trend was computed", "trend" in kinds)

    rate = [s for s in signals.signals if s.kind == "outlier_rate"]
    check_true("the bad segment was found", any("South" in s.headline for s in rate))
    for s in rate:
        check_true(f"rate signal carries SQL ({s.kind})",
                   s.evidence_sql.lower().startswith("select"))

    # The measure the diagnostics rejected must not appear in any headline.
    check_true("no signal is built on the corrupt column",
               not any("Hours" in s.headline for s in signals.signals))


def test_worst_value_priority() -> None:
    """'Rejected' is a failure; 'Pending' is a stage. Picking wrong asks the wrong
    question of every segment."""
    print("\nfailure-word priority")

    class Col:
        def __init__(self, values):
            self.values = values

    check("reject beats pending",
          trends._worst_outcome_value("t", Col(["Pending for Submission", "Rejected"])),
          "Rejected")
    check("pending is still used when it is all there is",
          trends._worst_outcome_value("t", Col(["Approved", "Pending"])), "Pending")
    check("no failure word means no signal",
          trends._worst_outcome_value("t", Col(["Alpha", "Beta"])), "")


def test_semantics() -> None:
    """The layer that says what things mean and which direction is good."""
    print("\nsemantic layer")
    from app.analysis import semantics

    config.STORAGE_DIR = _TMP
    semantics.save("""
domain: Field service
tables:
  tickets:
    description: Service tickets raised by dealers.
    grain: One row per ticket.
    dimensions:
      region: {means: Service region}
    outcomes:
      status:
        good: ['Approved']
        bad: ['Rejected']
    metrics:
      ticket_value:
        sql: SUM(amount)
        direction: higher_is_better
        means: Total value of tickets
      reject_rate:
        sql: 100.0 * SUM(CASE WHEN status='Rejected' THEN 1 ELSE 0 END)/COUNT(*)
        direction: lower_is_better
glossary:
  - term: BD
    means: Breakdown
    also: ['Failure']
  - term: PM
    means: Preventive Maintenance
""")
    data = semantics.load(force=True)
    check("one table parsed", len(data["tables"]), 1)
    check("two glossary terms", len(data["glossary"]), 2)

    info = semantics.table("tickets")
    check("outcome column read", info.outcomes[0].column, "status")
    check("bad states read", info.outcomes[0].bad, ["Rejected"])
    check("declared failure wins", semantics.bad_values("tickets", "status"), ["Rejected"])
    check("nothing declared means nothing returned",
          semantics.bad_values("tickets", "region"), [])
    directions = {m.name: m.direction for m in info.metrics}
    check("a rate is lower_is_better", directions["reject_rate"], "lower_is_better")
    check("a total is higher_is_better", directions["ticket_value"], "higher_is_better")

    # The glossary has to match on what a term MEANS, not only on the term itself --
    # somebody types "breakdown", the data says "BD".
    check_true("plain word finds the abbreviation",
               any(g.term == "BD" for g in semantics.relevant_glossary("count the breakdowns")))
    check_true("abbreviation finds the entry too",
               any(g.term == "BD" for g in semantics.relevant_glossary("how many BD jobs")))
    check_true("an unrelated question matches nothing",
               not semantics.relevant_glossary("what is the weather"))

    swaps = semantics.synonyms("how many breakdown jobs")
    check_true("a synonym query is produced", swaps)
    check_true("and it uses the data's own wording",
               any("BD" in x for x in swaps))

    rendered = semantics.render(question="breakdowns")
    check_true("render carries the metric definitions", "reject_rate" in rendered)
    check_true("render carries good/bad states", "Rejected" in rendered)

    # Malformed YAML must not take the whole app down.
    bad = semantics.path().with_suffix(".yaml")
    bad.write_text("tables: [unclosed", encoding="utf-8")
    broken = semantics.load(force=True)
    check_true("a broken file degrades quietly", "error" in broken)
    check("and yields no tables", len(broken["tables"]), 0)


def test_expansion() -> None:
    """Query rewriting, with the model switched off so the test is deterministic."""
    print("\nquery expansion")
    from app.analysis import semantics
    from app.retrieval import expand

    semantics.save("""
glossary:
  - term: BD
    means: Breakdown
    also: ['Failure']
""")
    semantics.load(force=True)
    out = expand.variants("how many breakdown jobs", use_model=False)
    check_true("glossary alone produces variants", out)
    check_true("the original is never repeated",
               "how many breakdown jobs" not in [x.lower() for x in out])
    check_true("the data's wording appears", any("BD" in x for x in out))
    check_true("bounded by the setting", len(out) <= config.EXPAND_QUERIES)

    check("no glossary hit means no variants",
          expand.variants("unrelated question about weather", use_model=False), [])

    parsed = expand._parse_list('["one", "two"]')
    check("a JSON array parses", parsed, ["one", "two"])
    check("a plain list parses too", expand._parse_list("- alpha\n- beta"), ["alpha", "beta"])

    # A reply cut off by max_tokens used to come back as ['[', 'json', 'real query",'].
    # Every one of those was then embedded and fused into the ranking as if it were a
    # phrasing of the question, so two of the four searches were spent on punctuation.
    truncated = ('```json\n[\n  "annual maintenance contract cancellation",\n'
                 '  "AMC end of')
    check("a truncated array keeps only its complete items",
          expand._parse_list(truncated), ["annual maintenance contract cancellation"])
    check("a fence on its own line is stripped",
          expand._parse_list('Here you go:\n```json\n["notice period"]\n```'),
          ["notice period"])
    for junk in ("[", "]", "```json", "{", ",", "  "):
        check_true(f"{junk!r} is not a search query", not expand._usable(junk))
    check_true("a real phrasing still is", expand._usable("notice period"))


def test_answer_modes() -> None:
    print("\nanswer modes")
    from app.core import llm

    config.ANSWER_MODE = "grounded"
    check_true("grounded refuses without documents",
               "Do not answer from your own knowledge" in llm._document_prompt(False))
    check_true("grounded stays document-only with documents",
               "Use ONLY the numbered context" in llm._document_prompt(True))

    config.ANSWER_MODE = "blended"
    check_true("blended may add general knowledge",
               "Beyond your documents" in llm._document_prompt(True))
    check_true("blended still answers when nothing matched",
               "Beyond your documents" in llm._document_prompt(False))
    check_true("and it must keep the two apart",
               "never attach a citation" in llm._document_prompt(True).lower())

    msgs = llm.build_messages("q", "ctx", knowledge="DG means diesel generator")
    check_true("domain knowledge reaches the prompt",
               "diesel generator" in msgs[0]["content"].lower())


def test_incremental_semantics() -> None:
    """Adding a spreadsheet must not cost you the edits you already made."""
    print("\nincremental semantics")
    from app.analysis import semantics

    config.STORAGE_DIR = _TMP
    original = """# why: rejection is the number the business watches
domain: Claims
tables:
  claims:
    description: hand written, do not lose me
    outcomes:
      status:
        good: ['Approved']
        bad: ['Rejected']
glossary:
  - term: BD
    means: Breakdown
"""
    addition = """tables:
  hr_people:
    description: Headcount by month.
    glossary:
      - term: PM
        means: Project Manager
glossary:
  - term: FTE
    means: Full Time Equivalent
"""
    merged = semantics.merge(original, addition)
    check_true("the comment survives", "# why: rejection" in merged)
    check_true("the hand-written line survives", "do not lose me" in merged)
    check_true("the new table is added", "hr_people" in merged)
    check_true("the new term is added", "FTE" in merged)
    check("a term already present is not duplicated", merged.count("term: BD"), 1)
    check("merging twice changes nothing more", semantics.merge(merged, addition), merged)

    # A table already described is never re-described, even if the draft covers it.
    again = semantics.merge(merged, """tables:
  claims:
    description: REPLACED
""")
    check_true("an existing table is left alone", "REPLACED" not in again)
    check_true("and its original text stands", "do not lose me" in again)

    semantics.save(merged)
    data = semantics.load(force=True)
    check("both tables parse", sorted(data["tables"]), ["claims", "hr_people"])

    # Scoping: PM means different things in the two datasets, and the table's own
    # definition has to win for its own questions.
    hr = semantics.glossary_for("hr_people")
    check("the table's own term is in scope", [g.term for g in hr][0], "PM")
    check("it means what that table says",
          next(g.means for g in hr if g.term == "PM"), "Project Manager")
    shared = [g.term for g in semantics.glossary_for("claims")]
    check_true("a shared term is still visible", "BD" in shared)
    check_true("another table's private term is not",
               "PM" not in [g.term for g in semantics.glossary_for("claims")])


def test_forgiving_yaml() -> None:
    """A model writing one bad line must not cost the other twenty-four."""
    print("\nsalvaging model YAML")
    from app.analysis import semantics

    text = ("glossary:\n"
            "  - term: MTBF\n"
            "    means: Mean Time Between Failures\n"
            '  - term: "9+1" topology\n'
            "    means: broken entry\n"
            "  - term: MDT\n"
            "    means: Mean Down Time\n")
    data, dropped = semantics.parse_forgiving(text)
    terms = [e["term"] for e in data["glossary"]]
    check("the good entries survive", terms, ["MTBF", "MDT"])
    check("exactly one entry was dropped", len(dropped), 1)
    check_true("and it was the broken one", "9+1" in dropped[0])

    clean, none = semantics.parse_forgiving("glossary:\n  - term: OK\n")
    check("valid YAML is untouched", none, [])
    check("and parses normally", clean["glossary"][0]["term"], "OK")


def test_file_scope() -> None:
    """Restricting an answer to chosen files."""
    print("\nfile scope")
    import numpy as np
    from app.core.store import Store

    store = Store(db_path=_TMP / "scope.db", vec_path=_TMP / "scope.f32")
    vecs = np.eye(3, 8, dtype=np.float32)
    for i, name in enumerate(["alpha.txt", "beta.txt", "gamma.txt"]):
        store.upsert_file_chunks(
            path=f"/data/{name}", rel_path=name, size=10, mtime=1.0,
            file_hash=f"h{i}", vectors=vecs[i:i + 1],
            chunks=[{"text": f"shared word unique{i}", "loc": "", "heading": ""}])

    ids = store.file_ids_for(["beta.txt"])
    check("a rel_path resolves to one file", len(ids), 1)
    check("an unknown file resolves to nothing", store.file_ids_for(["nope.txt"]), [])

    everywhere = store.search_bm25("shared", 10)
    scoped = store.search_bm25("shared", 10, ids)
    check("the term is in every file", len(everywhere), 3)
    check("scoping cuts it to one", len(scoped), 1)

    chunks = store.chunk_ids_for_files(ids)
    check("that file has one chunk", len(chunks), 1)
    check("and the scoped hit is that chunk", scoped[0][0], int(chunks[0]))

    # Dense search must scope *before* the top-k, or a small file never wins.
    query = np.zeros(8, dtype=np.float32); query[1] = 1.0     # points at beta
    dense_all = store.search_dense(query, 3)
    check("unscoped dense returns everything", len(dense_all), 3)
    away = store.file_ids_for(["gamma.txt"])
    dense_scoped = store.search_dense(query, 3, away)
    check("scoped dense returns only that file", len(dense_scoped), 1)
    check("even when it is not the best match",
          dense_scoped[0][0], int(store.chunk_ids_for_files(away)[0]))


def test_explaining_refusals() -> None:
    """A refusal has to say WHICH column is missing, not just that it cannot.

    Three cases, and the middle one is the subtle one: the question was about the
    spreadsheet, the spreadsheet cannot answer it, and document search then returns
    passages that look relevant. Without a signal the user is told "the passages do
    not contain this" and never learns their export has no open date.
    """
    print("\nexplaining refusals")
    from app.core import llm
    from app.main import _data_note

    class Fake:
        def __init__(self, missing="", why=""):
            self.missing, self.why = missing, why

    about_data = Fake(missing="claim_raised_date column",
                      why="the tables record sr_close_date but never when a claim was raised")
    check_true("a named missing column always surfaces",
               _data_note(about_data, "some passages were found"))
    check("and it uses the full sentence, not the bare column name",
          _data_note(about_data, "ctx"), about_data.why)

    about_prose = Fake(why="the question is about a report's wording")
    check("a prose question is not prefaced with SQL talk",
          _data_note(about_prose, "passages were found"), "")
    check_true("but it is explained when nothing was found either",
               _data_note(about_prose, ""))
    check("no table answer means no note", _data_note(None, ""), "")

    bare = Fake(missing="open_date")
    check_true("a bare column still produces a sentence",
               "open_date" in _data_note(bare, "ctx"))

    # The note has to lead, not trail: buried under a page of other instructions the
    # model follows the body and the reason never reaches the user.
    prompt = llm.build_messages("q", "ctx", data_note="no open date column")[0]["content"]
    check_true("the note is at the top", prompt.lstrip().startswith("FIRST"))
    check_true("and carries the reason", "no open date column" in prompt)
    plain = llm.build_messages("q", "ctx")[0]["content"]
    check_true("no note means no preamble", not plain.lstrip().startswith("FIRST"))

    check_true("the answer prompt refuses a vague explanation",
               "not good enough" in llm.TABLE_SYSTEM_PROMPT)


def test_business_signals(info) -> None:
    """The half that was missing: how the business is doing, not what is broken."""
    print("\nbusiness signals")
    from app.analysis import performance, semantics

    report = diagnose.scan_table(info)
    excluded = trends.bad_measures(report)
    signals = performance.scan_table(info, exclude=excluded).signals
    kinds = [s.kind for s in signals]

    check_true("the scale is stated", "headline" in kinds)
    check_true("segments are ranked", "ranking" in kinds)
    for s in signals:
        check_true(f"{s.kind} carries its query", s.evidence_sql.lower().startswith("select"))
    charted = [s for s in signals if s.rows and len(s.rows) > 1]
    check_true("and enough rows to draw", charted)

    # Direction is the whole point of declaring a metric: it turns a ranking into a
    # judgement. Read the wrong way round it inverts every conclusion.
    lower = performance.Measure(name="cost", sql="SUM(claim_amount)",
                                direction=semantics.LOWER, label="cost")
    higher = performance.Measure(name="value", sql="SUM(claim_amount)", label="value")
    check("lower_is_better is not good_high", lower.good_high, False)
    check("higher_is_better is", higher.good_high, True)

    dims = performance._dimensions(info)
    if dims:
        low = performance._ranking(info.name, dims[0], lower)
        if low:
            check_true("a lower-is-better ranking says so",
                       "LOWER is better" in low[0].detail)
            check_true("and does not quote a share of the total",
                       "% of the total" not in low[0].headline)
        high = performance._ranking(info.name, dims[0], higher)
        if high:
            check_true("a higher-is-better ranking quotes the share",
                       "% of the total" in high[0].headline)


def test_exclusion_is_narrow() -> None:
    """Six bad rows in two hundred thousand must not disqualify revenue.

    The first version excluded any column with a medium-or-worse outlier, which
    removed every money column and left the business analysis ranking segments by
    labour days -- describing entirely the wrong thing.
    """
    print("\nexclusion is narrow")

    class F:
        def __init__(self, kind, severity, column):
            self.kind, self.severity, self.column = kind, severity, column
            self.headline = f"{column} {severity}"

    class R:
        findings = [F("outlier", "high", "hour_meter"),      # genuinely corrupt
                    F("outlier", "medium", "amount"),        # six rows in 200k
                    F("sentinel", "medium", "quantity"),     # breaks AVG, not SUM
                    F("nulls", "high", "app_code")]          # empty, not corrupt

    check("only the corrupt column is banned", trends.bad_measures(R()), {"hour_meter"})
    caveats = trends.caveat_measures(R())
    check("the imperfect ones get a warning instead",
          sorted(caveats), ["amount", "quantity"])
    check_true("a missing column is neither banned nor warned",
               "app_code" not in caveats and "app_code" not in trends.bad_measures(R()))


def test_report_leads_with_business() -> None:
    print("\nreport structure")
    from app.analysis import agent

    check_true("the prompt forbids a data-quality audit",
               "NOT writing a data-quality audit" in agent.REPORT_SYSTEM)
    check_true("caveats are ordered last", "\"## Caveats\" - LAST" in agent.REPORT_SYSTEM)
    order = agent.REPORT_SYSTEM.index("Where it is working") <         agent.REPORT_SYSTEM.index("## Caveats")
    check("working comes before caveats", order, True)

    ev = {"performance": [], "trends": [], "diagnostics": [],
          "excluded_measures": [], "caveats": {}}
    text = agent._evidence_text(ev)
    check_true("business evidence is presented first",
               text.index("HOW THE BUSINESS IS DOING") < text.index("CAVEATS"))

    # One unreadable reply must not end the investigation.
    check("a truncated reply is marked unparsed",
          agent._parse('{"thought": "x", "action": "sql", "sql": "SELECT a FROM')["action"],
          "unparsed")
    check("a deliberate finish is still a finish",
          agent._parse('{"action":"finish","thought":"done"}')["action"], "finish")


def test_cleaning(info) -> None:
    """Fixes are proposed with a count, applied as a view, and never destructive."""
    print("\ndata cleaning")
    from app.analysis import cleaning
    from app.tables import sql as tsql

    config.STORAGE_DIR = _TMP
    rules = cleaning.propose(info.name)
    kinds = {r.kind for r in rules}
    check_true("the duplicate dealer spellings are caught", "map_value" in kinds)
    check_true("the absurd value is caught", "null_outlier" in kinds)
    check_true("the placeholder is caught", "null_sentinel" in kinds)
    check_true("every proposal is measured", all(r.affected > 0 for r in rules))
    check_true("and explains itself", all(len(r.reason) > 20 for r in rules))

    before_rows = tsql.run(f"SELECT COUNT(*) FROM {info.name}").rows[0][0]
    cleaning.save(rules)
    built = cleaning.build(info.name)
    view = built["view"]
    check("a view was created", view, info.name + cleaning.CLEAN_SUFFIX)

    # Non-destructive is the whole design: the raw table must be byte-identical.
    after_rows = tsql.run(f"SELECT COUNT(*) FROM {info.name}").rows[0][0]
    check("the raw table still has every row", after_rows, before_rows)
    check("and the raw values are untouched",
          tsql.run(f"SELECT COUNT(DISTINCT dealer) FROM {info.name}").rows[0][0], 2)
    check("while the view merges them",
          tsql.run(f"SELECT COUNT(DISTINCT dealer) FROM {view}").rows[0][0], 1)
    check("the view keeps every row",
          tsql.run(f"SELECT COUNT(*) FROM {view}").rows[0][0], before_rows)

    # The outlier must be gone from the average but still present in the raw table.
    raw_max = tsql.run(f"SELECT MAX(hours) FROM {info.name}").rows[0][0]
    clean_max = tsql.run(f"SELECT MAX(hours) FROM {view}").rows[0][0]
    check_true("the raw extreme value survives", raw_max > 1e9)
    check_true("but is set aside in the view", clean_max is None or clean_max < 1e9)

    # Disabling a rule removes its effect on the next build.
    for r in rules:
        if r.kind == "map_value":
            r.enabled = False
    cleaning.build(info.name, rules)
    check("a disabled rule stops applying",
          tsql.run(f"SELECT COUNT(DISTINCT dealer) FROM {view}").rows[0][0], 2)

    # Impact must count only what the rules touched. The row number is unique per
    # sheet, not across a union of sheets, so joining on it alone once paired every
    # row with its namesake in the other file and reported the whole table as changed.
    cleaning.build(info.name, rules)
    for r in rules:
        r.enabled = True
    cleaning.build(info.name, rules)
    touched = {r.column for r in rules if r.enabled}
    measured = {c["column"] for c in cleaning.impact(info.name).get("columns", [])}
    check_true("only changed columns are reported", measured <= touched)
    check_true("and the changed ones are", "dealer" in measured)

    # Dropping everything removes the view rather than leaving a stale one.
    cleaning.build(info.name, [])
    check_true("no rules means no view", info.name not in cleaning.cleaned_tables())


def test_unrelated_domain() -> None:
    """Nothing here may depend on the data being about claims."""
    print("\ndomain independence")
    from app.analysis import trends

    # A failure word from a completely different business still registers.
    class Col:
        def __init__(self, values):
            self.values, self.name = values, "status"

    for values, expected in [(["Active", "Churned"], "Churned"),
                             (["Paid", "Default"], "Default"),
                             (["Open", "Abandoned"], "Abandoned"),
                             (["Shipped", "Returned"], "Returned"),
                             (["Won", "Lost"], "Lost")]:
        check(f"{values} -> failure state",
              trends._worst_outcome_value("t", Col(values)), expected)
    check("a neutral pair yields nothing",
          trends._worst_outcome_value("t", Col(["Alpha", "Beta"])), "")

    # Dimension choice must not flip when a merge changes a count by one.
    class Dim:
        def __init__(self, n):
            self.role, self.n_distinct, self.name = "category", n, f"c{n}"
            self.values, self.label = [], f"c{n}"

    class Info:
        columns = [Dim(2), Dim(3), Dim(4), Dim(30), Dim(7)]

    order = [c.n_distinct for c in trends._pick(Info())["dims"]]
    check_true("mid-sized dimensions are preferred", order[0] in (7, 4))
    check_true("a two-value column is not first", order[0] != 2)


def test_transport() -> None:
    """Connections are made once, and never over a dead IPv6 path.

    The bug this pins: Google's endpoint resolves to eight IPv6 addresses before any
    IPv4 one. On a network with no working IPv6 httpx tried each in turn, so one
    /models call took 81 seconds while curl took 0.7 -- and /api/health, which probes
    every provider and which the UI polls, took a minute and a half.
    """
    print("\nnetwork transport")
    from app.core import llm

    check_true("IPv6 usability is decided, not assumed",
               isinstance(llm.ipv6_works(), bool))
    check("and the answer is cached", llm.ipv6_works(), llm.ipv6_works())

    a = llm.client_for("t", "https://example.invalid")
    b = llm.client_for("t", "https://example.invalid")
    check_true("the same provider reuses one client", a is b)
    c = llm.client_for("other", "https://example.invalid")
    check_true("a different provider gets its own", c is not a)

    # The timeout belongs to the request, not to the shared client. Storing it on
    # the client meant the health poll's 12 seconds could land on a chat request
    # that needed ten minutes, because both go through the same connection.
    import inspect
    # One streaming path now: every provider speaks the OpenAI protocol, Gemini
    # included through its compatibility endpoint, so the Ollama branch is gone.
    for fn in (llm._openai_stream,):
        src = inspect.getsource(fn)
        check_true(f"{fn.__name__} passes its own timeout",
                   "timeout=_timeout(timeout)" in src)
    check_true("the shared client carries no stored timeout",
               "existing.timeout" not in inspect.getsource(llm.client_for))

    # A closed client must be replaced rather than handed back unusable.
    a.close()
    check_true("a closed client is rebuilt",
               llm.client_for("t", "https://example.invalid") is not a)

    # Probing must be concurrent: serially the cost is the sum of every provider.
    src = inspect.getsource(llm.refresh_providers)
    check_true("providers are probed in parallel", "ThreadPoolExecutor" in src)


def test_router() -> None:




    print("\nmodel routing")
    from app.core import router

    agent = router.chain("agent")
    answer = router.chain("answer")
    check_true("the agent chain has a fallback", len(agent) >= 1)
    check_true("agent and answer lead with different models",
               agent[0] != answer[0] or config.AGENT_MODEL == config.ANSWER_MODEL)
    check_true("a rate-limit message is recognised",
               router._is_rate_limit(RuntimeError("Rate limit reached for model")))
    check_true("an ordinary error is not",
               not router._is_rate_limit(RuntimeError("connection refused")))

    # The pacing bucket must actually delay the second call.
    import time
    bucket = router._Bucket(rpm=120)              # 0.5s apart
    bucket.wait()
    start = time.monotonic()
    bucket.wait()
    check_true("calls are paced apart", time.monotonic() - start >= 0.4)


def main() -> int:
    print(f"scratch db: {config.TABLES_DB_PATH}\n")
    info = build_fixture()
    test_diagnostics(info)
    test_no_false_positives_on_meaningful_labels()
    test_trends(info)
    test_worst_value_priority()
    test_semantics()
    test_expansion()
    test_answer_modes()
    test_incremental_semantics()
    test_forgiving_yaml()
    test_file_scope()
    test_explaining_refusals()
    test_business_signals(info)
    test_exclusion_is_narrow()
    test_report_leads_with_business()
    test_cleaning(info)
    test_unrelated_domain()
    test_transport()
    test_router()
    print("\n" + "-" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all analysis tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
