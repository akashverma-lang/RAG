"""Tests for the structured-data path: detection, typing, import, SQL safety.

Runs entirely against a throwaway tables.db in a temp folder -- it never touches
your real storage.  Run with ``python -m pytest tests/test_tables.py`` or directly
with ``python tests/test_tables.py``.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                        # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="rag_tables_test_"))
config.TABLES_DB_PATH = _TMP / "tables.db"                    # before anything connects
config.DASHBOARDS_DB_PATH = _TMP / "dashboards.db"

from app.tables import catalog, charts, dashboard, detect, importer, sql   # noqa: E402

HEADER = ["Order ID", "Order Date", "Region", "Product Code",
          "Quantity", "Unit Price", "Claim Amount", "Status"]
ROWS = [
    ["ORD-001", "2025-07-04", "West", "0041290", 3, 100.5, 301.5, "Shipped"],
    ["ORD-002", "2025-08-11", "East", "0041291", 1, 250.0, 250.0, "Pending"],
    ["ORD-003", "2025-09-19", "West", "0041292", 7, 10.25, 71.75, "Shipped"],
    ["ORD-004", "2025-10-02", "North", "0041293", 2, 99.99, 199.98, "Cancelled"],
    ["ORD-005", "2025-11-23", "East", "0041294", 5, 20.0, 100.0, "Shipped"],
    ["ORD-006", "2025-12-30", "West", "0041295", 4, 60.5, 242.0, "Pending"],
]

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"  got={got!r} want={want!r}"))
    if not ok:
        _failures.append(label)


def check_true(label: str, got) -> None:
    check(label, bool(got), True)


def write_csv(path: Path, header=HEADER, rows=ROWS) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------- detection
def test_detection() -> None:
    print("detection and typing")
    probe = detect.probe_file(write_csv(_TMP / "orders.csv"))
    check_true("clean sheet is detected as a table", probe.ok)
    cols = {c.name: c for c in probe.sheets[0].columns}
    check("header row found", probe.sheets[0].header_row, 1)
    check("money column is numeric", cols["claim_amount"].type, "REAL")
    check("money column role", cols["claim_amount"].role, "number")
    check("integer column", cols["quantity"].type, "INTEGER")
    check("date column detected", cols["order_date"].role, "date")
    # The whole point of the leading-zero rule: 0041290 must not become 41290.
    check("zero-padded code stays text", cols["product_code"].type, "TEXT")
    check("zero-padded code is an id", cols["product_code"].role, "id")
    check("low-cardinality column is a category", cols["region"].role, "category")

    # A sheet with a title banner above the real header.
    banner = _TMP / "banner.csv"
    with open(banner, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Quarterly Report"])
        w.writerow([])
        w.writerow(HEADER)
        w.writerows(ROWS)
    p2 = detect.probe_file(banner)
    check_true("banner sheet still detected", p2.ok)
    check("header found below the banner", p2.sheets[0].header_row, 3)

    # Not a table: prose lines.
    prose = _TMP / "prose.csv"
    prose.write_text("just some notes\nabout nothing in particular\n", encoding="utf-8")
    check("prose is rejected", detect.probe_file(prose).ok, False)


# --------------------------------------------------------------------------- import
def test_import() -> None:
    print("\nimport and profiling")
    out = importer.import_file(write_csv(_TMP / "orders.csv"))
    check_true("import succeeded", out["ok"])
    check("row count", out["rows"], len(ROWS))

    infos = {t.name: t for t in catalog.list_tables()}
    table = infos["orders"]
    cols = {c.name: c for c in table.columns}
    check("profiled row count", table.n_rows, len(ROWS))
    check("category values are listed in full",
          sorted(cols["region"].values), ["East", "North", "West"])
    check("numeric range is profiled", (cols["quantity"].lo, cols["quantity"].hi), ("1", "7"))
    check_true("schema prompt names the table", "orders" in catalog.render_schema())
    check_true("schema prompt carries real literals", "'West'" in catalog.render_schema())

    # Money must be summable -- the regression that motivated MEASURE_HINT.
    got = sql.run("SELECT ROUND(SUM(claim_amount), 2) FROM orders")
    check("SUM over the money column", got.rows[0][0], 1165.23)
    # And the identifier must survive its leading zero.
    got = sql.run("SELECT product_code FROM orders WHERE order_id = 'ORD-001'")
    check("leading zero preserved", got.rows[0][0], "0041290")


def test_union_view() -> None:
    print("\nunion view over matching schemas")
    # Each half needs at least TABLE_MIN_ROWS rows of its own to be imported at all.
    half = [[f"ORD-1{i:02d}", *r[1:]] for i, r in enumerate(ROWS)]
    write_csv(_TMP / "q3.csv", rows=ROWS)
    write_csv(_TMP / "q4.csv", rows=half)
    importer.import_file(_TMP / "q3.csv")
    importer.import_file(_TMP / "q4.csv")
    made = importer.rebuild_union_views()
    check_true("a union view was created", made)
    view = made[0]
    got = sql.run(f"SELECT COUNT(*) FROM {view}")
    check("view spans every matching sheet", got.rows[0][0], len(ROWS) * 3)
    got = sql.run(f"SELECT COUNT(DISTINCT _source_file) FROM {view}")
    check("view keeps the source file", got.rows[0][0], 3)


# --------------------------------------------------------------------------- safety
def test_sql_guard() -> None:
    print("\nSQL safety")
    blocked = [
        ("DROP TABLE orders", "drop"),
        ("DELETE FROM orders", "delete"),
        ("SELECT 1; DELETE FROM orders", "two statements"),
        ("INSERT INTO orders VALUES (1)", "insert"),
        ("PRAGMA table_info(orders)", "pragma"),
        ("ATTACH DATABASE 'x.db' AS x", "attach"),
        ("UPDATE orders SET status = 'x'", "update"),
    ]
    for statement, label in blocked:
        check(f"blocked: {label}", sql.run(statement).ok, False)

    allowed = [
        "SELECT * FROM orders LIMIT 1",
        "  select region, count(*) from orders group by 1  ",
        "WITH x AS (SELECT * FROM orders) SELECT COUNT(*) FROM x",
        "SELECT * FROM orders WHERE status = 'Pending' LIMIT 1",
    ]
    for statement in allowed:
        r = sql.run(statement)
        check(f"allowed: {statement.strip()[:38]}", r.ok, True)

    # A literal that merely contains a banned word must not be rejected.
    r = sql.run("SELECT 'please update the record' AS note")
    check("keyword inside a string literal is fine", r.ok, True)

    # replace() is a SQLite function, not the REPLACE statement. Refusing it sent a
    # correct query back as "unsafe" and dropped the question to document search.
    r = sql.run("SELECT replace(status, 'Pending', 'Open') AS s FROM orders LIMIT 1")
    check("replace() is a function, not a write", r.ok, True)
    r = sql.run("SELECT trim(region) AS r FROM orders LIMIT 1")
    check("trim() is allowed", r.ok, True)

    # A semicolon inside a value is one statement, not two.
    r = sql.run("SELECT * FROM orders WHERE status = 'a;b' LIMIT 1")
    check("semicolon inside a literal is one statement", r.ok, True)

    # ...and a real second statement is still refused, literal or not.
    check("a real second statement is still blocked",
          sql.run("SELECT * FROM orders WHERE status='a' ; DROP TABLE orders").ok, False)
    check("a write disguised after a literal is blocked",
          sql.run("SELECT 'x;y' AS a; DELETE FROM orders").ok, False)

    # Read-only: even a valid write cannot reach the file.
    r = sql.run("SELECT COUNT(*) FROM orders")
    check("reads still work after the guards", r.ok, True)

    capped = sql.run("SELECT * FROM orders", limit=2)
    check("row cap applied", capped.n_rows, 2)
    check("truncation is reported", capped.truncated, True)


def test_partial_answers() -> None:
    """Half a question is still worth answering.

    The bug this pins: asked for total claim amount AND average resolution time,
    the planner saw that the table has a close date but no open date and declined
    the WHOLE question -- so a user who could have had the totals got a lecture on
    how to compute them in Excel instead.
    """
    print("\npartial answers")
    plan = sql._parse_plan(
        '{"mode":"sql","sql":"SELECT region, SUM(claim_amount) FROM orders GROUP BY 1",'
        '"unanswerable":"average resolution time: no open date column"}')
    check("it still answers with SQL", plan.mode, "sql")
    check_true("and the query survives", plan.sql.lower().startswith("select"))
    check_true("the missing half is carried", "open date" in plan.unanswerable)

    full = sql._parse_plan('{"mode":"sql","sql":"SELECT 1"}')
    check("nothing missing means an empty note", full.unanswerable, "")

    declined = sql._parse_plan('{"mode":"semantic","why":"about document prose"}')
    check("a genuinely unanswerable question still declines", declined.mode, "semantic")

    # The instruction the planner needs must actually be in the prompt.
    check_true("the prompt asks for partial answers",
               "ANY PART of it" in sql.SQL_SYSTEM)
    check_true("and explains when to decline outright",
               "NOTHING in the question" in sql.SQL_SYSTEM)


def test_no_hang() -> None:
    """The failure modes that froze the UI: a runaway query and a dead provider."""
    print("\nnothing can hang the request")
    import time
    import httpx
    from app.core import llm

    # An unbounded recursive CTE never finishes on its own; the watchdog has to stop
    # it. Without one, this query alone would hold a request thread for ever.
    endless = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
               "SELECT COUNT(*) FROM c")
    t = time.perf_counter()
    r = sql.run(endless, timeout=1.0)
    took = time.perf_counter() - t
    check("endless query is stopped", r.ok, False)
    check_true(f"stopped near its 1s budget (took {took:.1f}s)", took < 4.0)
    check_true("reported as a timeout", "timed out" in r.error)

    # The watchdog must not leak: a later fast query still works.
    time.sleep(1.2)
    check("watchdog does not affect later queries",
          sql.run("SELECT 1", timeout=1.0).rows, [(1,)])

    # An unreachable model falls back to document search instead of blocking.
    saved = llm.complete
    try:
        def dead(messages, model=None, temperature=None, timeout=None):
            raise httpx.ConnectError("connection refused")
        llm.complete = dead
        sql.llm = llm
        t = time.perf_counter()
        plan = sql.plan("how many orders are there?")
        check("unreachable model routes to document search", plan.mode, "semantic")
        check_true("and does so immediately", time.perf_counter() - t < 2.0)
    finally:
        llm.complete = saved
        sql.llm = llm

    # Connect/read budgets are bounded well below the total request budget.
    t = llm._timeout()
    check_true("connect timeout is short", t.connect <= 15)
    check_true("read timeout is bounded", t.read <= 120)


# --------------------------------------------------------------------------- intent
def test_intent() -> None:
    """Greetings must not reach either search path.

    The bug this pins: "Hi" was sent to the SQL planner together with the previous
    turn, and came back with the previous question's query -- so saying hello
    produced a chart, and a "Pin to dashboard" button under it.
    """
    print("\nmessage intent")
    from app.core import intent

    for greeting in ["Hi", "hii", "hello!", "hey there", "thanks", "Thank you", "ok",
                     "good morning", "bye", "how are you", "what can you do?"]:
        check(f"small talk: {greeting!r}", intent.is_small_talk(greeting), True)
    for real in ["how many claims are approved?", "total value by zone",
                 "tell me about SR 212609239", "hi, how many rows are there?"]:
        check(f"not small talk: {real!r}", intent.is_small_talk(real), False)

    # Only a genuine follow-up gets the conversation history.
    for followup in ["and West?", "what about that", "by month", "same for rejected"]:
        check(f"follow-up: {followup!r}", intent.is_follow_up(followup), True)
    for standalone in ["how many claims were approved in October",
                       "which dealer has the highest total value"]:
        check(f"standalone: {standalone!r}", intent.is_follow_up(standalone), False)
    check("a greeting is never a follow-up", intent.is_follow_up("Hi"), False)

    # And the planner refuses it without spending a request.
    plan = sql.plan("Hi", history=[{"role": "user", "content": "claims by zone"}])
    check("planner declines a greeting", plan.mode, "semantic")


# --------------------------------------------------------------------------- charts
def test_charts() -> None:
    """The chart is picked from the result shape, and the guards keep it drawable."""
    print("\nchart choice and guards")
    roles = {"product_code": "id", "order_date": "date", "region": "category"}

    check("one number is a stat tile",
          charts.infer(["total"], [[42]]).kind, "stat")
    check("a date axis is a line",
          charts.infer(["order_date", "n"], [["2025-01-01", 5], ["2025-02-01", 8]],
                       roles).kind, "line")
    check("short labels give vertical bars",
          charts.infer(["region", "n"], [["N", 5], ["S", 3]]).kind, "bar")
    check("long labels flip to horizontal",
          charts.infer(["region", "n"],
                       [["A very long dealer name indeed", 5], ["S", 3]]).kind, "hbar")
    check("two categories and a measure group",
          len(charts.infer(["region", "seg", "n"],
                           [["N", "A", 1], ["N", "B", 2], ["S", "A", 3]]).series), 2)
    check("two bare measures scatter",
          charts.infer(["x", "y"], [[1, 2], [3, 4], [5, 6]]).kind, "scatter")
    check("no numbers means a table",
          charts.infer(["a", "b"], [["x", "y"]]).kind, "table")

    # An identifier axis is the case that used to produce a 29,206-bar chart.
    ident = charts.infer(["product_code", "n"],
                         [[f"{i:07d}", i] for i in range(60)], roles)
    check("an identifier axis is never charted", ident.kind, "table")

    # Cardinality cap, and no silent "Others" bucket inventing a number.
    many = charts.infer(["region", "n"], [[f"cat{i}", 100 - i] for i in range(60)])
    check("categories are capped", len(many.labels), config.CHART_MAX_CATEGORIES)
    check_true("and the cap is disclosed", any("largest" in n for n in many.notes))
    check("the largest category survives the cap", many.labels[0], "cat0")
    check_true("no invented Others bucket",
               not any("other" in str(x).lower() for x in many.labels))

    # Null dates are dropped from a line and reported, not plotted as 1970.
    line = charts.infer(["order_date", "n"],
                        [["2025-01-01", 1], [None, 2], ["2025-02-01", 3]], roles)
    check("null dates are dropped", len(line.labels), 2)
    check_true("and reported", any("no order_date" in n for n in line.notes))

    # Blank categories get a visible label rather than vanishing.
    blank = charts.infer(["region", "n"], [[None, 5], ["S", 3]])
    check_true("blank category is labelled", charts.BLANK in blank.labels)

    # Long lines are thinned so the browser is not asked to draw thousands of points.
    long_line = charts.infer(["order_date", "n"],
                             [[f"2025-01-{(i % 28) + 1:02d}", i]
                              for i in range(config.CHART_MAX_POINTS * 3)], roles)
    check_true("long lines are downsampled",
               len(long_line.labels) <= config.CHART_MAX_POINTS + 1)

    # A table fallback is capped for the DOM.
    big = charts.infer(["a", "b"], [[f"x{i}", f"y{i}"] for i in range(400)])
    check("table fallback is capped", len(big.rows), charts.TABLE_DISPLAY_ROWS)

    # A donut is only offered when the parts actually make a whole.
    share = charts.infer(["region", "n"], [["N", 5], ["S", 3], ["E", 2]])
    check_true("donut offered for a few positive parts", "donut" in charts.options_for(share))
    negative = charts.infer(["region", "n"], [["N", 5], ["S", -3]])
    check_true("donut refused for negative values",
               "donut" not in charts.options_for(negative))


# --------------------------------------------------------------------------- panels
def test_dashboard() -> None:
    print("\ndashboards")
    did = dashboard.create_dashboard("Test board")
    dashboard.add_panel(did, "By region",
                        "SELECT region, COUNT(*) AS n FROM orders GROUP BY 1")
    dashboard.add_panel(did, "Total", "SELECT SUM(claim_amount) AS total FROM orders")

    run = dashboard.run_dashboard(did)
    check("both panels ran", len(run["panels"]), 2)
    check("bar for the grouped panel", run["panels"][0]["chart"]["kind"], "bar")
    check("stat for the single value", run["panels"][1]["chart"]["kind"], "stat")
    check("no panel errored", [p["error"] for p in run["panels"]], ["", ""])

    # A write can never be saved as a panel.
    try:
        dashboard.add_panel(did, "bad", "DELETE FROM orders")
        check("a write is refused at save time", "accepted", "refused")
    except sql.UnsafeSQL:
        check("a write is refused at save time", "refused", "refused")

    # ...and stored SQL is validated again on every run, not trusted because it was
    # safe when it was pinned.
    db = dashboard.connect()
    with dashboard._lock:                                   # noqa: SLF001
        db.execute("UPDATE panels SET sql='DROP TABLE orders' WHERE title='Total'")
        db.commit()
    run2 = dashboard.run_dashboard(did)
    tampered = [p for p in run2["panels"] if p["title"] == "Total"][0]
    check_true("tampered SQL is rejected at run time", bool(tampered["error"]))
    check("and nothing was dropped",
          sql.run("SELECT COUNT(*) FROM orders").ok, True)

    # A panel reading one member of a union view says so.
    members = [t for t in catalog.list_tables(with_columns=False) if t.kind == "table"]
    views = [t for t in catalog.list_tables(with_columns=False) if t.kind == "view"]
    if views and members:
        member = views[0].members[0]
        pid = dashboard.add_panel(did, "one file",
                                  f"SELECT region, COUNT(*) AS n FROM {member} GROUP BY 1")
        run3 = dashboard.run_dashboard(did)
        one = [p for p in run3["panels"] if p["id"] == pid][0]
        check_true("a single-file panel is flagged",
                   any(views[0].name in n for n in one["chart"]["notes"]))

    dashboard.delete_dashboard(did)
    check("deleting a dashboard removes it", dashboard.get_dashboard(did), None)


def test_chart_override() -> None:
    """Choosing a chart type must never be a one-way door.

    The bug: options were derived from the *drawn* kind, so picking "table" left
    "table" as the only option -- and the forced chart carried no rows, so the
    panel showed "nothing to show" with no way back.
    """
    print("\nswitching chart type")
    cols, rows = ["region", "n"], [["East", 5], ["West", 3], ["North", 2]]
    auto = charts.infer(cols, rows)
    options = charts.options_for(auto)
    check_true("more than one option to begin with", len(options) > 1)

    forced = charts.infer(cols, rows, prefer="table")
    check("the forced kind is used", forced.kind, "table")
    check("the natural kind is remembered", forced.natural, auto.kind)
    check("the options do not collapse", charts.options_for(forced), options)
    check("rows travel with it, so the table renders", len(forced.rows), 3)
    check_true("and so do the columns", forced.columns == cols)

    back = charts.infer(cols, rows, prefer="hbar")
    check("switching back works", back.kind, "hbar")

    # Every chart carries its rows, whatever the kind.
    for kind in charts.options_for(auto):
        got = charts.infer(cols, rows, prefer=kind)
        check_true(f"rows present as {kind}", bool(got.rows))

    # An override that makes no sense for the data is refused.
    line = charts.infer(["d", "n"], [["2025-01-01", 1], ["2025-02-01", 2]],
                        {"d": "date"})
    check("a nonsense override is ignored",
          charts.infer(["d", "n"], [["2025-01-01", 1], ["2025-02-01", 2]],
                       {"d": "date"}, prefer="donut").kind, line.kind)

    # Stacking is offered only where parts add up; donut only for a single series.
    multi = charts.infer(["region", "seg", "n"],
                         [["N", "A", 1], ["N", "B", 2], ["S", "A", 3]])
    check_true("stacked offered for multi-series", "sbar" in charts.options_for(multi))
    check_true("donut not offered for multi-series",
               "donut" not in charts.options_for(multi))
    check_true("area offered for a line", "area" in charts.options_for(line))


def test_titles_and_filters() -> None:
    print("\nauto titles and the date filter")
    from app.tables import filters

    check("measure by dimension",
          charts.suggest_title(charts.infer(["region", "claims"], [["E", 1]])),
          "Claims by Region")
    check("a single value names itself",
          charts.suggest_title(charts.infer(["total_value"], [[9.5]])), "Total value")
    check("falls back to the question",
          charts.suggest_title(charts.infer(["a"], [["x"]]), "what is in the contract?"),
          "A")

    # Only a plain ISO date is ever accepted into the SQL.
    for bad in ["2025-13-99'; DROP TABLE orders --", "yesterday", "2025/10/01", ""]:
        out, touched = filters.apply_dates("SELECT * FROM orders", bad, "")
        check(f"rejected date {bad[:22]!r}", touched, [])

    date_cols = {"orders": "order_date"}
    out, touched = filters.apply_dates(
        "SELECT region, COUNT(*) AS n FROM orders GROUP BY 1 ORDER BY 2 DESC",
        "2025-08-01", "2025-10-31", date_cols)
    check("the table was filtered", touched, ["orders"])
    # GROUP/ORDER used to be swallowed as if they were table aliases.
    check_true("GROUP BY survives", "GROUP BY 1" in out)
    check_true("ORDER BY survives", "ORDER BY 2 DESC" in out)
    check("and it still runs", sql.run(out).ok, True)

    out, touched = filters.apply_dates(
        "SELECT COUNT(*) FROM orders o WHERE o.region = 'West'",
        "2025-08-01", "", date_cols)
    check("an alias is preserved", sql.run(out).ok, True)

    # A table name inside a string literal is not a FROM clause.
    out, touched = filters.apply_dates(
        "SELECT 'from orders is text' AS note", "2025-08-01", "", date_cols)
    check("string literals are left alone", touched, [])
    check("the literal survives", sql.run(out).rows[0][0], "from orders is text")

    # Filtering actually narrows the data, and clearing restores it.
    full = sql.run("SELECT COUNT(*) FROM orders").rows[0][0]
    narrowed_sql, _ = filters.apply_dates("SELECT COUNT(*) FROM orders",
                                          "2025-09-01", "2025-12-31", date_cols)
    narrowed = sql.run(narrowed_sql).rows[0][0]
    check_true("the range narrows the result", narrowed < full)
    cleared, touched = filters.apply_dates("SELECT COUNT(*) FROM orders", "", "", date_cols)
    check("clearing restores the original query", sql.run(cleared).rows[0][0], full)


def main() -> int:
    print(f"scratch db: {config.TABLES_DB_PATH}\n")
    test_detection()
    test_import()
    test_union_view()
    test_sql_guard()
    test_partial_answers()
    test_no_hang()
    test_intent()
    test_charts()
    test_chart_override()
    test_titles_and_filters()
    test_dashboard()
    print("\n" + ("-" * 60))
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all table tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
