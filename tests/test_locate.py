"""Finding a file on disk from however the user happened to refer to it.

The index is stubbed with a fixed set of file rows, so these test the matching rules
rather than whatever happens to be indexed on this machine. The content fallback is
stubbed too: whether the embedder can find a passage is the retrieval tests' job.

Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import os
import re
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


def _rows(root: str) -> list[dict]:
    def row(rel, size=1000, chunks=2):
        return {"path": os.path.join(root, *rel.split("/")), "rel_path": rel,
                "name": rel.split("/")[-1], "ext": os.path.splitext(rel)[1].lower(),
                "size": size, "n_chunks": chunks, "status": "ok"}
    return [
        row("sheets/sales_2025.xlsx"),
        row("sheets/customers.csv"),
        row("Process Quality Index Kagal.xlsx"),
        row("AK_Analysis.xlsx"),
        row("reports/employee_handbook.docx"),
        row("reports/roadmap.pptx"),
        row("reports/quarterly_report.pdf"),
        row("scans/invoice_photo.png"),
        row("260621 Kirloskar RAMS Study to SB Energy_Prepared for OpenAI V2 (3).pdf",
            size=3_300_000, chunks=128),
    ]


def main() -> int:
    from app import config
    from app.retrieval import locate

    root = str(config.DATA_DIR)
    rows = _rows(root)

    # The store and the content search are both replaced: this file is about the
    # matching rules, and a real index would make the results depend on the machine.
    import app.core.store as store_mod
    from app.retrieval import search as search_mod

    class FakeStore:
        """Stands in for the real store, including its name index.

        search_files_by_name is what locate() now calls first: at scale it is an FTS
        lookup, so the fake matches on whole words the same way rather than handing
        back everything, or the test would not exercise the shortlist path at all.
        """

        def files_for_lookup(self, limit=50000):
            return rows[:limit]

        def search_files_by_name(self, terms, limit=300):
            want = {str(t).lower() for t in terms if str(t).strip()}
            out = []
            for r in rows:
                words = set(re.findall(r"[a-z0-9]+", (r["name"] + " " + r["rel_path"]).lower()))
                if want & words or any(
                        w.startswith(t) for t in want for w in words):
                    out.append(r)
            return out[:limit]

    real_get = store_mod.get_store
    real_search = search_mod.search
    store_mod.get_store = lambda: FakeStore()
    # The RAMS pdf is the only thing whose *contents* anything is about here.
    rams = rows[-1]
    search_mod.search = lambda q, **k: [
        {**rams, "text": "reliability and availability of gas engine generators",
         "score": 0.9}]

    try:
        print("-" * 60)
        print("is this a question about where, or about what?")
        print("-" * 60)
        for q in ["where is the RAMS report?",
                  "what is the location of sales_2025.xlsx",
                  "which folder is the kirloskar file in",
                  "where's my sales spreadsheet",
                  "locate the file about gas engines",
                  "show me the path of the PQI file"]:
            check(f"asks where: {q[:44]}", locate.is_location_question(q))
        for q in ["what is the purpose of the RAMS report?",
                  "where does the report say the deadline is",
                  "how many claims were there in 2025?",
                  "summarise the key points",
                  "hi"]:
            check(f"not a location question: {q[:38]}",
                  not locate.is_location_question(q))

        print("\n" + "-" * 60)
        print("resolving however it was referred to")
        print("-" * 60)
        cases = [
            ("where is sales_2025.xlsx",                 "sales_2025.xlsx", "exact name"),
            ("where is my sales spreadsheet",            "sales_2025.xlsx", "name + type word"),
            ("which folder holds the customers csv",     "customers.csv",   "name + extension"),
            ("where is my PQI spreadsheet",              "Process Quality Index Kagal.xlsx", "acronym"),
            ("where is the AK analysis file",            "AK_Analysis.xlsx", "underscored name"),
            ("where is the employee handbook",           "employee_handbook.docx", "two words"),
            ("where is the roadmap",                     "roadmap.pptx",    "one word"),
            ("where is the invoice photo",               "invoice_photo.png", "image"),
            ("locate the file about gas engine reliability",
             "260621 Kirloskar RAMS Study to SB Energy_Prepared for OpenAI V2 (3).pdf",
             "by contents only"),
        ]
        for q, want, how in cases:
            found = locate.find(q)
            got = found[0].name if found else "(nothing)"
            check(f"{how}: {q[:40]}", got == want, got[:46])

        # A stated file type is a constraint, not a hint: asking for a spreadsheet
        # must not return the PDF that happens to mention sales.
        found = locate.find("where is the sales spreadsheet")
        check("a stated type keeps the answer in that format",
              found and found[0].ext_ok if hasattr(found[0], "ext_ok") else
              (found and found[0].name.endswith(".xlsx")),
              found[0].name if found else "-")

        # A name typed in full that matches nothing must not be answered with a
        # different file the content search happens to like.
        check("a typed name that does not exist finds nothing",
              locate.find("where is zzqqxx_nonexistent_thing.pdf") == [],
              str([(f.name, f.score)
                   for f in locate.find("where is zzqqxx_nonexistent_thing.pdf")][:2]))
        check("and the caller can tell a name was typed",
              locate.named_in("where is zzqqxx_nonexistent_thing.pdf")
              == ["zzqqxx_nonexistent_thing.pdf"],
              str(locate.named_in("where is zzqqxx_nonexistent_thing.pdf")))
        # Without a typed name, guessing from contents is exactly what is wanted.
        check("a vague description still falls back to contents",
              bool(locate.find("where is the file about gas engines")))

        print("\n" + "-" * 60)
        print("the answer, and the route to the file")
        print("-" * 60)
        found = locate.find("where is my sales spreadsheet")
        text = locate.render("where is my sales spreadsheet", found)
        check("the absolute path is stated", found[0].path in text)
        check("the folder is stated", os.path.dirname(found[0].path) in text)
        check("a flowchart is included", "```mermaid" in text and "flowchart TD" in text)

        segs = locate.segments_for(found[0].path)
        check("the route starts at the drive",
              segs[0].startswith("Local Disk") or segs[0].startswith("/"), segs[0])
        check("the route ends at the file", segs[-1] == "sales_2025.xlsx", segs[-1])
        check("the route passes through its folders", "sheets" in segs, " -> ".join(segs))

        # The flowchart is rendered by the page's own parser, which treats [] and |
        # as syntax. A file named with them must not produce a broken drawing.
        odd = locate._mermaid(["Local Disk (D:)", "data", "odd [name] | here", "f.pdf"])
        check("brackets and pipes are removed from labels",
              "[name]" not in odd and "|" not in odd.replace("-->", ""), odd.splitlines()[-1])

        check("sources are emitted for the citation deck",
              len(locate.sources_for(found)) == len(found)
              and locate.sources_for(found)[0]["path"] == found[0].path)

    finally:
        store_mod.get_store = real_get
        search_mod.search = real_search

    print("\n" + "-" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all locate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
