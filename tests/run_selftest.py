"""Self-test: build sample documents, index them, and check what comes back.

Runs against a throwaway data folder and a throwaway index, so it never touches
your real DATA_DIR or storage.  Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = "  pass  ", "  FAIL  "


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print((PASS if ok else FAIL) + name + (f"   {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)


# Questions whose answer lives in ordinary text, and the file that should win.
TEXT_CASES = [
    ("How much revenue did Acme make in Q3 2025?", "quarterly_report.pdf"),
    ("How many days of annual leave do employees get?", "employee_handbook.docx"),
    ("Can I work from home?", "employee_handbook.docx"),
    ("Who manages the Warsaw office?", "employee_handbook.docx"),
    ("What is planned for Q2 2026?", "roadmap.pptx"),
    ("What did we spend on cloud hosting?", "sales_2025.xlsx"),
    ("Which customer pays the highest MRR?", "customers.csv"),
    ("How often do passwords rotate?", "policy.html"),
    ("What is the support phone number?", "readme.txt"),
    ("Why was the Nordic launch postponed?", "notes.md"),
    ("When does the Nordwind contract expire?", "mail.eml"),
    ("data retention period", "config.json"),
]

# Questions whose answer exists ONLY inside a picture - these prove OCR works.
OCR_CASES = [
    ("What is the invoice number?", "invoice_photo.png", "88421"),
    ("How many days of termination notice?", "service_agreement_scan.pdf", "90"),
    ("What is the Q4 revenue forecast?", "forecast_appendix.pdf", "6.1"),
    ("What is the safety code?", "employee_handbook.docx", "7719"),
    ("What is our market share?", "roadmap.pptx", "34"),
]


def main(keep: bool = False) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_selftest_"))
    os.environ["DATA_DIR"] = str(tmp / "data")
    os.environ["STORAGE_DIR"] = str(tmp / "store")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    report = Report()
    try:
        print(f"workspace: {tmp}\n")
        from tests.make_fixtures import build

        data = build(tmp / "data")
        n_files = sum(1 for f in data.rglob("*") if f.is_file())
        print(f"built {n_files} sample files\n")

        from app import config
        from app.ingestion import ocr

        print(f"OCR engine: {ocr.engine_name() or 'none (pip install rapidocr-onnxruntime)'}")
        print(f"embedding : {config.EMBED_MODEL}\n")

        # ---------------------------------------------------------------- index
        from app.ingestion.indexer import index_once

        summary = index_once(rebuild=True)
        report.check("every sample file indexed",
                     summary["failed"] == 0 and summary["added"] == n_files,
                     f"added={summary['added']}/{n_files} failed={summary['failed']}")
        report.check("chunks were written", summary["chunks"] > 0,
                     f"{summary['chunks']} chunks")

        summary2 = index_once(rebuild=False)
        report.check("re-index skips unchanged files",
                     summary2["skipped"] == summary2["total"] and summary2["added"] == 0,
                     f"skipped={summary2['skipped']}/{summary2['total']}")

        # ---------------------------------------------------------------- retrieval
        print("\n--- retrieval (plain text) ---")
        from app.retrieval import search as retrieval

        hits_ok = 0
        for question, expected in TEXT_CASES:
            hits = retrieval.search(question, top_k=3)
            names = [h["name"] for h in hits]
            ok = bool(names) and names[0] == expected
            hits_ok += ok
            if not ok:
                print(f"        {question!r} -> {names[:3]} (wanted {expected})")
        report.check("text questions find the right file first",
                     hits_ok >= len(TEXT_CASES) - 1, f"{hits_ok}/{len(TEXT_CASES)}")

        # ---------------------------------------------------------------- ocr
        print("\n--- retrieval (text that exists only inside images) ---")
        if ocr.available():
            ocr_ok = 0
            for question, expected, token in OCR_CASES:
                hits = retrieval.search(question, top_k=4)
                names = [h["name"] for h in hits]
                found = expected in names
                token_ok = any(token in h["text"] for h in hits if h["name"] == expected)
                ocr_ok += bool(found and token_ok)
                print(f"        {'ok ' if found and token_ok else 'MISS'} "
                      f"{question:<40} -> {names[0] if names else '-'}")
            report.check("image-only questions are answerable",
                         ocr_ok >= len(OCR_CASES) - 1, f"{ocr_ok}/{len(OCR_CASES)}")
        else:
            print("        skipped - no OCR engine installed")

        # ---------------------------------------------------------------- llm
        print("\n--- language model ---")
        from app.core import llm

        health = llm.health()
        report.check("LLM reachable with the configured model", health["model_ready"],
                     health["hint"] or f"{health['model']} @ {health['endpoint']}")

    finally:
        if keep:
            print(f"\nkept workspace: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s): {', '.join(report.failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--keep" in sys.argv))
