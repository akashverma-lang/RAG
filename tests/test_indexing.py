"""Incremental indexing: what gets redone on a second run, and what must not.

Runs against a throwaway data folder and a throwaway index, so it never touches
your real DATA_DIR or storage.  Invoked by ``python rag.py test``.

Both cases here are regressions with a visible cost:

* A spreadsheet that is already a SQL table and then gets edited used to fall
  through to the text path.  The old rows stayed in tables.db, so "how many claims"
  kept answering from the version before the edit -- confidently, and with no sign
  on screen that anything was stale.
* A file that could not be parsed was re-parsed on every single run, because the
  skip only honoured files whose last run had succeeded.  On a folder holding a few
  hundred scans that came back blank, re-OCRing them was most of the run.
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
_failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print((PASS if ok else FAIL) + name + (f"   got={got} want={want}" if not ok else ""))
    if not ok:
        _failures.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_indexing_"))
    data = tmp / "data"
    data.mkdir(parents=True)
    os.environ["DATA_DIR"] = str(data)
    os.environ["STORAGE_DIR"] = str(tmp / "store")
    os.environ["OCR_ENABLED"] = "false"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from app import config
    from app.core.store import Store
    from app.ingestion.indexer import Indexer
    from app.tables import catalog

    store = Store(config.DB_PATH, config.VEC_PATH)
    indexer = Indexer(store)

    def write_csv(n_rows: int) -> None:
        rows = ["region,status,amount"]
        rows += [f"North,Open,{i * 10}" for i in range(1, n_rows + 1)]
        (data / "claims.csv").write_text("\n".join(rows), encoding="utf-8")

    def run() -> dict:
        indexer.start()
        indexer._thread.join(300)          # noqa: SLF001 - the test drives it directly
        return indexer.progress.snapshot()

    def sql_rows() -> int:
        con = catalog.connect_readonly()
        try:
            names = [r["name"] for r in con.execute(
                "SELECT name FROM _catalog_tables WHERE kind='table'")]
            if not names:
                return 0
            return int(con.execute(f'SELECT COUNT(*) FROM "{names[0]}"').fetchone()[0])
        finally:
            con.close()

    try:
        print("\nan edited spreadsheet is re-imported into SQL")
        write_csv(10)
        run()
        check("the first import lands in SQL", sql_rows(), 10)
        check("and is not embedded as text", store.stats()["chunks"], 0)

        write_csv(25)
        run()
        check("the edit reaches SQL", sql_rows(), 25)
        check("and is still a table, not text chunks", store.stats()["chunks"], 0)

        snap = run()
        check("an untouched sheet is skipped", snap["skipped"], 1)
        check("with nothing re-imported", snap["tables"], 0)
        check("and the row count holds", sql_rows(), 25)

        print("\na file that cannot be read is not retried every run")
        broken = data / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4\nthis is not actually a pdf\n")
        check("it is recorded as failed once", run()["failed"], 1)
        snap = run()
        check("the next run does not re-parse it", snap["failed"], 0)
        check("it is skipped instead", snap["skipped"], 2)

        broken.write_bytes(b"%PDF-1.4\nstill not a pdf, but different bytes\n")
        check("but a changed file IS retried", run()["failed"], 1)

        _check_compaction(check)
        _check_progress(check)
    finally:
        store.db.close()
        try:
            catalog.connect().close()
        except Exception:                                   # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "-" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all indexing tests passed")
    return 0


class _Buffer:
    """A stream that is explicitly not a terminal, for the redirected case."""

    encoding = "utf-8"

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> None:
        self.parts.append(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    @property
    def value(self) -> str:
        return "".join(self.parts)


def _check_progress(check) -> None:
    """The estimate has to be honest about a folder whose file types differ in cost."""
    import time

    from app.ingestion.progress import ConsoleBar, Meter, human_bytes, human_time

    print("\nthe remaining-time estimate")
    check("an unknown duration says so", human_time(None), "estimating")
    check("seconds", human_time(45), "45s")
    check("minutes", human_time(195), "3m 15s")
    check("hours", human_time(7400), "2h 03m")
    check("bytes", human_bytes(2_500_000), "2.4 MB")

    def started_run(plan_files):
        """A meter that believes it has been going a while.

        Nothing is estimated in the first second and a half, because a rate measured
        over a few milliseconds is noise -- so a test that runs in 200ms would never
        see an estimate at all.
        """
        meter = Meter()
        meter.plan(plan_files)
        meter.started -= 10.0
        return meter

    m = Meter()
    m.plan([(".txt", 1000)] * 10 + [(".pdf", 1000)] * 10)
    first = m.snapshot()
    check("nothing is promised before there is evidence", first["eta"], None)
    check("the plan is counted", first["files_total"], 20)

    # Cheap files first. The expensive type has not been seen, so it must NOT be
    # costed at the cheap rate -- that is what made the old estimate promise forty
    # seconds and then climb.
    m = started_run([(".txt", 1000)] * 10 + [(".pdf", 1000)] * 10)
    m2 = started_run([(".txt", 1000)] * 20)
    for _ in range(10):
        m.complete(".txt", 1000, 0.01)
        m2.complete(".txt", 1000, 0.01)
        time.sleep(0.06)
    cheap_only, same_type = m.snapshot(), m2.snapshot()
    check("an estimate is offered once the run is under way",
          cheap_only["eta"] is not None and same_type["eta"] is not None, True)
    check("an untried file type is not assumed to be cheap",
          cheap_only["eta"] >= same_type["eta"], True)

    # Now the expensive ones arrive: the estimate must reflect them.
    for _ in range(3):
        m.complete(".pdf", 1000, 0.5)
        time.sleep(0.06)
    check("a slow file type keeps the estimate above zero",
          (m.snapshot()["eta"] or 0) > 0, True)

    for _ in range(7):
        m.complete(".pdf", 1000, 0.5)
    done = m.snapshot()
    check("everything done means nothing left", done["eta"], 0.0)
    check("and the bar is full", done["pct"], 100.0)
    check("every file is accounted for", done["files_done"], 20)

    # observe() adds cost without retiring a file twice.
    before = m.snapshot()["files_done"]
    m.observe(".pdf", 1000, 0.2)
    check("observing extra cost does not double-count", m.snapshot()["files_done"], before)

    print("\nthe progress bar")
    snap = {"pct": 42.0, "files_done": 500, "files_total": 1203,
            "bytes_per_sec": 2_000_000, "eta": 193.0}
    for width in (120, 100, 80, 60, 45):
        line = ConsoleBar(stream=_Buffer(), width=width).render(snap, "indexing", "a.pdf")
        check(f"fits a {width}-column window", len(line) <= width - 1, True)
        check(f"keeps the estimate at {width} columns", "ETA" in line, True)

    live = ConsoleBar(stream=_Buffer(), width=90, interactive=True)
    live.draw(snap, "indexing", "a.pdf")
    live.draw(snap, "indexing", "b.pdf")
    check("a terminal gets an in-place redraw", live.stream.value.count("\r") >= 2, True)

    piped = ConsoleBar(stream=_Buffer(), width=90, interactive=False)
    for _ in range(200):
        piped.draw(snap, "indexing", "a.pdf")
    check("a redirected run is not 200 lines of carriage returns",
          piped.stream.value.count("\r"), 0)
    check("it writes one line instead",
          len([ln for ln in piped.stream.value.splitlines() if ln.strip()]), 1)
    check("and a non-tty stream is detected on its own",
          ConsoleBar(stream=_Buffer(), width=90).interactive, False)


def _check_compaction(check) -> None:
    """Re-indexing a file abandons its old vectors; compaction must reclaim them.

    Until it did, the vector file only ever grew -- re-index the same folder ten
    times and nine tenths of every similarity pass was spent multiplying rows that
    no chunk pointed at any more.
    """
    import numpy as np

    from app import config
    from app.core.store import Store

    print("\ncompaction reclaims abandoned vector rows")
    rng = np.random.default_rng(7)
    dim = 32
    vec_path = config.STORAGE_DIR / "compact_test.f32"
    store = Store(config.STORAGE_DIR / "compact_test.db", vec_path)

    def put(name: str, n: int) -> None:
        chunks = [{"text": f"{name} chunk {i}", "loc": str(i), "heading": ""}
                  for i in range(n)]
        store.upsert_file_chunks(
            path=name, rel_path=name, size=100, mtime=1.0, file_hash=f"{name}{n}",
            chunks=chunks, vectors=rng.normal(size=(n, dim)).astype(np.float32))

    try:
        for name, n in (("a", 4), ("b", 5), ("c", 3), ("a", 4), ("b", 5), ("a", 4)):
            put(name, n)
        check("dead rows accumulate as files are rewritten", store.vector_usage(), (25, 12))

        query = rng.normal(size=dim).astype(np.float32)
        before = store.search_dense(query, 12)

        out = store.compact_vectors()
        check("the run reports what it reclaimed", out["freed"], (25 - 12) * 4 * dim)
        check("the file is now exactly the live rows", store.vector_usage(), (12, 12))
        check("and is that size on disk", vec_path.stat().st_size, 12 * 4 * dim)

        after = store.search_dense(query, 12)
        check("the same chunks rank the same", [c for c, _ in after], [c for c, _ in before])
        check("with the same scores",
              [round(s, 6) for _, s in after], [round(s, 6) for _, s in before])
        rows = store.get_chunks([c for c, _ in after])
        check("and every row still resolves", len(rows), len(after))

        check("a clean index is left alone", store.compact_vectors()["compacted"], False)
        check("nothing is truncated by that", vec_path.stat().st_size, 12 * 4 * dim)
    finally:
        store.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
