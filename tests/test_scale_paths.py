"""The operations whose cost must not follow the size of the index.

Some work is allowed to grow with the corpus -- an index run reads every file, and a
gap analysis groups every row. What must not grow is anything that runs on a timer or
on every question, because that is what decides whether the app is usable at ten
million documents rather than at ten thousand.

Every check here was a real regression, measured at 200,000 files and 2M chunks:

* ``locate.find`` loaded every file row and scored it in Python: 2,669ms, linear, so
  ten million files would have been minutes per question.
* ``stats()`` made three full table passes, and the page polls it every 15 seconds:
  173ms a poll for numbers that only move when an index run ends.
* ``file_ids_for`` had no index on rel_path, so scoping a search scanned the table.

The test builds a small index and a larger one and compares, because an absolute
millisecond figure means nothing on someone else's machine -- the shape of the curve
does.

Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
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


SMALL, LARGE = 2_000, 40_000

# Enough distinct words that a two-word lookup lands on a handful of files at either
# size, which is what a real corpus of file names looks like.
WORDS = ["kirloskar", "maintenance", "quarterly", "invoice", "roadmap", "handbook",
         "forecast", "appendix", "audit", "warranty", "logistics", "throughput",
         "calibration", "downtime", "inventory", "compliance", "retrofit", "tender",
         "commissioning", "spares", "overhaul", "turbine", "gearbox", "alternator",
         "switchgear", "bearing", "coolant", "emission", "vibration", "pipeline"]


def build(store, n_files: int) -> None:
    rows = []
    exts = [".pdf", ".docx", ".xlsx", ".csv"]
    folders = ["reports", "sheets", "archive/2025", ""]
    for i in range(1, n_files + 1):
        folder = folders[i % len(folders)]
        ext = exts[i % len(exts)]
        # Distinctive, the way real file names are. When every name shares a word
        # the match count grows with the corpus and so does the ranking, which
        # measures the generator rather than the index.
        name = f"{WORDS[i % len(WORDS)]}_{i}_{WORDS[(i * 7) % len(WORDS)]}{ext}"
        rel = (folder + "/" if folder else "") + name
        rows.append((i, f"/data/{name}", rel, name, ext, 1000 + i, 0.0, f"h{i}", 2, 0.0))
    store.db.executemany(
        "INSERT INTO files (id, path, rel_path, name, ext, size, mtime, hash,"
        " n_chunks, indexed_at, status) VALUES (?,?,?,?,?,?,?,?,?,?,'ok')", rows)
    store.db.executemany(
        "INSERT INTO chunks (id, file_id, ord, text, n_chars, loc, heading, vec_row)"
        " VALUES (?,?,?,?,?,?,'',?)",
        [(i, (i % n_files) + 1, i % 5, f"text body {i} about maintenance", 30,
          f"p. {i % 9}", i) for i in range(1, n_files * 2 + 1)])
    # One file with a name nothing else shares, in both corpora. Looking it up is
    # the property under test: a real file name identifies one file, so an index
    # lookup should cost the same whether there are two thousand files or ten
    # million. Searching a word the generator sprinkles everywhere would instead
    # measure how many rows the generator made match.
    store.db.execute(
        "INSERT INTO files (id, path, rel_path, name, ext, size, mtime, hash,"
        " n_chunks, indexed_at, status) VALUES"
        " (1000001,'/data/zzsentinel_marker.pdf','zzsentinel_marker.pdf',"
        " 'zzsentinel_marker.pdf','.pdf',1,0,'hs',1,0,'ok')")
    store.db.commit()
    store.rebuild_name_index()


def fastest(fn, n: int = 3) -> float:
    fn()
    best = 1e9
    for _ in range(n):
        a = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - a)
    return best


def main() -> int:
    from app.core.store import Store

    tmp = Path(tempfile.mkdtemp(prefix="scalepaths_"))
    try:
        stores = {}
        for label, n in (("small", SMALL), ("large", LARGE)):
            d = tmp / label
            d.mkdir(parents=True, exist_ok=True)
            st = Store(d / "index.db", d / "vectors.f32")
            build(st, n)
            stores[label] = st

        print("-" * 66)
        print(f"cost at {SMALL:,} files vs {LARGE:,} files ({LARGE // SMALL}x the data)")
        print("-" * 66)

        def compare(label, make, budget):
            """A path is flat if x20 the data does not cost it x20 the time."""
            small = fastest(make(stores["small"]))
            large = fastest(make(stores["large"]))
            growth = large / small if small > 0 else 1.0
            ok = growth <= budget
            check(f"{label}", ok,
                  f"{small * 1000:.1f}ms -> {large * 1000:.1f}ms "
                  f"({growth:.1f}x for {LARGE // SMALL}x the data, budget {budget}x)")
            return growth

        # The page polls this every 15 seconds. It must not notice the corpus at all.
        compare("stats() does not recount per poll",
                lambda s: (lambda: s.stats()), 4.0)

        # A name that identifies one file, as real names do. This is the check that
        # matters: before the index, locate() read every row and scored it in Python,
        # so the same lookup cost twenty times more at twenty times the size.
        compare("finding one named file stays flat",
                lambda s: (lambda: s.search_files_by_name(["zzsentinel", "marker"])),
                4.0)

        # A word in no file at all: the miss must be a lookup too, not a scan.
        compare("a name that matches nothing stays flat",
                lambda s: (lambda: s.search_files_by_name(["zzqqxxabsent"])), 4.0)

        compare("scoping a search to chosen files stays flat",
                lambda s: (lambda: s.file_ids_for(
                    ["reports/quarterly review 5 volume.docx",
                     "sheets/quarterly review 9 volume.pdf"])), 4.0)

        compare("fetching chunks by id stays flat",
                lambda s: (lambda: s.get_chunks(list(range(1, 41)))), 4.0)

        compare("keyword search stays flat",
                lambda s: (lambda: s.search_bm25("maintenance", 40, [])), 6.0)

        print("\n" + "-" * 66)
        print("the name index is correct, not just fast")
        print("-" * 66)
        st = stores["large"]
        hits = st.search_files_by_name(["kirloskar"])
        check("a lookup returns something", bool(hits), f"{len(hits)} hits")
        check("a lookup is capped", len(hits) <= 300, f"{len(hits)}")
        # Underscores, dots and hyphens split words: "handbook" has to find
        # employee_handbook.docx, which is why they are separators and not token
        # characters.
        st.db.execute(
            "INSERT INTO files (id, path, rel_path, name, ext, size, mtime, hash,"
            " n_chunks, indexed_at, status) VALUES"
            " (999999,'/x','x','employee_zzglossary.docx','.docx',1,0,'h',1,0,'ok')")
        st.db.commit()
        st._index_name(999999, "employee_zzglossary.docx", "x")
        st.db.commit()
        found = [r["name"] for r in st.search_files_by_name(["zzglossary"])]
        check("an underscored name is found by one of its words",
              "employee_zzglossary.docx" in found, str(found[:3]))

        check("a word in no file finds nothing",
              st.search_files_by_name(["zzqqxxnothing"]) == [])

        print("\n" + "-" * 66)
        print("counts are cached but not stale")
        print("-" * 66)
        before = st.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        st.db.execute(
            "INSERT INTO files (id, path, rel_path, name, ext, size, mtime, hash,"
            " n_chunks, indexed_at, status) VALUES"
            " (999998,'/y','y','y.pdf','.pdf',1,0,'h',1,0,'ok')")
        st.db.commit()
        # A new index run changes last_index, which is what retires the cache.
        st.set_meta("last_index", "2026-01-01 00:00:00")
        after = st.stats()["files"]
        check("an index run retires the cached counts", after == before + 1,
              f"{before} -> {after}")
        # And the cache really is a cache: without a new index run, a row inserted
        # behind its back is not picked up until the TTL or the next run.
        st.db.execute(
            "INSERT INTO files (id, path, rel_path, name, ext, size, mtime, hash,"
            " n_chunks, indexed_at, status) VALUES"
            " (999997,'/z','z','z.pdf','.pdf',1,0,'h',1,0,'ok')")
        st.db.commit()
        check("counts are actually cached between runs",
              st.stats()["files"] == after, f"{st.stats()['files']} vs {after}")

        for s in stores.values():
            s.db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "-" * 66)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all scale-path tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
