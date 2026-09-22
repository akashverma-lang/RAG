"""Incremental indexing of the data folder.

Only files whose (size, mtime) changed are hashed; only files whose *content* hash
changed are re-parsed and re-embedded.  Files that disappeared from disk are
dropped from the index.  Extraction and chunking run on a thread pool while the
main thread batches embeddings, which keeps the CPU busy without exploding memory.

Nothing here holds the file list.  The walk streams every path it finds into a
staging table and everything afterwards is a query against that table: the totals
the progress meter needs are one GROUP BY, the files that vanished are an anti-join,
and the work queue is a cursor.  Holding it in Python instead -- one list of paths
plus a set of resolved names to diff against the index -- cost several gigabytes at
twenty million files, and none of the work could start until the whole walk had
finished.
"""
from __future__ import annotations

import hashlib
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app import config
from app.core import embed
from app.core.store import Store, get_store
from app.ingestion import chunk as chunker
from app.ingestion import extractors
from app.ingestion.progress import Meter
from app.tables import catalog as table_catalog
from app.tables import importer as table_importer


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(block):
            h.update(data)
    return h.hexdigest()


@dataclass(frozen=True)
class FileRef:
    """One file to consider. Resolved once, in the walk, never again."""

    path: Path
    full: str
    size: int
    mtime: float
    ext: str


def iter_files(root: Path):
    """Walk the data folder, skipping ignored dirs, temp files and oversized files."""
    max_bytes = int(config.MAX_FILE_MB * 1024 * 1024)
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if entry.name.lower() not in config.IGNORE_DIRS and not entry.name.startswith("~"):
                        stack.append(entry)
                    continue
                name, ext = entry.name, entry.suffix.lower()
                if name.startswith(("~$", ".~")) or name.endswith((".tmp", ".crdownload", ".part")):
                    continue
                if ext not in config.INCLUDE_EXT or not extractors.supported(ext):
                    continue
                stat = entry.stat()
                if stat.st_size == 0 or stat.st_size > max_bytes:
                    continue
                yield entry, stat
            except (PermissionError, OSError):
                continue


_STAGING = """
CREATE TABLE IF NOT EXISTS scan_queue (
    path  TEXT PRIMARY KEY,
    size  INTEGER NOT NULL,
    mtime REAL    NOT NULL,
    ext   TEXT    NOT NULL
);
"""


@dataclass
class Progress:
    running: bool = False
    phase: str = "idle"
    total: int = 0
    done: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    failed: int = 0
    chunks: int = 0
    tables: int = 0
    table_rows: int = 0
    current: str = ""
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    log: list[str] = field(default_factory=list)
    meter: Meter | None = None

    def snapshot(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
        state = {k: v for k, v in self.__dict__.items() if k != "meter"}
        # The estimate comes from the meter, which weighs a byte of scanned PDF
        # against a byte of plain text instead of calling both "one file".
        live = self.meter.snapshot() if self.meter else {}
        finished = not self.running and bool(self.started)
        return {
            **state,
            "log": self.log[-40:],
            "elapsed": round(elapsed, 1),
            "eta": 0.0 if finished else live.get("eta"),
            "pct": 100.0 if finished else live.get("pct", 0.0),
            "bytes_done": live.get("bytes_done", 0),
            "bytes_total": live.get("bytes_total", 0),
            "bytes_per_sec": live.get("bytes_per_sec", 0.0),
        }


# One embedding call should be given a full batch. Handing it one file at a time
# means a folder of short notes never fills one, and the per-call overhead -- model
# setup, padding to the longest sequence in the batch -- is paid once per file
# instead of once per batch.
_EMBED_FLUSH_TEXTS = max(32, config.EMBED_BATCH)
_EMBED_FLUSH_CHARS = 4_000_000        # ...but never hold more than this in memory
# A batch is also capped by how many files it covers. A file only counts as done
# once its vectors are written, so a batch spanning sixty short notes would hold the
# bar still and then jump sixty files at once -- and the remaining time would drop
# to zero while that batch was in fact still being embedded.
_EMBED_FLUSH_FILES = 16


class Indexer:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store or get_store()
        self.progress = Progress()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._batch: list[dict] = []
        self._batch_texts = 0
        self._batch_chars = 0

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def cancel(self) -> None:
        self._cancel.set()

    def start(self, rebuild: bool = False) -> bool:
        with self._lock:
            if self.running:
                return False
            self._cancel.clear()
            self._batch, self._batch_texts, self._batch_chars = [], 0, 0
            self.progress = Progress(running=True, phase="scanning", started=time.time(),
                                     meter=Meter())
            self._thread = threading.Thread(target=self._run, args=(rebuild,),
                                            name="indexer", daemon=True)
            self._thread.start()
            return True

    def _log(self, msg: str) -> None:
        self.progress.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        if len(self.progress.log) > 400:
            del self.progress.log[:200]

    # ------------------------------------------------------------------ worker
    def _run(self, rebuild: bool) -> None:
        p = self.progress
        try:
            root = config.DATA_DIR
            if not root.exists():
                raise FileNotFoundError(
                    f"DATA_DIR does not exist: {root}  (set it in {config.ROOT / '.env'})"
                )
            if rebuild:
                self._log("rebuilding index from scratch")
                self.store.reset()
                if config.TABLES_ENABLED:
                    table_catalog.reset()

            p.phase = "scanning"
            n_files, n_bytes = self._stage_scan(root)
            p.total = n_files
            self._log(f"found {n_files:,} indexable files "
                      f"({n_bytes / 1024 / 1024:,.0f} MB) under {root}")

            # Files that vanished, found by asking the database rather than by
            # building two sets of twenty million path strings and subtracting them.
            p.phase = "reconciling"
            for gone in self._vanished():
                self.store.delete_file(gone)
                if config.TABLES_ENABLED:
                    for table in table_catalog.tables_for_source(gone):
                        table_catalog.drop_table(table)
                p.removed += 1
            if p.removed:
                self._log(f"removed {p.removed:,} deleted file(s) from the index")

            p.phase = "indexing"
            embed.get_embedder()          # load the model once, before the pool starts
            self.store.set_meta("embed_model", embed.get_embedder().name)

            pending: set = set()
            it = self._queued()
            max_inflight = max(2, config.INGEST_WORKERS * 2)
            with ThreadPoolExecutor(max_workers=config.INGEST_WORKERS,
                                    thread_name_prefix="extract") as pool:
                exhausted = False
                while not self._cancel.is_set():
                    while not exhausted and len(pending) < max_inflight:
                        try:
                            ref = next(it)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.add(pool.submit(self._prepare, ref))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        self._consume(fut.result())   # flushes itself when full
                self._flush_batch()                   # whatever is left over
            if config.TABLES_ENABLED and not self._cancel.is_set():
                # Sheets that share a schema (quarterly exports, monthly dumps) only
                # answer questions about the whole period once they are unioned.
                p.phase = "linking tables"
                for made in table_importer.rebuild_union_views(self._log):
                    self._log(f"VIEW   {made}")

            if not self._cancel.is_set():
                self._compact()
                self._build_ann()

            self._clear_staging()

            p.phase = "cancelled" if self._cancel.is_set() else "done"
            self.store.set_meta("last_index", time.strftime("%Y-%m-%d %H:%M:%S"))
            # A database made before the file-name index existed gains one here.
            # It is a no-op once the two counts agree.
            self.store.rebuild_name_index()
            self._log(
                f"{p.phase}: +{p.added} new, ~{p.updated} updated, ={p.skipped} unchanged, "
                f"-{p.removed} removed, !{p.failed} failed, {p.chunks} chunks written"
                + (f", {p.tables} table(s) / {p.table_rows:,} rows" if p.tables else "")
            )
        except Exception as exc:                              # noqa: BLE001
            p.error = f"{type(exc).__name__}: {exc}"
            p.phase = "error"
            self._log(f"ERROR {p.error}")
            traceback.print_exc()
        finally:
            p.running = False
            p.finished = time.time()
            p.current = ""

    def _build_ann(self) -> None:
        """Keep the approximate index in step with what was just indexed.

        Rows added since the last build are searched exactly, which is correct but
        linear -- so the index is rebuilt once the tail has grown enough to be worth
        the cost, rather than after every run.
        """
        if not config.ANN_ENABLED:
            return
        try:
            stats = self.store.ann_stats()
            live = stats.get("rows_live", 0)
            if live < config.ANN_MIN_VECTORS:
                return
            tail = stats.get("rows_uncovered", 0)
            if stats.get("built") and tail < config.ANN_REBUILD_AFTER:
                return
            p = self.progress
            p.phase = "building the search index"
            self._log(f"building approximate index over {live:,} vectors")
            out = self.store.build_ann(
                on_progress=lambda m: setattr(p, "current", m))
            if out.get("built"):
                self._log(
                    f"search index ready: {out['n_vectors']:,} vectors, "
                    f"{out['n_cells']:,} cells, "
                    f"{out['bytes_codes'] / 2**20:,.0f} MB "
                    f"(exact would be {out['bytes_exact'] / 2**20:,.0f} MB) "
                    f"in {out['seconds']:.0f}s")
            elif out.get("reason"):
                self._log(f"approximate index skipped: {out['reason']}")
        except Exception as exc:                              # noqa: BLE001
            # Never fail a good index run over a derived artifact; exact search is
            # still there and still correct.
            self._log(f"approximate index skipped: {type(exc).__name__}: {exc}")

    def _compact(self) -> None:
        """Reclaim the vector rows that re-indexed files left behind.

        Every search multiplies the whole vector matrix, so rows no chunk points at
        any more are re-read on every single question. Left alone the file only
        grows; the end of an index run is the one moment nothing is searching.
        """
        ratio = config.VECTOR_COMPACT_RATIO
        if ratio <= 0:
            return
        try:
            rows, live = self.store.vector_usage()
            if rows <= 0 or (rows - live) < rows * ratio:
                return
            self.progress.phase = "compacting"
            self.progress.current = "reclaiming unused vectors"
            out = self.store.compact_vectors()
            if out.get("compacted"):
                self._log(f"compacted vectors: {rows:,} -> {out['live']:,} rows "
                          f"({out['freed'] / (1024 * 1024):.1f} MB freed)")
        except Exception as exc:                              # noqa: BLE001
            # Never fail a good index run over housekeeping.
            self._log(f"vector compaction skipped: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ scanning
    def _stage_scan(self, root: Path) -> tuple[int, int]:
        """Walk the tree into a staging table. Returns (files, bytes).

        Batched inserts, so peak memory is one batch regardless of how many files
        are down there. The meter is planned from a GROUP BY at the end rather than
        from a list, which is the same information for a constant amount of memory.
        """
        db = self.store.db
        with self.store._lock:                            # noqa: SLF001
            db.executescript(_STAGING)
            db.execute("DELETE FROM scan_queue")
            db.commit()

        batch: list[tuple] = []
        n_files = 0

        def flush() -> None:
            if not batch:
                return
            with self.store._lock:                        # noqa: SLF001
                db.executemany(
                    "INSERT OR REPLACE INTO scan_queue(path,size,mtime,ext) "
                    "VALUES(?,?,?,?)", batch)
                db.commit()
            batch.clear()

        for entry, stat in iter_files(root):
            batch.append((str(entry.resolve()), stat.st_size, stat.st_mtime,
                          entry.suffix.lower()))
            n_files += 1
            if len(batch) >= 5000:
                flush()
                self.progress.total = n_files
                self.progress.current = f"found {n_files:,} files"
        flush()

        with self.store._lock:                            # noqa: SLF001
            rows = db.execute("SELECT ext, COUNT(*) n, COALESCE(SUM(size),0) b "
                              "FROM scan_queue GROUP BY ext").fetchall()
        n_bytes = sum(int(r["b"]) for r in rows)
        if self.progress.meter:
            self.progress.meter.plan_bulk(
                [(r["ext"], int(r["n"]), int(r["b"])) for r in rows])
        return n_files, n_bytes

    def _clear_staging(self) -> None:
        """Empty the work queue once the run is over.

        The table itself is kept -- it is the same shape next time -- but twenty
        million rows of paths are not worth carrying around in the index file
        between runs.
        """
        try:
            with self.store._lock:                        # noqa: SLF001
                self.store.db.execute("DELETE FROM scan_queue")
                self.store.db.commit()
        except Exception:                                 # noqa: BLE001
            pass

    def _vanished(self):
        """Indexed paths that the walk did not find, streamed from an anti-join."""
        db = self.store.db
        while True:
            with self.store._lock:                        # noqa: SLF001
                rows = db.execute(
                    "SELECT f.path FROM files f "
                    "LEFT JOIN scan_queue s ON s.path = f.path "
                    "WHERE s.path IS NULL LIMIT 500").fetchall()
            if not rows:
                return
            for r in rows:
                yield r["path"]

    def _queued(self):
        """The work queue, a page at a time. Never a list of every file."""
        db = self.store.db
        last = ""
        while not self._cancel.is_set():
            with self.store._lock:                        # noqa: SLF001
                rows = db.execute(
                    "SELECT path,size,mtime,ext FROM scan_queue "
                    "WHERE path > ? ORDER BY path LIMIT 1000", (last,)).fetchall()
            if not rows:
                return
            last = rows[-1]["path"]
            for r in rows:
                yield FileRef(path=Path(r["path"]), full=r["path"], size=int(r["size"]),
                              mtime=float(r["mtime"]), ext=r["ext"])

    # ------------------------------------------------------------------ per file
    def _prepare(self, ref: "FileRef") -> dict:
        """Runs on the pool: decide if work is needed, then parse + chunk."""
        path = ref.path
        result = {"ref": ref, "path": path, "action": "skip", "chunks": [], "hash": "",
                  "secs": 0.0, "full": ref.full}
        clock = time.perf_counter()
        try:
            full = ref.full
            prev = self.store.file_fingerprint(full)
            same_bytes_on_disk = bool(prev and prev["size"] == ref.size
                                      and abs(prev["mtime"] - ref.mtime) < 1e-6)
            unchanged = bool(prev and prev["status"] == "ok" and same_bytes_on_disk)

            # A file that could not be read last time is not worth reading again until
            # it changes.  Retrying it every run means re-OCRing every scan that came
            # back blank and re-parsing every corrupt PDF, for the same failure -- on a
            # folder with a few hundred of those it is most of the run.
            if prev and prev["status"] == "error" and same_bytes_on_disk:
                return result

            # A genuinely tabular sheet goes to SQL instead of the embedder: top-k
            # retrieval can never count or total a 100k-row sheet, and loading it
            # takes seconds where embedding it would take hours.  A spreadsheet
            # indexed as text before tables existed looks unchanged but still has to
            # migrate, so candidates are probed before the skip is honoured -- and a
            # sheet the probe rejects falls straight back through to the text path
            # without being re-embedded.
            probe = None
            if table_importer.is_candidate(path):
                already = bool(table_catalog.tables_for_source(full))
                # An edited workbook that is already a SQL table has to be re-imported,
                # not quietly re-chunked as text: the old rows stay in tables.db and
                # every "how many" keeps answering from the version before the edit.
                if not already or not unchanged:
                    self.progress.current = path.name
                    probe = table_importer.detect.probe_file(path)
                    if probe.ok:
                        result["hash"] = sha256_file(path)
                        if already and prev and prev["hash"] == result["hash"]:
                            self.store.touch_file(prev["id"], ref.mtime)
                            return result                      # same bytes, new timestamp
                        result["probe"] = probe
                        result["action"] = "table"
                        return result

            if unchanged:
                return result                                  # untouched since last run
            file_hash = sha256_file(path)
            result["hash"] = file_hash
            if prev and prev["status"] == "ok" and prev["hash"] == file_hash:
                self.store.touch_file(prev["id"], ref.mtime)
                return result                                  # same bytes, new timestamp
            self.progress.current = path.name
            blocks = extractors.extract(path)
            result["chunks"] = chunker.chunk_blocks(blocks)
            result["action"] = "update" if prev else "add"
        except Exception as exc:                               # noqa: BLE001
            result["action"] = "fail"
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # What this file actually cost to read. The meter turns it into a rate
            # per byte for this file type, which is what makes the estimate hold up
            # when the folder changes from notes to scanned contracts.
            result["secs"] = time.perf_counter() - clock
        return result

    def _consume_table(self, r: dict, rel: str) -> None:
        """Load one spreadsheet into SQL. Runs on the indexer thread, like embedding."""
        p = self.progress
        ref = r["ref"]
        path: Path = ref.path
        stat = ref
        try:
            out = table_importer.import_file(path, probe=r.get("probe"),
                                             on_progress=lambda m: setattr(p, "current", m))
        except Exception as exc:                               # noqa: BLE001
            p.failed += 1
            self.store.record_failure(r["full"], rel, stat.size, stat.mtime,
                                      f"table import: {type(exc).__name__}: {exc}")
            self._log(f"FAILED {rel}: {exc}")
            return
        if not out["ok"]:
            p.failed += 1
            self.store.record_failure(r["full"], rel, stat.size, stat.mtime,
                                      f"not tabular: {out.get('reason', '')}")
            self._log(f"SKIPPED {rel}: {out.get('reason', '')}")
            return

        # Recorded with zero chunks so the fingerprint check still skips it next run.
        self.store.upsert_file_chunks(
            path=r["full"], rel_path=rel, size=stat.size, mtime=stat.mtime,
            file_hash=r["hash"], chunks=[], vectors=None,
        )
        p.tables += len(out["tables"])
        p.table_rows += out["rows"]
        p.added += 1
        self._log(f"TABLE  {rel} -> {', '.join(out['tables'])} "
                  f"({out['rows']:,} rows in {out['took']}s)")

    def _settle(self, r: dict, extra_secs: float = 0.0) -> None:
        """One file is finished with, however it ended. Tell the meter what it cost."""
        p = self.progress
        p.done += 1
        if p.meter:
            ref = r["ref"]
            p.meter.complete(ref.ext, ref.size, float(r.get("secs", 0.0)) + extra_secs)

    def _observe(self, r: dict, secs: float) -> None:
        """Extra cost for a file already retired from the count (its embedding)."""
        if self.progress.meter:
            ref = r["ref"]
            self.progress.meter.observe(ref.ext, ref.size, secs)

    def _consume(self, r: dict) -> None:
        """Runs on the indexer thread: queue a file for embedding, or finish it now."""
        p = self.progress
        ref = r["ref"]
        path: Path = ref.path
        stat = ref
        rel = self._rel(ref)
        action = r["action"]

        if action == "skip":
            p.skipped += 1
            self._settle(r)
            return
        if action == "fail":
            p.failed += 1
            self.store.record_failure(r.get("full", str(path)), rel,
                                      stat.size, stat.mtime, r.get("error", ""))
            self._log(f"FAILED {rel}: {r.get('error', '')}")
            self._settle(r)
            return
        if action == "table":
            clock = time.perf_counter()
            self._consume_table(r, rel)
            self._settle(r, time.perf_counter() - clock)
            return

        chunks = r["chunks"]
        if not chunks:
            p.failed += 1
            self.store.record_failure(r["full"], rel, stat.size, stat.mtime,
                                      "no extractable text")
            self._log(f"EMPTY  {rel}")
            self._settle(r)
            return

        # Held back rather than embedded on its own, so several files ride in one
        # call to the model. The file is still written in a single transaction of
        # its own when the batch is flushed, so a crash can never leave one half
        # indexed.
        texts = [chunker.embed_text(c, path.name, rel) for c in chunks]
        self._batch.append({"r": r, "rel": rel, "chunks": chunks, "texts": texts})
        self._batch_texts += len(texts)
        self._batch_chars += sum(len(t) for t in texts)
        if (self._batch_texts >= _EMBED_FLUSH_TEXTS
                or self._batch_chars >= _EMBED_FLUSH_CHARS
                or len(self._batch) >= _EMBED_FLUSH_FILES):
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Embed every queued file in one call, then persist them one at a time."""
        batch, self._batch = self._batch, []
        self._batch_texts = self._batch_chars = 0
        if not batch:
            return
        p = self.progress
        p.current = (f"embedding {len(batch)} file(s)" if len(batch) > 1
                     else batch[0]["rel"])
        clock = time.perf_counter()
        texts: list[str] = []
        for item in batch:
            texts.extend(item["texts"])
        try:
            vectors = embed.encode_documents(texts)
            if vectors.shape[0] != len(texts):
                raise RuntimeError("embedding count mismatch")
        except Exception as exc:                               # noqa: BLE001
            # One bad batch must not lose every file in it, so each is retried on
            # its own and only the file that actually fails is recorded as failed.
            self._log(f"batch embedding failed ({exc}); retrying file by file")
            for item in batch:
                self._persist(item, None, 0.0)
            return

        # The call was made for the whole batch, so its cost is shared out by the
        # share of it each file asked for.
        spent = time.perf_counter() - clock
        at = 0
        for item in batch:
            n = len(item["texts"])
            share = spent * (n / len(texts)) if texts else 0.0
            self._persist(item, vectors[at:at + n], share)
            at += n

    def _persist(self, item: dict, vectors, embed_secs: float) -> None:
        """Write one file's chunks. vectors is None when the batch has to be redone."""
        p = self.progress
        r, rel, chunks = item["r"], item["rel"], item["chunks"]
        stat = r["ref"]
        clock = time.perf_counter()
        try:
            if vectors is None:
                vectors = embed.encode_documents(item["texts"])
                if vectors.shape[0] != len(chunks):
                    raise RuntimeError("embedding count mismatch")
            self.store.upsert_file_chunks(
                path=r["full"], rel_path=rel, size=stat.size, mtime=stat.mtime,
                file_hash=r["hash"], chunks=chunks, vectors=vectors,
            )
        except Exception as exc:                               # noqa: BLE001
            p.failed += 1
            self.store.record_failure(r["full"], rel, stat.size, stat.mtime,
                                      f"{type(exc).__name__}: {exc}")
            self._log(f"FAILED {rel}: {exc}")
            self._settle(r, embed_secs + (time.perf_counter() - clock))
            return

        p.chunks += len(chunks)
        if r["action"] == "add":
            p.added += 1
        else:
            p.updated += 1
        self._log(f"{'ADDED ' if r['action'] == 'add' else 'UPDATED'} {rel} "
                  f"({len(chunks)} chunks)")
        self._settle(r, embed_secs + (time.perf_counter() - clock))

    @staticmethod
    def _rel(ref: "FileRef") -> str:
        """Path relative to DATA_DIR. Uses the walk's resolve, never a fresh one."""
        try:
            return Path(ref.full).relative_to(config.DATA_DIR.resolve()).as_posix()
        except Exception:                                     # noqa: BLE001
            return ref.path.as_posix()


_indexer: Indexer | None = None


def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def index_once(rebuild: bool = False, show_progress: bool = True) -> dict:
    """Blocking index run with a live console bar - used by "python rag.py index"."""
    from app.ingestion.progress import ConsoleBar, human_bytes, human_time

    ix = get_indexer()
    ix.start(rebuild=rebuild)
    bar = ConsoleBar() if show_progress else None

    # Four redraws a second: fast enough that the remaining time visibly counts down
    # rather than stepping, cheap enough to be free next to reading the files.
    while ix.running:
        if bar:
            p = ix.progress
            bar.draw(p.meter.snapshot() if p.meter else {}, p.phase, p.current)
        time.sleep(0.25)

    snap = ix.progress.snapshot()
    if bar:
        moved = snap.get("bytes_done", 0)
        elapsed = snap.get("elapsed", 0.0) or 0.0
        average = moved / elapsed if elapsed > 0 else 0.0
        bar.done(f"  {snap['phase']} in {human_time(elapsed)}  -  "
                 f"{snap['done']:,} files, {human_bytes(moved)} read, "
                 f"{human_bytes(average)}/s average")
    return snap
