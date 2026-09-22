"""Persistence: SQLite (metadata + FTS5/BM25) and a flat float32 vector file.

Design notes
------------
* One SQLite database holds files, chunks and a standalone FTS5 table used for BM25.
* Vectors live in a raw float32 file, one row per chunk, appended on write and read
  back with np.memmap.  Rows belonging to deleted chunks become tombstones and are
  reclaimed by compaction -- this keeps writes O(1) and search a single matmul.
* All vectors are L2-normalised on write, so cosine similarity is a plain dot product.
* A sidecar file holds the chunk id of every vector row, in the same order, so the
  row -> chunk mapping is memory-mapped rather than rebuilt from SQL.  It used to be
  reconstructed by selecting every chunk in the database and filling an array a row
  at a time; at a hundred million chunks that is minutes of work before the first
  search can run, repeated every time an index run marks the vectors dirty.
* Past a threshold the exact matmul is replaced by an IVF-PQ index (see ann.py).
  It is a derived artifact: the float32 vectors remain the source of truth, rows the
  index does not cover yet are still searched exactly, so results are never stale.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app import config

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY,
    path       TEXT UNIQUE NOT NULL,
    rel_path   TEXT NOT NULL,
    name       TEXT NOT NULL,
    ext        TEXT NOT NULL,
    size       INTEGER NOT NULL,
    mtime      REAL    NOT NULL,
    hash       TEXT    NOT NULL,
    n_chunks   INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'ok',
    error      TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,
    file_id  INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord      INTEGER NOT NULL,
    loc      TEXT    NOT NULL DEFAULT '',
    heading  TEXT    NOT NULL DEFAULT '',
    text     TEXT    NOT NULL,
    n_chars  INTEGER NOT NULL,
    vec_row  INTEGER NOT NULL DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id, ord);
CREATE INDEX IF NOT EXISTS idx_chunks_vec  ON chunks(vec_row);

-- External content: the index points back at chunks.text instead of keeping its
-- own copy. A standalone FTS5 table stores the whole corpus a second time, so the
-- text was on disk twice -- measured at 100k chunks of realistic vocabulary, the
-- database went from 336 MB to 206 MB by not doing that, and to 165 MB with
-- detail=none. At a hundred million chunks that is the difference between 328 GB
-- and 161 GB.
--
-- detail=none drops the per-token positions, which costs phrase and NEAR queries.
-- Nothing here issues one: _fts_query builds an OR of single terms and ranking is
-- bm25(), both of which work unchanged.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content = chunks,
    content_rowid = id,
    detail = none,
    tokenize = "unicode61 remove_diacritics 2"
);

-- Scoping a search to chosen files looks rel_path up by value, which was a table
-- scan: 62ms at 200k files, and linear from there.
CREATE INDEX IF NOT EXISTS idx_files_rel ON files(rel_path);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

-- File names, tokenised, so "where is the kirloskar rams study" becomes an index
-- lookup instead of a scan. Names are short, so this one keeps its own content --
-- external content would save nothing and cost a join on every lookup.
-- Underscores, hyphens and dots SEPARATE words here rather than joining them.
-- Keeping them as token characters made "employee_handbook.docx" a single token, so
-- searching for "handbook" found nothing and every lookup fell through to the scan.
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    name,
    rel_path,
    tokenize = "unicode61 remove_diacritics 2 separators '_-.'"
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_FTS_SAFE = re.compile(r"[^\w\s]+", re.UNICODE)


class Store:
    def __init__(self, db_path: Path | None = None, vec_path: Path | None = None) -> None:
        self.db_path = Path(db_path or config.DB_PATH)
        self.vec_path = Path(vec_path or config.VEC_PATH)
        self.rowid_path = self.vec_path.with_suffix(".rowids")
        self.ann_path = self.vec_path.with_name(self.vec_path.stem + "_ann")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self._fts_external = self._detect_fts()
        self._vecs: np.ndarray | None = None      # memmap (n_rows, dim)
        self._row2id: np.ndarray | None = None    # row -> chunk id, -1 = tombstone
        self._dirty = True
        self._ann = None                          # ann.IVFPQIndex, loaded on demand
        self._ann_checked = 0.0

    def _detect_fts(self) -> bool:
        """Is the search index external-content, or the older self-contained kind?

        A database written before the change still works and must keep working --
        only the delete path differs -- so the shape is read from the schema rather
        than assumed. Rebuilding is how the space gets reclaimed, and that is the
        user's call, not something to do to them on startup.
        """
        row = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE name='chunks_fts'").fetchone()
        return bool(row and "content" in (row["sql"] or "").lower()
                    and "content_rowid" in (row["sql"] or "").lower())

    # ------------------------------------------------------------------ sql helpers
    # One connection is shared by the ingest worker pool and the web server, so every
    # statement goes through this lock -- sqlite3 raises SQLITE_MISUSE otherwise.
    def _q(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def _q1(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._q1("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            self.db.commit()

    @property
    def dim(self) -> int:
        return int(self.get_meta("dim", 0) or 0)

    # ------------------------------------------------------------------ files
    def file_fingerprint(self, path: str) -> sqlite3.Row | None:
        return self._q1("SELECT id, size, mtime, hash, status FROM files WHERE path=?", (path,))

    def is_indexed(self, path: str) -> bool:
        """Is this exact file in the index? Gates what /api/open will serve."""
        return self._q1("SELECT 1 FROM files WHERE path=?", (path,)) is not None

    def all_paths(self) -> set[str]:
        return {r["path"] for r in self._q("SELECT path FROM files")}

    def touch_file(self, file_id: int, mtime: float) -> None:
        with self._lock:
            self.db.execute("UPDATE files SET mtime=? WHERE id=?", (mtime, file_id))
            self.db.commit()

    def delete_file(self, path: str) -> None:
        """Remove a file and every trace of its chunks (vector rows become tombstones)."""
        with self._lock:
            row = self.db.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
            if not row:
                return
            self._purge_chunks(row["id"])
            self.db.execute("DELETE FROM files_fts WHERE rowid=?", (row["id"],))
            self.db.execute("DELETE FROM files WHERE id=?", (row["id"],))
            self.db.commit()
        self._dirty = True

    def _purge_chunks(self, file_id: int) -> None:
        rows = self.db.execute(
            "SELECT id, vec_row, text FROM chunks WHERE file_id=?",
            (file_id,)).fetchall()
        if rows:
            self._fts_delete(rows)
            self.db.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            self._tombstone_rows([r["vec_row"] for r in rows])
        self._dirty = True

    def _fts_delete(self, rows) -> None:
        """Remove rows from the search index, whichever shape it has.

        An external-content table keeps no copy of the text, so it cannot work out
        what to un-index on its own -- the original text has to be handed back to
        it, and it has to happen before the row leaves ``chunks``.
        """
        if self._fts_external:
            self.db.executemany(
                "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
                "VALUES('delete', ?, ?)",
                [(r["id"], r["text"]) for r in rows])
        else:
            self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?",
                                [(r["id"],) for r in rows])

    def _index_name(self, file_id: int, name: str, rel_path: str) -> None:
        """Keep the file-name index in step with one row.

        Done explicitly rather than with triggers: a trigger fires once per row inside
        the bulk insert that indexing does, and cannot be batched or deferred.
        """
        try:
            self.db.execute("DELETE FROM files_fts WHERE rowid=?", (file_id,))
            self.db.execute(
                "INSERT INTO files_fts(rowid, name, rel_path) VALUES(?,?,?)",
                (file_id, name or "", rel_path or ""))
        except sqlite3.Error as exc:
            print(f"[store] name index not updated ({exc})", flush=True)

    def rebuild_name_index(self) -> int:
        """Fill the name index from the files table.

        Runs once for a database created before the index existed, and is cheap
        enough afterwards to be worth calling rather than reasoning about.
        """
        with self._lock:
            try:
                have = self._q1("SELECT COUNT(*) n FROM files_fts")["n"] or 0
                want = self._q1("SELECT COUNT(*) n FROM files")["n"] or 0
                if have == want:
                    return 0
                self.db.execute("DELETE FROM files_fts")
                self.db.execute(
                    "INSERT INTO files_fts(rowid, name, rel_path) "
                    "SELECT id, name, rel_path FROM files")
                self.db.commit()
                return want
            except sqlite3.Error as exc:
                print(f"[store] could not build the name index ({exc})", flush=True)
                return 0

    def record_failure(self, path: str, rel_path: str, size: int, mtime: float, error: str) -> None:
        with self._lock:
            self.db.execute(
                """INSERT INTO files(path, rel_path, name, ext, size, mtime, hash, n_chunks,
                                     indexed_at, status, error)
                   VALUES(?,?,?,?,?,?,'',0,?,'error',?)
                   ON CONFLICT(path) DO UPDATE SET
                       size=excluded.size, mtime=excluded.mtime,
                       indexed_at=excluded.indexed_at, status='error',
                       error=excluded.error, n_chunks=0""",
                (path, rel_path, Path(path).name, Path(path).suffix.lower(),
                 size, mtime, time.time(), error[:500]),
            )
            row = self.db.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
            if row:
                self._index_name(row["id"], Path(path).name, rel_path)
            self.db.commit()

    # ------------------------------------------------------------------ writes
    def upsert_file_chunks(
        self,
        *,
        path: str,
        rel_path: str,
        size: int,
        mtime: float,
        file_hash: str,
        chunks: Sequence[dict],
        vectors: np.ndarray | None,
    ) -> int:
        """Replace all chunks of one file atomically. vectors is (len(chunks), dim)."""
        with self._lock:
            cur = self.db.cursor()
            row = cur.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
            if row:
                self._purge_chunks(row["id"])
                file_id = row["id"]
                cur.execute(
                    """UPDATE files SET rel_path=?, name=?, ext=?, size=?, mtime=?, hash=?,
                              n_chunks=?, indexed_at=?, status='ok', error=NULL WHERE id=?""",
                    (rel_path, Path(path).name, Path(path).suffix.lower(), size, mtime,
                     file_hash, len(chunks), time.time(), file_id),
                )
                self._index_name(file_id, Path(path).name, rel_path)
            else:
                cur.execute(
                    """INSERT INTO files(path, rel_path, name, ext, size, mtime, hash,
                                         n_chunks, indexed_at, status)
                       VALUES(?,?,?,?,?,?,?,?,?,'ok')""",
                    (path, rel_path, Path(path).name, Path(path).suffix.lower(), size,
                     mtime, file_hash, len(chunks), time.time()),
                )
                file_id = cur.lastrowid
                self._index_name(file_id, Path(path).name, rel_path)

            base_row = self._append_vectors(vectors) if (vectors is not None and len(chunks)) else -1
            new_ids: list[int] = []
            for i, ch in enumerate(chunks):
                vec_row = base_row + i if base_row >= 0 else -1
                cur.execute(
                    """INSERT INTO chunks(file_id, ord, loc, heading, text, n_chars, vec_row)
                       VALUES(?,?,?,?,?,?,?)""",
                    (file_id, i, ch.get("loc", ""), ch.get("heading", ""),
                     ch["text"], len(ch["text"]), vec_row),
                )
                new_ids.append(int(cur.lastrowid))
                cur.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?,?)",
                            (cur.lastrowid, ch["text"]))
            if base_row >= 0:
                self._append_rowids(new_ids)
            self.db.commit()
        self._dirty = True
        return file_id

    # ------------------------------------------------------------------ row -> id
    def _append_rowids(self, chunk_ids: Sequence[int]) -> None:
        """Keep the sidecar exactly as long as the vector file, in the same order."""
        if not chunk_ids:
            return
        with open(self.rowid_path, "ab") as fh:
            fh.write(np.asarray(chunk_ids, dtype=np.int64).tobytes())

    def _tombstone_rows(self, rows: Sequence[int]) -> None:
        """Mark vector rows as dead in the sidecar, without moving anything."""
        rows = [int(r) for r in rows if r is not None and int(r) >= 0]
        if not rows or not self.rowid_path.exists():
            return
        total = self.rowid_path.stat().st_size // 8
        rows = [r for r in rows if r < total]
        if not rows:
            return
        mm = np.memmap(self.rowid_path, dtype=np.int64, mode="r+", shape=(total,))
        mm[np.asarray(rows, dtype=np.int64)] = -1
        mm.flush()
        del mm

    def _rebuild_rowids(self, n_rows: int) -> np.ndarray:
        """Recreate the sidecar from SQL. Runs once, for an index written before it."""
        row2id = np.full(n_rows, -1, dtype=np.int64)
        cur = self.db.execute("SELECT id, vec_row FROM chunks WHERE vec_row >= 0")
        while True:
            batch = cur.fetchmany(100_000)
            if not batch:
                break
            arr = np.asarray(batch, dtype=np.int64)
            keep = arr[:, 1] < n_rows
            row2id[arr[keep, 1]] = arr[keep, 0]
        row2id.tofile(self.rowid_path)
        return row2id

    def _append_vectors(self, vectors: np.ndarray) -> int:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        dim = int(vectors.shape[1])
        known = self.dim
        if known and known != dim:
            raise ValueError(f"embedding dim changed ({known} -> {dim}); run a full rebuild")
        if not known:
            self.set_meta("dim", dim)
        size = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        base_row = size // (4 * dim)
        with open(self.vec_path, "ab") as fh:
            fh.write(vectors.tobytes())
        self._vecs = None
        return base_row

    # ------------------------------------------------------------------ vectors
    def _load_vectors(self) -> None:
        dim = self.dim
        if not dim or not self.vec_path.exists() or self.vec_path.stat().st_size < 4 * dim:
            self._vecs, self._row2id = None, None
            self._dirty = False
            return
        n_rows = self.vec_path.stat().st_size // (4 * dim)
        self._vecs = np.memmap(self.vec_path, dtype=np.float32, mode="r", shape=(n_rows, dim))
        # Memory-mapped, not rebuilt: the sidecar is already the answer.
        have = self.rowid_path.stat().st_size // 8 if self.rowid_path.exists() else -1
        if have != n_rows:
            print(f"[store] rebuilding the row->chunk sidecar ({n_rows:,} rows)",
                  flush=True)
            self._row2id = self._rebuild_rowids(n_rows)
        else:
            self._row2id = np.memmap(self.rowid_path, dtype=np.int64, mode="r",
                                     shape=(n_rows,))
        self._dirty = False

    # ------------------------------------------------------------------ compaction
    def vector_usage(self) -> tuple[int, int]:
        """(rows in the file, rows still referenced by a chunk)."""
        dim = self.dim
        if not dim or not self.vec_path.exists():
            return 0, 0
        rows = self.vec_path.stat().st_size // (4 * dim)
        live = self._q1("SELECT COUNT(*) n FROM chunks WHERE vec_row >= 0")
        return int(rows), int((live["n"] if live else 0) or 0)

    def compact_vectors(self) -> dict:
        """Rewrite the vector file with the dead rows removed.

        Re-indexing a file appends its new vectors and abandons the old ones, which
        keeps writes O(1) but means the file only ever grows: re-index the same
        folder ten times and nine tenths of it is rows nothing points at any more.
        They are still read on every single search, because the similarity pass
        multiplies the whole matrix.

        The rewrite goes to a temporary file and is swapped in at the end, so an
        interruption leaves the original untouched rather than a half-written index.
        """
        with self._lock:
            dim = self.dim
            rows, live = self.vector_usage()
            if not dim or rows <= 0 or live >= rows:
                return {"compacted": False, "rows": rows, "live": live, "freed": 0}

            keep = [(int(r["id"]), int(r["vec_row"])) for r in self.db.execute(
                "SELECT id, vec_row FROM chunks WHERE vec_row >= 0 ORDER BY vec_row")]
            keep = [(cid, vrow) for cid, vrow in keep if vrow < rows]

            source = np.memmap(self.vec_path, dtype=np.float32, mode="r",
                               shape=(rows, dim))
            tmp = self.vec_path.with_suffix(self.vec_path.suffix + ".compact")
            try:
                with open(tmp, "wb") as fh:
                    for i in range(0, len(keep), 4096):
                        block = [vrow for _cid, vrow in keep[i:i + 4096]]
                        fh.write(np.ascontiguousarray(source[block]).tobytes())
                del source
                self._vecs, self._row2id = None, None      # drop the mapping first
                self.vec_path.unlink()
                tmp.replace(self.vec_path)
            except BaseException:
                self._vecs, self._row2id = None, None
                tmp.unlink(missing_ok=True)
                raise

            self.db.executemany("UPDATE chunks SET vec_row=? WHERE id=?",
                                [(new_row, cid) for new_row, (cid, _old)
                                 in enumerate(keep)])
            self.db.commit()
            self._dirty = True
            return {"compacted": True, "rows": rows, "live": len(keep),
                    "freed": (rows - len(keep)) * 4 * dim}

    def file_ids_for(self, rel_paths: Iterable[str]) -> list[int]:
        """File ids for a list of rel_paths, for scoping a search to chosen files."""
        wanted = [p for p in dict.fromkeys(rel_paths) if p]
        if not wanted:
            return []
        out: list[int] = []
        for i in range(0, len(wanted), 400):
            batch = wanted[i:i + 400]
            marks = ",".join("?" * len(batch))
            out += [int(r["id"]) for r in self._q(
                f"SELECT id FROM files WHERE rel_path IN ({marks}) "
                f"OR path IN ({marks})", batch + batch)]
        return sorted(set(out))

    def chunk_ids_for_files(self, file_ids: Sequence[int]) -> np.ndarray:
        if not file_ids:
            return np.zeros(0, dtype=np.int64)
        marks = ",".join("?" * len(file_ids))
        rows = self._q(f"SELECT id FROM chunks WHERE file_id IN ({marks})",
                       list(file_ids))
        return np.asarray([int(r["id"]) for r in rows], dtype=np.int64)

    # ------------------------------------------------------------------ ann
    def ann(self):
        """The IVF-PQ index, if one has been built. Cheap: it only maps files."""
        from app.core import ann as ann_mod

        if self._ann is not None:
            return self._ann
        idx = ann_mod.IVFPQIndex(self.ann_path)
        if not idx.exists():
            return None
        try:
            self._ann = idx.load()
        except Exception as exc:                              # noqa: BLE001
            print(f"[store] ANN index unreadable, using exact search ({exc})",
                  flush=True)
            self._ann = None
        return self._ann

    def drop_ann(self) -> None:
        import shutil

        self._ann = None
        if self.ann_path.exists():
            shutil.rmtree(self.ann_path, ignore_errors=True)

    def ann_stats(self) -> dict:
        from app.core import ann as ann_mod

        idx = self._ann or ann_mod.IVFPQIndex(self.ann_path)
        out = idx.stats()
        rows, live = self.vector_usage()
        out["rows_total"] = rows
        out["rows_live"] = live
        out["rows_uncovered"] = max(0, rows - int(out.get("covers_rows", 0)))
        return out

    def build_ann(self, on_progress=None) -> dict:
        """(Re)build the approximate index over every live vector row.

        The float32 vectors stay exactly as they are -- this is a derived artifact,
        so a failed or interrupted build costs nothing but time, and deleting the
        directory falls back to exact search.
        """
        from app.core import ann as ann_mod

        with self._lock:
            if self._dirty or self._vecs is None:
                self._load_vectors()
            vecs, row2id = self._vecs, self._row2id
            dim = self.dim
        if vecs is None or row2id is None or not len(vecs) or not dim:
            return {"built": False, "reason": "no vectors"}
        live = int((np.asarray(row2id) >= 0).sum())
        if live < config.ANN_MIN_VECTORS:
            return {"built": False, "reason": f"only {live:,} vectors; "
                                              f"exact search is faster below "
                                              f"{config.ANN_MIN_VECTORS:,}"}
        n_rows = len(vecs)

        def blocks():
            step = ann_mod.BLOCK
            for start in range(0, n_rows, step):
                stop = min(start + step, n_rows)
                ids = np.asarray(row2id[start:stop])
                keep = ids >= 0                       # skip tombstones
                if not keep.any():
                    continue
                rows = np.arange(start, stop, dtype=np.int64)[keep]
                yield np.asarray(vecs[start:stop])[keep], ids[keep], rows

        stats = ann_mod.IVFPQIndex.build(
            self.ann_path, blocks, dim=dim, n_vectors=live, covers_rows=n_rows,
            n_cells=config.ANN_CELLS or 0, on_progress=on_progress)
        self._ann = None                              # reopen against the new files
        return {"built": True, **stats.__dict__}

    def _exact_rows(self, rows: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Exact inner products for a handful of rows, read straight from the file."""
        if self._vecs is None:
            return np.zeros(len(rows), dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return np.asarray(self._vecs[np.asarray(rows, dtype=np.int64)]) @ q

    def search_dense(self, query_vec: np.ndarray, k: int,
                     file_ids: Sequence[int] | None = None) -> list[tuple[int, float]]:
        out = self.search_dense_many(np.asarray(query_vec).reshape(1, -1), k, file_ids)
        return out[0] if out else []

    def search_dense_many(self, query_vecs: np.ndarray, k: int,
                          file_ids: Sequence[int] | None = None,
                          ) -> list[list[tuple[int, float]]]:
        """Rank every query in one pass.

        Query expansion searches the same index three or four times per question.
        Done one at a time that is three or four full reads of the vector memmap;
        stacked into a single (n_rows x dim) @ (dim x n_queries) product it is one,
        and the row mask and the file scope are computed once instead of per query.
        """
        queries = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
        if not len(queries):
            return []
        with self._lock:
            if self._dirty or self._vecs is None:
                self._load_vectors()
            vecs, row2id = self._vecs, self._row2id
        if vecs is None or row2id is None or not len(vecs):
            return [[] for _ in queries]

        index = self.ann() if config.ANN_ENABLED else None
        if index is not None and index.covers_rows > 0:
            return self._search_ann(index, queries, k, file_ids, vecs, row2id)

        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-12)
        allowed = row2id >= 0
        if file_ids:
            # Scoped before the top-k is taken, not after: filtering afterwards
            # would return nothing whenever the chosen file is not already winning.
            keep = self.chunk_ids_for_files(file_ids)
            allowed &= np.isin(row2id, keep)
        n_allowed = int(allowed.sum())
        if n_allowed <= 0:
            return [[] for _ in queries]
        k = min(k, n_allowed)

        sims = vecs @ queries.T               # cosine: vectors are pre-normalised
        sims = np.where(allowed[:, None], sims, -np.inf)
        results: list[list[tuple[int, float]]] = []
        for col in range(sims.shape[1]):
            column = sims[:, col]
            top = np.argpartition(-column, k - 1)[:k]
            top = top[np.argsort(-column[top])]
            results.append([(int(row2id[r]), float(column[r]))
                            for r in top if row2id[r] >= 0 and np.isfinite(column[r])])
        return results

    @staticmethod
    def nprobe_for(index) -> int:
        """Cells to open: the configured floor, or a share of the index if larger.

        Accuracy follows the fraction of the corpus a query examines. Cells grow
        with the corpus, so a fixed count examines an ever smaller share of it and
        the answers get quietly worse as the index grows -- which is the one failure
        mode nobody notices.
        """
        cells = int(index.meta.get("n_cells", 1))
        want = config.ANN_NPROBE
        if config.ANN_SCAN_FRACTION > 0:
            want = max(want, int(round(cells * config.ANN_SCAN_FRACTION)))
        return max(1, min(want, cells))

    def _search_ann(self, index, queries: np.ndarray, k: int,
                    file_ids, vecs, row2id) -> list[list[tuple[int, float]]]:
        """Approximate over the indexed rows, exact over whatever arrived after it.

        An index run appends vectors; rebuilding the ANN index after every one would
        be wasteful, so rows past ``covers_rows`` are simply scanned exactly and
        merged in. The tail is small by construction, so this costs little -- and it
        means a document is searchable the moment it is indexed, never pending.
        """
        allowed_ids = None
        if file_ids:
            allowed_ids = np.sort(self.chunk_ids_for_files(file_ids))
            if not len(allowed_ids):
                return [[] for _ in queries]

        results = index.search(
            queries, k, nprobe=self.nprobe_for(index), allowed=allowed_ids,
            rescore=self._exact_rows,
            shortlist=max(k * 8, config.ANN_SHORTLIST))

        tail_start = index.covers_rows
        if tail_start < len(vecs):
            tail_ids = np.asarray(row2id[tail_start:])
            live = tail_ids >= 0
            if allowed_ids is not None:
                live &= np.isin(tail_ids, allowed_ids)
            if live.any():
                tail = np.asarray(vecs[tail_start:])[live]
                ids = tail_ids[live]
                q = queries / np.maximum(
                    np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
                sims = tail @ q.T                     # (n_tail, n_queries)
                for qi in range(len(queries)):
                    take = min(k, len(ids))
                    top = np.argpartition(-sims[:, qi], take - 1)[:take]
                    merged = results[qi] + [(int(ids[r]), float(sims[r, qi]))
                                            for r in top]
                    merged.sort(key=lambda pair: -pair[1])
                    results[qi] = merged[:k]
        return results

    # ------------------------------------------------------------------ bm25
    @staticmethod
    def _fts_query(text: str) -> str:
        """Turn free text into a safe FTS5 OR-query (operators stripped, terms quoted)."""
        terms = [t for t in _FTS_SAFE.sub(" ", text).split() if len(t) > 1][:32]
        return " OR ".join('"' + t + '"' for t in terms)

    def search_bm25(self, query: str, k: int,
                    file_ids: Sequence[int] | None = None) -> list[tuple[int, float]]:
        expr = self._fts_query(query)
        if not expr:
            return []
        try:
            if file_ids:
                marks = ",".join("?" * len(file_ids))
                rows = self._q(
                    "SELECT f.rowid AS rowid, bm25(chunks_fts) AS score "
                    "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                    f"WHERE chunks_fts MATCH ? AND c.file_id IN ({marks}) "
                    "ORDER BY score LIMIT ?",
                    (expr, *file_ids, k),
                )
            else:
                rows = self._q(
                    "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (expr, k),
                )
        except sqlite3.OperationalError:
            return []
        # SQLite bm25() returns "more negative == better"; flip it so bigger == better.
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    # ------------------------------------------------------------------ reads
    def get_chunks(self, ids: Iterable[int]) -> dict[int, dict]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        if not ids:
            return {}
        out: dict[int, dict] = {}
        for i in range(0, len(ids), 400):
            batch = ids[i:i + 400]
            marks = ",".join("?" * len(batch))
            for r in self._q(
                f"""SELECT c.id, c.file_id, c.ord, c.loc, c.heading, c.text,
                           f.name, f.rel_path, f.path, f.ext
                      FROM chunks c JOIN files f ON f.id = c.file_id
                     WHERE c.id IN ({marks})""",
                batch,
            ):
                out[int(r["id"])] = dict(r)
        return out

    def neighbors(self, file_id: int, ord_: int, window: int) -> list[dict]:
        if window <= 0:
            return []
        rows = self._q(
            """SELECT c.id, c.file_id, c.ord, c.loc, c.heading, c.text,
                      f.name, f.rel_path, f.path, f.ext
                 FROM chunks c JOIN files f ON f.id = c.file_id
                WHERE c.file_id=? AND c.ord BETWEEN ? AND ? ORDER BY c.ord""",
            (file_id, ord_ - window, ord_ + window),
        )
        return [dict(r) for r in rows]

    def _counts(self) -> dict:
        """The expensive part of stats(): three full passes over the big tables.

        Cached because the page polls health every fifteen seconds and these numbers
        only move when an index run finishes. Measured at 200k files and 2M chunks
        the uncached version took 173ms a poll; at ten million files it would be
        seconds, continuously, for a number that had not changed.
        """
        now = time.time()
        stamp = self.get_meta("last_index", "")
        cached = getattr(self, "_counts_cache", None)
        if cached and cached[0] == stamp and now - cached[1] < config.STATS_TTL:
            return cached[2]
        f = self._q1(
            "SELECT COUNT(*) n, COALESCE(SUM(size),0) b, "
            "COALESCE(SUM(status='error'),0) e FROM files"
        )
        c = self._q1("SELECT COUNT(*) n FROM chunks")
        by_ext = [dict(r) for r in self._q(
            "SELECT ext, COUNT(*) n FROM files WHERE status='ok' "
            "GROUP BY ext ORDER BY n DESC LIMIT 20")]
        out = {"files": f["n"] or 0, "failed": f["e"] or 0, "bytes": f["b"] or 0,
               "chunks": c["n"] or 0, "by_ext": by_ext}
        self._counts_cache = (stamp, now, out)
        return out

    def stats(self) -> dict:
        counts = self._counts()
        return {
            **counts,
            "dim": self.dim,
            "last_index": self.get_meta("last_index", ""),
            "embed_model": self.get_meta("embed_model", ""),
            "fts_external": self._fts_external,
        }

    def _stats_uncached(self) -> dict:
        f = self._q1(
            "SELECT COUNT(*) n, COALESCE(SUM(size),0) b, "
            "COALESCE(SUM(status='error'),0) e FROM files"
        )
        c = self._q1("SELECT COUNT(*) n FROM chunks")
        by_ext = [dict(r) for r in self._q(
            "SELECT ext, COUNT(*) n FROM files WHERE status='ok' "
            "GROUP BY ext ORDER BY n DESC LIMIT 20")]
        return {
            "files": f["n"] or 0,
            "failed": f["e"] or 0,
            "bytes": f["b"] or 0,
            "chunks": c["n"] or 0,
            "dim": self.dim,
            "by_ext": by_ext,
            "last_index": self.get_meta("last_index", ""),
            "embed_model": self.get_meta("embed_model", ""),
            "fts_external": self._fts_external,
        }

    def search_files_by_name(self, terms: Sequence[str], limit: int = 300) -> list[dict]:
        """Files whose name or folder matches any of these words.

        An index lookup, so the cost follows the number of matches rather than the
        size of the corpus. Falls back to a scan only when the FTS table is missing,
        which happens on a database created before it existed.
        """
        words = [_FTS_SAFE.sub(" ", str(t)).strip() for t in terms if str(t).strip()]
        words = [w for w in words if len(w) >= 2]
        if not words:
            return []
        picked = words[:8]
        sql = ("SELECT f.path, f.rel_path, f.name, f.ext, f.size, f.n_chunks "
               "FROM files_fts x JOIN files f ON f.rowid = x.rowid "
               "WHERE files_fts MATCH ? AND f.status = 'ok' "
               "ORDER BY bm25(files_fts) LIMIT ?")
        try:
            # Every word first. A common word makes the OR form match a large share
            # of the corpus, and ranking that many rows costs as much as the scan it
            # replaced -- measured at 800ms where every file shared a token. AND is
            # selective enough that the ranking stays small, and the OR below still
            # catches the case where one word was simply wrong.
            rows = []
            if len(picked) > 1:
                rows = self._q(sql, (" AND ".join(f'"{w}"*' for w in picked), limit))
            if not rows:
                rows = self._q(sql, (" OR ".join(f'"{w}"*' for w in picked), limit))
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            print(f"[store] file-name search unavailable ({exc}); scanning", flush=True)
            return self.files_for_lookup(limit=limit)

    def list_files(self, limit: int = 500, query: str = "") -> list[dict]:
        if query:
            rows = self._q(
                """SELECT rel_path, ext, size, n_chunks, status, error, indexed_at FROM files
                    WHERE rel_path LIKE ? ORDER BY rel_path LIMIT ?""",
                (f"%{query}%", limit),
            )
        else:
            rows = self._q(
                """SELECT rel_path, ext, size, n_chunks, status, error, indexed_at FROM files
                    ORDER BY indexed_at DESC LIMIT ?""",
                (limit,),
            )
        return [dict(r) for r in rows]

    def files_for_lookup(self, limit: int = 50000) -> list[dict]:
        """Every indexed file with the columns needed to locate it on disk.

        Separate from list_files because that one feeds the Manage screen and
        deliberately does not hand out absolute paths; resolving "where is it" needs
        exactly those, plus the bare name to match against.
        """
        rows = self._q(
            """SELECT path, rel_path, name, ext, size, n_chunks, status FROM files
                WHERE status = 'ok' ORDER BY rel_path LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in rows]

    def reset(self) -> None:
        """Drop the whole index (used by 'rebuild from scratch')."""
        with self._lock:
            # 'delete-all' is the only way to empty an external-content index;
            # a plain DELETE has no text to work from.
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')"
                            if self._fts_external else "DELETE FROM chunks_fts")
            self.db.executescript(
                "DELETE FROM chunks; DELETE FROM files; DELETE FROM meta; "
                "DELETE FROM files_fts;")
            self.db.commit()
            self.db.execute("VACUUM")
            self._vecs, self._row2id = None, None
        for path in (self.vec_path, self.rowid_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    with open(path, "wb"):
                        pass
        self.drop_ann()
        self._dirty = True


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
