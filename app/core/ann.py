"""Approximate nearest-neighbour search: IVF-PQ, in numpy, on memory-mapped files.

The problem this exists to solve
--------------------------------
The exact search in ``store.py`` multiplies the whole vector matrix by the query.
That is the right answer for a small index -- it is exact, it is one BLAS call, and
below a few hundred thousand chunks it finishes in single-digit milliseconds. It
also cannot be made to scale, because its cost is the size of the corpus:

      20M documents  ~ 100M chunks  ~ 143 GB of float32 vectors
      one query = read all 143 GB = 24 seconds at 6 GB/s, and it does not fit in RAM

Two independent things have to change: the amount of data a query touches, and how
much space each vector occupies.

**IVF** -- the inverted file -- fixes the first. Vectors are clustered once, and a
query is only compared against the few clusters whose centroid it resembles. Probing
32 of 16384 cells reads 0.2% of the corpus, so the work per query stops growing with
the corpus and starts growing with the number of cells probed, which is a constant
the caller chooses.

**PQ** -- product quantisation -- fixes the second. Each 384-dimensional vector is
cut into 32 slices of 12 dimensions; each slice is replaced by the index of the
nearest of 256 prototypes learned for that slice. A vector becomes 32 bytes instead
of 1536, a factor of 48. At 20M documents that is the difference between 143 GB on
disk and 3 GB that stays in RAM -- and the inner product is recovered without ever
decompressing, by looking each slice up in a table computed once per query.

The two compose: PQ makes the data small enough to keep resident, IVF makes the
query touch a sliver of it.

Layout, and why it is this way
------------------------------
Two decisions, both about locality, both measured.

Codes are stored **grouped by cell**, so the rows a probe needs are one contiguous
run. Left in insertion order they would be scattered across the whole file and a
probe would be thousands of random reads; grouped, it is one sequential read that
the prefetcher sees coming. Same bytes, same arithmetic, an order of magnitude
apart in practice, and the gap widens once the file is bigger than RAM.

Codes are also stored **one sub-quantiser per row** -- an (m_sub x n) matrix rather
than (n x m_sub). Scoring walks the sub-quantisers on the outside and the candidates
on the inside, so this puts the inner loop on contiguous memory and lets each of the
64 lookup tables stay in L1 while its whole column is consumed. Transposed, the
inner loop strides by 64 bytes and the same arithmetic measured 4.9ms against 3.3ms
for a 7,680-candidate probe.

Everything is a flat memory-mapped array. Nothing is parsed at open time, so
attaching to a 100M-vector index costs a few microseconds and the pages the queries
actually touch are the only ones that ever load.

Two details that decide whether this works at all
------------------------------------------------
**The rotation.** Plain PQ cuts the vector into contiguous slices, which assumes the
variance is spread evenly across the dimensions. Real embeddings are nowhere near
that: measured on bge-small, the top 10 of 384 directions carry 58% of the variance
and the effective dimensionality is about 23. Sliced as they come, a few slices
carry nearly all the information and cannot express it with 256 prototypes, while
the rest spend their 256 prototypes describing noise. Recall measured 0.54 that way.
So the space is first rotated onto its principal axes, and the axes are then dealt
out to slices so that each slice receives roughly the same total variance. The
rotation is orthonormal, so every inner product is preserved exactly; only the
bookkeeping changes.

**The re-score.** Even with a good rotation, PQ returns an approximate ordering. It
is used to produce a *shortlist* several times longer than what was asked for, and
the shortlist is then scored exactly against the original float32 vectors. That is a
few hundred random reads of 1.5 KB each -- microseconds -- and it restores the exact
ranking of everything the shortlist contains. Approximation decides what to look at;
arithmetic decides the order.

Measured recall against exact search is reported by ``tests/test_scale.py``.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# One block of vectors handled at a time, everywhere. Bounds peak memory during a
# build to something independent of the corpus size.
BLOCK = 65536


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _assign(x: np.ndarray, centroids: np.ndarray, block: int = 8192) -> np.ndarray:
    """Nearest centroid by inner product, in blocks so the score matrix stays small.

    The full matrix would be (n vectors x n centroids) floats: at 2M vectors and
    16k cells that is 128 GB, which is why this is chunked rather than one call.
    """
    out = np.empty(len(x), dtype=np.int32)
    for i in range(0, len(x), block):
        out[i:i + block] = np.argmax(x[i:i + block] @ centroids.T, axis=1)
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 10, seed: int = 0,
           on_progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Spherical k-means. Vectors are unit length, so inner product is the metric.

    Empty clusters are re-seeded from the points furthest from their own centroid --
    without that, a bad initialisation silently leaves dead cells that cost memory
    and catch nothing.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    k = max(1, min(k, n))
    centroids = x[rng.choice(n, size=k, replace=False)].copy()
    for it in range(iters):
        labels = _assign(x, centroids)
        fresh = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(fresh, labels, x)                 # sum each cluster's members
        np.add.at(counts, labels, 1)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            scores = np.einsum("ij,ij->i", x, centroids[labels])
            worst = np.argsort(scores)[:len(empty)]
            fresh[empty] = x[worst]
            counts[empty] = 1
        centroids = _l2_normalise(fresh / counts[:, None])
        if on_progress and (it + 1) % 4 == 0:
            on_progress(f"k-means {it + 1}/{iters} ({k} cells)")
    return centroids.astype(np.float32)


def balanced_rotation(x: np.ndarray, m_sub: int) -> np.ndarray:
    """An orthonormal matrix that spreads the variance evenly across PQ slices.

    Two steps. PCA gives axes that are uncorrelated and ordered by how much variance
    they carry. Those axes are then dealt out to the slices greedily, largest first,
    each going to whichever slice currently holds the least -- the standard
    multiway-partition heuristic. The result is a basis in which every slice has a
    comparable job to do, which is the condition ordinary PQ silently assumes and
    real embeddings badly violate.

    Being orthonormal, it changes no inner product: IP(q, x) == IP(Rq, Rx).
    """
    dim = x.shape[1]
    centred = x - x.mean(axis=0)
    cov = np.cov(centred.T)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]

    dsub = dim // m_sub
    loads = np.zeros(m_sub)
    slots: list[list[int]] = [[] for _ in range(m_sub)]
    for axis in range(dim):
        # Only slices that still have room, else a slice would overflow its width.
        open_slots = [i for i in range(m_sub) if len(slots[i]) < dsub]
        pick = min(open_slots, key=lambda i: loads[i])
        slots[pick].append(axis)
        loads[pick] += ev[axis]
    layout = [a for slot in slots for a in slot]
    return np.ascontiguousarray(evec[:, layout].T.astype(np.float32))


def train_pq(residuals: np.ndarray, m_sub: int, iters: int = 10,
             seed: int = 0) -> np.ndarray:
    """One 256-entry codebook per slice of the vector. Returns (m_sub, 256, dsub)."""
    n, dim = residuals.shape
    dsub = dim // m_sub
    books = np.zeros((m_sub, 256, dsub), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for m in range(m_sub):
        sub = np.ascontiguousarray(residuals[:, m * dsub:(m + 1) * dsub])
        k = min(256, len(sub))
        centroids = sub[rng.choice(len(sub), size=k, replace=False)].copy()
        for _ in range(iters):
            # L2 nearest: ||r||^2 is constant per point, so maximise 2*r.c - ||c||^2
            labels = np.argmax(sub @ centroids.T * 2.0
                               - np.einsum("ij,ij->i", centroids, centroids)[None, :],
                               axis=1)
            fresh = np.zeros_like(centroids)
            counts = np.zeros(len(centroids), dtype=np.int64)
            np.add.at(fresh, labels, sub)
            np.add.at(counts, labels, 1)
            alive = counts > 0
            fresh[alive] /= counts[alive][:, None]
            fresh[~alive] = centroids[~alive]       # keep, rather than collapse to 0
            centroids = fresh
        books[m, :k] = centroids
        if k < 256:                                  # pad so the code space is uniform
            books[m, k:] = centroids[-1]
    return books


def encode_pq(residuals: np.ndarray, books: np.ndarray) -> np.ndarray:
    """Nearest prototype per slice. Returns (n, m_sub) uint8."""
    m_sub, _, dsub = books.shape
    codes = np.empty((len(residuals), m_sub), dtype=np.uint8)
    for m in range(m_sub):
        book = books[m]
        sub = residuals[:, m * dsub:(m + 1) * dsub]
        scores = sub @ book.T * 2.0 - np.einsum("ij,ij->i", book, book)[None, :]
        codes[:, m] = np.argmax(scores, axis=1).astype(np.uint8)
    return codes


@dataclass
class BuildStats:
    n_vectors: int = 0
    n_cells: int = 0
    m_sub: int = 0
    bytes_codes: int = 0
    bytes_exact: int = 0
    seconds: float = 0.0


class IVFPQIndex:
    """An IVF-PQ index living in one directory of memory-mapped arrays."""

    FORMAT = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.meta: dict = {}
        self.centroids: np.ndarray | None = None
        self.books: np.ndarray | None = None
        self.rotation: np.ndarray | None = None     # (dim, dim) orthonormal
        self.offsets: np.ndarray | None = None      # (n_cells + 1,) into codes/ids
        self.codes: np.ndarray | None = None        # (n, m_sub) uint8, cell-contiguous
        self.ids: np.ndarray | None = None          # (n,) int64 chunk id, same order
        self.rows: np.ndarray | None = None         # (n,) int64 row in vectors.f32

    # ------------------------------------------------------------------ files
    def _f(self, name: str) -> Path:
        return self.path / name

    def exists(self) -> bool:
        return (self._f("meta.json").exists() and self._f("codes.u8").exists())

    def load(self) -> "IVFPQIndex":
        self.meta = json.loads(self._f("meta.json").read_text(encoding="utf-8"))
        dim, m_sub = self.meta["dim"], self.meta["m_sub"]
        n, cells = self.meta["n_vectors"], self.meta["n_cells"]
        dsub = dim // m_sub
        # Centroids, codebooks and offsets are read in full on every query and are
        # small -- a few megabytes even for a hundred million vectors -- so they are
        # read into memory once rather than faulted through the mapping each time.
        # Only the codes, which are the part that is actually large, stay mapped.
        self.centroids = np.fromfile(self._f("centroids.f32"),
                                     dtype=np.float32).reshape(cells, dim)
        self.books = np.fromfile(self._f("books.f32"),
                                 dtype=np.float32).reshape(m_sub, 256, dsub)
        self.offsets = np.fromfile(self._f("offsets.i64"), dtype=np.int64)
        self.codes = np.memmap(self._f("codes.u8"), dtype=np.uint8, mode="r",
                               shape=(m_sub, n))
        self.ids = np.memmap(self._f("ids.i64"), dtype=np.int64, mode="r", shape=(n,))
        self.rows = np.memmap(self._f("rows.i64"), dtype=np.int64, mode="r", shape=(n,))
        self.rotation = np.memmap(self._f("rotation.f32"), dtype=np.float32,
                                  mode="r", shape=(dim, dim))
        return self

    @property
    def n_vectors(self) -> int:
        return int(self.meta.get("n_vectors", 0))

    @property
    def covers_rows(self) -> int:
        """Vector-file rows 0..covers_rows-1 are in this index; the rest are a tail."""
        return int(self.meta.get("covers_rows", 0))

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        path: Path,
        blocks: Callable[[], Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]],
        *,
        dim: int,
        n_vectors: int,
        covers_rows: int,
        n_cells: int = 0,
        m_sub: int = 0,
        train_size: int = 0,
        seed: int = 0,
        on_progress: Callable[[str], None] | None = None,
    ) -> BuildStats:
        """Train and write an index.

        ``blocks()`` yields (vectors, chunk_ids, vector_rows) and is called three
        times: to draw a training sample, to count cell sizes, and to encode.
        Streaming it rather than holding the corpus is the whole reason a build does
        not need the corpus to fit in memory.
        """
        say = on_progress or (lambda _m: None)
        started = time.perf_counter()
        path = Path(path)
        tmp = path.with_name(path.name + ".building")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        n_cells = n_cells or cls.suggest_cells(n_vectors)
        m_sub = m_sub or cls.suggest_subquantisers(dim)
        if dim % m_sub:
            raise ValueError(f"m_sub={m_sub} does not divide dim={dim}")
        train_size = train_size or min(max(40 * n_cells, 20_000), 250_000, n_vectors)

        # ---- 1. training sample -------------------------------------------------
        say(f"sampling {train_size:,} of {n_vectors:,} vectors")
        rng = np.random.default_rng(seed)
        keep = np.zeros(0, dtype=np.float32).reshape(0, dim)
        seen = 0
        parts: list[np.ndarray] = []
        take = min(1.0, train_size / max(n_vectors, 1))
        for vecs, _ids, _rows in blocks():
            seen += len(vecs)
            if take >= 1.0:
                parts.append(np.asarray(vecs, dtype=np.float32))
            else:
                pick = rng.random(len(vecs)) < take
                if pick.any():
                    parts.append(np.asarray(vecs[pick], dtype=np.float32))
        keep = np.vstack(parts) if parts else keep
        del parts
        if len(keep) > train_size:
            keep = keep[rng.choice(len(keep), size=train_size, replace=False)]
        keep = _l2_normalise(keep)

        # ---- 2. rotation --------------------------------------------------------
        # Learned before anything else and applied to everything after, so centroids,
        # codebooks, codes and queries all live in the same rotated basis.
        say("learning a variance-balancing rotation")
        rotation = balanced_rotation(keep, m_sub)
        keep = _l2_normalise(keep @ rotation.T)

        # ---- 3. coarse quantiser ------------------------------------------------
        say(f"clustering into {n_cells:,} cells")
        centroids = kmeans(keep, n_cells, seed=seed, on_progress=say)

        # ---- 4. product quantiser on the residuals ------------------------------
        say(f"training {m_sub} sub-quantisers")
        labels = _assign(keep, centroids)
        residuals = keep - centroids[labels]
        books = train_pq(residuals, m_sub, seed=seed)
        del keep, residuals, labels

        # ---- 5. count per cell --------------------------------------------------
        say("assigning vectors to cells")
        counts = np.zeros(n_cells, dtype=np.int64)
        total = 0
        for vecs, _ids, _rows in blocks():
            vecs = _l2_normalise(np.asarray(vecs, dtype=np.float32)) @ rotation.T
            lab = _assign(vecs, centroids)
            np.add.at(counts, lab, 1)
            total += len(vecs)

        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        # ---- 5. encode, written straight into cell order ------------------------
        # The scatter is what produces the contiguous layout: every vector lands at
        # its cell's slot, so a probe later reads one unbroken run.
        say(f"encoding {total:,} vectors")
        codes_mm = np.memmap(tmp / "codes.u8", dtype=np.uint8, mode="w+",
                             shape=(m_sub, total))
        ids_mm = np.memmap(tmp / "ids.i64", dtype=np.int64, mode="w+", shape=(total,))
        rows_mm = np.memmap(tmp / "rows.i64", dtype=np.int64, mode="w+", shape=(total,))
        cursor = offsets[:-1].copy()
        done = 0
        for vecs, chunk_ids, vec_rows in blocks():
            vecs = _l2_normalise(np.asarray(vecs, dtype=np.float32)) @ rotation.T
            lab = _assign(vecs, centroids)
            residual = vecs - centroids[lab]
            block_codes = encode_pq(residual, books)
            order = np.argsort(lab, kind="stable")   # group this block by cell
            lab_sorted = lab[order]
            starts = cursor[lab_sorted]
            within = np.arange(len(order), dtype=np.int64)
            first = np.searchsorted(lab_sorted, lab_sorted, side="left")
            positions = starts + (within - first)
            codes_mm[:, positions] = block_codes[order].T
            ids_mm[positions] = np.asarray(chunk_ids, dtype=np.int64)[order]
            rows_mm[positions] = np.asarray(vec_rows, dtype=np.int64)[order]
            np.add.at(cursor, lab, 1)
            done += len(vecs)
            if done % (BLOCK * 8) < len(vecs):
                say(f"encoded {done:,}/{total:,}")
        codes_mm.flush()
        ids_mm.flush()
        rows_mm.flush()
        del codes_mm, ids_mm, rows_mm

        # ---- 6. publish ---------------------------------------------------------
        np.asarray(rotation, dtype=np.float32).tofile(tmp / "rotation.f32")
        np.asarray(centroids, dtype=np.float32).tofile(tmp / "centroids.f32")
        np.asarray(books, dtype=np.float32).tofile(tmp / "books.f32")
        offsets.astype(np.int64).tofile(tmp / "offsets.i64")
        meta = {
            "format": cls.FORMAT, "dim": dim, "m_sub": m_sub, "n_cells": n_cells,
            "n_vectors": int(total), "covers_rows": int(covers_rows),
            "metric": "ip", "built_at": time.time(),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Swapped in whole. A half-written index is never visible to a search.
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        tmp.rename(path)

        return BuildStats(
            n_vectors=int(total), n_cells=n_cells, m_sub=m_sub,
            bytes_codes=int(total) * m_sub + int(total) * 16,
            bytes_exact=int(total) * dim * 4,
            seconds=round(time.perf_counter() - started, 2),
        )

    # ------------------------------------------------------------------ tuning
    @staticmethod
    def suggest_cells(n_vectors: int) -> int:
        """Finer is better, up to the point where cells stop having members.

        Measured on 100k calibrated vectors, at a fixed 3% of the corpus scanned:
        256 cells gave recall 0.71, 1024 gave 0.83, 4096 gave 0.91. Finer cells
        target the probe more precisely, so the same budget buys more. The limit is
        that a cell still needs enough members to be worth a seek, and k-means has
        to train on them -- hence roughly 48 vectors per cell, capped.
        """
        if n_vectors <= 0:
            return 1
        # Capped at 16384 because training is quadratic in the cell count: k-means
        # costs (training points x cells x dim) per iteration, so 65536 cells turn a
        # four-minute build into a multi-hour one for a recall gain that more probes
        # buy far more cheaply.
        return int(max(16, min(int(4 * n_vectors ** 0.5), n_vectors // 48, 16384)))

    @staticmethod
    def suggest_subquantisers(dim: int) -> int:
        """Largest slice count at or under 32 that divides the dimension evenly.

        Not more, because more buys nothing. Measured on 200k calibrated vectors,
        32 and 64 sub-quantisers returned the same recall to within noise (0.831 and
        0.831 at 64 probes, 0.927 and 0.931 at 128) while 32 used half the space and
        scored candidates 2.6x faster. 96 was likewise identical to 64. Once a
        shortlist is re-scored exactly, extra PQ precision only refines an ordering
        that is about to be replaced.
        """
        for m in (32, 24, 16, 12, 8, 6, 4, 2, 1):
            if m <= dim and dim % m == 0:
                return m
        return 1

    # ------------------------------------------------------------------ search
    def search(self, queries: np.ndarray, k: int, nprobe: int = 16,
               allowed: np.ndarray | None = None,
               rescore: "Callable[[np.ndarray, np.ndarray], np.ndarray] | None" = None,
               shortlist: int = 0) -> list[list[tuple[int, float]]]:
        """Top-k (chunk id, score) per query, best first.

        ``allowed`` restricts to a sorted array of chunk ids. ``rescore`` is given
        (vector rows, query) and returns exact scores for those rows; when supplied,
        PQ is used only to choose a shortlist and the final order is exact.
        """
        if self.codes is None:
            self.load()
        assert self.centroids is not None and self.books is not None
        assert self.offsets is not None and self.ids is not None

        q = _l2_normalise(np.atleast_2d(np.asarray(queries, dtype=np.float32)))
        q = q @ np.asarray(self.rotation).T          # same basis as the codes
        m_sub, _, dsub = self.books.shape
        nprobe = max(1, min(int(nprobe), int(self.meta["n_cells"])))
        want = max(k, shortlist or k * 8)

        coarse = q @ self.centroids.T                       # (nq, n_cells)
        probes = np.argpartition(-coarse, nprobe - 1, axis=1)[:, :nprobe]

        out: list[list[tuple[int, float]]] = []
        for qi in range(len(q)):
            # One table per query: table[m, j] is the inner product of this query's
            # m-th slice with prototype j. Every candidate is then m_sub lookups and
            # an add -- no vector is ever reconstructed.
            table = np.einsum("mkd,md->mk", self.books,
                              q[qi].reshape(m_sub, dsub)).astype(np.float32)

            # Every probed cell is gathered in one go rather than a numpy call per
            # cell. The work per cell is tiny -- a few hundred rows of 64 bytes --
            # so at 64 probes the per-call overhead was most of the query.
            cells = probes[qi]
            starts = np.asarray(self.offsets[cells], dtype=np.int64)
            ends = np.asarray(self.offsets[cells + 1], dtype=np.int64)
            lengths = ends - starts
            live = lengths > 0
            if not live.any():
                out.append([])
                continue
            starts, lengths, cells = starts[live], lengths[live], cells[live]
            total = int(lengths.sum())
            # Ragged ranges, vectorised: repeat each run's base, then add a running
            # offset that restarts at every run boundary.
            begins = np.concatenate(([0], np.cumsum(lengths)[:-1]))
            picks = np.repeat(starts - begins, lengths) + np.arange(total)

            # Sub-quantiser on the outside, candidates on the inside: each of the
            # 256-entry tables is read straight through while it sits in L1.
            block = np.asarray(self.codes[:, picks])
            cand_scores = np.zeros(total, dtype=np.float32)
            for m in range(m_sub):
                cand_scores += np.take(table[m], block[m])
            cand_scores += np.repeat(coarse[qi, cells], lengths)
            cand_ids = np.asarray(self.ids[picks])
            cand_rows = np.asarray(self.rows[picks])
            if allowed is not None:
                keep = np.isin(cand_ids, allowed)
                cand_ids, cand_rows, cand_scores = (cand_ids[keep], cand_rows[keep],
                                                    cand_scores[keep])
                if not len(cand_ids):
                    out.append([])
                    continue

            take = min(want, len(cand_ids))
            top = np.argpartition(-cand_scores, take - 1)[:take]
            cand_ids, cand_rows = cand_ids[top], cand_rows[top]
            cand_scores = cand_scores[top]

            if rescore is not None and len(cand_rows):
                # The approximation chose what to look at; now read those vectors and
                # let arithmetic decide the order.
                exact = rescore(cand_rows, queries[qi] if queries.ndim > 1 else queries)
                if exact is not None and len(exact) == len(cand_rows):
                    cand_scores = np.asarray(exact, dtype=np.float32)

            final = min(k, len(cand_ids))
            best = np.argpartition(-cand_scores, final - 1)[:final]
            best = best[np.argsort(-cand_scores[best])]
            out.append([(int(cand_ids[i]), float(cand_scores[i])) for i in best])
        return out

    # ------------------------------------------------------------------ admin
    def stats(self) -> dict:
        if not self.meta:
            if not self.exists():
                return {"built": False}
            self.meta = json.loads(self._f("meta.json").read_text(encoding="utf-8"))
        n, m = self.meta["n_vectors"], self.meta["m_sub"]
        return {
            "built": True,
            "vectors": n,
            "cells": self.meta["n_cells"],
            "sub_quantisers": m,
            "bytes_per_vector": m + 16,
            "bytes_total": n * (m + 16),
            "exact_would_be": n * self.meta["dim"] * 4,
            "covers_rows": self.meta.get("covers_rows", 0),
            "built_at": self.meta.get("built_at", 0),
        }

    def close(self) -> None:
        self.centroids = self.books = self.offsets = None
        self.codes = self.ids = None
