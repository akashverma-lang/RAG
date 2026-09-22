"""The approximate index: does it return the right things, and does it hold up?

Runs against throwaway storage. Invoked by ``python rag.py test``.

The vectors here are synthesised rather than embedded, because embedding a corpus
large enough to be interesting would take days and would measure the embedder. They
are drawn to match the real embedding space -- which was measured, not assumed: real
bge-small embeddings have an effective dimensionality of about 23 inside their 384,
carry 58% of their variance in ten directions, and sit at a mean cosine of 0.57 to
each other. Isotropic random vectors have none of that, and structure is exactly
what an inverted file trades on, so benchmarking on them would measure a corpus
nobody has.
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


def _space(np, dim: int = 384, seed: int = 0):
    """A generator whose first two moments match measured embeddings."""
    rng = np.random.default_rng(seed)
    # A spectrum that decays like the real one: a handful of strong directions and
    # a long tail, rather than 384 equal ones.
    sd = (1.0 / (1.0 + np.arange(dim) / 6.0) ** 1.1).astype(np.float32)
    basis, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    basis = basis.astype(np.float32)
    mean = basis[:, 0] * 1.6                       # the anisotropy: a common direction

    def sample(n: int) -> "np.ndarray":
        coef = rng.standard_normal((n, dim)).astype(np.float32) * sd
        v = coef @ basis.T + mean
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v.astype(np.float32)

    def topical(n: int, n_topics: int, spread: float = 0.68):
        coef = rng.standard_normal((n_topics, dim)).astype(np.float32) * sd * 3.0
        topics = coef @ basis.T + mean
        topics /= np.linalg.norm(topics, axis=1, keepdims=True)
        which = rng.integers(0, n_topics, size=n)
        v = topics[which] * (1 - spread) + sample(n) * spread
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v.astype(np.float32), which

    return sample, topical, rng


def main() -> int:
    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="rag_scale_"))
    os.environ["STORAGE_DIR"] = str(tmp)
    os.environ["DATA_DIR"] = str(tmp / "data")
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from app import config
    from app.core.ann import IVFPQIndex
    from app.core.store import Store

    sample, topical, rng = _space(np)
    DIM, N, NQ = 384, 60_000, 60
    corpus, _topic = topical(N, n_topics=300)
    pick = rng.choice(N, size=NQ, replace=False)
    queries = corpus[pick] * 0.75 + sample(NQ) * 0.25
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries.astype(np.float32)

    store = Store(tmp / "index.db", tmp / "vectors.f32")
    store.set_meta("dim", DIM)
    try:
        print(f"\nthe sidecar that replaces a full table scan ({N:,} vectors)")
        for i in range(0, N, 20_000):
            j = min(i + 20_000, N)
            store.upsert_file_chunks(
                path=f"f{i}", rel_path=f"f{i}", size=1, mtime=1.0, file_hash=f"h{i}",
                chunks=[{"text": f"c{c}", "loc": "", "heading": ""}
                        for c in range(i, j)],
                vectors=corpus[i:j])
        check("a row->chunk sidecar is written beside the vectors",
              store.rowid_path.exists())
        check("it is exactly one int64 per vector row",
              store.rowid_path.stat().st_size == N * 8,
              f"{store.rowid_path.stat().st_size:,} bytes")
        store._dirty = True
        store._load_vectors()
        ids = np.asarray(store._row2id)
        check("every row maps to a live chunk", int((ids >= 0).sum()) == N)
        rows = store.get_chunks([int(ids[0]), int(ids[N - 1])])
        check("and the mapping points at the right chunks",
              rows.get(int(ids[0]), {}).get("text") == "c0"
              and rows.get(int(ids[N - 1]), {}).get("text") == f"c{N-1}")

        # Deleting must poke holes in the sidecar, not shift anything.
        store.delete_file("f0")
        store._dirty = True
        store._load_vectors()
        after = np.asarray(store._row2id)
        check("deleting a file tombstones its rows in place",
              int((after[:20_000] >= 0).sum()) == 0 and len(after) == N)
        check("and leaves the rest untouched",
              int((after[20_000:] >= 0).sum()) == N - 20_000)

        print("\nexact search is kept while the corpus is small")
        out = store.build_ann()
        check("no index is built below the threshold",
              not out.get("built") and "reason" in out, str(out.get("reason", ""))[:60])

        print(f"\nthe approximate index ({N:,} vectors)")
        config.ANN_MIN_VECTORS = 1000
        t = time.perf_counter()
        out = store.build_ann()
        build_s = time.perf_counter() - t
        check("it builds", bool(out.get("built")),
              f"{out.get('n_cells', 0):,} cells, {out.get('m_sub', 0)} sub-quantisers, "
              f"{build_s:.0f}s")
        if not out.get("built"):
            return 1
        ratio = out["bytes_exact"] / max(out["bytes_codes"], 1)
        check("and is far smaller than the vectors", ratio > 10,
              f"{out['bytes_codes']/2**20:.1f} MB vs "
              f"{out['bytes_exact']/2**20:.0f} MB exact ({ratio:.0f}x)")

        # Ground truth, from the exact path.
        store._ann = None
        config.ANN_ENABLED = False
        t = time.perf_counter()
        exact = store.search_dense_many(queries, 40)
        exact_ms = (time.perf_counter() - t) / NQ * 1000
        truth = [set(i for i, _ in r) for r in exact]

        config.ANN_ENABLED = True
        store._ann = None
        t = time.perf_counter()
        got = store.search_dense_many(queries, 40)
        ann_ms = (time.perf_counter() - t) / NQ * 1000
        recall = float(np.mean([len(set(i for i, _ in g) & tr) / 40
                                for g, tr in zip(got, truth)]))
        check("it agrees with exact search on most of the top 40", recall >= 0.70,
              f"recall@40 {recall:.3f} at nprobe {config.ANN_NPROBE} "
              f"({ann_ms:.2f} ms vs {exact_ms:.2f} ms exact)")
        # Probing more cells must never find less: if it does, the cell ordering or
        # the candidate merge is wrong, and the accuracy dial does not mean anything.
        curve = [(p, float(np.mean([len(set(i for i, _ in g) & tr) / 40
                                    for g, tr in zip(_probe(store, config, queries, p),
                                                     truth)])))
                 for p in (16, 64, 256)]
        check("more probes never find less",
              all(b >= a - 1e-9 for (_, a), (_, b) in zip(curve, curve[1:])),
              "  ".join(f"{p}:{r:.3f}" for p, r in curve))
        check("and enough probes converge on exact", curve[-1][1] >= 0.95,
              f"recall@40 {curve[-1][1]:.3f} at nprobe 256")
        config.ANN_NPROBE = 64
        store._ann = None

        # Scores must be the real inner products, not the quantised approximations:
        # fusion and the rerank threshold both read them.
        best_id, best_score = got[0][0]
        row = int(np.flatnonzero(np.asarray(store._row2id) == best_id)[0])
        true_score = float(corpus[row] @ queries[0])
        check("the score returned is the exact inner product",
              abs(true_score - best_score) < 1e-4,
              f"{best_score:.5f} vs {true_score:.5f}")

        print("\nrows added after the build are still found")
        extra, _ = topical(3000, n_topics=300)
        marker = extra[7].copy()
        store.upsert_file_chunks(
            path="late", rel_path="late", size=1, mtime=1.0, file_hash="late",
            chunks=[{"text": f"late{c}", "loc": "", "heading": ""} for c in range(3000)],
            vectors=extra)
        store._dirty = True
        store._ann = None
        hits = store.search_dense_many(np.asarray([marker]), 5)[0]
        found = store.get_chunks([i for i, _ in hits])
        check("a chunk indexed after the last build is retrievable",
              any(v["text"] == "late7" for v in found.values()),
              "the tail is searched exactly and merged in")

        print("\nscoping to chosen files still works")
        file_ids = store.file_ids_for(["f20000"])
        scoped = store.search_dense_many(queries[:3], 10, file_ids)
        allowed = set(int(x) for x in store.chunk_ids_for_files(file_ids))
        check("every scoped hit comes from the chosen file",
              all(i in allowed for r in scoped for i, _ in r),
              f"{sum(len(r) for r in scoped)} hits")

        print("\nthe probe count has to grow with the index")
        # Accuracy follows the share of the corpus examined, and cells grow with the
        # corpus -- so a fixed probe count examines less and less of it. Measured: 64
        # probes covered 8% of a 60k corpus and returned recall 0.98, but only 1.1%
        # of a 2M corpus, where it returned 0.74.
        config.ANN_SCAN_FRACTION = 0.02
        config.ANN_NPROBE = 64

        class _Idx:
            def __init__(self, cells):
                self.meta = {"n_cells": cells}

        shares = [(cells, Store.nprobe_for(_Idx(cells)) / cells)
                  for cells in (800, 5656, 16384)]
        check("the share of the index scanned stops shrinking",
              all(share >= 0.019 for _c, share in shares[1:]),
              "  ".join(f"{c} cells:{s*100:.1f}%" for c, s in shares))
        config.ANN_SCAN_FRACTION = 0.0
        check("and it can be pinned to a constant cost instead",
              Store.nprobe_for(_Idx(16384)) == 64,
              "ANN_SCAN_FRACTION=0 uses the nprobe floor alone")
        config.ANN_SCAN_FRACTION = 0.02

        print("\nsizing")
        for docs in (1_000_000, 20_000_000):
            chunks = docs * 5
            cells = IVFPQIndex.suggest_cells(chunks)
            m = IVFPQIndex.suggest_subquantisers(DIM)
            exact_gb = chunks * DIM * 4 / 2**30
            idx_gb = chunks * (m + 16) / 2**30
            check(f"{docs:,} documents stays within memory", idx_gb < 16,
                  f"{chunks:,} chunks: {idx_gb:.1f} GB index vs "
                  f"{exact_gb:.0f} GB exact, {cells:,} cells")
    finally:
        store.db.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "-" * 60)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all scale tests passed")
    return 0


def _probe(store, config, queries, nprobe):
    config.ANN_NPROBE = nprobe
    store._ann = None
    return store.search_dense_many(queries, 40)


if __name__ == "__main__":
    raise SystemExit(main())
