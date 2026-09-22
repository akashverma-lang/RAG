"""Local embedding + reranking backends, in order of preference.

1. fastembed      - ONNX runtime, no torch, quantised models, fast on CPU  (default)
2. sentence-transformers - if you already have torch and want other models

Everything runs on your machine; nothing is sent anywhere, and neither backend needs
a server of its own -- the model is downloaded once and read from disk after that.
"""
from __future__ import annotations

import threading
from typing import Iterable, Sequence

import numpy as np

from app import config


class EmbedderError(RuntimeError):
    pass


class BaseEmbedder:
    name = "base"
    dim = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode several queries at once. Overridden where the backend can batch."""
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        return np.asarray([self.encode_query(t) for t in texts], dtype=np.float32)


class FastEmbedEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(config.MODEL_CACHE),
            threads=None,
        )
        self.dim = int(self.encode_query("dimension probe").shape[-1])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = list(self.model.passage_embed(list(texts), batch_size=config.EMBED_BATCH))
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self.model.query_embed([text]))[0], dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        return np.asarray(list(self.model.query_embed(list(texts),
                                                      batch_size=config.EMBED_BATCH)),
                          dtype=np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name, cache_folder=str(config.MODEL_CACHE))
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=config.EMBED_BATCH,
                              normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        return np.asarray(
            self.model.encode([config.QUERY_PREFIX + t for t in texts],
                              batch_size=config.EMBED_BATCH,
                              normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


_ST_DEFAULT = "BAAI/bge-small-en-v1.5"

_embedder: BaseEmbedder | None = None
_reranker = None
_lock = threading.Lock()


def _build(backend: str) -> BaseEmbedder:
    model = config.EMBED_MODEL
    if backend == "fastembed":
        return FastEmbedEmbedder(model)
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model or _ST_DEFAULT)
    raise EmbedderError(f"unknown EMBED_BACKEND: {backend}")


def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is not None:
            return _embedder
        order = ([config.EMBED_BACKEND] if config.EMBED_BACKEND != "auto"
                 else ["fastembed", "sentence_transformers"])
        errors = []
        for backend in order:
            try:
                _embedder = _build(backend)
                print(f"[embed] using {_embedder.name} (dim={_embedder.dim})", flush=True)
                return _embedder
            except Exception as exc:                      # noqa: BLE001 - try the next backend
                errors.append(f"{backend}: {exc}")
        raise EmbedderError(
            "no embedding backend available.\n  " + "\n  ".join(errors) +
            "\nInstall one with:  pip install fastembed"
        )


def encode_documents(texts: Sequence[str]) -> np.ndarray:
    return get_embedder().encode_documents(texts)


def encode_query(text: str) -> np.ndarray:
    return get_embedder().encode_query(text)


def encode_queries(texts: Sequence[str]) -> np.ndarray:
    """One embedding call for every phrasing of a question, not one per phrasing."""
    return get_embedder().encode_queries(list(texts))


# --------------------------------------------------------------------------- rerank
def get_reranker():
    """Cross-encoder that rescores candidates against the query. Optional."""
    global _reranker
    if _reranker is not None or not config.RERANK_ENABLED:
        return _reranker
    with _lock:
        if _reranker is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _reranker = TextCrossEncoder(
                    model_name=config.RERANK_MODEL, cache_dir=str(config.MODEL_CACHE)
                )
                print(f"[rerank] using {config.RERANK_MODEL}", flush=True)
            except Exception as exc:                      # noqa: BLE001
                print(f"[rerank] disabled ({exc})", flush=True)
                _reranker = False
    return _reranker or None


def rerank(query: str, docs: Sequence[str]) -> list[float] | None:
    model = get_reranker()
    if not model or not docs:
        return None
    try:
        return [float(s) for s in model.rerank(query, list(docs))]
    except Exception as exc:                              # noqa: BLE001
        print(f"[rerank] failed, falling back to fusion order ({exc})", flush=True)
        return None


def warmup() -> dict:
    emb = get_embedder()
    get_reranker()
    return {"embedder": emb.name, "dim": emb.dim}
