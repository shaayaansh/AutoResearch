"""Dense (embedding) search with BAAI/bge-base-en-v1.5, stored as a plain numpy matrix.

Why numpy and no vector DB: ~18k papers x 768 dims x 4 bytes = ~55 MB. Exact search is a single
matrix-vector product (a few ms), so an approximate-nearest-neighbour index or a vector database
would add complexity and lose accuracy for no speed benefit at this size.

Documents are embedded from RAW text (title + abstract) — no stemming or stopword removal; the
model has its own tokenizer and needs natural language.

bge models are asymmetric: queries get an instruction prefix, documents don't.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import INDEX_DIR
from .corpus import Corpus

log = logging.getLogger(__name__)

DENSE_DIR = INDEX_DIR / "dense"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Bump if dense_text() changes: the index stores it and refuses to load a mismatch.
TEXT_VERSION = "v1"


def dense_text(paper: dict) -> str:
    return f"{paper.get('title', '')}. {paper.get('abstract', '')}".strip()


def pick_device() -> str:
    """Use whatever accelerator is available: NVIDIA GPU, then Apple GPU, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    device = device or pick_device()
    model = SentenceTransformer(name, device=device)
    if device == "cuda":
        model.half()  # fp16 on NVIDIA: ~2x faster, negligible quality loss for retrieval
    log.info("Loaded %s on %s (max_seq_length=%d)", name, device, model.max_seq_length)
    return model, device


@dataclass
class DenseHit:
    paper_id: str
    score: float  # cosine similarity
    rank: int


@dataclass
class DenseResult:
    hits: list[DenseHit]
    n_docs: int


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, paper_ids: list[str], meta: dict):
        self.embeddings = embeddings  # (n_docs, dim), float32, L2-normalized
        self.paper_ids = paper_ids
        self.meta = meta
        self._model = None

    # ---- build / persist ---------------------------------------------------------
    @classmethod
    def build(cls, corpus: Corpus, model_name: str = DEFAULT_MODEL, batch_size: int = 64,
              device: str | None = None) -> "DenseIndex":
        model, device = load_model(model_name, device)
        texts = [dense_text(p) for p in corpus.papers]
        t = time.perf_counter()
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        secs = time.perf_counter() - t
        log.info("Embedded %d docs in %.1fs on %s", len(texts), secs, device)
        meta = {"model": model_name, "dim": int(emb.shape[1]), "text_version": TEXT_VERSION,
                "max_seq_length": model.max_seq_length, "device": device,
                "build_seconds": round(secs, 1), "n_docs": len(texts)}
        idx = cls(emb, corpus.paper_ids, meta)
        idx._model = model
        return idx

    def save(self, path: Path = DENSE_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        (path / "paper_ids.json").write_text(json.dumps(self.paper_ids))
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))
        log.info("Saved dense index to %s (%.1f MB)", path, self.embeddings.nbytes / 1e6)

    @classmethod
    def load(cls, path: Path = DENSE_DIR) -> "DenseIndex":
        if not (path / "embeddings.npy").exists():
            raise FileNotFoundError(f"No dense index at {path}; run `python -m autoresearch build-index`")
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("text_version") != TEXT_VERSION:
            raise RuntimeError(f"Dense index text format {meta.get('text_version')} != code {TEXT_VERSION}; "
                               "rebuild: python -m autoresearch build-index --dense")
        emb = np.load(path / "embeddings.npy")
        paper_ids = json.loads((path / "paper_ids.json").read_text())
        return cls(emb, paper_ids, meta)

    # ---- query -------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:  # lazy: loading the model costs a few seconds
            self._model, _ = load_model(self.meta["model"])
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        """A search question: gets bge's retrieval instruction prefix."""
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.meta["model"].lower() else ""
        return self._encode(prefix + query)

    def embed_document(self, text: str) -> np.ndarray:
        """Document-like text (e.g. a HyDE abstract): embedded exactly like the corpus, no prefix,
        so the comparison is document-to-document."""
        return self._encode(text)

    def _encode(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32)

    def search(self, query: str, k: int = 50) -> DenseResult:
        return self.search_vector(self.embed_query(query), k)

    def search_vector(self, q: np.ndarray, k: int = 50) -> DenseResult:
        scores = self.embeddings @ q  # cosine similarity, since both sides are normalized
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        hits = [DenseHit(self.paper_ids[i], float(scores[i]), r) for r, i in enumerate(top, start=1)]
        return DenseResult(hits, len(scores))
