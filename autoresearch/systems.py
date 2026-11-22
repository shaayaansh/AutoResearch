"""Registry of retrieval systems. Every version of the system registers here, so the CLI and
scripts/run_system.py run exactly the same code, and old versions stay runnable for comparison.

A system is a function  search(query, k) -> (hits, diagnostics)
  hits:        ranked objects with .paper_id, .score, .rank (and optionally .component_ranks)
  diagnostics: dict of whatever the system knows about how it got the results
"""

from __future__ import annotations

from typing import Callable


def _bm25():
    from .bm25 import TOKENIZER_VERSION, BM25Index

    idx = BM25Index.load()

    def run(q, k):
        r = idx.search(q, k=k)
        return r.hits, {"tokenizer_version": TOKENIZER_VERSION, "query_tokens": r.query_tokens,
                        "n_matched": r.n_matched, "n_docs": r.n_docs, "term_doc_freq": r.term_doc_freq}
    return run


def _dense():
    from .dense import DenseIndex

    idx = DenseIndex.load()
    idx.embed_query("warm-up")  # load the model now so it isn't counted in the first query's latency
    return lambda q, k: (idx.search(q, k=k).hits,
                         {"model": idx.meta["model"], "text_version": idx.meta["text_version"]})


def _hybrid():
    from .bm25 import TOKENIZER_VERSION, BM25Index
    from .dense import DenseIndex
    from .hybrid import HybridSearcher

    dense = DenseIndex.load()
    dense.embed_query("warm-up")  # see _dense()
    hs = HybridSearcher(BM25Index.load(), dense)

    def run(q, k):
        r = hs.search(q, k=k)
        return r.hits, {"tokenizer_version": TOKENIZER_VERSION, "dense_model": dense.meta["model"],
                        **r.diagnostics}
    return run


SYSTEMS: dict[str, Callable] = {"bm25": _bm25, "dense": _dense, "hybrid": _hybrid}


def get_system(name: str):
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    return SYSTEMS[name]()
