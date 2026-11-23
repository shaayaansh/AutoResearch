"""Registry of retrieval systems. Every version of the system registers here, so the CLI and
scripts/run_system.py run exactly the same code, and old versions stay runnable for comparison.

A system is a function  search(query, k) -> (hits, diagnostics)
  hits:        ranked objects with .paper_id, .score, .rank (and optionally .component_ranks)
  diagnostics: dict of whatever the system knows about how it got the results
"""

from __future__ import annotations

from typing import Callable


def _bm25(**_):
    from .bm25 import TOKENIZER_VERSION, BM25Index

    idx = BM25Index.load()

    def run(q, k):
        r = idx.search(q, k=k)
        return r.hits, {"tokenizer_version": TOKENIZER_VERSION, "query_tokens": r.query_tokens,
                        "n_matched": r.n_matched, "n_docs": r.n_docs, "term_doc_freq": r.term_doc_freq}
    return run


def _dense(**_):
    from .dense import DenseIndex

    idx = DenseIndex.load()
    idx.embed_query("warm-up")  # load the model now so it isn't counted in the first query's latency
    return lambda q, k: (idx.search(q, k=k).hits,
                         {"model": idx.meta["model"], "text_version": idx.meta["text_version"]})


def _hybrid(**_):
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


def _planned(mode: str, planner_model: str = "gpt-5.6-luna"):
    def make(**_):
        from .bm25 import TOKENIZER_VERSION, BM25Index
        from .dense import DenseIndex
        from .llm import LLM
        from .planned import PlannedSearcher
        from .planner import PLANNER_VERSION, Planner

        dense = DenseIndex.load()
        dense.embed_query("warm-up")
        planner = Planner(LLM(model=planner_model, reasoning_effort="low"))
        ps = PlannedSearcher(BM25Index.load(), dense, planner, mode=mode)

        def run(q, k):
            hits, diag = ps.search(q, k=k)
            return hits, {"tokenizer_version": TOKENIZER_VERSION, "dense_model": dense.meta["model"],
                          "planner_version": PLANNER_VERSION, **diag}
        return run
    return make


def _agent(loop: bool, model: str = "gpt-5.6-luna"):
    """v3. loop=False: verifier only (one round, re-rank by verdicts). loop=True: the full agent."""
    def make(verbose: bool = False):
        from .bm25 import TOKENIZER_VERSION, BM25Index
        from .corpus import Corpus
        from .dense import DenseIndex
        from .llm import LLM
        from .agent import AgentSearcher
        from .planner import PLANNER_VERSION, REVISER_VERSION, Planner, Reviser
        from .verifier import VERIFIER_VERSION, VerifierPool

        dense = DenseIndex.load()
        dense.embed_query("warm-up")
        llm = LLM(model=model, reasoning_effort="low")  # one client shared by all three roles
        searcher = AgentSearcher(BM25Index.load(), dense, Corpus.load(), Planner(llm),
                                 VerifierPool(llm), Reviser(llm) if loop else None, verbose=verbose)

        def run(q, k):
            hits, diag = searcher.search(q, k=k)
            return hits, {"tokenizer_version": TOKENIZER_VERSION, "dense_model": dense.meta["model"],
                          "planner_version": PLANNER_VERSION, "verifier_version": VERIFIER_VERSION,
                          "reviser_version": REVISER_VERSION if loop else None, **diag}
        return run
    return make


SYSTEMS: dict[str, Callable] = {
    "bm25": _bm25,                    # v0
    "dense": _dense,                  # v1 component
    "hybrid": _hybrid,                # v1
    "aspects": _planned("aspects"),   # v2 ablation: planner aspects, dense uses descriptions
    "hyde": _planned("hyde"),         # v2 / v2.1: planner aspects, dense uses HyDE abstracts
    "verify": _agent(loop=False),     # v3 ablation: + one verifier per aspect, no loop
    "agent": _agent(loop=True),       # v3: verifiers + planner revision loop
}


def get_system(name: str, **opts):
    """opts (e.g. verbose=True) go to systems that support them; others ignore them."""
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    return SYSTEMS[name](**opts)
