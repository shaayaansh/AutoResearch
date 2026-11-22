"""Hybrid search: BM25 + dense, fused with Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))      (k = 60)

RRF uses only RANKS, never raw scores, so BM25 scores (unbounded, ~0-15) and cosine similarities
(~0.3-0.9) never need to be normalized onto one scale. A document absent from a retriever's list
contributes nothing from that retriever. k = 60 dampens the advantage of the very top ranks so
that agreement between retrievers matters more than one retriever's #1.

Each retriever contributes its top `depth` (100) results, so a paper ranked ~70 by both can still
end up in the fused top 50.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25Index
from .dense import DenseIndex

RRF_K = 60
DEPTH = 100


@dataclass
class FusedHit:
    paper_id: str
    score: float                 # RRF score
    rank: int
    component_ranks: dict = field(default_factory=dict)  # e.g. {"bm25": 3, "dense": 41}


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> list[FusedHit]:
    """rankings: retriever name -> ordered list of paper ids (best first)."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict] = {}
    for name, ids in rankings.items():
        for r, pid in enumerate(ids, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + r)
            ranks.setdefault(pid, {})[name] = r
    # tie-break by best single rank, then id, so the output is deterministic
    order = sorted(scores, key=lambda p: (-scores[p], min(ranks[p].values()), p))
    return [FusedHit(p, scores[p], i, ranks[p]) for i, p in enumerate(order, start=1)]


@dataclass
class HybridResult:
    hits: list[FusedHit]
    diagnostics: dict


class HybridSearcher:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, depth: int = DEPTH, k: int = RRF_K):
        self.bm25, self.dense, self.depth, self.k = bm25, dense, depth, k

    def search(self, query: str, k: int = 50) -> HybridResult:
        b = self.bm25.search(query, k=self.depth)
        d = self.dense.search(query, k=self.depth)
        b_ids = [h.paper_id for h in b.hits]
        d_ids = [h.paper_id for h in d.hits]
        fused = rrf({"bm25": b_ids, "dense": d_ids}, k=self.k)[:k]

        top = fused
        diag = {
            "rrf_k": self.k, "depth": self.depth,
            "bm25": {"query_tokens": b.query_tokens, "n_matched": b.n_matched,
                     "term_doc_freq": b.term_doc_freq},
            # how much the two retrievers agree, and where the fused list came from
            "overlap_at_depth": len(set(b_ids) & set(d_ids)),
            "fused_from_both": sum(len(h.component_ranks) == 2 for h in top),
            "fused_bm25_only": sum(list(h.component_ranks) == ["bm25"] for h in top),
            "fused_dense_only": sum(list(h.component_ranks) == ["dense"] for h in top),
        }
        return HybridResult(fused, diag)
