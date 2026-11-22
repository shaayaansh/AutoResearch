"""v2: planner-driven search.

    question ─► planner (LLM) ─► aspects ─► per aspect: BM25(bm25_query) + dense(probe) ─► RRF
                                                                                          │
                              final top-k ◄── RRF across aspect lists ◄───────────────────┘

Two dense probes, so each piece can be measured on its own (ablation):
  mode="aspects"  dense search embeds the aspect's plain description (as a query, with prefix)
  mode="hyde"     dense search embeds the aspect's hypothetical abstract (as a document, no prefix)
BM25 always uses the planner's keyword query.

Merging across aspects uses RRF again: each aspect's list counts equally, so one aspect with many
matching papers can't crowd out the others, and papers relevant to several aspects rise.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .bm25 import BM25Index
from .dense import DenseIndex
from .hybrid import DEPTH, RRF_K, rrf
from .planner import Plan, Planner

MODES = ("aspects", "hyde")


@dataclass
class PlannedHit:
    paper_id: str
    score: float
    rank: int
    component_ranks: dict = field(default_factory=dict)  # aspect index -> rank in that aspect's list
    aspects: list[str] = field(default_factory=list)      # names of aspects that retrieved it


class PlannedSearcher:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, planner: Planner, mode: str = "hyde",
                 depth: int = DEPTH, k: int = RRF_K):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.bm25, self.dense, self.planner, self.mode = bm25, dense, planner, mode
        self.depth, self.k = depth, k
        # One persistent loop: the async OpenAI client is bound to the loop it first ran on, so a
        # fresh asyncio.run() per question would fail on the second question.
        self._loop = asyncio.new_event_loop()

    def _dense_probe(self, aspect):
        if self.mode == "hyde":
            return self.dense.embed_document(aspect.hyde_abstract)
        return self.dense.embed_query(aspect.description)

    def search(self, question: str, k: int = 50) -> tuple[list[PlannedHit], dict]:
        t0 = time.perf_counter()
        plan, usage, cached = self._loop.run_until_complete(self.planner.plan(question))
        t_plan = time.perf_counter() - t0

        aspect_lists: dict[str, list[str]] = {}
        aspect_diag = []
        for i, a in enumerate(plan.aspects, start=1):
            b = self.bm25.search(a.bm25_query, k=self.depth)
            d = self.dense.search_vector(self._dense_probe(a), k=self.depth)
            b_ids = [h.paper_id for h in b.hits]
            d_ids = [h.paper_id for h in d.hits]
            fused = rrf({"bm25": b_ids, "dense": d_ids}, k=self.k)[: self.depth]
            aspect_lists[f"a{i}"] = [h.paper_id for h in fused]
            aspect_diag.append({
                "id": f"a{i}", "name": a.name, "description": a.description,
                "bm25_query": a.bm25_query, "bm25_tokens": b.query_tokens, "bm25_n_matched": b.n_matched,
                "hyde_abstract": a.hyde_abstract,  # stored in both modes: the run keeps its full plan
                "dense_probe": "hyde_abstract" if self.mode == "hyde" else "description",
                "bm25_dense_overlap": len(set(b_ids) & set(d_ids)),
                # this aspect's own ranked list (before merging across aspects), for later analysis
                "retrieved": [{"paper_id": h.paper_id, "rrf": round(h.score, 6), **h.component_ranks}
                              for h in fused],
            })

        merged = rrf(aspect_lists, k=self.k)[:k]
        names = {f"a{i}": a.name for i, a in enumerate(plan.aspects, start=1)}
        hits = [PlannedHit(h.paper_id, h.score, h.rank, h.component_ranks,
                           [names[a] for a in h.component_ranks]) for h in merged]

        # how evenly the final list covers the aspects
        coverage = {names[a]: sum(a in h.component_ranks for h in hits) for a in aspect_lists}
        diag = {
            "mode": self.mode, "planner_model": self.planner.llm.model,
            "plan_cached": cached, "plan_latency_s": round(t_plan, 3),
            "llm_usage": usage.as_dict(), "n_aspects": len(plan.aspects),
            "aspects": aspect_diag, "final_coverage_by_aspect": coverage,
            "final_multi_aspect": sum(len(h.component_ranks) > 1 for h in hits),
            "rrf_k": self.k, "depth": self.depth,
        }
        return hits, diag
