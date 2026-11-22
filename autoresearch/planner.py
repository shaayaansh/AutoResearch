"""Planner: one LLM call that turns a research question into a search plan.

For each ASPECT of the question the planner writes three things:
  description    plain-language statement of the aspect (used by the `aspects` ablation)
  bm25_query     keywords for BM25: exact terms, synonyms, acronyms + expansions
  hyde_abstract  a hypothetical abstract of the ideal paper for this aspect (HyDE, Gao et al. 2022),
                 embedded for dense search instead of the question

Plans are cached per (question, model, prompt version) in data/cache/plans.jsonl, so re-running
the system on the same question is deterministic and free. Bump PLANNER_VERSION when the prompt
or schema changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from . import DATA_DIR
from .llm import LLM, Usage

PLANNER_VERSION = "v1"
PLAN_CACHE = DATA_DIR / "cache" / "plans.jsonl"

PLANNER_INSTRUCTIONS = """\
You are an expert ML/NLP researcher planning a literature search. The search runs over the \
titles and abstracts of papers from ACL, EMNLP, NAACL and EACL (2025-2026), using two \
retrievers: BM25 keyword search and dense embedding search.

Given a research question, split it into its ASPECTS: the distinct sub-topics a complete \
literature review of the question must cover. Use 2 to 5 aspects. A narrow question needs only \
2; do not pad. Aspects should not overlap heavily, and together they should cover every part of \
the question.

For each aspect write:

1. `name`: 2-6 words.

2. `description`: one or two plain sentences stating what papers for this aspect are about.

3. `bm25_query`: 6-15 space-separated search terms for a keyword engine that matches words, \
not meaning. Include the field's exact technical terms, common synonyms, and both acronyms and \
their expansions (e.g. "RAG retrieval-augmented generation"). Rules: no quotes, no boolean \
operators, no question words or filler; avoid words that appear in almost every NLP paper \
(model, language, LLM, method, approach, task, paper, results) unless the aspect is about them.

4. `hyde_abstract`: a 100-150 word abstract of an ideal paper on this aspect, written exactly as \
it would appear in ACL/EMNLP proceedings: a problem statement, the approach, how it is \
evaluated, and the findings. Use the field's real terminology. It is a search probe, not a \
real paper, so plausible specifics are fine, but do not invent named benchmarks or models.
"""


class Aspect(BaseModel):
    name: str = Field(description="2-6 word name of the aspect")
    description: str = Field(description="One or two plain sentences describing the aspect")
    bm25_query: str = Field(description="6-15 space-separated keyword search terms")
    hyde_abstract: str = Field(description="100-150 word hypothetical abstract of an ideal paper")


class Plan(BaseModel):
    aspects: list[Aspect] = Field(description="2 to 5 aspects of the research question")


class Planner:
    def __init__(self, llm: LLM, cache_path: Path = PLAN_CACHE, use_cache: bool = True):
        self.llm = llm
        self.cache_path = cache_path
        self.use_cache = use_cache
        self._cache: dict[tuple, dict] = {}
        if use_cache and cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                r = json.loads(line)
                self._cache[(r["question"], r["model"], r["planner_version"])] = r

    async def plan(self, question: str) -> tuple[Plan, Usage, bool]:
        """Returns (plan, usage of this call, whether it came from the cache)."""
        key = (question, self.llm.model, PLANNER_VERSION)
        if self.use_cache and key in self._cache:
            return Plan.model_validate(self._cache[key]["plan"]), Usage(), True

        plan, usage = await self.llm.parse(instructions=PLANNER_INSTRUCTIONS,
                                           input=f"Research question:\n{question}", schema=Plan)
        plan.aspects = plan.aspects[:5]  # enforce the upper bound even if the model overshoots
        record = {"question": question, "model": self.llm.model, "planner_version": PLANNER_VERSION,
                  "plan": plan.model_dump(), "usage": usage.as_dict()}
        self._cache[key] = record
        if self.use_cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return plan, usage, False
