"""Verifier: an LLM that reads retrieved abstracts and decides, per aspect, which papers are
relevant. Part of the SYSTEM (v3), deliberately different from the evaluation judge
(scripts/llm_judge.py): different prompt, different output (keep/drop + short reason instead of
a 1-5 grade), and it judges relevance to ONE ASPECT of the question rather than to the question.

Design:
- The orchestrator creates one verifier per aspect; verifiers run in parallel.
- Each verifier reads its aspect's candidates in batches of ~10 papers per LLM call. Seeing
  several papers at once lets it describe the *pattern* of failures ("7 of 10 are general RLHF,
  not about domain agents"), which the planner uses to rewrite queries in the next round.
- Verdicts are cached per (question, aspect, paper, model, version): papers already checked for
  an aspect are never re-checked in later rounds, and re-runs are free and deterministic.
- The relevance standard never changes between rounds, so the loop cannot "succeed" by lowering
  the bar.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from . import DATA_DIR
from .llm import LLM, Usage

VERIFIER_VERSION = "v1"
VERIFY_CACHE = DATA_DIR / "cache" / "verifications.jsonl"

VERIFIER_INSTRUCTIONS = """\
You are a careful ML/NLP researcher screening search results for a literature review. You \
will be given a research question, ONE aspect of that question, and a numbered batch of papers \
(title, venue, abstract). For each paper decide whether it is RELEVANT to this aspect of the \
question.

A paper is relevant only if BOTH hold:
1. Its main topic substantially addresses this aspect (not just a shared keyword or a passing \
mention).
2. It is about the question's core subject. Example: for an aspect about preference \
optimization in a question about domain-specific agents, a general RLHF paper with no agents is \
NOT relevant.

Rules:
- Judge relevance, not quality or novelty. Use only the title and abstract.
- Shared keywords are not enough: watch for the same word used with a different meaning.
- Apply the same standard to every paper and every batch.
- Give every paper a verdict, using its number from the batch.

Also report, for the next search round:
- `failure_pattern`: in one or two sentences, WHY the irrelevant papers in this batch were \
retrieved, using these categories where they fit: wrong meaning (keyword matched a different \
concept), lost core subject (right method/topic but not about the question's subject), too \
generic (broad match), off-topic. Mention concrete topics, e.g. "6 papers are general RLHF \
without agents". Write "none" if all papers are relevant.
- `missing`: one sentence on a sub-topic this aspect clearly needs that none of these papers \
covers, or "none".
"""


class Verdict(BaseModel):
    paper: int = Field(description="The paper's number in the batch")
    relevant: bool
    reason: str = Field(description="At most 15 words")


class BatchVerdicts(BaseModel):
    verdicts: list[Verdict]
    failure_pattern: str
    missing: str


def format_batch(question: str, aspect_name: str, aspect_desc: str, papers: list[dict]) -> str:
    lines = [f"Research question:\n{question}\n", f"Aspect: {aspect_name}\n{aspect_desc}\n", "Papers:"]
    for i, p in enumerate(papers, start=1):
        abstract = p.get("abstract") or "(no abstract available)"
        lines.append(f"[{i}] {p['title']}\nVenue: {p['venue']} {p['year']}\nAbstract: {abstract}\n")
    return "\n".join(lines)


@dataclass
class AspectReport:
    """What one aspect's verifier found in one round."""
    aspect_id: str
    n_checked: int = 0
    n_relevant: int = 0
    relevant_ids: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    n_cached: int = 0
    n_errors: int = 0


class VerifierPool:
    """Creates and runs one verifier per aspect, in parallel, with a shared cache."""

    def __init__(self, llm: LLM, batch_size: int = 10, concurrency: int = 16,
                 cache_path: Path = VERIFY_CACHE, use_cache: bool = True):
        self.llm, self.batch_size, self.use_cache, self.cache_path = llm, batch_size, use_cache, cache_path
        self.sem = asyncio.Semaphore(concurrency)
        self._cache: dict[tuple, dict] = {}
        if use_cache and cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                r = json.loads(line)
                self._cache[self._key(r["question"], r["aspect"], r["paper_id"])] = r

    def _key(self, question, aspect_desc, paper_id):
        return (question, aspect_desc, paper_id, self.llm.model, VERIFIER_VERSION)

    def cached(self, question: str, aspect_desc: str, paper_id: str) -> dict | None:
        return self._cache.get(self._key(question, aspect_desc, paper_id))

    def _store(self, record: dict) -> None:
        self._cache[self._key(record["question"], record["aspect"], record["paper_id"])] = record
        if self.use_cache:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _verify_batch(self, question, aspect, papers, usage: Usage, trace: list | None = None):
        text = format_batch(question, aspect["name"], aspect["description"], papers)
        async with self.sem:
            out, u = await self.llm.parse(instructions=VERIFIER_INSTRUCTIONS, input=text,
                                          schema=BatchVerdicts, cache_key="verifier")
        usage.add(u)
        if trace is not None:  # the exact conversation, for inspection (see scripts/show_trace.py)
            trace.append({"role": "verifier", "aspect_id": aspect["id"], "cached": False,
                          "paper_ids": [p["id"] for p in papers], "input": text,
                          "output": out.model_dump(), "usage": u.as_dict()})
        return out

    async def verify_aspect(self, question: str, aspect: dict, papers: list[dict],
                            usage: Usage, trace: list | None = None) -> AspectReport:
        """aspect: {"id", "name", "description"}; papers: corpus dicts to check (in rank order)."""
        rep = AspectReport(aspect["id"])
        todo, from_cache = [], []
        for p in papers:
            hit = self.cached(question, aspect["description"], p["id"])
            if hit:
                from_cache.append({"paper_id": p["id"], "relevant": hit["relevant"], "reason": hit["reason"]})
                rep.n_cached += 1
                rep.n_checked += 1
                if hit["relevant"]:
                    rep.n_relevant += 1
                    rep.relevant_ids.append(p["id"])
            else:
                todo.append(p)

        if trace is not None and from_cache:
            trace.append({"role": "verifier", "aspect_id": aspect["id"], "cached": True,
                          "verdicts": from_cache})
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        results = await asyncio.gather(*(self._verify_batch(question, aspect, b, usage, trace)
                                         for b in batches), return_exceptions=True)
        for batch, res in zip(batches, results):
            if isinstance(res, Exception):
                rep.n_errors += len(batch)
                if trace is not None:
                    trace.append({"role": "verifier", "aspect_id": aspect["id"], "error": repr(res),
                                  "paper_ids": [p["id"] for p in batch]})
                continue
            by_num = {v.paper: v for v in res.verdicts}
            for i, p in enumerate(batch, start=1):
                v = by_num.get(i)
                if v is None:  # the model skipped this paper: leave it unverified
                    rep.n_errors += 1
                    continue
                self._store({"question": question, "aspect": aspect["description"], "paper_id": p["id"],
                             "model": self.llm.model, "verifier_version": VERIFIER_VERSION,
                             "relevant": v.relevant, "reason": v.reason})
                rep.n_checked += 1
                if v.relevant:
                    rep.n_relevant += 1
                    rep.relevant_ids.append(p["id"])
            if res.failure_pattern and res.failure_pattern.strip().lower() != "none":
                rep.failure_patterns.append(res.failure_pattern.strip())
            if res.missing and res.missing.strip().lower() != "none":
                rep.missing.append(res.missing.strip())
        return rep

    async def verify_all(self, question: str, work: list[tuple[dict, list[dict]]]) -> tuple[list[AspectReport], Usage]:
        """work: [(aspect, papers_to_check), ...] -> one report per aspect, all aspects in parallel."""
        usage = Usage()
        reports = await asyncio.gather(*(self.verify_aspect(question, a, ps, usage) for a, ps in work))
        return list(reports), usage
