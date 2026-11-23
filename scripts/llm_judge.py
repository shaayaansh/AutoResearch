#!/usr/bin/env python3
"""Grade a saved run with an LLM judge. Knows nothing about how the results were produced.

    python scripts/llm_judge.py eval/runs/<run_id>
    python scripts/llm_judge.py eval/runs/<run_id> --model gpt-5.6-terra --reasoning-effort medium

Reads   <run_dir>/results.jsonl  (query_id, rank, paper_id, score) + run_meta.json
Writes  <run_dir>/judgments_<model>.jsonl   one grade + reason per retrieved paper
        <run_dir>/metrics_<model>.json      per-query metrics, tokens and cost
Judging a run again with another model writes new files next to the old ones.

Judging is POINTWISE: each paper gets its own call, independent of the others.
  + no position bias / "lost in the middle" across a long multi-paper prompt
  + a paper's grade doesn't depend on what else was retrieved, so grades are reusable
    across systems: they are cached per (question, paper, model, prompt version) in
    eval/judgments_cache.jsonl, and re-judging a paper another system also found is free
  - question + rubric are re-sent each call
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402
from autoresearch.corpus import Corpus  # noqa: E402
from autoresearch.llm import LLM, Usage  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
CACHE_PATH = EVAL_DIR / "judgments_cache.jsonl"
PROMPT_VERSION = "v1"
RELEVANT_AT = 4  # score >= 4 counts as "relevant" for precision

JUDGE_INSTRUCTIONS = """\
You are an expert ML/NLP researcher helping build a literature review. You will be given a \
research question and ONE paper (title and abstract). Judge how relevant the paper is to the \
question.

Research questions are often broad and have several parts or aspects. A paper that addresses \
only ONE aspect well can still be highly relevant.

Grade relevance on this 1-5 scale:

5 - Directly relevant. The paper's main contribution squarely addresses the question or one of \
its aspects. A researcher answering the question would definitely cite it.
4 - Highly relevant. The paper substantially addresses an aspect of the question, but either \
covers it only in part or it is a major (not the sole) focus of the paper. Very likely cited.
3 - Somewhat relevant. The paper is on a closely related topic, or provides a method, dataset, \
benchmark or finding that would usefully inform an answer, without addressing the question \
itself. Might be cited as related work or background.
2 - Marginally relevant. The paper shares the general area or some keywords, but would not \
meaningfully help answer the question.
1 - Not relevant. Different topic, or the overlap is superficial (e.g. same word, different meaning).

Rules:
- Judge RELEVANCE to the question, not the paper's quality, novelty or how convincing it is. \
Even if the question asks for "the most effective" methods, any paper that studies or proposes \
such methods is relevant.
- Base your judgment only on the title and abstract provided. Do not assume content that is \
not stated.
- Shared keywords are not enough: check that the paper is about what the question is about.
- In `aspects_addressed`, list which parts of the question the paper addresses (short phrases); \
leave it empty if none.
- In `reason`, give one or two sentences justifying the grade.
"""


class Judgment(BaseModel):
    reason: str = Field(description="One or two sentences justifying the grade")
    aspects_addressed: list[str] = Field(description="Parts of the question this paper addresses")
    score: Literal[1, 2, 3, 4, 5]


def paper_for(row: dict, corpus) -> dict:
    """Our own runs reference corpus papers by id. External runs (e.g. Google Scholar) may contain
    papers outside the corpus; those rows carry the paper's title/abstract/venue inline."""
    if "paper" in row:
        p = row["paper"]
        return {"id": row["paper_id"], "title": p["title"], "abstract": p.get("abstract", ""),
                "venue": p.get("venue") or "", "year": p.get("year") or "", "url": p.get("url", "")}
    return corpus.get(row["paper_id"])


def format_input(question: str, paper: dict) -> str:
    # Question first, paper last: calls for the same question share a prefix (prompt caching).
    abstract = paper.get("abstract") or "(no abstract available)"
    venue = f"{paper.get('venue') or ''} {paper.get('year') or ''}".strip() or "unknown"
    return (f"Research question:\n{question}\n\n"
            f"Paper title: {paper['title']}\nVenue: {venue}\n"
            f"Abstract: {abstract}")


# ---------------------------------------------------------------------------------------
# Cache: append-only JSONL keyed by (question text, paper, model, prompt version).
# Keyed on the question TEXT, not its id, so editing a question never reuses stale grades.
# ---------------------------------------------------------------------------------------
class JudgmentCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._data: dict[tuple, dict] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                r = json.loads(line)
                self._data[self.key(r["question"], r["paper_id"], r["model"], r["prompt_version"])] = r

    @staticmethod
    def key(question, paper_id, model, version=PROMPT_VERSION):
        return (question, paper_id, model, version)

    def get(self, *key):
        return self._data.get(self.key(*key))

    def put(self, r: dict) -> None:
        self._data[self.key(r["question"], r["paper_id"], r["model"], r["prompt_version"])] = r
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def judge_one(llm: LLM, cache: JudgmentCache, sem: asyncio.Semaphore,
                    qid: str, question: str, paper: dict) -> tuple[dict, Usage | None]:
    hit = cache.get(question, paper["id"], llm.model)
    if hit:
        return hit, None
    async with sem:
        try:
            j, u = await llm.parse(instructions=JUDGE_INSTRUCTIONS, input=format_input(question, paper),
                                   schema=Judgment, cache_key=f"judge-{qid}")
        except Exception as e:  # record the failure, keep going
            return {"paper_id": paper["id"], "error": str(e)}, None
    r = {"question": question, "paper_id": paper["id"], "model": llm.model,
         "prompt_version": PROMPT_VERSION, "score": j.score,
         "aspects_addressed": j.aspects_addressed, "reason": j.reason, "usage": u.as_dict()}
    cache.put(r)
    return r, u


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def precision_at(scores: list, n: int):
    top = [s for s in scores[:n] if s is not None]
    return round(sum(s >= RELEVANT_AT for s in top) / len(top), 3) if top else None


def query_metrics(scores: list) -> dict:
    valid = [s for s in scores if s is not None]
    dist = Counter(valid)
    return {
        "n_judged": len(valid), "n_errors": len(scores) - len(valid),
        "score_dist": {str(k): dist.get(k, 0) for k in range(5, 0, -1)},
        "mean_score": round(sum(valid) / len(valid), 3) if valid else None,
        f"n_relevant(>={RELEVANT_AT})": sum(s >= RELEVANT_AT for s in valid),
        "P@10": precision_at(scores, 10), "P@20": precision_at(scores, 20),
        "P@50": precision_at(scores, 50),
    }


async def run(args) -> None:
    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    by_query = defaultdict(list)
    for line in (run_dir / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        by_query[r["query_id"]].append(r)

    corpus = Corpus.load()
    llm = LLM(model=args.model, reasoning_effort=args.reasoning_effort or None)
    cache = JudgmentCache()
    sem = asyncio.Semaphore(args.concurrency)
    total = Usage()
    metrics = {"run_id": meta["run_id"], "system": meta["system"], "k": meta["k"],
               "judge_model": args.model, "reasoning_effort": args.reasoning_effort,
               "prompt_version": PROMPT_VERSION, "relevant_threshold": RELEVANT_AT, "queries": {}}

    out_path = run_dir / f"judgments_{args.model}.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for qid, rows in by_query.items():
            question = meta["queries"][qid]
            rows.sort(key=lambda r: r["rank"])
            t0 = time.perf_counter()
            results = await asyncio.gather(*(
                judge_one(llm, cache, sem, qid, question, paper_for(r, corpus)) for r in rows))

            usage, n_cached, scores = Usage(), 0, []
            for row, (j, u) in zip(rows, results):
                if u is None and "error" not in j:
                    n_cached += 1
                elif u is not None:
                    usage.add(u)
                p = paper_for(row, corpus)
                scores.append(j.get("score"))
                out.write(json.dumps({
                    "query_id": qid, "rank": row["rank"], "paper_id": row["paper_id"],
                    "system_score": row["score"], "title": p["title"],
                    "venue": f"{p['venue']} {p['year']}", "url": p["url"],
                    "judge_score": j.get("score"), "aspects_addressed": j.get("aspects_addressed"),
                    "reason": j.get("reason"), "error": j.get("error"),
                }, ensure_ascii=False) + "\n")
            total.add(usage)

            m = query_metrics(scores)
            m.update(question=question, n_cached=n_cached, usage=usage.as_dict(),
                     elapsed_s=round(time.perf_counter() - t0, 1))
            metrics["queries"][qid] = m
            print(f"{qid}: mean {m['mean_score']} · relevant {m[f'n_relevant(>={RELEVANT_AT})']}/{m['n_judged']}"
                  f" · P@10 {m['P@10']} · dist {m['score_dist']} · ${usage.cost_usd:.4f}"
                  f" ({n_cached} cached, {m['n_errors']} errors) · {m['elapsed_s']}s")

    metrics["total_usage"] = total.as_dict()
    (run_dir / f"metrics_{args.model}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nTotal: {total.calls} calls · {total.input_tokens:,} in ({total.cached_tokens:,} cached)"
          f" · {total.output_tokens:,} out ({total.reasoning_tokens:,} reasoning) · ${total.cost_usd:.4f}")
    print(f"Wrote {out_path.name} and metrics_{args.model}.json in {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="eval/runs/<run_id> produced by scripts/run_system.py")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--reasoning-effort", default="low", help="'' to omit")
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
