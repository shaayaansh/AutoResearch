#!/usr/bin/env python3
"""Compare judged runs against a shared POOL of relevance judgments (TREC-style pooling).

    python scripts/compare_runs.py eval/runs/<bm25_run> eval/runs/<dense_run> eval/runs/<hybrid_run>

Each run must already be graded by scripts/llm_judge.py with the same judge model. No LLM calls
happen here: this script only reads judgments and computes metrics.

Why pooling: nobody knows every relevant paper among ~18k. The pool for a question is the union
of the top-k papers of ALL compared systems, all graded by the same judge. Recall is then
"share of the relevant papers that ANY system found", which is a lower bound on the true
denominator but fair across the systems being compared (the standard TREC approach).

Metrics per question (relevant = judge score >= 4; graded gain = score - 1, i.e. 0..4):
  P@10, P@50       share of the top 10 / 50 that are relevant
  R@10, R@50       share of the pool's relevant papers found in the top 10 / 50
  nDCG@10, @50     graded ranking quality vs. the ideal ordering of the pool (1.0 = perfect)
  MRR              1 / rank of the first relevant paper
  unique_rel       relevant papers ONLY this run found (what it adds over the others)

Writes eval/comparisons/<timestamp>/comparison.json and comparison.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
RELEVANT_AT = 4
METRICS = ["P@10", "P@50", "R@10", "R@50", "nDCG@10", "nDCG@50", "MRR", "unique_rel"]


def gain(score: int) -> int:
    return max(score - 1, 0)


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def load_run(run_dir: Path, judge_model: str) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    jpath = run_dir / f"judgments_{judge_model}.jsonl"
    if not jpath.exists():
        sys.exit(f"{run_dir.name} has no {jpath.name}; run: python scripts/llm_judge.py {run_dir}")
    ranked = defaultdict(list)  # question text -> [(rank, paper_id, score)]
    for line in jpath.open(encoding="utf-8"):
        r = json.loads(line)
        q = meta["queries"][r["query_id"]]
        ranked[q].append((r["rank"], r["paper_id"], r["judge_score"]))
    for q in ranked:
        ranked[q].sort()
    label = meta["system"]
    return {"dir": run_dir, "meta": meta, "label": label, "ranked": dict(ranked)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--judge-model", default="gpt-5.6-luna")
    args = ap.parse_args()
    if len(args.runs) < 2:
        sys.exit("Give at least two runs to compare.")

    runs = [load_run(r.resolve(), args.judge_model) for r in args.runs]
    labels = [r["label"] for r in runs]
    if len(set(labels)) < len(labels):  # same system twice: fall back to run ids
        for r in runs:
            r["label"] = r["meta"]["run_id"]

    # Questions every run answered, matched by TEXT (ids could be renumbered)
    questions = set.intersection(*(set(r["ranked"]) for r in runs))
    qid_of = {q: qid for qid, q in runs[0]["meta"]["queries"].items()}
    questions = sorted(questions, key=lambda q: int(qid_of[q][1:]) if qid_of[q][1:].isdigit() else 0)

    # ---- the pool: every (question, paper) graded in any run --------------------------
    pool: dict[str, dict[str, int]] = {}
    for q in questions:
        grades = {}
        for r in runs:
            for _, pid, s in r["ranked"][q]:
                if s is not None:
                    grades[pid] = s
        pool[q] = grades

    # ---- metrics -------------------------------------------------------------------------
    per_q = {r["label"]: {} for r in runs}
    for q in questions:
        relevant = {p for p, s in pool[q].items() if s >= RELEVANT_AT}
        ideal = sorted((gain(s) for s in pool[q].values()), reverse=True)
        found_by = defaultdict(set)
        for r in runs:
            for _, pid, s in r["ranked"][q]:
                if pid in relevant:
                    found_by[pid].add(r["label"])
        for r in runs:
            ids = [pid for _, pid, _ in r["ranked"][q]]
            scores = [pool[q].get(pid, 0) for pid in ids]
            rel = [pid in relevant for pid in ids]
            first = next((i + 1 for i, x in enumerate(rel) if x), None)
            m = {}
            for n in (10, 50):
                m[f"P@{n}"] = sum(rel[:n]) / min(n, len(rel)) if rel else 0.0
                m[f"R@{n}"] = sum(rel[:n]) / len(relevant) if relevant else None
                idcg = dcg(ideal[:n])
                m[f"nDCG@{n}"] = dcg([gain(s) for s in scores[:n]]) / idcg if idcg else None
            m["MRR"] = 1 / first if first else 0.0
            m["unique_rel"] = sum(1 for pid in ids if pid in relevant and found_by[pid] == {r["label"]})
            m["pool_relevant"] = len(relevant)
            per_q[r["label"]][qid_of[q]] = m

    def macro(label, metric):
        v = [m[metric] for m in per_q[label].values() if m[metric] is not None]
        return sum(v) / len(v) if v else None

    summary = {r["label"]: {m: macro(r["label"], m) for m in METRICS} for r in runs}
    for r in runs:
        summary[r["label"]]["unique_rel"] = sum(m["unique_rel"] for m in per_q[r["label"]].values())

    # ---- output --------------------------------------------------------------------------
    out_dir = EVAL_DIR / "comparisons" / f"{datetime.now():%Y%m%d-%H%M%S}-{'-vs-'.join(r['label'] for r in runs)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_sizes = {qid_of[q]: {"judged": len(pool[q]), "relevant": sum(s >= RELEVANT_AT for s in pool[q].values())}
                  for q in questions}
    result = {"runs": {r["label"]: r["meta"]["run_id"] for r in runs}, "judge_model": args.judge_model,
              "relevant_threshold": RELEVANT_AT, "gain": "score - 1", "pool": pool_sizes,
              "summary": summary, "per_query": per_q}
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))

    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = [f"# Run comparison ({args.judge_model} judge, relevant = score >= {RELEVANT_AT})", "",
          "Runs: " + ", ".join(f"`{l}` = `{rid}`" for l, rid in result["runs"].items()), "",
          "## Macro-average over questions (unique_rel = total)", "",
          "| system | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for l in summary:
        md.append(f"| {l} | " + " | ".join(fmt(summary[l][m]) for m in METRICS) + " |")
    md += ["", "## Per question: R@50 / nDCG@50 (pool rel = relevant papers found by any system)", "",
           "| q | pool rel | " + " | ".join(summary) + " |", "|---|---|" + "---|" * len(summary)]
    for q in questions:
        qid = qid_of[q]
        md.append(f"| {qid} | {pool_sizes[qid]['relevant']} | " + " | ".join(
            f"{fmt(per_q[l][qid]['R@50'])} / {fmt(per_q[l][qid]['nDCG@50'])}" for l in summary) + " |")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
