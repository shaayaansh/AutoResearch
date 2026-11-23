#!/usr/bin/env python3
"""Compare judged runs on their own top-k only. No pooling, no recall.

    python scripts/compare_topk.py eval/runs/<google_scholar> eval/runs/<bm25> eval/runs/<hybrid> eval/runs/<hyde> -k 10

Every system is scored only on the first k papers it returned, using the grades the same LLM
judge gave them (judgments_<model>.jsonl, produced by scripts/llm_judge.py). Runs with k > 10
(our systems use k = 50) are simply cut at k.

Metrics per question (relevant = judge score >= 4; scores 1-5):
  P@5, P@k      share of the top 5 / top k that are relevant
  mean@k        average judge score of the top k
  MRR           1 / rank of the first relevant paper (0 if none in the top k)
  nDCG@k        graded ranking quality against an absolute ideal (every slot a 5);
                gain = score - 1, log2 discount. Needs no pool: it only uses this run's own grades.

Writes eval/comparisons/<timestamp>-top<k>-<systems>/comparison.md and .json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
RELEVANT_AT = 4
METRICS = ["P@5", "P@k", "mean@k", "MRR", "nDCG@k"]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def load(run_dir: Path, judge_model: str, k: int) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    jpath = run_dir / f"judgments_{judge_model}.jsonl"
    if not jpath.exists():
        sys.exit(f"{run_dir.name} has no {jpath.name}; run: python scripts/llm_judge.py {run_dir}")
    by_q: dict[str, list] = {}
    for line in jpath.open(encoding="utf-8"):
        r = json.loads(line)
        if r["rank"] <= k:
            by_q.setdefault(r["query_id"], []).append((r["rank"], r["judge_score"], r["title"]))
    for q in by_q:
        by_q[q].sort()
    return {"label": meta["system"], "run_id": meta["run_id"], "by_q": by_q}


def metrics(ranked: list, k: int) -> dict:
    scores = [s for _, s, _ in ranked[:k]]
    if any(s is None for s in scores):
        raise ValueError("ungraded paper in the top k (judge error?) - re-run llm_judge.py")
    rel = [s >= RELEVANT_AT for s in scores]
    first = next((i + 1 for i, x in enumerate(rel) if x), None)
    return {
        "P@5": sum(rel[:5]) / min(5, len(rel)) if rel else 0.0,
        "P@k": sum(rel) / len(rel) if rel else 0.0,
        "mean@k": sum(scores) / len(scores) if scores else 0.0,
        "MRR": 1 / first if first else 0.0,
        "nDCG@k": dcg([s - 1 for s in scores]) / dcg([4] * k),
        "n": len(scores),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--judge-model", default="gpt-5.6-luna")
    args = ap.parse_args()

    runs = [load(r.resolve(), args.judge_model, args.k) for r in args.runs]
    labels = [r["label"] for r in runs]
    if len(set(labels)) < len(labels):
        for r in runs:
            r["label"] = r["run_id"]
    qids = sorted(set.intersection(*(set(r["by_q"]) for r in runs)),
                  key=lambda q: int(q[1:]) if q[1:].isdigit() else 0)
    if not qids:
        sys.exit("No questions in common across these runs.")

    per_q = {r["label"]: {q: metrics(r["by_q"][q], args.k) for q in qids} for r in runs}
    summary = {l: {m: sum(per_q[l][q][m] for q in qids) / len(qids) for m in METRICS} for l in per_q}
    short = {l: [q for q in qids if per_q[l][q]["n"] < args.k] for l in per_q}

    k = args.k
    name = lambda m: m.replace("@k", f"@{k}")
    md = [f"# Top-{k} comparison ({args.judge_model} judge, relevant = score >= {RELEVANT_AT})", "",
          f"Questions: {len(qids)} ({', '.join(qids)}). Each system is scored on its own top {k} only.", "",
          "Runs: " + ", ".join(f"`{r['label']}` = `{r['run_id']}`" for r in runs), "",
          "## Macro-average over questions", "",
          "| system | " + " | ".join(name(m) for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for l in sorted(summary, key=lambda l: -summary[l]["nDCG@k"]):
        md.append(f"| {l} | " + " | ".join(f"{summary[l][m]:.3f}" for m in METRICS) + " |")
    md += ["", f"## Per question: P@{k} / mean score@{k}", "",
           "| q | " + " | ".join(per_q) + " |", "|---|" + "---|" * len(per_q)]
    for q in qids:
        md.append(f"| {q} | " + " | ".join(f"{per_q[l][q]['P@k']:.1f} / {per_q[l][q]['mean@k']:.2f}"
                                          for l in per_q) + " |")
    if any(short.values()):
        md += ["", "Note: fewer than k results for " +
               "; ".join(f"{l}: {', '.join(v)}" for l, v in short.items() if v)]

    out_dir = EVAL_DIR / "comparisons" / f"{datetime.now():%Y%m%d-%H%M%S}-top{k}-{'-vs-'.join(per_q)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n")
    (out_dir / "comparison.json").write_text(json.dumps(
        {"k": k, "judge_model": args.judge_model, "relevant_threshold": RELEVANT_AT,
         "runs": {r["label"]: r["run_id"] for r in runs}, "questions": qids,
         "summary": summary, "per_query": per_q}, indent=2))
    print("\n".join(md))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
