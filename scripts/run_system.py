#!/usr/bin/env python3
"""Run the system on every eval question and save what it returned. No judging happens here.

    python scripts/run_system.py                         # BM25, top 50, all of eval/queries.jsonl
    python scripts/run_system.py --system hybrid -k 50      # bm25 | dense | hybrid | aspects | hyde
    python scripts/run_system.py --system bm25 -k 50 --queries q1 q3

Writes eval/runs/<run_id>/:
    results.jsonl   one line per (query, rank): query_id, rank, paper_id, score
    run_meta.json   which system/version/settings/corpus produced the results
Then grade it with:  python scripts/llm_judge.py eval/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402
from autoresearch.corpus import Corpus  # noqa: E402
from autoresearch.systems import SYSTEMS, get_system  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"


def load_queries(path: Path, only: list[str] | None) -> list[dict]:
    qs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return [q for q in qs if not only or q["id"] in only]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="bm25")
    ap.add_argument("-k", type=int, default=50)
    ap.add_argument("--queries-file", type=Path, default=EVAL_DIR / "queries.jsonl")
    ap.add_argument("--queries", nargs="+", help="query ids to run (default: all)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="don't narrate the agent loop (verify/agent systems narrate by default)")
    args = ap.parse_args()

    queries = load_queries(args.queries_file, args.queries)
    if not queries:
        sys.exit("No queries selected.")
    corpus_size = len(Corpus.load())
    system = get_system(args.system, verbose=not args.quiet)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.system}-k{args.k}"
    out_dir = EVAL_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = {}
    llm_total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"{args.system} k={args.k}", unit="query"):
            t = time.perf_counter()
            hits, diag = system(q["question"], args.k)
            latency = round(time.perf_counter() - t, 4)
            trace = diag.pop("trace", None)
            if trace is not None:  # v3: every planner/verifier/reviser exchange, one line each
                with (out_dir / "trace.jsonl").open("a", encoding="utf-8") as tf:
                    for ev in trace:
                        tf.write(json.dumps({"query_id": q["id"], **ev}, ensure_ascii=False) + "\n")
            per_query[q["id"]] = {"n_results": len(hits), "latency_s": latency, **diag}
            for key in llm_total:  # systems that call an LLM report its usage per question
                llm_total[key] += diag.get("llm_usage", {}).get(key, 0)
            for h in hits:
                row = {"query_id": q["id"], "rank": h.rank, "paper_id": h.paper_id,
                       "score": round(float(h.score), 6)}
                if getattr(h, "component_ranks", None):  # hybrid: rank in each retriever
                    row["component_ranks"] = h.component_ranks
                f.write(json.dumps(row) + "\n")
            matched = f" · {diag['n_matched']:,} of {diag['n_docs']:,} matched" if "n_matched" in diag else ""
            if "n_aspects" in diag:
                matched += (f" · {diag['n_aspects']} aspects"
                            f"{' (cached plan)' if diag['plan_cached'] else ''}"
                            f" · coverage {list(diag['final_coverage_by_aspect'].values())}"
                            f" · ${diag['llm_usage']['cost_usd']:.4f}")
            if "n_rounds" in diag:  # v3: verifiers (+ loop)
                matched += (f" · {diag['n_rounds']} round(s), stop: {diag['stop_reason']}"
                            f" · verified relevant {diag['n_verified_relevant']}"
                            f" · top-{args.k} tiers {diag['final_tiers']}")
            if "overlap_at_depth" in diag:
                matched += (f" · top-{diag['depth']} overlap {diag['overlap_at_depth']} · fused: "
                            f"{diag['fused_from_both']} both / {diag['fused_bm25_only']} bm25 / "
                            f"{diag['fused_dense_only']} dense")
            tqdm.write(f"{q['id']}: {len(hits)} results in {latency * 1000:.0f} ms{matched}")

    meta = {
        "run_id": run_id, "system": args.system, "k": args.k,
        "queries_file": str(args.queries_file.relative_to(PROJECT_ROOT)),
        "queries": {q["id"]: q["question"] for q in queries},
        "corpus_size": corpus_size, "llm_usage_total": {**llm_total, "cost_usd": round(llm_total["cost_usd"], 6)},
        "per_query": per_query,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if llm_total["calls"]:
        print(f"\nSystem LLM usage: {llm_total['calls']} calls · ${llm_total['cost_usd']:.4f}")
    print(f"\nSaved {out_dir}\nNext: python scripts/llm_judge.py {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
