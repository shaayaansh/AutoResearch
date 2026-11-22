#!/usr/bin/env python3
"""Run the system on every eval question and save what it returned. No judging happens here.

    python scripts/run_system.py                         # BM25, top 50, all of eval/queries.jsonl
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
from autoresearch.bm25 import TOKENIZER_VERSION, BM25Index  # noqa: E402
from autoresearch.corpus import Corpus  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"


def load_queries(path: Path, only: list[str] | None) -> list[dict]:
    qs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return [q for q in qs if not only or q["id"] in only]


# Each system: name -> function(question, k) -> (list of (paper_id, score), diagnostics dict).
# Diagnostics are whatever the system knows about how it got the results (saved in run_meta.json
# for later analysis). New versions plug in here.
def make_bm25():
    index = BM25Index.load()

    def run(question, k):
        res = index.search(question, k=k)
        diag = {"tokenizer_version": TOKENIZER_VERSION, "query_tokens": res.query_tokens,
                "n_matched": res.n_matched,
                "n_docs": res.n_docs, "term_doc_freq": res.term_doc_freq}
        return [(h.paper_id, h.score) for h in res.hits], diag

    return run


SYSTEMS = {"bm25": make_bm25}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="bm25")
    ap.add_argument("-k", type=int, default=50)
    ap.add_argument("--queries-file", type=Path, default=EVAL_DIR / "queries.jsonl")
    ap.add_argument("--queries", nargs="+", help="query ids to run (default: all)")
    args = ap.parse_args()

    queries = load_queries(args.queries_file, args.queries)
    if not queries:
        sys.exit("No queries selected.")
    corpus_size = len(Corpus.load())
    system = SYSTEMS[args.system]()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{args.system}-k{args.k}"
    out_dir = EVAL_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = {}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"{args.system} k={args.k}", unit="query"):
            t = time.perf_counter()
            hits, diag = system(q["question"], args.k)
            latency = round(time.perf_counter() - t, 4)
            per_query[q["id"]] = {"n_results": len(hits), "latency_s": latency, **diag}
            for rank, (pid, score) in enumerate(hits, start=1):
                f.write(json.dumps({"query_id": q["id"], "rank": rank, "paper_id": pid,
                                    "score": round(float(score), 4)}) + "\n")
            matched = f" · {diag['n_matched']:,} of {diag['n_docs']:,} matched" if "n_matched" in diag else ""
            tqdm.write(f"{q['id']}: {len(hits)} results in {latency * 1000:.0f} ms{matched}")

    meta = {
        "run_id": run_id, "system": args.system, "k": args.k,
        "queries_file": str(args.queries_file.relative_to(PROJECT_ROOT)),
        "queries": {q["id"]: q["question"] for q in queries},
        "corpus_size": corpus_size, "per_query": per_query,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_dir}\nNext: python scripts/llm_judge.py {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
