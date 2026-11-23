"""Read the agent's conversation for one question: planner -> verifiers -> reviser -> ...

    python scripts/show_trace.py eval/runs/<agent-run-id> --query q5
    python scripts/show_trace.py eval/runs/<agent-run-id> --query q5 --round 1 --role verifier
    python scripts/show_trace.py eval/runs/<agent-run-id> --query q5 --full   # exact prompts too

trace.jsonl (written by run_system.py for the verify/agent systems) holds every LLM exchange:
  planner   input = the question,              output = aspects + queries
  verifier  input = a numbered batch of papers, output = verdicts + failure_pattern + missing
            (cached=true: verdicts reused from an earlier run, no LLM call)
  reviser   input = coverage + notes per aspect (exactly what the planner read),
            output = reasoning, done, updated queries, new aspects
The fixed system prompts are in autoresearch/verifier.py and autoresearch/planner.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoresearch.corpus import Corpus  # noqa: E402

W = 100


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(line, W, initial_indent=indent, subsequent_indent=indent) if line else ""
                     for line in text.splitlines())


def titles_from_batch(text: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"^\[(\d+)\] (.+)$", text, re.M)}


def show(ev: dict, corpus: Corpus, full: bool) -> None:
    role, tag = ev["role"], " (cached)" if ev.get("cached") else ""
    if role == "planner":
        print(f"\n=== PLANNER{tag} ===")
        if full:
            print(wrap(ev["input"]))
        for i, a in enumerate(ev["output"]["aspects"], start=1):
            print(f"  a{i} {a['name']}\n{wrap(a['description'])}\n    bm25: {a['bm25_query']}")
            if full:
                print(wrap("hyde: " + a["hyde_abstract"]))
    elif role == "verifier":
        if "error" in ev:
            print(f"\n--- verifier {ev['aspect_id']}: FAILED {ev['error']} ({len(ev['paper_ids'])} papers unverified)")
            return
        if ev.get("cached"):
            print(f"\n--- verifier {ev['aspect_id']}{tag}: {len(ev['verdicts'])} verdicts")
            for v in ev["verdicts"]:
                t = corpus.get(v["paper_id"])["title"]
                print(f"  {'✓' if v['relevant'] else '✗'} {t[:70]:70}  {v['reason']}")
            return
        out = ev["output"]
        n_rel = sum(v["relevant"] for v in out["verdicts"])
        print(f"\n--- verifier {ev['aspect_id']}: batch of {len(ev['paper_ids'])}, {n_rel} relevant")
        if full:
            print(wrap(ev["input"]))
        titles = titles_from_batch(ev["input"])
        for v in out["verdicts"]:
            print(f"  {'✓' if v['relevant'] else '✗'} [{v['paper']}] {titles.get(v['paper'], '?')[:66]:66}  {v['reason']}")
        print(wrap("why not: " + out["failure_pattern"]))
        print(wrap("missing: " + out["missing"]))
    elif role == "reviser":
        out = ev["output"]
        print(f"\n=== REVISER{tag} — what the planner read ===")
        print(wrap(ev["input"]))
        print(f"\n=== REVISER — what it decided (done={out['done']}) ===")
        print(wrap(out["reasoning"]))
        for u in out["updates"]:
            print(f"  update {u['aspect_id']}: {u['reason']}\n    bm25: {u['bm25_query']}")
            if full:
                print(wrap("hyde: " + u["hyde_abstract"]))
        for a in out["new_aspects"]:
            print(f"  NEW aspect {a['name']}: {a['description']}\n    bm25: {a['bm25_query']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--query", required=True, help="query id, e.g. q5")
    ap.add_argument("--round", type=int, help="only this round (0 = planning)")
    ap.add_argument("--role", choices=["planner", "verifier", "reviser"])
    ap.add_argument("--full", action="store_true", help="also print the exact inputs and HyDE abstracts")
    args = ap.parse_args()

    path = args.run_dir / "trace.jsonl"
    if not path.exists():
        sys.exit(f"No {path}: only the verify/agent systems write a trace.")
    events = [e for e in map(json.loads, path.open(encoding="utf-8")) if e["query_id"] == args.query]
    if not events:
        sys.exit(f"No trace for {args.query} in {path}")
    corpus = Corpus.load()
    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    print(f"{args.query}: {meta['queries'][args.query]}")
    last = None
    for ev in events:
        if args.round is not None and ev["round"] != args.round:
            continue
        if args.role and ev["role"] != args.role:
            continue
        if ev["round"] != last and ev["round"] > 0:
            print(f"\n################ ROUND {ev['round']} ################")
        last = ev["round"]
        show(ev, corpus, args.full)
    d = meta["per_query"][args.query]
    print(f"\nstop: {d['stop_reason']} after {d['n_rounds']} round(s) · tiers in top-{meta['k']}: "
          f"{d['final_tiers']} · ${d['llm_usage']['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
