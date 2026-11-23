#!/usr/bin/env python3
"""Turn the manually collected Google Scholar results into a normal run folder, so the same
LLM judge can grade them.

    python scripts/import_google_scholar.py
    python scripts/llm_judge.py eval/runs/<run_id>          # printed at the end

Reads  eval/manual/google_scholar/q*.jsonl   (top 10 per question, with abstracts; see
                                              scripts/fetch_gs_abstracts.py)
Writes eval/runs/<timestamp>-google_scholar-k10/results.jsonl + run_meta.json

- Results that are in our corpus keep their corpus paper_id, so the judgments already made for
  our own systems are reused (same question + same paper = same cached grade).
- Results outside the corpus get a stable id ("ext:" + hash of the normalized title) and carry
  their title / abstract / venue inline; llm_judge.py grades them from that, with the same
  prompt, rubric and model as every other paper.
- Google Scholar is NOT restricted to our corpus: all 10 results count. This compares what a
  researcher actually gets from each tool's top 10.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
GS_DIR = EVAL_DIR / "manual" / "google_scholar"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (t or "").lower())).strip()


def venue_from_source_line(line: str) -> str:
    """Google Scholar's green line is 'Authors - Venue, Year - host'; take the middle part."""
    parts = line.split(" - ")
    if len(parts) < 3:
        return ""
    return re.sub(r",?\s*\d{4}\s*$", "", parts[1]).strip(" ,…")


def ext_id(title: str) -> str:
    return "ext:" + hashlib.sha1(norm(title.rstrip("…")).encode()).hexdigest()[:12]


def main():
    queries = {json.loads(l)["id"]: json.loads(l)["question"]
               for l in (EVAL_DIR / "queries.jsonl").open(encoding="utf-8") if l.strip()}
    files = sorted(GS_DIR.glob("q*.jsonl"), key=lambda p: int(p.stem[1:]))
    if not files:
        sys.exit(f"No files in {GS_DIR}")

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-google_scholar-k10"
    out_dir = EVAL_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query, problems = {}, []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for f in files:
            rows = sorted((json.loads(l) for l in f.open(encoding="utf-8") if l.strip()),
                          key=lambda r: r["rank"])
            qid = f.stem
            if qid not in queries:
                problems.append(f"{qid}: not in eval/queries.jsonl, skipped")
                continue
            for r in rows:
                if not r.get("abstract"):
                    problems.append(f"{qid} #{r['rank']}: no abstract (run fetch_gs_abstracts.py or paste one)")
                row = {"query_id": qid, "rank": r["rank"], "score": round(1 / r["rank"], 6),
                       "in_corpus": bool(r.get("in_corpus")), "abstract_source": r.get("abstract_source")}
                if r.get("in_corpus") and r.get("paper_id"):
                    row["paper_id"] = r["paper_id"]
                else:
                    row["paper_id"] = ext_id(r["title"])
                    row["paper"] = {"title": r.get("matched_title") or r["title"],
                                    "abstract": r.get("abstract", ""),
                                    "venue": r.get("venue") or venue_from_source_line(r.get("source_line", "")),
                                    "year": r.get("year") or "",
                                    "url": r.get("doi") or r.get("openalex_id") or r.get("arxiv_id") or ""}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            per_query[qid] = {"n_results": len(rows),
                              "n_in_corpus": sum(bool(r.get("in_corpus")) for r in rows),
                              "abstract_sources": dict(Counter(r.get("abstract_source") for r in rows)),
                              "gs_filters": rows[0].get("gs_filters") if rows else None}
            print(f"{qid}: {len(rows)} results ({per_query[qid]['n_in_corpus']} in our corpus) · "
                  f"abstracts {per_query[qid]['abstract_sources']}")

    meta = {"run_id": run_id, "system": "google_scholar", "k": 10, "external": True,
            "collection": "manual: Google Scholar web UI, top 10 per question, exact question text",
            "queries": {q: queries[q] for q in per_query}, "per_query": per_query,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if problems:
        print("\nWarnings:\n  " + "\n  ".join(problems))
    print(f"\nSaved {out_dir}\nNext: python scripts/llm_judge.py {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
