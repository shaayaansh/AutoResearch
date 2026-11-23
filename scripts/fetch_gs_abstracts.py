#!/usr/bin/env python3
"""Add abstracts to the manually collected Google Scholar results, so the LLM judge can grade
them exactly like our own systems' papers (title + abstract).

    python scripts/fetch_gs_abstracts.py                  # all eval/manual/google_scholar/q*.jsonl
    python scripts/fetch_gs_abstracts.py --only q4 q10
    OPENALEX_EMAIL=you@x.com python scripts/fetch_gs_abstracts.py   # optional: OpenAlex "polite pool"

For each result without an abstract:
  1. If its title matches a paper in our corpus -> take the abstract from the corpus
     (and record paper_id, so the judgment cache is reused).
  2. Otherwise search OpenAlex (free, official API, no key) by title, accept the best candidate
     only if its title matches (normalized similarity >= 0.92, or a prefix match when Google
     Scholar truncated the title with "…"), and rebuild the abstract from OpenAlex's
     inverted index.
  3. Still missing -> search arXiv (free, official API; full abstracts for every preprint) by title,
     with the same strict title matching. Many journal papers have an arXiv version whose abstract
     OpenAlex can't show (publisher restrictions).
  4. If nothing matches, the row is flagged (abstract_source = "missing") and listed at the end
     so you can paste those few abstracts by hand.

Rows are updated in place; rows that already have an abstract are skipped, so re-running is safe.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import PROJECT_ROOT  # noqa: E402
from autoresearch.corpus import Corpus  # noqa: E402

GS_DIR = PROJECT_ROOT / "eval" / "manual" / "google_scholar"
OPENALEX = "https://api.openalex.org/works"
MIN_SIM = 0.92


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (t or "").lower())).strip()


def title_match(gs_title: str, cand: str, truncated: bool) -> float:
    a, b = norm(gs_title.rstrip("…")), norm(cand)
    if not a or not b:
        return 0.0
    if truncated and b.startswith(a) and len(a) >= 30:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def rebuild_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def openalex_search(title: str, email: str | None) -> list[dict]:
    params = {"search": title.rstrip("…"), "per-page": 10,
              "select": "id,display_name,abstract_inverted_index,publication_year,primary_location,doi"}
    if email:
        params["mailto"] = email
    url = f"{OPENALEX}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AutoResearch/0.1"}),
                                        timeout=30) as r:
                return json.load(r).get("results", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
    return []


ARXIV = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(title: str) -> list[dict]:
    words = re.findall(r"[A-Za-z0-9]+", title.rstrip("…"))[:12]  # arXiv search syntax: ti:word AND ...
    query = " AND ".join(f"ti:{w}" for w in words)
    url = f"{ARXIV}?{urllib.parse.urlencode({'search_query': query, 'max_results': 10})}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AutoResearch/0.1"}),
                                        timeout=30) as r:
                root = ET.fromstring(r.read())
            out = []
            for e in root.findall("a:entry", ATOM):
                t = " ".join((e.findtext("a:title", "", ATOM) or "").split())
                a = " ".join((e.findtext("a:summary", "", ATOM) or "").split())
                out.append({"title": t, "abstract": a, "id": e.findtext("a:id", "", ATOM),
                            "year": (e.findtext("a:published", "", ATOM) or "")[:4]})
            return out
        except (urllib.error.HTTPError, urllib.error.URLError, ET.ParseError):
            time.sleep(3 * (attempt + 1))
    return []


def venue_of(work: dict) -> str | None:
    src = ((work.get("primary_location") or {}).get("source") or {})
    return src.get("display_name")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="query ids, e.g. q4 q10 (default: all files)")
    ap.add_argument("--min-sim", type=float, default=MIN_SIM)
    args = ap.parse_args()

    files = sorted(GS_DIR.glob("q*.jsonl"), key=lambda p: int(p.stem[1:]))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    if not files:
        sys.exit(f"No files in {GS_DIR}")

    corpus = Corpus.load()
    by_title = {}
    for p in corpus.papers:
        by_title.setdefault(norm(p["title"]), p)
    email = os.getenv("OPENALEX_EMAIL")

    stats = {"corpus": 0, "openalex": 0, "arxiv": 0, "missing": 0, "skipped": 0}
    missing = []
    for f in files:
        rows = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
        for r in rows:
            if r.get("abstract"):
                stats["skipped"] += 1
                continue
            truncated = r.get("title_truncated", r["title"].endswith("…"))
            # 1) our corpus
            p = by_title.get(norm(r["title"]))
            if p is None and truncated:
                key = norm(r["title"].rstrip("…"))
                p = next((v for k, v in by_title.items() if len(key) >= 30 and k.startswith(key)), None)
            if p is not None:
                r.update(in_corpus=True, paper_id=p["id"], abstract=p.get("abstract", ""),
                         abstract_source="corpus", matched_title=p["title"],
                         venue=f"{p['venue']} {p['year']}")
                stats["corpus"] += 1
                continue
            # 2) OpenAlex
            r["in_corpus"] = False
            try:
                cands = openalex_search(r["title"], email)
            except Exception as e:  # network problem: leave the row untouched, report it
                print(f"  {f.stem} #{r['rank']}: OpenAlex error {e}")
                cands = []
            time.sleep(0.15)  # well under OpenAlex's rate limit
            scored = sorted(((title_match(r["title"], c.get("display_name", ""), truncated), c)
                             for c in cands), key=lambda x: -x[0])
            best = next(((s, c) for s, c in scored if s >= args.min_sim
                         and rebuild_abstract(c.get("abstract_inverted_index"))), None)
            if best:
                s, c = best
                for k in ("closest_openalex_title", "closest_score", "closest_had_abstract"):
                    r.pop(k, None)
                r.update(abstract=rebuild_abstract(c["abstract_inverted_index"]), abstract_source="openalex",
                         openalex_id=c.get("id"), matched_title=c.get("display_name"),
                         match_score=round(s, 3), venue=venue_of(c), year=c.get("publication_year"),
                         doi=c.get("doi"))
                stats["openalex"] += 1
            else:
                # 3) arXiv fallback
                ax = arxiv_search(r["title"])
                time.sleep(3)  # arXiv asks for at most one request every 3 seconds
                ax_scored = sorted(((title_match(r["title"], c["title"], truncated), c) for c in ax),
                                   key=lambda x: -x[0])
                if ax_scored and ax_scored[0][0] >= args.min_sim and ax_scored[0][1]["abstract"]:
                    s, c = ax_scored[0]
                    r.update(abstract=c["abstract"], abstract_source="arxiv", arxiv_id=c["id"],
                             matched_title=c["title"], match_score=round(s, 3), year=c["year"] or None)
                    for k in ("closest_openalex_title", "closest_score", "closest_had_abstract"):
                        r.pop(k, None)
                    stats["arxiv"] += 1
                    continue
                top = scored[0] if scored else (0.0, {})
                r.update(abstract="", abstract_source="missing",
                         closest_openalex_title=top[1].get("display_name"),
                         closest_score=round(top[0], 3),
                         closest_had_abstract=bool(top[1].get("abstract_inverted_index")))
                stats["missing"] += 1
                missing.append(f"{f.stem} #{r['rank']}: {r['title']}")
        tmp = f.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(f)
        n_ok = sum(bool(r.get("abstract")) for r in rows)
        print(f"{f.stem}: {n_ok}/{len(rows)} with abstract")

    print(f"\nFrom corpus: {stats['corpus']} · from OpenAlex: {stats['openalex']} · "
          f"from arXiv: {stats['arxiv']} · "
          f"missing: {stats['missing']} · already had one: {stats['skipped']}")
    if missing:
        print("\nNo abstract found (paste these by hand into the 'abstract' field, "
              "and set \"abstract_source\": \"manual\"):")
        print("\n".join("  " + m for m in missing))


if __name__ == "__main__":
    main()
