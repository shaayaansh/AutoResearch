"""Command-line entry point.

    python -m autoresearch build-index
    python -m autoresearch search "structural similarity between stories" -k 20
    python -m autoresearch search "..." --explain     # show the BM25 tokens
"""

from __future__ import annotations

import argparse
import logging
import textwrap
import time

from .bm25 import BM25Index, explain_query
from .corpus import Corpus
from .tracing import Tracer


def cmd_build_index(args) -> None:
    corpus = Corpus.load()
    t = time.perf_counter()
    BM25Index.build(corpus).save()
    print(f"Indexed {len(corpus)} papers in {time.perf_counter() - t:.1f}s")


def cmd_search(args) -> None:
    corpus = Corpus.load()
    index = BM25Index.load()
    tracer = Tracer("search")

    if args.explain:
        print("BM25 tokens:", explain_query(args.query), "\n")

    t = time.perf_counter()
    res = index.search(args.query, k=args.k)
    latency = time.perf_counter() - t
    hits = res.hits
    tracer.log("search", retriever="bm25", query=args.query, k=args.k,
               query_tokens=res.query_tokens, n_matched=res.n_matched, n_docs=res.n_docs,
               term_doc_freq=res.term_doc_freq, latency_s=round(latency, 4),
               hits=[{"paper_id": h.paper_id, "rank": h.rank, "score": round(h.score, 3)} for h in hits])

    for h in hits:
        p = corpus.get(h.paper_id)
        print(f"{h.rank:>3}. [{h.score:6.2f}] {p['title']}")
        print(f"     {p['venue']} {p['year']} ({p['track']})  {p['url']}")
        if args.abstracts:
            print(textwrap.indent(textwrap.fill(p["abstract"], 100), "     "))
        print()
    print(f"{len(hits)} shown · {res.n_matched:,} of {res.n_docs:,} papers matched at least one term "
          f"· {latency * 1000:.0f} ms")
    print("papers containing each term:", res.term_doc_freq)
    print(f"trace: {tracer.path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="autoresearch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-index", help="build the BM25 index from data/raw").set_defaults(fn=cmd_build_index)

    s = sub.add_parser("search", help="keyword search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=20)
    s.add_argument("--abstracts", action="store_true", help="print abstracts too")
    s.add_argument("--explain", action="store_true", help="show the tokens BM25 matches on")
    s.set_defaults(fn=cmd_search)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
