"""Command-line entry point.

    python -m autoresearch build-index                 # BM25 + dense (dense: ~minutes, once)
    python -m autoresearch build-index --bm25          # only BM25 (seconds)
    python -m autoresearch build-index --dense         # only embeddings
    python -m autoresearch search "structural similarity between stories" -k 20
    python -m autoresearch search "..." --retriever dense
    python -m autoresearch search "..." --retriever bm25 --explain   # show BM25 tokens
"""

from __future__ import annotations

import argparse
import logging
import textwrap
import time

from .corpus import Corpus
from .systems import SYSTEMS, get_system
from .tracing import Tracer


def cmd_build_index(args) -> None:
    corpus = Corpus.load()
    both = not (args.bm25 or args.dense)
    if args.bm25 or both:
        from .bm25 import BM25Index

        t = time.perf_counter()
        BM25Index.build(corpus).save()
        print(f"BM25: indexed {len(corpus)} papers in {time.perf_counter() - t:.1f}s")
    if args.dense or both:
        from .dense import DenseIndex

        idx = DenseIndex.build(corpus, batch_size=args.batch_size, device=args.device)
        idx.save()
        print(f"Dense: embedded {len(corpus)} papers with {idx.meta['model']} on {idx.meta['device']} "
              f"in {idx.meta['build_seconds']}s")


def cmd_search(args) -> None:
    corpus = Corpus.load()
    search = get_system(args.retriever)
    tracer = Tracer("search")

    if args.explain:
        from .bm25 import explain_query

        print("BM25 tokens:", explain_query(args.query), "\n")

    t = time.perf_counter()
    hits, diag = search(args.query, args.k)
    latency = time.perf_counter() - t
    tracer.log("search", retriever=args.retriever, query=args.query, k=args.k,
               latency_s=round(latency, 4), diagnostics=diag,
               hits=[{"paper_id": h.paper_id, "rank": h.rank, "score": round(h.score, 4),
                      **({"component_ranks": h.component_ranks} if hasattr(h, "component_ranks") else {})}
                     for h in hits])

    for h in hits:
        p = corpus.get(h.paper_id)
        extra = f"  {h.component_ranks}" if hasattr(h, "component_ranks") else ""
        print(f"{h.rank:>3}. [{h.score:7.4f}] {p['title']}{extra}")
        print(f"     {p['venue']} {p['year']} ({p['track']})  {p['url']}")
        if args.abstracts:
            print(textwrap.indent(textwrap.fill(p["abstract"], 100), "     "))
        print()

    summary = f"{len(hits)} shown · {latency * 1000:.0f} ms"
    if "n_matched" in diag:
        summary += f" · {diag['n_matched']:,} of {diag['n_docs']:,} papers matched at least one term"
    if "overlap_at_depth" in diag:
        summary += (f" · BM25/dense overlap in top {diag['depth']}: {diag['overlap_at_depth']}"
                    f" · fused list: {diag['fused_from_both']} from both, {diag['fused_bm25_only']} BM25-only,"
                    f" {diag['fused_dense_only']} dense-only")
    print(summary)
    if "term_doc_freq" in diag:
        print("papers containing each term:", diag["term_doc_freq"])
    print(f"trace: {tracer.path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="autoresearch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="build search indexes from data/raw (default: all)")
    b.add_argument("--bm25", action="store_true", help="build only the BM25 index")
    b.add_argument("--dense", action="store_true", help="build only the dense index")
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")
    b.set_defaults(fn=cmd_build_index)

    s = sub.add_parser("search", help="search the corpus")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=20)
    s.add_argument("--retriever", choices=sorted(SYSTEMS), default="hybrid")
    s.add_argument("--abstracts", action="store_true", help="print abstracts too")
    s.add_argument("--explain", action="store_true", help="show the tokens BM25 matches on")
    s.set_defaults(fn=cmd_search)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
