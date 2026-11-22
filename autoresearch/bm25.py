"""BM25 keyword search over title + abstract, backed by the `bm25s` library.

Tokenization (BM25 only — dense search will see raw text):
  plural acronyms -> singular (LLMs -> LLM) -> lowercase -> stopword removal
  (bm25s list + extra function words) -> Snowball (Porter2) stemming
so "agents" and "agent" match. Stemming is not perfect ("agentic" stays "agent"),
so check `explain_query` when a keyword query behaves oddly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from . import INDEX_DIR
from .corpus import Corpus, bm25_text

log = logging.getLogger(__name__)

BM25_DIR = INDEX_DIR / "bm25"
_stemmer = Stemmer.Stemmer("english")

# Bump whenever tokenization changes: the index stores it, and loading an index built with a
# different version fails loudly instead of silently mis-matching query and document tokens.
TOKENIZER_VERSION = "v2"

# bm25s' default English list (33 words, from Lucene) misses common question words, pronouns
# and auxiliaries ("how", "we", "do", "what", "from", "can"), which then match ~every paper.
# Extended with grammatical function words only; no topical words.
EXTRA_STOPWORDS = """
how what which where when why who whom whose whether
do does did doing done have has had having been being were am
can could would should may might must shall
we us our ours ourselves you your yours i me my mine he him his she her hers
its itself them themselves those
from about between among through during before after above below over under
against within without upon onto toward towards via per across along around
than so too very just also only both either neither nor each every any all some
most more much many few other another same own here
""".split()
STOPWORDS = sorted(set(bm25s.stopwords.STOPWORDS_EN) | set(EXTRA_STOPWORDS))

# "LLMs" -> "LLM", "VLMs" -> "VLM", "GNNs" -> "GNN": the stemmer leaves "llms" alone, so plural
# acronyms would never match their singular. Applied before lowercasing (needs the capitals).
_PLURAL_ACRONYM = re.compile(r"\b([A-Z]{2,})s\b")


def normalize(text: str) -> str:
    return _PLURAL_ACRONYM.sub(r"\1", text)


def tokenize(texts: list[str]):
    return bm25s.tokenize([normalize(t) for t in texts], lower=True, stopwords=STOPWORDS,
                          stemmer=_stemmer, show_progress=False)


def explain_query(query: str) -> list[str]:
    """The tokens BM25 actually matches on — useful for debugging keyword queries."""
    tok = tokenize([query])
    inv = {i: t for t, i in tok.vocab.items()}
    return [inv[i] for i in tok.ids[0]]


@dataclass
class Hit:
    paper_id: str
    score: float
    rank: int  # 1-based


@dataclass
class SearchResult:
    hits: list[Hit]
    query_tokens: list[str]       # what BM25 actually matched on
    n_matched: int                # docs with score > 0 (contain at least one query token)
    n_docs: int                   # corpus size
    term_doc_freq: dict[str, int]  # per query token: how many docs contain it


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, paper_ids: list[str]):
        self.retriever = retriever
        self.paper_ids = paper_ids

    # ---- build / persist ---------------------------------------------------------
    @classmethod
    def build(cls, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        log.info("Tokenizing %d documents...", len(corpus))
        tokens = tokenize([bm25_text(p) for p in corpus.papers])
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(tokens, show_progress=False)
        return cls(retriever, corpus.paper_ids)

    def save(self, path: Path = BM25_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path))
        (path / "paper_ids.json").write_text(json.dumps(self.paper_ids))
        (path / "tokenizer.json").write_text(json.dumps({"version": TOKENIZER_VERSION}))
        log.info("Saved BM25 index to %s", path)

    @classmethod
    def load(cls, path: Path = BM25_DIR) -> "BM25Index":
        if not (path / "paper_ids.json").exists():
            raise FileNotFoundError(f"No BM25 index at {path}; run `python -m autoresearch build-index`")
        tok_file = path / "tokenizer.json"
        built_with = json.loads(tok_file.read_text())["version"] if tok_file.exists() else "v1"
        if built_with != TOKENIZER_VERSION:
            raise RuntimeError(f"BM25 index was built with tokenizer {built_with}, code uses "
                               f"{TOKENIZER_VERSION}. Rebuild: python -m autoresearch build-index")
        retriever = bm25s.BM25.load(str(path))
        paper_ids = json.loads((path / "paper_ids.json").read_text())
        return cls(retriever, paper_ids)

    # ---- query -------------------------------------------------------------------
    def search(self, query: str, k: int = 50) -> SearchResult:
        tokens = explain_query(query)
        n_docs = len(self.paper_ids)
        if not tokens:  # query was all stopwords
            return SearchResult([], [], 0, n_docs, {})

        # Score every document (cheap at this corpus size), so we also learn how many
        # documents match at all — a measure of how broad the query is.
        scores = self.retriever.get_scores(tokens)
        n_matched = int((scores > 0).sum())

        # Top-k: argpartition is O(n), then sort only the k winners.
        k = min(k, n_matched)
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-scores[top], kind="stable")]
        hits = [Hit(self.paper_ids[i], float(scores[i]), r) for r, i in enumerate(top, start=1)]

        vocab = self.retriever.vocab_dict
        term_df = {
            t: int((self.retriever.get_scores([t]) > 0).sum()) if t in vocab else 0
            for t in dict.fromkeys(tokens)  # unique, order kept
        }
        return SearchResult(hits, tokens, n_matched, n_docs, term_df)
