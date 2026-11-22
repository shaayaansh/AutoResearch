"""Load the unified JSONL files in data/raw into one in-memory corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import RAW_DIR

log = logging.getLogger(__name__)


class Corpus:
    """All papers, in a fixed order. A paper's position in `papers` is its doc index,
    which is what the search indexes store; `paper_ids` maps it back to a stable id."""

    def __init__(self, papers: list[dict]):
        self.papers = papers
        self.paper_ids = [p["id"] for p in papers]
        self._by_id = {p["id"]: p for p in papers}

    def __len__(self) -> int:
        return len(self.papers)

    def get(self, paper_id: str) -> dict:
        return self._by_id[paper_id]

    @classmethod
    def load(cls, raw_dir: Path = RAW_DIR) -> "Corpus":
        papers, seen = [], set()
        files = sorted(raw_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files in {raw_dir}; run scripts/fetch_proceedings.py")
        for f in files:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    p = json.loads(line)
                    if p["id"] in seen:
                        continue
                    seen.add(p["id"])
                    papers.append(p)
        # Sort so doc indices are deterministic no matter how files were fetched.
        papers.sort(key=lambda p: p["id"])
        log.info("Loaded %d papers from %d files", len(papers), len(files))
        return cls(papers)


def bm25_text(paper: dict) -> str:
    """Text indexed by BM25. The title is repeated so its terms weigh more:
    titles are short and keyword-dense, abstracts are long and diluted."""
    title = paper.get("title", "")
    return f"{title}\n{title}\n{paper.get('abstract', '')}"
