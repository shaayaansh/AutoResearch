"""AutoResearch: agentic literature search over ML/NLP conference proceedings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
RUNS_DIR = PROJECT_ROOT / "runs"
