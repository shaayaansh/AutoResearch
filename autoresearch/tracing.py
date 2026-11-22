"""Minimal structured tracing: every run appends JSON events to runs/<run_id>.jsonl.

Later versions log planner decisions, tool calls, verifier outputs and costs here;
the reflector and the eval harness both read these files.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import RUNS_DIR


class Tracer:
    def __init__(self, name: str = "run", runs_dir: Path = RUNS_DIR):
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{name}-{uuid.uuid4().hex[:6]}"
        self.path = runs_dir / f"{self.run_id}.jsonl"
        self._t0 = time.perf_counter()

    def log(self, event: str, **data) -> None:
        record = {
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.perf_counter() - self._t0, 4),
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
