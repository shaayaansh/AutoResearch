#!/usr/bin/env python3
"""
Fetch conference proceedings (title, abstract, authors, links) into a unified JSONL corpus.

Sources
-------
* OpenReview API v2  -> ICLR, NeurIPS   (accepted papers are found by their `venueid`)
* ACL Anthology      -> ACL, EMNLP, EACL, NAACL      (via the official `acl-anthology` package, which
                                         shallow-clones the Anthology's XML data from GitHub)

Output
------
data/raw/<venue>_<year>.jsonl   one paper per line, same schema for every source
data/raw/manifest.json          what was fetched, when, and how many papers

Usage
-----
    python scripts/fetch_proceedings.py                       # all venues, 2025 + 2026
    python scripts/fetch_proceedings.py --venues iclr acl --years 2025
    python scripts/fetch_proceedings.py --force               # re-download existing files

Optional: set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD if guest access gets rate-limited.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("fetch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw"

OPENREVIEW_VENUES = {
    # venue key -> OpenReview venue id template. Accepted papers carry exactly this venueid;
    # rejected/withdrawn ones get suffixes like ".../Rejected_Submission".
    "iclr": "ICLR.cc/{year}/Conference",
    "neurips": "NeurIPS.cc/{year}/Conference",
}

ACL_VENUES = {"acl", "emnlp", "eacl", "naacl"}  # EACL/NAACL are not annual

# Which ACL Anthology volumes count as "the proceedings".
#  ACL uses long/short, EMNLP uses main; Findings live in a separate `<year>.findings` collection.
DEFAULT_ACL_VOLUMES = ["long", "short", "main", "findings"]


# --------------------------------------------------------------------------------------
# Unified schema — every source is normalized into this, so downstream code never cares
# where a paper came from.
# --------------------------------------------------------------------------------------
@dataclass
class PaperRecord:
    id: str                    # globally unique, e.g. "openreview:abc123" or "acl:2025.acl-long.1"
    venue: str                 # "ICLR", "NeurIPS", "ACL", "EMNLP"
    year: int
    track: str                 # e.g. "Oral", "Poster", "long", "findings"
    title: str
    abstract: str
    authors: list[str]
    keywords: list[str] = field(default_factory=list)
    url: str = ""
    pdf_url: str = ""
    source: str = ""           # "openreview" | "acl_anthology"


# --------------------------------------------------------------------------------------
# OpenReview
# --------------------------------------------------------------------------------------
def _or_value(content: dict, key: str, default=None):
    """API v2 wraps every content field as {"value": ...}."""
    item = content.get(key)
    if isinstance(item, dict):
        return item.get("value", default)
    return default if item is None else item


def _openreview_track(venue_str: str, venue_name: str, year: int) -> str:
    # venue string looks like "ICLR 2025 Oral" / "NeurIPS 2025 poster"
    track = venue_str.replace(venue_name, "").replace(str(year), "").strip()
    return track.title() or "Accepted"


def normalize_openreview_note(note, venue_name: str, year: int) -> PaperRecord | None:
    c = note.content or {}
    title = (_or_value(c, "title") or "").strip()
    if not title:
        return None
    pdf = _or_value(c, "pdf") or ""
    return PaperRecord(
        id=f"openreview:{note.id}",
        venue=venue_name,
        year=year,
        track=_openreview_track(_or_value(c, "venue") or "", venue_name, year),
        title=title,
        abstract=(_or_value(c, "abstract") or "").strip(),
        authors=list(_or_value(c, "authors") or []),
        keywords=list(_or_value(c, "keywords") or []),
        url=f"https://openreview.net/forum?id={note.forum or note.id}",
        pdf_url=f"https://openreview.net{pdf}" if pdf.startswith("/") else pdf,
        source="openreview",
    )


def make_openreview_client():
    import openreview

    user, pw = os.getenv("OPENREVIEW_USERNAME"), os.getenv("OPENREVIEW_PASSWORD")
    if not (user and pw):
        log.warning("OPENREVIEW_USERNAME/PASSWORD not set; anonymous access may be blocked.")
    return openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net", username=user, password=pw
    )


def fetch_openreview(venue: str, year: int, client, retries: int = 3) -> list[PaperRecord]:
    venue_id = OPENREVIEW_VENUES[venue].format(year=year)
    venue_name = venue_id.split(".cc")[0]  # "ICLR" / "NeurIPS"
    log.info("OpenReview: fetching accepted papers for %s", venue_id)

    for attempt in range(1, retries + 1):
        try:
            # get_all_notes handles pagination internally
            notes = client.get_all_notes(content={"venueid": venue_id})
            break
        except Exception as e:  # network hiccups / rate limits (HTTP 429)
            if "ChallengeRequired" in str(e) or "'status': 403" in str(e):
                # Anonymous API access is gated behind a browser challenge; retrying won't help.
                raise RuntimeError(
                    "OpenReview requires sign-in for API access. Set OPENREVIEW_USERNAME and "
                    "OPENREVIEW_PASSWORD (your openreview.net account) and re-run."
                ) from None
            if attempt == retries:
                raise
            wait = 10 * attempt
            log.warning("  attempt %d failed (%s); retrying in %ds", attempt, e, wait)
            time.sleep(wait)

    records = [r for n in notes if (r := normalize_openreview_note(n, venue_name, year))]
    return records


# --------------------------------------------------------------------------------------
# ACL Anthology
# --------------------------------------------------------------------------------------
_anthology = None


def get_anthology():
    """Load once; the first call shallow-clones the Anthology repo (a few minutes)."""
    global _anthology
    if _anthology is None:
        from acl_anthology import Anthology

        log.info("ACL Anthology: cloning/updating data repo (first run takes a few minutes)...")
        _anthology = Anthology.from_repo()
    return _anthology


def _text(markup) -> str:
    return markup.as_text().strip() if markup is not None else ""


def normalize_acl_paper(paper, venue_name: str, year: int, track: str) -> PaperRecord | None:
    if paper.is_frontmatter or paper.is_deleted:
        return None
    pdf = paper.pdf
    return PaperRecord(
        id=f"acl:{paper.full_id}",
        venue=venue_name,
        year=year,
        track=track,
        title=_text(paper.title),
        abstract=_text(paper.abstract),
        authors=[a.name.as_first_last() for a in paper.authors],
        url=paper.web_url,
        pdf_url=pdf.url if pdf is not None else "",
        source="acl_anthology",
    )


def _acl_volumes(venue: str, year: int, wanted: list[str]):
    """Yield (track, Volume) pairs for the main-conference + Findings volumes."""
    anth = get_anthology()

    main = anth.get_collection(f"{year}.{venue}")
    if main is not None:
        for vol in main.volumes():
            if vol.id in wanted and not vol.is_workshop:
                yield vol.id, vol

    if "findings" in wanted:
        findings = anth.get_volume(f"{year}.findings-{venue}")
        if findings is not None:
            yield "findings", findings


def fetch_acl(venue: str, year: int, wanted_volumes: list[str]) -> list[PaperRecord]:
    venue_name = venue.upper()
    records: list[PaperRecord] = []
    for track, vol in _acl_volumes(venue, year, wanted_volumes):
        before = len(records)
        records.extend(
            r for p in vol.papers() if (r := normalize_acl_paper(p, venue_name, year, track))
        )
        log.info("  %s: %d papers", vol.full_id, len(records) - before)
    return records


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def write_jsonl(path: Path, records: list[PaperRecord]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: never leave a half-written file behind


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venues", nargs="+", default=["iclr", "neurips", "acl", "emnlp"],
                    choices=sorted(OPENREVIEW_VENUES) + sorted(ACL_VENUES))
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--acl-volumes", nargs="+", default=DEFAULT_ACL_VOLUMES,
                    help="ACL Anthology volume ids to include (e.g. add 'demo' 'industry')")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-fetch even if output exists")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    or_client = None
    for venue in args.venues:
        for year in args.years:
            key = f"{venue}_{year}"
            out_file = args.out / f"{key}.jsonl"
            if out_file.exists() and not args.force:
                log.info("skip %s (exists; use --force to refresh)", key)
                continue
            try:
                if venue in OPENREVIEW_VENUES:
                    or_client = or_client or make_openreview_client()
                    records = fetch_openreview(venue, year, or_client)
                else:
                    records = fetch_acl(venue, year, args.acl_volumes)
            except Exception as e:
                log.error("FAILED %s: %s", key, e)
                manifest[key] = {"status": "error", "error": str(e)}
                continue

            if not records:
                # e.g. proceedings not published yet
                log.warning("no papers found for %s — not available yet?", key)
                manifest[key] = {"status": "unavailable", "count": 0}
                continue

            write_jsonl(out_file, records)
            n_abs = sum(bool(r.abstract) for r in records)
            log.info("wrote %s: %d papers (%d with abstracts)", out_file.name, len(records), n_abs)
            manifest[key] = {
                "status": "ok",
                "count": len(records),
                "with_abstract": n_abs,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "file": out_file.name,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nSummary:")
    for k, v in sorted(manifest.items()):
        print(f"  {k:<14} {v.get('status'):<12} {v.get('count', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
