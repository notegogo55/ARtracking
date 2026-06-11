"""HARP <-> NOAA AR cross-identification via the official JSOC mapping file.

The mapping (derived from NOAA SRS by the HMI team) lives at
http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt
with one `HARPNUM NOAA_ARS` row per HARP (NOAA_ARS comma-separated).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HARP_NOAA_MAPPING_URL = (
    "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"
)


def parse_harp_noaa_mapping(text: str) -> pd.DataFrame:
    """Parse the mapping file text into a DataFrame [harpnum, noaa_ars(list[int])]."""
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("HARPNUM"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            harpnum = int(parts[0])
            noaa_ars = [int(x) for x in parts[1].split(",") if x]
        except ValueError:
            log.warning("skipping malformed mapping line: %r", line)
            continue
        rows.append({"harpnum": harpnum, "noaa_ars": noaa_ars})
    if not rows:
        raise ValueError("no rows parsed from HARP/NOAA mapping text")
    return pd.DataFrame(rows)


def fetch_harp_noaa_mapping(
    cache_dir: str | Path,
    url: str = HARP_NOAA_MAPPING_URL,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (or load cached) official HARP/NOAA mapping. Network on cache miss."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "all_harps_with_noaa_ars.txt"
    if refresh or not cache_file.exists():
        import requests

        log.info("downloading HARP/NOAA mapping from %s", url)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_file.write_text(resp.text, encoding="utf-8")
    return parse_harp_noaa_mapping(cache_file.read_text(encoding="utf-8"))


def harps_for_noaa(mapping: pd.DataFrame, noaa: int) -> list[int]:
    """All HARP numbers whose NOAA_ARS list contains `noaa` (usually exactly one)."""
    hits = mapping[mapping["noaa_ars"].map(lambda ars: noaa in ars)]
    return sorted(int(h) for h in hits["harpnum"])


def resolve_harp(mapping: pd.DataFrame, noaa: int) -> int:
    """The unique HARP for a NOAA AR; raises if zero or several match."""
    hits = harps_for_noaa(mapping, noaa)
    if len(hits) != 1:
        raise ValueError(f"NOAA {noaa}: expected exactly one HARP, found {hits}")
    return hits[0]
