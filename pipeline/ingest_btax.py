"""Download active business tax certificate CSVs from seshat.datasd.org."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Primary: sandiego.gov treasurer page (A-K, L-Z split)
# Fallback: seshat.datasd.org open data CDN
SOURCES: dict[str, list[str]] = {
    "btax_active_1": [
        "https://www.sandiego.gov/sites/default/files/2023-12/tr_active1.csv",
        "https://seshat.datasd.org/sd_businesses_active/sd_businesses_active_1_datasd.csv",
    ],
    "btax_active_2": [
        "https://www.sandiego.gov/sites/default/files/2023-12/tr_active2.csv",
        "https://seshat.datasd.org/sd_businesses_active/sd_businesses_active_2_datasd.csv",
    ],
}


def download(name: str, url: str, *, force: bool = False) -> Path | None:
    """Download a single CSV. Skips if file exists and force=False."""
    dest = RAW_DIR / f"{name}.csv"
    if dest.exists() and not force:
        print(f"  [skip] {name} (already exists, {dest.stat().st_size:,} bytes)")
        return dest

    print(f"  [download] {name} ...")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  [done] {name} -> {dest.stat().st_size:,} bytes")
        return dest
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            print(f"  [warn] {name}: 403 forbidden, skipping")
            return None
        raise


def ingest(*, force: bool = False) -> list[Path]:
    """Download all business tax cert CSVs. Returns list of downloaded paths."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, urls in SOURCES.items():
        result = None
        for url in urls:
            result = download(name, url, force=force)
            if result is not None:
                break
        if result is not None:
            paths.append(result)
    return paths


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest(force=force)
