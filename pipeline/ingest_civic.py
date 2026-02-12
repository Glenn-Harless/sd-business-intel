"""Copy aggregated parquets from sibling civic data projects."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

AGG_DIR = Path(__file__).resolve().parent.parent / "data" / "aggregated"
SIBLING_ROOT = Path(__file__).resolve().parent.parent.parent  # ../

# source parquet → destination name (with civic_ prefix)
CIVIC_SOURCES: dict[str, str] = {
    "sd-housing-permits/data/aggregated/construction_by_zip.parquet": "civic_permits.parquet",
    "sd-climate-action/data/aggregated/solar_by_zip.parquet": "civic_solar.parquet",
    "sd-public-safety/data/aggregated/crime_by_zip.parquet": "civic_crime.parquet",
    "sd-get-it-done/data/aggregated/response_by_neighborhood.parquet": "civic_311.parquet",
    "sd-get-it-done/data/aggregated/monthly_trends.parquet": "civic_311_monthly.parquet",
    "sd-homelessness/data/aggregated/pit_geography.parquet": "civic_homelessness.parquet",
}


def ingest(*, force: bool = False) -> list[Path]:
    """Copy civic parquets from sibling projects into data/aggregated/."""
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    copied = []

    for rel_path, dest_name in CIVIC_SOURCES.items():
        src = SIBLING_ROOT / rel_path
        dest = AGG_DIR / dest_name

        if not src.exists():
            print(f"  [skip] {dest_name}: source not found ({src})")
            continue

        if dest.exists() and not force:
            src_mtime = datetime.fromtimestamp(src.stat().st_mtime)
            dest_mtime = datetime.fromtimestamp(dest.stat().st_mtime)
            if src_mtime <= dest_mtime:
                print(f"  [skip] {dest_name} (up to date)")
                copied.append(dest)
                continue

        shutil.copy2(src, dest)
        mod_date = datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d")
        print(f"  [copy] {dest_name} (source modified {mod_date})")
        copied.append(dest)

    return copied


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest(force=force)
