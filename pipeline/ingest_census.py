"""Fetch Census ACS 5-year demographic data by ZCTA for San Diego area."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import httpx

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ACS_URL = "https://api.census.gov/data/2022/acs/acs5"

# Variables to fetch
VARIABLES = [
    "B01003_001E",  # total population
    "B01002_001E",  # median age
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25064_001E",  # median gross rent
    "B15003_001E",  # total education (25+)
    "B15003_022E",  # bachelor's degree
    "B15003_023E",  # master's degree
    "B15003_024E",  # professional degree
    "B15003_025E",  # doctorate
]

# San Diego area ZCTAs (91900-92199 covers most of SD county)
SD_ZIP_MIN = 91900
SD_ZIP_MAX = 92199


def ingest(*, force: bool = False) -> Path | None:
    """Fetch Census ACS data for SD-area ZCTAs. Saves to data/raw/census_acs.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "census_acs.csv"

    if dest.exists() and not force:
        print(f"  [skip] census_acs (already exists, {dest.stat().st_size:,} bytes)")
        return dest

    print("  [download] census ACS data ...")

    params = {
        "get": ",".join(["NAME"] + VARIABLES),
        "for": "zip code tabulation area:*",
    }

    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key

    resp = httpx.get(ACS_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()

    # first row is header
    header = rows[0]
    data = rows[1:]

    # find the ZCTA column
    zcta_col = header.index("zip code tabulation area")

    # filter to SD-area ZCTAs
    sd_rows = []
    for row in data:
        try:
            zcta = int(row[zcta_col])
            if SD_ZIP_MIN <= zcta <= SD_ZIP_MAX:
                sd_rows.append(row)
        except (ValueError, IndexError):
            continue

    # write CSV
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sd_rows)

    print(f"  [done] census_acs -> {len(sd_rows)} ZCTAs, {dest.stat().st_size:,} bytes")
    return dest


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest(force=force)
