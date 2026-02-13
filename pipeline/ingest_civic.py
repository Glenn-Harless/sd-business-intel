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
    "sd-get-it-done/data/aggregated/top_problem_types.parquet": "civic_311_services.parquet",
    "sd-housing-permits/data/aggregated/approval_timelines.parquet": "civic_permit_timelines.parquet",
    "sd-climate-action/data/aggregated/energy_by_zip_annual.parquet": "civic_energy.parquet",
    # Phase 3: map point layers
    "sd-get-it-done/data/aggregated/map_points.parquet": "map_311.parquet",
    "sd-housing-permits/data/aggregated/map_points.parquet": "map_permits.parquet",
    "sd-public-safety/data/aggregated/map_points.parquet": "map_crime.parquet",
    "sd-climate-action/data/aggregated/solar_map_points.parquet": "map_solar.parquet",
    # Phase 3d: crime detail
    "sd-public-safety/data/aggregated/crime_by_type.parquet": "civic_crime_detail.parquet",
    "sd-public-safety/data/aggregated/temporal_patterns.parquet": "civic_crime_temporal.parquet",
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

    # Build zip centroids from Census Gazetteer
    centroid_path = build_zip_centroids(force=force)
    if centroid_path:
        copied.append(centroid_path)

    return copied


def build_zip_centroids(*, force: bool = False) -> Path | None:
    """Download Census Gazetteer ZCTA centroids and save as parquet."""
    dest = AGG_DIR / "zip_centroids.parquet"
    if dest.exists() and not force:
        print("  [skip] zip_centroids.parquet (already exists)")
        return dest

    import csv
    import io
    import urllib.request
    import zipfile

    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip"
    print(f"  [download] Census ZCTA centroids from {url}")

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"  [error] failed to download centroids: {e}")
        return None

    # Extract the .txt from the zip archive
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
            if not txt_names:
                print("  [error] no .txt file found in zip archive")
                return None
            text = zf.read(txt_names[0]).decode("utf-8")
    except Exception as e:
        print(f"  [error] failed to extract centroids zip: {e}")
        return None

    # Tab-delimited with columns: GEOID, ALAND, AWATER, ALAND_SQMI, AWATER_SQMI, INTPTLAT, INTPTLONG
    # Note: Gazetteer file has fixed-width padding — keys/values need stripping
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = []
    for row in reader:
        stripped = {k.strip(): v.strip() for k, v in row.items()}
        zcta = stripped.get("GEOID", "")
        lat = stripped.get("INTPTLAT", "")
        lng = stripped.get("INTPTLONG", "")
        if zcta and lat and lng:
            rows.append({"zip_code": zcta, "lat": float(lat), "lng": float(lng)})

    if not rows:
        print("  [error] no centroids parsed from Gazetteer file")
        return None

    # Filter to SD-profiled zips using neighborhood_profile
    import duckdb
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    if np_path.exists():
        con = duckdb.connect()
        sd_zips = set(
            con.execute(f"SELECT DISTINCT zip_code FROM '{np_path}'")
            .fetchdf()["zip_code"].tolist()
        )
        con.close()
        rows = [r for r in rows if r["zip_code"] in sd_zips]

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "zip_code": [r["zip_code"] for r in rows],
        "lat": [r["lat"] for r in rows],
        "lng": [r["lng"] for r in rows],
    })
    pq.write_table(table, dest)
    print(f"  [write] zip_centroids.parquet ({len(rows)} zips)")
    return dest


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest(force=force)
