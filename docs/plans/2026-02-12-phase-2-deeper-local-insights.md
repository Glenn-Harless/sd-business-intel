# Phase 2: Deeper Local Insights — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add area-level profiles, temporal trends, and dashboard query-layer refactor to the SD Business Intel platform.

**Architecture:** Pipeline builds new area + trend parquets from existing data. Query layer (`api/queries.py`) gets 7 new functions + updates. Dashboard stops duplicating SQL and imports from query layer directly. API/MCP get matching new endpoints/tools.

**Tech Stack:** DuckDB (pipeline + queries), FastAPI, FastMCP, Streamlit, Pydantic, Plotly

**Source spec:** `phase_2_spec.md`

---

## Task 1: Fix business date parsing bug

**Files:**
- Modify: `pipeline/transform.py:212-214` (the 3 TRY_CAST lines)
- Modify: `pipeline/transform.py:216` (status CASE expression also uses TRY_CAST)

**Context:** The pipeline reads CSVs with `all_varchar=true`, then uses `TRY_CAST("START DT" AS DATE)`. Dates in the CSVs are `MM/DD/YYYY` format (e.g., `"10/11/2019"`) which doesn't match DuckDB's default ISO format. All 51,511 rows currently have NULL dates.

**Step 1: Fix the 3 date TRY_CAST lines**

In `pipeline/transform.py`, find these three lines (~212-214):

```python
            {f'TRY_CAST("{start_col}" AS DATE) AS start_date,' if start_col else "NULL AS start_date,"}
            {f'TRY_CAST("{create_col}" AS DATE) AS created_date,' if create_col else "NULL AS created_date,"}
            {f'TRY_CAST("{exp_col}" AS DATE) AS expiration_date,' if exp_col else "NULL AS expiration_date,"}
```

Replace with:

```python
            {f"TRY_CAST(STRPTIME(\"{start_col}\", '%m/%d/%Y') AS DATE) AS start_date," if start_col else "NULL AS start_date,"}
            {f"TRY_CAST(STRPTIME(\"{create_col}\", '%m/%d/%Y') AS DATE) AS created_date," if create_col else "NULL AS created_date,"}
            {f"TRY_CAST(STRPTIME(\"{exp_col}\", '%m/%d/%Y') AS DATE) AS expiration_date," if exp_col else "NULL AS expiration_date,"}
```

**Step 2: Fix the status CASE expression**

Find (~line 216):

```python
            CASE
                WHEN {f'TRY_CAST("{exp_col}" AS DATE) IS NULL OR TRY_CAST("{exp_col}" AS DATE) >= CURRENT_DATE' if exp_col else "TRUE"}
                THEN 'active' ELSE 'inactive'
            END AS status
```

Replace with:

```python
            CASE
                WHEN {f"TRY_CAST(STRPTIME(\"{exp_col}\", '%m/%d/%Y') AS DATE) IS NULL OR TRY_CAST(STRPTIME(\"{exp_col}\", '%m/%d/%Y') AS DATE) >= CURRENT_DATE" if exp_col else "TRUE"}
                THEN 'active' ELSE 'inactive'
            END AS status
```

**Step 3: Verify the fix**

Run:
```bash
uv run python -c "
import duckdb
con = duckdb.connect()
# test parsing a known date from the CSV
result = con.execute(\"SELECT TRY_CAST(STRPTIME('10/11/2019', '%m/%d/%Y') AS DATE) AS d\").fetchone()
print(f'Parsed date: {result[0]}')
assert str(result[0]) == '2019-10-11', f'Expected 2019-10-11, got {result[0]}'
print('Date parsing fix verified')
"
```

Expected: `Parsed date: 2019-10-11` and `Date parsing fix verified`

**Step 4: Commit**

```bash
git add pipeline/transform.py
git commit -m "fix: use STRPTIME for business date parsing (MM/DD/YYYY format)"
```

---

## Task 2: Add ZIP_TO_AREA mapping dict

**Files:**
- Modify: `pipeline/transform.py` (add new dict after `ZIP_TO_COMM_PLAN`, ~line 135)

**Context:** `ZIP_TO_NEIGHBORHOOD` has 82 entries (lines 16-99). `ZIP_TO_COMM_PLAN` has 33 entries (lines 102-134). The new `ZIP_TO_AREA` maps each of the 82 profiled zips to one of ~20 area names.

**Step 1: Verify which 82 zips are in the profile**

Run:
```bash
uv run python -c "
import duckdb
con = duckdb.connect()
zips = con.execute(\"SELECT DISTINCT zip_code FROM 'data/aggregated/neighborhood_profile.parquet' ORDER BY zip_code\").fetchdf()['zip_code'].tolist()
print(f'{len(zips)} profiled zips:')
for z in zips:
    print(f'  {z}')
"
```

Use this output to build the final mapping. Only include zips that appear in this list.

**Step 2: Add ZIP_TO_AREA dict**

After `ZIP_TO_COMM_PLAN` (after line 134), add:

```python
# zip → area mapping for area-level aggregation (only profiled zips)
ZIP_TO_AREA: dict[str, str] = {
    # Downtown
    "92101": "Downtown",
    # Uptown / North Park
    "92103": "Uptown / North Park",
    "92104": "Uptown / North Park",
    "92116": "Uptown / North Park",
    # Golden Hill / City Heights
    "92102": "Golden Hill / City Heights",
    "92105": "Golden Hill / City Heights",
    # Barrio Logan / Logan Heights
    "92113": "Barrio Logan / Logan Heights",
    # Pacific Beach
    "92109": "Pacific Beach",
    # Ocean Beach / Point Loma
    "92106": "Ocean Beach / Point Loma",
    "92107": "Ocean Beach / Point Loma",
    # Mission Valley / Linda Vista
    "92108": "Mission Valley / Linda Vista",
    "92110": "Mission Valley / Linda Vista",
    "92111": "Mission Valley / Linda Vista",
    # Clairemont
    "92117": "Clairemont",
    # La Jolla / University City
    "92037": "La Jolla / University City",
    "92122": "La Jolla / University City",
    # Sorrento Valley / Mira Mesa
    "92121": "Sorrento Valley / Mira Mesa",
    "92126": "Sorrento Valley / Mira Mesa",
    # Rancho Bernardo / Scripps Ranch
    "92127": "Rancho Bernardo / Scripps Ranch",
    "92128": "Rancho Bernardo / Scripps Ranch",
    "92131": "Rancho Bernardo / Scripps Ranch",
    # Carmel Valley / Rancho Penasquitos
    "92129": "Carmel Valley / Rancho Penasquitos",
    "92130": "Carmel Valley / Rancho Penasquitos",
    # Carlsbad
    "92008": "Carlsbad",
    "92009": "Carlsbad",
    "92010": "Carlsbad",
    "92011": "Carlsbad",
    # Oceanside
    "92054": "Oceanside",
    "92056": "Oceanside",
    "92057": "Oceanside",
    "92058": "Oceanside",
    # Encinitas / Del Mar / Solana Beach
    "92007": "Encinitas / Del Mar / Solana Beach",
    "92014": "Encinitas / Del Mar / Solana Beach",
    "92024": "Encinitas / Del Mar / Solana Beach",
    "92075": "Encinitas / Del Mar / Solana Beach",
    # Escondido
    "92025": "Escondido",
    "92026": "Escondido",
    "92027": "Escondido",
    "92029": "Escondido",
    # Vista / San Marcos
    "92069": "Vista / San Marcos",
    "92078": "Vista / San Marcos",
    "92081": "Vista / San Marcos",
    "92083": "Vista / San Marcos",
    "92084": "Vista / San Marcos",
    # Chula Vista / National City
    "91910": "Chula Vista / National City",
    "91911": "Chula Vista / National City",
    "91913": "Chula Vista / National City",
    "91914": "Chula Vista / National City",
    "91915": "Chula Vista / National City",
    "91950": "Chula Vista / National City",
    # La Mesa / Lemon Grove
    "91941": "La Mesa / Lemon Grove",
    "91942": "La Mesa / Lemon Grove",
    "91945": "La Mesa / Lemon Grove",
    # El Cajon
    "92019": "El Cajon",
    "92020": "El Cajon",
    "92021": "El Cajon",
    # South Bay / San Ysidro
    "92154": "South Bay / San Ysidro",
    "92173": "South Bay / San Ysidro",
    "91932": "South Bay / San Ysidro",
    "92139": "South Bay / San Ysidro",
    # East County
    "91901": "East County",
    "91977": "East County",
    "91978": "East County",
    "92040": "East County",
    "92065": "East County",
    "92071": "East County",
    # Poway / Rancho Santa Fe
    "92064": "Poway / Rancho Santa Fe",
    "92067": "Poway / Rancho Santa Fe",
    "92091": "Poway / Rancho Santa Fe",
}
# Any profiled zip NOT in this dict goes to "Other"
```

**Step 3: Cross-reference against profiled zips**

After adding the dict, verify coverage:

```bash
uv run python -c "
from pipeline.transform import ZIP_TO_AREA, ZIP_TO_NEIGHBORHOOD
profiled = set(ZIP_TO_NEIGHBORHOOD.keys())
mapped = set(ZIP_TO_AREA.keys())
unmapped = profiled - mapped
print(f'Profiled: {len(profiled)}, Mapped: {len(mapped)}')
print(f'Unmapped profiled zips (will go to Other): {sorted(unmapped)}')
areas = sorted(set(ZIP_TO_AREA.values()))
print(f'Areas: {len(areas)}')
for a in areas:
    zips = sorted(z for z, area in ZIP_TO_AREA.items() if area == a)
    print(f'  {a}: {zips}')
"
```

Drop any zips from the dict that aren't in the profiled set. Add unmapped profiled zips to an "Other" entry or assign to the most sensible area.

**Step 4: Commit**

```bash
git add pipeline/transform.py
git commit -m "feat: add ZIP_TO_AREA mapping for ~20 area-level groupings"
```

---

## Task 3: Add 311 monthly trends to civic ingestion

**Files:**
- Modify: `pipeline/ingest_civic.py:14-20` (add entry to CIVIC_SOURCES dict)

**Context:** The `CIVIC_SOURCES` dict maps sibling project paths to local filenames. We need to add the 311 monthly trends parquet. The source has columns: `request_month_start (DATE), total_requests, closed_requests, avg_resolution_days, median_resolution_days`.

**Step 1: Add the new source**

In `pipeline/ingest_civic.py`, add to the `CIVIC_SOURCES` dict (after line 18):

```python
    "sd-get-it-done/data/aggregated/monthly_trends.parquet": "civic_311_monthly.parquet",
```

**Step 2: Verify the source exists**

```bash
uv run python -c "
from pathlib import Path
src = Path('/Users/glennharless/dev-brain/sd-get-it-done/data/aggregated/monthly_trends.parquet')
print(f'Exists: {src.exists()}, Size: {src.stat().st_size} bytes')
"
```

Expected: `Exists: True, Size: 3231 bytes`

**Step 3: Commit**

```bash
git add pipeline/ingest_civic.py
git commit -m "feat: add 311 monthly trends to civic ingestion"
```

---

## Task 4: Build area-level aggregated parquets

**Files:**
- Modify: `pipeline/transform.py` — add `_build_area_profiles()` function, call it from `_build_aggregates()`

**Context:** This creates `area_profile.parquet` and `area_business_by_category.parquet`. Also injects `area` column into `neighborhood_profile.parquet`.

**Step 1: Add `_build_area_profiles()` function**

Add this function before `_build_city_averages()` (~line 497):

```python
def _build_area_profiles(
    con: duckdb.DuckDBPyConnection, has_biz: bool, has_demo: bool
) -> None:
    """Build area_profile.parquet and area_business_by_category.parquet."""
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    if not np_path.exists():
        print("  [skip] area parquets (no neighborhood_profile)")
        return

    # Load ZIP_TO_AREA into a temp table
    rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE zip_area(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO zip_area VALUES (?, ?)", rows)

    # Handle unmapped profiled zips → "Other"
    con.execute(f"""
        INSERT INTO zip_area
        SELECT np.zip_code, 'Other'
        FROM '{np_path}' np
        WHERE np.zip_code NOT IN (SELECT zip_code FROM zip_area)
    """)

    # 1. area_profile.parquet — aggregate neighborhood_profile by area
    con.execute(f"""
        COPY (
            SELECT
                za.area,
                LIST(DISTINCT za.zip_code ORDER BY za.zip_code) AS zip_codes,
                COUNT(DISTINCT za.zip_code) AS zip_count,
                SUM(np.population) AS population,
                ROUND(SUM(np.median_age * np.population) / NULLIF(SUM(np.population), 0), 1) AS median_age,
                ROUND(SUM(np.median_income * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_income,
                ROUND(SUM(np.median_home_value * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_home_value,
                ROUND(SUM(np.median_rent * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_rent,
                ROUND(SUM(np.pct_bachelors_plus * np.population) / NULLIF(SUM(np.population), 0), 1) AS pct_bachelors_plus,
                SUM(np.active_count) AS active_count,
                SUM(np.total_count) AS total_count,
                ROUND(1000.0 * SUM(np.active_count) / NULLIF(SUM(np.population), 0), 1) AS businesses_per_1k,
                SUM(np.new_permits) AS new_permits,
                SUM(np.permit_valuation) AS permit_valuation,
                SUM(np.solar_installs) AS solar_installs,
                SUM(np.crime_count) AS crime_count,
                ROUND(SUM(np.median_311_days * np.total_311_requests) / NULLIF(SUM(np.total_311_requests), 0), 1) AS median_311_days,
                SUM(np.total_311_requests) AS total_311_requests
            FROM zip_area za
            JOIN '{np_path}' np ON za.zip_code = np.zip_code
            GROUP BY za.area
            ORDER BY SUM(np.population) DESC
        ) TO '{AGG_DIR}/area_profile.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print("  [done] area_profile.parquet")

    # 2. area_business_by_category.parquet — category breakdown per area
    biz_by_zip_path = AGG_DIR / "business_by_zip.parquet"
    if biz_by_zip_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    za.area,
                    bz.category,
                    SUM(bz.active_count) AS active_count,
                    SUM(bz.total_count) AS total_count,
                    ROUND(1000.0 * SUM(bz.active_count) / NULLIF(ap.population, 0), 2) AS per_1k
                FROM zip_area za
                JOIN '{biz_by_zip_path}' bz ON za.zip_code = bz.zip_code
                JOIN '{AGG_DIR}/area_profile.parquet' ap ON za.area = ap.area
                GROUP BY za.area, bz.category, ap.population
                ORDER BY za.area, SUM(bz.active_count) DESC
            ) TO '{AGG_DIR}/area_business_by_category.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] area_business_by_category.parquet")
```

**Step 2: Inject `area` column into neighborhood_profile**

In `_build_neighborhood_profile()`, after the `zn` table is created (~line 367), add the area column to `select_parts`:

After line 371 (`from_parts = ["zn"]`), add:

```python
    # area mapping
    area_rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE za_lookup(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO za_lookup VALUES (?, ?)", area_rows)
    select_parts.append("COALESCE(za_lookup.area, 'Other') AS area")
    from_parts.append("LEFT JOIN za_lookup ON zn.zip_code = za_lookup.zip_code")
```

**Step 3: Call `_build_area_profiles` from `_build_aggregates`**

In `_build_aggregates()`, after the `_build_city_averages` call (~line 347), add:

```python
    # 7. area profiles — area-level aggregation
    _build_area_profiles(con, has_biz, has_demo)
```

**Step 4: Verify by running pipeline**

```bash
uv run python -m pipeline.build --force
```

Then verify:

```bash
uv run python -c "
import duckdb
con = duckdb.connect()
# Check area_profile
ap = con.execute(\"SELECT area, zip_count, population, businesses_per_1k FROM 'data/aggregated/area_profile.parquet' ORDER BY population DESC\").fetchdf()
print(f'area_profile: {len(ap)} areas')
print(ap.to_string())
print()
# Check area column in neighborhood_profile
np = con.execute(\"SELECT zip_code, area FROM 'data/aggregated/neighborhood_profile.parquet' WHERE area IS NOT NULL LIMIT 10\").fetchdf()
print(f'neighborhood_profile area column sample:')
print(np.to_string())
print()
# Check business dates fixed
biz = con.execute(\"SELECT COUNT(*) as total, COUNT(start_date) as with_start FROM 'data/processed/businesses.parquet'\").fetchdf()
print(f'Business dates: {biz.to_string()}')
"
```

Expected: ~20 areas, area column populated in neighborhood_profile, most business rows have non-NULL start_date.

**Step 5: Commit**

```bash
git add pipeline/transform.py
git commit -m "feat: add area-level aggregated parquets and area column to profiles"
```

---

## Task 5: Build trend parquets

**Files:**
- Modify: `pipeline/transform.py` — add `_build_trends()` function, call it from `_build_aggregates()`

**Context:** Creates `trend_business_formation.parquet` from fixed business dates. Ingests `civic_311_monthly.parquet` with year/month extraction. Existing civic parquets (permits, crime, solar) already have year-level data and don't need new parquets.

**Step 1: Add `_build_trends()` function**

Add after `_build_area_profiles()`:

```python
def _build_trends(con: duckdb.DuckDBPyConnection) -> None:
    """Build trend parquets for temporal analysis."""
    # 1. trend_business_formation — new businesses per zip per year
    biz_path = PROCESSED_DIR / "businesses.parquet"
    if biz_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    zip_code,
                    EXTRACT(YEAR FROM start_date)::INTEGER AS year,
                    COUNT(*) AS new_businesses
                FROM '{biz_path}'
                WHERE start_date IS NOT NULL
                  AND EXTRACT(YEAR FROM start_date) >= 2015
                GROUP BY zip_code, EXTRACT(YEAR FROM start_date)
                ORDER BY zip_code, year
            ) TO '{AGG_DIR}/trend_business_formation.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        count = con.execute(f"""
            SELECT COUNT(*) FROM '{AGG_DIR}/trend_business_formation.parquet'
        """).fetchone()[0]
        print(f"  [done] trend_business_formation.parquet -> {count:,} rows")
    else:
        print("  [skip] trend_business_formation.parquet (no businesses)")

    # 2. trend_311_monthly — extract year/month from DATE column
    monthly_path = AGG_DIR / "civic_311_monthly.parquet"
    if monthly_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    EXTRACT(YEAR FROM request_month_start)::INTEGER AS year,
                    EXTRACT(MONTH FROM request_month_start)::INTEGER AS month,
                    total_requests,
                    closed_requests,
                    avg_resolution_days,
                    median_resolution_days
                FROM '{monthly_path}'
                ORDER BY year, month
            ) TO '{AGG_DIR}/trend_311_monthly.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        count = con.execute(f"""
            SELECT COUNT(*) FROM '{AGG_DIR}/trend_311_monthly.parquet'
        """).fetchone()[0]
        print(f"  [done] trend_311_monthly.parquet -> {count:,} rows")
    else:
        print("  [skip] trend_311_monthly.parquet (no 311 monthly data)")
```

**Step 2: Call from `_build_aggregates`**

After the `_build_area_profiles` call, add:

```python
    # 8. trend parquets
    _build_trends(con)
```

**Step 3: Run pipeline and verify**

```bash
uv run python -m pipeline.build --force
```

```bash
uv run python -c "
import duckdb
con = duckdb.connect()
tbf = con.execute(\"SELECT year, COUNT(*) as zips, SUM(new_businesses) as total FROM 'data/aggregated/trend_business_formation.parquet' GROUP BY year ORDER BY year\").fetchdf()
print('Business formation by year:')
print(tbf.to_string())
print()
t311 = con.execute(\"SELECT year, COUNT(*) as months, SUM(total_requests) as total FROM 'data/aggregated/trend_311_monthly.parquet' GROUP BY year ORDER BY year\").fetchdf()
print('311 monthly by year:')
print(t311.to_string())
"
```

Expected: Business formation rows from 2015+ across multiple zips. 311 monthly from 2016-2026.

**Step 4: Commit**

```bash
git add pipeline/transform.py
git commit -m "feat: add trend parquets for business formation and 311 monthly"
```

---

## Task 6: Add area query functions to query layer

**Files:**
- Modify: `api/queries.py` — add 5 new area functions + update `get_filters()` + update `get_neighborhood_profile()`

**Step 1: Update `get_filters()` to include areas**

In `get_filters()` (~line 58), after the `statuses` assignment and before `con.close()`, add:

```python
    # areas from area_profile
    areas = []
    area_path = _AGG / "area_profile.parquet"
    if area_path.exists():
        areas = sorted(
            con.execute(
                f"SELECT DISTINCT area FROM '{area_path}' ORDER BY area"
            ).fetchdf()["area"].tolist()
        )
```

And update the return dict to include `"areas": areas`.

**Step 2: Update `get_neighborhood_profile()` to include `area`**

In `get_neighborhood_profile()`, the response dict is built across lines 236-285. Find where `"neighborhood"` is set and add the `area` field. The neighborhood_profile.parquet now has an `area` column, so:

After the line that sets `"neighborhood"`:

```python
        "area": _clean(row.get("area")),
```

**Step 3: Add `get_areas()` function**

Add after `get_rankings()`:

```python
def get_areas() -> list[dict]:
    """Get all areas with summary metrics."""
    path = _q("data/aggregated/area_profile.parquet")
    return _run(f"""
        SELECT
            area,
            zip_count,
            population,
            active_count,
            businesses_per_1k,
            median_income
        FROM '{path}'
        ORDER BY population DESC
    """)
```

**Step 4: Add `get_area_profile()` function**

```python
def get_area_profile(area: str) -> dict:
    """Get full area profile with demographics, businesses, civic signals."""
    ap_path = _q("data/aggregated/area_profile.parquet")
    row = _run_one(f"SELECT * FROM '{ap_path}' WHERE area = $1", [area])
    if not row:
        return {}

    # top categories for area
    abc_path = _q("data/aggregated/area_business_by_category.parquet")
    top_cats = _run(f"""
        SELECT category, active_count, total_count, per_1k
        FROM '{abc_path}'
        WHERE area = $1
        ORDER BY active_count DESC
        LIMIT 10
    """, [area])

    # city avg for comparison
    avg_path = _q("data/aggregated/city_averages.parquet")
    avg = _run_one(f"SELECT * FROM '{avg_path}'")

    # city avg per-1k for each category
    bz_path = _q("data/aggregated/business_by_zip.parquet")
    demo_path = _q("data/processed/demographics.parquet")
    for cat in top_cats:
        city_cat = _run_one(f"""
            SELECT ROUND(1000.0 * SUM(bz.active_count) / NULLIF(
                (SELECT SUM(population) FROM '{demo_path}'), 0
            ), 2) AS city_avg_per_1k
            FROM '{bz_path}' bz
            WHERE bz.category = $1
              AND bz.zip_code IN (SELECT zip_code FROM '{demo_path}')
        """, [cat["category"]])
        cat["city_avg_per_1k"] = _clean(city_cat.get("city_avg_per_1k")) if city_cat else None

    # comparison to avg
    comparison = {}
    compare_fields = [
        ("population", "avg_population"),
        ("median_income", "avg_median_income"),
        ("median_home_value", "avg_median_home_value"),
        ("median_rent", "avg_median_rent"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus"),
        ("active_count", "avg_active_businesses"),
        ("businesses_per_1k", "avg_businesses_per_1k"),
    ]
    for local_key, avg_key in compare_fields:
        local_val = row.get(local_key)
        avg_val = avg.get(avg_key) if avg else None
        if local_val is not None and avg_val is not None and avg_val != 0:
            comparison[local_key] = {
                "value": _clean(local_val),
                "city_avg": _clean(avg_val),
                "vs_avg_pct": _clean(round(100 * (local_val - avg_val) / abs(avg_val), 1)),
            }

    narrative = _build_area_narrative(row, avg)

    zip_codes = row.get("zip_codes", [])

    return {
        "area": row.get("area"),
        "zip_codes": zip_codes if isinstance(zip_codes, list) else [],
        "zip_count": _clean(row.get("zip_count")),
        "demographics": {
            "population": _clean(row.get("population")),
            "median_age": _clean(row.get("median_age")),
            "median_income": _clean(row.get("median_income")),
            "median_home_value": _clean(row.get("median_home_value")),
            "median_rent": _clean(row.get("median_rent")),
            "pct_bachelors_plus": _clean(row.get("pct_bachelors_plus")),
        },
        "business_landscape": {
            "active_count": _clean(row.get("active_count")),
            "total_count": _clean(row.get("total_count")),
            "businesses_per_1k": _clean(row.get("businesses_per_1k")),
            "top_categories": [{k: _clean(v) for k, v in c.items()} for c in top_cats],
        },
        "civic_signals": {
            "new_permits": _clean(row.get("new_permits")),
            "permit_valuation": _clean(row.get("permit_valuation")),
            "solar_installs": _clean(row.get("solar_installs")),
            "crime_count": _clean(row.get("crime_count")),
            "median_311_days": _clean(row.get("median_311_days")),
            "total_311_requests": _clean(row.get("total_311_requests")),
        },
        "comparison_to_avg": comparison,
        "narrative": narrative,
    }
```

**Step 5: Add `compare_areas()` function**

```python
def compare_areas(area_a: str, area_b: str) -> dict:
    """Compare two areas head-to-head."""
    profile_a = get_area_profile(area_a)
    profile_b = get_area_profile(area_b)

    if not profile_a or not profile_b:
        return {"error": "One or both areas not found"}

    compare_metrics = [
        ("population", "demographics"),
        ("median_income", "demographics"),
        ("median_home_value", "demographics"),
        ("median_rent", "demographics"),
        ("pct_bachelors_plus", "demographics"),
        ("active_count", "business_landscape"),
        ("businesses_per_1k", "business_landscape"),
        ("new_permits", "civic_signals"),
        ("crime_count", "civic_signals"),
        ("solar_installs", "civic_signals"),
        ("median_311_days", "civic_signals"),
    ]

    head_to_head = {}
    for metric, section in compare_metrics:
        val_a = profile_a.get(section, {}).get(metric)
        val_b = profile_b.get(section, {}).get(metric)
        diff = None
        if val_a is not None and val_b is not None:
            diff = _clean(round(100 * (val_a - val_b) / abs(val_b), 1)) if val_b != 0 else None
        head_to_head[metric] = {
            "area_a": _clean(val_a),
            "area_b": _clean(val_b),
            "difference": diff,
        }

    return {
        "area_a": profile_a,
        "area_b": profile_b,
        "head_to_head": head_to_head,
        "narrative": _build_area_comparison_narrative(profile_a, profile_b),
    }
```

**Step 6: Add `get_area_rankings()` function**

```python
def get_area_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Rank areas by a chosen metric."""
    ap_path = _q("data/aggregated/area_profile.parquet")
    direction = "DESC" if sort_desc else "ASC"

    if category:
        abc_path = _q("data/aggregated/area_business_by_category.parquet")
        if sort_by == "category_per_1k":
            order_col = "abc.per_1k"
        else:
            order_col = f"ap.{sort_by}"
        rows = _run(f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {order_col} {direction} NULLS LAST) AS rank,
                ap.area,
                ap.{sort_by} AS sort_value,
                '{sort_by}' AS sort_metric,
                abc.active_count AS category_active,
                abc.per_1k AS category_per_1k,
                '{category}' AS category,
                ap.population,
                ap.median_income,
                ap.active_count
            FROM '{ap_path}' ap
            LEFT JOIN '{abc_path}' abc ON ap.area = abc.area AND abc.category = $1
            ORDER BY {order_col} {direction} NULLS LAST
            LIMIT {limit}
        """, [category])
    else:
        rows = _run(f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {sort_by} {direction} NULLS LAST) AS rank,
                area,
                {sort_by} AS sort_value,
                '{sort_by}' AS sort_metric,
                NULL AS category,
                NULL AS category_active,
                NULL AS category_per_1k,
                population,
                median_income,
                active_count
            FROM '{ap_path}'
            ORDER BY {sort_by} {direction} NULLS LAST
            LIMIT {limit}
        """)

    return [{k: _clean(v) for k, v in r.items()} for r in rows]
```

**Step 7: Add `get_area_zips()` function**

```python
def get_area_zips(area: str) -> list[dict]:
    """Get all zips in an area with key metrics."""
    np_path = _q("data/aggregated/neighborhood_profile.parquet")
    return _run(f"""
        SELECT
            zip_code,
            neighborhood,
            population,
            active_count,
            businesses_per_1k,
            median_income,
            crime_count,
            new_permits
        FROM '{np_path}'
        WHERE area = $1
        ORDER BY active_count DESC
    """, [area])
```

**Step 8: Commit**

```bash
git add api/queries.py
git commit -m "feat: add area query functions (get_areas, get_area_profile, compare_areas, get_area_rankings, get_area_zips)"
```

---

## Task 7: Add trend query functions to query layer

**Files:**
- Modify: `api/queries.py` — add `get_zip_trends()`, `get_area_trends()`, narrative helpers

**Step 1: Add `get_zip_trends()` function**

```python
def get_zip_trends(zip_code: str) -> dict:
    """Get year-over-year trend data for a zip."""
    trends = {}

    # Business formation
    tbf_path = _AGG / "trend_business_formation.parquet"
    if tbf_path.exists():
        rows = _run(f"""
            SELECT year, new_businesses AS count
            FROM '{tbf_path}'
            WHERE zip_code = $1
            ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["business_formation"] = rows

    # Permits
    cp_path = _AGG / "civic_permits.parquet"
    if cp_path.exists():
        rows = _run(f"""
            SELECT year, SUM(permit_count) AS count
            FROM '{cp_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["permits"] = rows

    # Crime
    cc_path = _AGG / "civic_crime.parquet"
    if cc_path.exists():
        rows = _run(f"""
            SELECT year, SUM(count) AS count
            FROM '{cc_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["crime"] = rows

    # Solar
    cs_path = _AGG / "civic_solar.parquet"
    if cs_path.exists():
        rows = _run(f"""
            SELECT year, SUM(solar_count) AS count
            FROM '{cs_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["solar"] = rows

    return trends
```

**Step 2: Add `_add_yoy()` helper**

```python
def _add_yoy(rows: list[dict]) -> None:
    """Add yoy_pct to each row (in-place)."""
    for i, row in enumerate(rows):
        if i == 0 or rows[i - 1]["count"] is None or rows[i - 1]["count"] == 0:
            row["yoy_pct"] = None
        else:
            row["yoy_pct"] = _clean(
                round(100 * (row["count"] - rows[i - 1]["count"]) / abs(rows[i - 1]["count"]), 1)
            )
        row["count"] = _clean(row["count"])
        row["year"] = _clean(row["year"])
```

**Step 3: Add `get_area_trends()` function**

```python
def get_area_trends(area: str) -> dict:
    """Get year-over-year trend data aggregated across an area's zips."""
    from pipeline.transform import ZIP_TO_AREA
    zips = [z for z, a in ZIP_TO_AREA.items() if a == area]
    if not zips:
        return {}

    placeholders = ", ".join(f"'{z}'" for z in zips)
    trends = {}

    # Business formation
    tbf_path = _AGG / "trend_business_formation.parquet"
    if tbf_path.exists():
        rows = _run(f"""
            SELECT year, SUM(new_businesses) AS count
            FROM '{tbf_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["business_formation"] = rows

    # Permits
    cp_path = _AGG / "civic_permits.parquet"
    if cp_path.exists():
        rows = _run(f"""
            SELECT year, SUM(permit_count) AS count
            FROM '{cp_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["permits"] = rows

    # Crime
    cc_path = _AGG / "civic_crime.parquet"
    if cc_path.exists():
        rows = _run(f"""
            SELECT year, SUM(count) AS count
            FROM '{cc_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["crime"] = rows

    # Solar
    cs_path = _AGG / "civic_solar.parquet"
    if cs_path.exists():
        rows = _run(f"""
            SELECT year, SUM(solar_count) AS count
            FROM '{cs_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["solar"] = rows

    return trends
```

**Step 4: Add `_build_area_narrative()` and `_build_area_comparison_narrative()`**

Model these on the existing `_build_narrative()` (line 335) and `_build_comparison_narrative()` (line 402). Same metric comparison logic, but adapted language:

```python
def _build_area_narrative(row: dict, avg: dict) -> str:
    """Generate a one-sentence narrative comparing an area to city averages."""
    if not avg:
        return ""

    area = row.get("area", "this area")
    zip_count = row.get("zip_count", 0)
    pop = row.get("population")
    pop_str = f"pop {pop:,.0f}" if pop else ""

    metrics = [
        ("active_count", "avg_active_businesses", "businesses", "good", "fewer"),
        ("businesses_per_1k", "avg_businesses_per_1k", "businesses per 1k residents", "good", "fewer"),
        ("median_income", "avg_median_income", "median income", "good", "lower"),
        ("median_rent", "avg_median_rent", "median rent", "bad", "lower"),
        ("median_home_value", "avg_median_home_value", "home values", "bad", "lower"),
        ("crime_count", "avg_crime_count", "crime incidents", "bad", "fewer"),
        ("new_permits", "avg_new_permits", "new permits", "good", "fewer"),
        ("solar_installs", "avg_solar_installs", "solar installs", "good", "fewer"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus", "college-educated residents", "good", "fewer"),
    ]

    scored = []
    for row_key, avg_key, label, higher_is, less_word in metrics:
        local = row.get(row_key)
        city = avg.get(avg_key)
        if local is None or city is None or city == 0:
            continue
        ratio = local / city
        if 0.9 <= ratio <= 1.1:
            continue
        magnitude = abs(ratio - 1)
        is_higher = ratio > 1
        if ratio > 10 or (ratio > 0 and ratio < 0.1):
            mag_str = f"{ratio:.0f}x" if is_higher else f"1/{1/ratio:.0f}th the"
        elif ratio >= 2:
            mag_str = f"{ratio:.1f}x"
        elif ratio <= 0.5:
            mag_str = f"{100*(1-ratio):.0f}% {less_word}"
        elif is_higher:
            mag_str = f"{100*(ratio-1):.0f}% more"
        else:
            mag_str = f"{100*(1-ratio):.0f}% {less_word}"

        is_positive = (is_higher and higher_is == "good") or (not is_higher and higher_is == "bad")
        scored.append((magnitude, is_positive, f"{mag_str} {label}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]
    positives = [s[2] for s in top if s[1]]
    negatives = [s[2] for s in top if not s[1]]

    prefix = f"{area.lower()} ({zip_count} zip codes, {pop_str})" if pop_str else f"{area.lower()} ({zip_count} zip codes)"

    if not positives and not negatives:
        return f"{prefix} is close to city average across most metrics."

    parts = []
    if positives:
        parts.append(", ".join(positives))
    if negatives:
        neg_str = ", ".join(negatives)
        parts.append(f"but {neg_str}" if positives else neg_str)

    return f"compared to the avg sd area: {prefix} has {' — '.join(parts)}."


def _build_area_comparison_narrative(profile_a: dict, profile_b: dict) -> str:
    """Generate a comparison narrative for two areas."""
    name_a = profile_a.get("area", "area a")
    name_b = profile_b.get("area", "area b")

    metrics = [
        ("population", "demographics", "population", "neutral"),
        ("median_income", "demographics", "median income", "good"),
        ("businesses_per_1k", "business_landscape", "businesses per capita", "good"),
        ("active_count", "business_landscape", "total businesses", "good"),
        ("crime_count", "civic_signals", "crime incidents", "bad"),
        ("new_permits", "civic_signals", "new permits", "good"),
        ("median_311_days", "civic_signals", "311 response time", "bad"),
    ]

    a_wins = []
    b_wins = []
    for metric, section, label, higher_is in metrics:
        val_a = profile_a.get(section, {}).get(metric)
        val_b = profile_b.get(section, {}).get(metric)
        if val_a is None or val_b is None or val_b == 0:
            continue
        ratio = val_a / val_b
        if 0.9 <= ratio <= 1.1:
            continue
        pct = abs(ratio - 1) * 100
        desc = f"{pct:.0f}% {'more' if ratio > 1 else 'fewer'} {label}"
        a_better = (ratio > 1 and higher_is == "good") or (ratio < 1 and higher_is == "bad")
        if higher_is == "neutral":
            continue
        if a_better:
            a_wins.append((pct, desc))
        else:
            b_wins.append((pct, desc))

    a_wins.sort(reverse=True)
    b_wins.sort(reverse=True)
    a_top = [w[1] for w in a_wins[:3]]
    b_top = [w[1] for w in b_wins[:3]]

    if not a_top and not b_top:
        return f"{name_a.lower()} and {name_b.lower()} are similar across most metrics."

    parts = []
    if a_top:
        parts.append(f"{name_a.lower()} has {', '.join(a_top)}")
    if b_top:
        parts.append(f"{name_b.lower()} has {', '.join(b_top)}")

    return " — ".join(parts) + "."
```

**Step 5: Commit**

```bash
git add api/queries.py
git commit -m "feat: add trend query functions, area narratives, and YoY helpers"
```

---

## Task 8: Add Pydantic models + fix NeighborhoodProfile

**Files:**
- Modify: `api/models.py`

**Step 1: Fix `NeighborhoodProfile` — add missing fields**

Find the `NeighborhoodProfile` class (~line 66) and add:

```python
    narrative: str = ""
    area: str | None = None
```

**Step 2: Add new models**

Add after the existing models:

```python
class AreaSummary(BaseModel):
    area: str
    zip_count: int
    population: int | None = None
    active_count: int | None = None
    businesses_per_1k: float | None = None
    median_income: int | None = None


class AreaProfile(BaseModel):
    area: str
    zip_codes: list[str] = []
    zip_count: int
    demographics: Demographics
    business_landscape: BusinessLandscape
    civic_signals: CivicSignals
    comparison_to_avg: dict[str, ComparisonValue] = {}
    narrative: str = ""


class AreaComparison(BaseModel):
    area_a: AreaProfile
    area_b: AreaProfile
    head_to_head: dict[str, HeadToHeadMetric] = {}
    narrative: str = ""


class AreaRankingRow(BaseModel):
    rank: int
    area: str
    sort_metric: str
    sort_value: float | int | None = None
    category: str | None = None
    category_active: int | None = None
    category_per_1k: float | None = None
    population: int | None = None
    median_income: int | None = None
    active_count: int | None = None


class TrendPoint(BaseModel):
    year: int
    count: int | None = None
    yoy_pct: float | None = None


class TrendSeries(BaseModel):
    business_formation: list[TrendPoint] = []
    permits: list[TrendPoint] = []
    crime: list[TrendPoint] = []
    solar: list[TrendPoint] = []


class AreaZipSummary(BaseModel):
    zip_code: str
    neighborhood: str | None = None
    population: int | None = None
    active_count: int | None = None
    businesses_per_1k: float | None = None
    median_income: int | None = None
    crime_count: int | None = None
    new_permits: int | None = None
```

**Step 3: Commit**

```bash
git add api/models.py
git commit -m "feat: add area/trend Pydantic models, fix NeighborhoodProfile narrative field"
```

---

## Task 9: Add API endpoints

**Files:**
- Modify: `api/main.py`

**Step 1: Add new endpoints**

Add after existing endpoints, before the end of the file:

```python
@app.get("/areas", response_model=list[AreaSummary])
def areas():
    return queries.get_areas()


@app.get("/area-profile", response_model=AreaProfile)
def area_profile(area: str = Query(..., description="Area name")):
    result = queries.get_area_profile(area)
    if not result:
        raise HTTPException(404, f"Area '{area}' not found")
    return result


@app.get("/compare-areas", response_model=AreaComparison)
def compare_areas(
    area_a: str = Query(..., description="First area"),
    area_b: str = Query(..., description="Second area"),
):
    result = queries.compare_areas(area_a, area_b)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/area-rankings", response_model=list[AreaRankingRow])
def area_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
):
    return queries.get_area_rankings(sort_by, sort_desc, category, limit)


@app.get("/area-zips", response_model=list[AreaZipSummary])
def area_zips(area: str = Query(..., description="Area name")):
    return queries.get_area_zips(area)


@app.get("/zip-trends", response_model=TrendSeries)
def zip_trends(zip: str = Query(..., alias="zip", description="Zip code")):
    return queries.get_zip_trends(zip)


@app.get("/area-trends", response_model=TrendSeries)
def area_trends(area: str = Query(..., description="Area name")):
    return queries.get_area_trends(area)
```

**Step 2: Add imports**

Add to the imports at the top:

```python
from api.models import AreaSummary, AreaProfile, AreaComparison, AreaRankingRow, AreaZipSummary, TrendSeries
```

**Step 3: Verify endpoints**

```bash
uvicorn api.main:app --reload &
sleep 2
curl -s http://localhost:8000/areas | python -m json.tool | head -20
curl -s "http://localhost:8000/area-profile?area=Downtown" | python -m json.tool | head -30
curl -s "http://localhost:8000/zip-trends?zip=92101" | python -m json.tool | head -20
kill %1
```

**Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat: add 7 new API endpoints for areas and trends"
```

---

## Task 10: Add MCP tools

**Files:**
- Modify: `api/mcp_server.py`

**Step 1: Add new MCP tools**

Add after existing tools:

```python
@mcp.tool()
def get_areas() -> list[dict]:
    """List all San Diego areas with summary metrics.

    Returns area names, zip counts, population, business counts, and income.
    Use this to see what areas are available for deeper analysis."""
    return queries.get_areas()


@mcp.tool()
def get_area_profile(area: str) -> dict:
    """Get full area profile for a San Diego area.

    Returns demographics, business landscape, civic signals, comparison
    to city averages, and a narrative summary. Areas group nearby zip codes."""
    return queries.get_area_profile(area)


@mcp.tool()
def compare_areas(area_a: str, area_b: str) -> dict:
    """Compare two San Diego areas head-to-head.

    Returns both area profiles plus a comparison showing differences
    in demographics, business landscape, and civic signals."""
    return queries.compare_areas(area_a, area_b)


@mcp.tool()
def get_area_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Rank San Diego areas by a chosen metric.

    Sort by demographics, business metrics, or civic signals.
    Optionally filter by business category for category-specific density."""
    return queries.get_area_rankings(sort_by, sort_desc, category, limit)


@mcp.tool()
def get_area_zips(area: str) -> list[dict]:
    """Get all zip codes within a San Diego area with key metrics.

    Enables drill-down from area-level to zip-level analysis."""
    return queries.get_area_zips(area)


@mcp.tool()
def get_zip_trends(zip_code: str) -> dict:
    """Get year-over-year trend data for a San Diego zip code.

    Returns time-series data for business formation, permits, crime,
    and solar with YoY change percentages."""
    return queries.get_zip_trends(zip_code)


@mcp.tool()
def get_area_trends(area: str) -> dict:
    """Get year-over-year trend data for a San Diego area.

    Returns aggregated time-series across the area's zip codes
    for business formation, permits, crime, and solar."""
    return queries.get_area_trends(area)
```

**Step 2: Commit**

```bash
git add api/mcp_server.py
git commit -m "feat: add 7 new MCP tools for areas and trends"
```

---

## Task 11: Refactor dashboard to call query layer directly

**Files:**
- Modify: `dashboard/app.py` — replace duplicated SQL/narrative with imports from `api.queries`

**Context:** This is the critical refactor (spec 2.3.6). The dashboard currently has its own DuckDB query helper, duplicated `_build_narrative()`, duplicated `_build_comparison_narrative()`, and multiple `@st.cache_data` functions that duplicate query-layer logic. After this task, the dashboard calls `api.queries.*` for data and only handles presentation.

**Step 1: Replace the DuckDB query helper and imports**

Remove the `query()` helper (lines 46-49) and add:

```python
from api import queries
```

Keep `duckdb` import only if still needed for any remaining direct queries.

**Step 2: Replace `_load_profile()` with query layer call**

Replace the cached loader with:

```python
@st.cache_data(ttl=3600)
def _load_profile(zip_code: str):
    result = queries.get_neighborhood_profile(zip_code)
    return result
```

This now returns a dict (not a DataFrame). Update downstream code that accesses `row.get(...)` or `row["key"]` — the query layer already returns dicts, so most access patterns stay the same.

**Step 3: Replace `_load_city_avg()` with query layer**

The city averages are embedded in the profile's `comparison_to_avg` field. This function can be simplified or removed.

**Step 4: Remove duplicated `_build_narrative()` and `_build_comparison_narrative()`**

Delete `_build_narrative()` (lines 301-365) and `_build_comparison_narrative()` (lines 368-436). The narrative is now returned as part of the profile dict from the query layer.

**Step 5: Replace `_load_rankings()` with query layer**

Replace with:

```python
@st.cache_data(ttl=3600)
def _load_rankings(sort_by, sort_desc, category, limit):
    rows = queries.get_rankings(sort_by, sort_desc, category, limit)
    return pd.DataFrame(rows)
```

**Step 6: Replace other cached loaders similarly**

Each existing `_load_*` function should call the corresponding `queries.*` function and convert to DataFrame where needed.

**Step 7: Verify dashboard still works**

```bash
streamlit run dashboard/app.py
```

Walk through: explorer tab, compare tab, rankings tab. Verify all data loads, narratives display, charts render.

**Step 8: Commit**

```bash
git add dashboard/app.py
git commit -m "refactor: dashboard calls api.queries directly, remove duplicated SQL/narrative"
```

**Note:** This is the most complex task — it touches many parts of `app.py`. Take care with the dict-to-DataFrame conversions. The query layer returns `list[dict]` which converts via `pd.DataFrame(result)`. Single-row results return `dict` which can be used directly or converted via `pd.Series(result)`.

---

## Task 12: Dashboard area navigation + area explorer tab

**Files:**
- Modify: `dashboard/app.py` — add area/zip toggle, area dropdown, drill-down, area explorer

**Step 1: Add area-first sidebar navigation**

Replace the existing sidebar zip selection with a level picker:

```python
level = st.sidebar.radio("explore by", ["area", "zip code"], index=0, horizontal=True)
```

When `level == "area"`:
- Show area dropdown (populated from `queries.get_areas()`)
- Below it, optional "drill into zip" dropdown with zips in the selected area (from `queries.get_area_zips(area)`)

When `level == "zip code"`:
- Show existing zip dropdown (unchanged)

**Step 2: Build area explorer tab content**

When area selected (no zip drill):
- Area summary metrics (population, businesses, per_1k, income) with `st.metric(delta=...)`
- Constituent zips mini-table from `queries.get_area_zips(area)`
- Business categories chart from area profile's `top_categories`
- Area narrative from profile

**Step 3: Verify**

```bash
streamlit run dashboard/app.py
```

Test: select area mode, choose an area, see area profile. Drill into a zip, see zip profile with breadcrumb.

**Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add area-first navigation and area explorer tab"
```

---

## Task 13: Dashboard trend indicators + area compare/rankings

**Files:**
- Modify: `dashboard/app.py`

**Step 1: Add trend indicators to profile cards**

For both zip and area profiles, use `st.metric()` with `delta` parameter:

```python
# Get trends
trends = queries.get_zip_trends(zip_code)  # or get_area_trends(area)

# Show metric with trend
permits_trend = trends.get("permits", [])
latest_yoy = permits_trend[-1]["yoy_pct"] if permits_trend else None
st.metric("new permits", value=profile["civic_signals"]["new_permits"],
          delta=f"{latest_yoy:+.0f}% yoy" if latest_yoy is not None else None)
```

Only show delta when trend data is available.

**Step 2: Add area compare tab**

When area mode is active in compare tab:
- Two area dropdowns instead of zip dropdowns
- Call `queries.compare_areas(area_a, area_b)`
- Same layout: narrative, metrics table, category comparison chart

**Step 3: Add area rankings tab**

When area mode is active in rankings tab:
- Same controls (metric, category, sort, limit)
- Call `queries.get_area_rankings(...)`
- Display as dataframe

**Step 4: Verify all tabs**

```bash
streamlit run dashboard/app.py
```

Test: area compare, area rankings, trend arrows on profiles.

**Step 5: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add trend indicators, area compare and area rankings tabs"
```

---

## Task 14: Polish and verification

**Step 1: Full pipeline rebuild**

```bash
uv run python -m pipeline.build --force
```

Verify output:
- `area_profile.parquet` (~20 rows)
- `area_business_by_category.parquet` (~500 rows)
- `trend_business_formation.parquet` (800+ rows)
- `trend_311_monthly.parquet` (~118 rows)
- `neighborhood_profile.parquet` has `area` column
- Business dates are no longer NULL

**Step 2: API verification**

```bash
uvicorn api.main:app --reload &
sleep 2
# Test each new endpoint
curl -s http://localhost:8000/areas | python -m json.tool
curl -s "http://localhost:8000/area-profile?area=Downtown" | python -m json.tool
curl -s "http://localhost:8000/compare-areas?area_a=Downtown&area_b=Carlsbad" | python -m json.tool
curl -s "http://localhost:8000/area-rankings?sort_by=businesses_per_1k" | python -m json.tool
curl -s "http://localhost:8000/area-zips?area=Carlsbad" | python -m json.tool
curl -s "http://localhost:8000/zip-trends?zip=92101" | python -m json.tool
curl -s "http://localhost:8000/area-trends?area=Downtown" | python -m json.tool
# Test existing endpoints still work
curl -s "http://localhost:8000/neighborhood-profile?zip=92101" | python -m json.tool
curl -s http://localhost:8000/filters | python -m json.tool
kill %1
```

**Step 3: MCP verification**

Use MCP tools via Claude Code:
- `get_areas()` → list of ~20 areas
- `get_area_profile("Carlsbad")` → full profile with narrative
- `compare_areas("Downtown", "Carlsbad")` → head-to-head
- `get_zip_trends("92101")` → multi-year data
- `get_area_trends("Downtown")` → aggregated trends

**Step 4: Dashboard verification**

```bash
streamlit run dashboard/app.py
```

Test:
- Area mode: select area, see area profile
- Drill-down: area → zip
- Zip mode: existing behavior unchanged
- Compare: area vs area, zip vs zip
- Rankings: area rankings, zip rankings
- Trend indicators on profile cards
- Toggle between area/zip modes

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: phase 2 polish — verify all endpoints, MCP tools, and dashboard"
```
