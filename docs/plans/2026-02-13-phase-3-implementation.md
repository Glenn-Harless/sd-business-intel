# Phase 3: Spatial Intelligence & Competitive Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-layer interactive maps, a competitor analysis tab, trend visualizations, and enhanced crime detail to the SD Business Intel dashboard.

**Architecture:** Extends the existing pipeline→queries→API/MCP→dashboard pattern. Pipeline adds 6 new civic parquets + 1 zip centroid parquet via `CIVIC_SOURCES`. Query layer adds 7 new functions. API/MCP adds 5 endpoints/tools. Dashboard gets map layers, a new competitors tab, plotly trend charts, and detailed crime views.

**Tech Stack:** DuckDB, pydeck (HexagonLayer, ScatterplotLayer), plotly, Streamlit, FastAPI, FastMCP

---

### Task 1: Add map point and crime detail parquets to pipeline

**Files:**
- Modify: `pipeline/ingest_civic.py`

**Step 1: Add 6 new entries to CIVIC_SOURCES dict**

Add these entries after the existing `civic_energy.parquet` entry (line 23):

```python
CIVIC_SOURCES: dict[str, str] = {
    # ... existing 9 entries ...
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
```

**Step 2: Run pipeline to copy the new parquets**

Run: `uv run python -m pipeline.ingest_civic`
Expected: 6 new `[copy]` lines for map_311, map_permits, map_crime, map_solar, civic_crime_detail, civic_crime_temporal

**Step 3: Verify the files exist**

Run: `ls -la data/aggregated/map_*.parquet data/aggregated/civic_crime_detail.parquet data/aggregated/civic_crime_temporal.parquet`
Expected: 6 files totaling ~62MB

**Step 4: Commit**

```bash
git add pipeline/ingest_civic.py data/aggregated/map_311.parquet data/aggregated/map_permits.parquet data/aggregated/map_crime.parquet data/aggregated/map_solar.parquet data/aggregated/civic_crime_detail.parquet data/aggregated/civic_crime_temporal.parquet
git commit -m "feat: add Phase 3 map points and crime detail to pipeline"
```

---

### Task 2: Add zip centroid parquet to pipeline

**Files:**
- Modify: `pipeline/ingest_civic.py`

The Census Gazetteer file provides ZCTA centroids. We need a small pipeline step that downloads the TSV once and creates `data/aggregated/zip_centroids.parquet` filtered to our 82 profiled zips.

**Step 1: Add a `build_zip_centroids()` function to `pipeline/ingest_civic.py`**

Add after the `ingest()` function:

```python
def build_zip_centroids(*, force: bool = False) -> Path | None:
    """Download Census Gazetteer ZCTA centroids and save as parquet."""
    dest = AGG_DIR / "zip_centroids.parquet"
    if dest.exists() and not force:
        print("  [skip] zip_centroids.parquet (already exists)")
        return dest

    import csv
    import io
    import urllib.request

    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.txt"
    print(f"  [download] Census ZCTA centroids from {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [error] failed to download centroids: {e}")
        return None

    # Tab-delimited with columns: GEOID, ALAND, AWATER, ALAND_SQMI, AWATER_SQMI, INTPTLAT, INTPTLONG
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = []
    for row in reader:
        zcta = row.get("GEOID", "").strip()
        lat = row.get("INTPTLAT", "").strip()
        lng = row.get("INTPTLONG", "").strip()
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
```

**Step 2: Call `build_zip_centroids()` from the `ingest()` function**

Add at the end of `ingest()`, before `return copied`:

```python
    # Build zip centroids from Census Gazetteer
    centroid_path = build_zip_centroids(force=force)
    if centroid_path:
        copied.append(centroid_path)
```

**Step 3: Run the pipeline to generate centroids**

Run: `uv run python -m pipeline.ingest_civic --force`
Expected: `[download] Census ZCTA centroids...` then `[write] zip_centroids.parquet (XX zips)`

**Step 4: Verify the centroid file**

Run: `uv run python3 -c "import duckdb; con = duckdb.connect(); print(con.execute(\"SELECT COUNT(*), MIN(lat), MAX(lat) FROM 'data/aggregated/zip_centroids.parquet'\").fetchone())"`
Expected: tuple showing ~82 zips with lat range ~32.5-33.1

**Step 5: Commit**

```bash
git add pipeline/ingest_civic.py data/aggregated/zip_centroids.parquet
git commit -m "feat: add zip centroid parquet from Census Gazetteer"
```

---

### Task 3: Add query functions for map points, city trends, competitors, and crime detail

**Files:**
- Modify: `api/queries.py`

Add 7 new query functions to `api/queries.py`.

**Step 1: Add `get_zip_centroids()` helper**

Add after the `_add_yoy()` function:

```python
def get_zip_centroids() -> dict[str, tuple[float, float]]:
    """Load zip centroids as {zip_code: (lat, lng)} dict."""
    path = _AGG / "zip_centroids.parquet"
    if not path.exists():
        return {}
    rows = _run(f"SELECT zip_code, lat, lng FROM '{path}'")
    return {r["zip_code"]: (r["lat"], r["lng"]) for r in rows}
```

**Step 2: Add `get_map_points()`**

**IMPORTANT:** The centroid lookup is a separate `_run_one()` call, so it always uses `$1` regardless of what `idx` is at. The spatial bounds are injected as literal floats (safe — computed from our own centroid data, not user input).

```python
def get_map_points(
    layer: str,
    zip_code: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 50000,
) -> list[dict]:
    """Get lat/lng points for a map layer, optionally filtered by location and time."""
    layer_map = {
        "311": ("map_311.parquet", "request_year", "lng"),
        "permits": ("map_permits.parquet", "approval_year", "lng"),
        "crime": ("map_crime.parquet", "year", "lng"),
        "solar": ("map_solar.parquet", "year", "lng"),
    }
    if layer not in layer_map:
        return []

    filename, year_col, lng_col = layer_map[layer]
    path = _q(f"data/aggregated/{filename}")

    clauses = []
    params = []
    idx = 1

    if year_min is not None:
        clauses.append(f"{year_col} >= ${idx}")
        params.append(year_min)
        idx += 1
    if year_max is not None:
        clauses.append(f"{year_col} <= ${idx}")
        params.append(year_max)
        idx += 1

    # Spatial filter: if zip_code provided, filter to bounding box around zip centroid.
    # This is a SEPARATE query (via _run_one), so it always uses $1.
    if zip_code:
        centroid_path = _q("data/aggregated/zip_centroids.parquet")
        centroid = _run_one(
            f"SELECT lat, lng FROM '{centroid_path}' WHERE zip_code = $1",
            [zip_code],
        )
        if centroid and centroid.get("lat"):
            lat, lng = float(centroid["lat"]), float(centroid["lng"])
            # ~3 mile bounding box (0.05 degrees lat/lng)
            clauses.append(f"lat BETWEEN {lat - 0.05} AND {lat + 0.05}")
            clauses.append(f"{lng_col} BETWEEN {lng - 0.05} AND {lng + 0.05}")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM '{path}' {where} LIMIT {min(limit, 100000)}"
    return _run(sql, params if params else None)
```

**Step 3: Add `get_city_trends()`**

City trends computes per-zip averages (city total / number of zips with data per year), so the comparison line is at the same scale as an individual zip.

```python
def get_city_trends() -> dict:
    """Get city-wide per-zip average time-series for all trend metrics.

    Returns average-per-zip values (not city totals) so charts can
    compare a single zip against the typical zip on the same y-axis.
    """
    trends = {}

    tbf_path = _AGG / "trend_business_formation.parquet"
    if tbf_path.exists():
        rows = _run(f"""
            SELECT year,
                   ROUND(SUM(new_businesses) * 1.0 / COUNT(DISTINCT zip_code), 1) AS count
            FROM '{tbf_path}'
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["business_formation"] = rows

    cp_path = _AGG / "civic_permits.parquet"
    if cp_path.exists():
        rows = _run(f"""
            SELECT year,
                   ROUND(SUM(permit_count) * 1.0 / COUNT(DISTINCT zip_code), 1) AS count
            FROM '{cp_path}'
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["permits"] = rows

    cc_path = _AGG / "civic_crime.parquet"
    if cc_path.exists():
        rows = _run(f"""
            SELECT year,
                   ROUND(SUM(count) * 1.0 / COUNT(DISTINCT zip_code), 1) AS count
            FROM '{cc_path}'
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["crime"] = rows

    cs_path = _AGG / "civic_solar.parquet"
    if cs_path.exists():
        rows = _run(f"""
            SELECT year,
                   ROUND(SUM(solar_count) * 1.0 / COUNT(DISTINCT zip_code), 1) AS count
            FROM '{cs_path}'
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["solar"] = rows

    return trends
```

**Step 4: Add `get_competitors()`**

The "nearby" filter uses geographic proximity: zips within ~0.1 degrees (~7 miles) of the selected zip's centroid.

```python
def get_competitors(category: str, zip_code: str) -> dict:
    """Get competitor businesses in a category for a zip code + nearby context."""
    biz_path = _PROCESSED / "businesses.parquet"
    if not biz_path.exists():
        return {"businesses": [], "nearby_zips": [], "density": None, "city_avg_density": None}

    # Businesses in this category + zip
    businesses = _run(f"""
        SELECT business_name, address, zip_code, start_date, status
        FROM '{biz_path}'
        WHERE category = $1 AND zip_code = $2 AND status = 'active'
        ORDER BY business_name
    """, [category, zip_code])

    # Population for density
    np_path = _q("data/aggregated/neighborhood_profile.parquet")
    pop_row = _run_one(f"SELECT population FROM '{np_path}' WHERE zip_code = $1", [zip_code])
    population = pop_row.get("population") if pop_row else None

    density = None
    if population and population > 0:
        density = round(1000.0 * len(businesses) / population, 2)

    # City avg density for this category (total/total method)
    bz_path = _q("data/aggregated/business_by_zip.parquet")
    demo_path = _q("data/aggregated/demographics_by_zip.parquet")
    city_avg_row = _run_one(f"""
        SELECT ROUND(1000.0 * SUM(bz.active_count) / NULLIF(
            (SELECT SUM(population) FROM '{demo_path}'), 0
        ), 2) AS city_avg_density
        FROM '{bz_path}' bz
        WHERE bz.category = $1
          AND bz.zip_code IN (SELECT zip_code FROM '{demo_path}')
    """, [category])
    city_avg_density = _clean(city_avg_row.get("city_avg_density")) if city_avg_row else None

    # Nearby zips: geographically close (within ~0.1 deg / ~7 miles)
    centroid_path = _q("data/aggregated/zip_centroids.parquet")
    nearby = _run(f"""
        WITH center AS (
            SELECT lat, lng FROM '{centroid_path}' WHERE zip_code = $2
        )
        SELECT bz.zip_code, np.neighborhood, bz.active_count,
               CASE WHEN np.population > 0
                    THEN ROUND(1000.0 * bz.active_count / np.population, 2)
                    ELSE NULL END AS per_1k
        FROM '{bz_path}' bz
        JOIN '{np_path}' np ON bz.zip_code = np.zip_code
        JOIN '{centroid_path}' zc ON bz.zip_code = zc.zip_code
        CROSS JOIN center c
        WHERE bz.category = $1
          AND bz.active_count > 0
          AND ABS(zc.lat - c.lat) <= 0.1
          AND ABS(zc.lng - c.lng) <= 0.1
        ORDER BY bz.active_count DESC
        LIMIT 20
    """, [category, zip_code])

    return {
        "zip_code": zip_code,
        "category": category,
        "count": len(businesses),
        "businesses": businesses,
        "density": density,
        "city_avg_density": city_avg_density,
        "nearby_zips": [{k: _clean(v) for k, v in n.items()} for n in nearby],
    }
```

**Step 5: Add `get_crime_detail()` and `get_crime_temporal()`**

```python
def get_crime_detail(year: int | None = None) -> list[dict]:
    """Get city-wide offense group breakdown."""
    path = _q("data/aggregated/civic_crime_detail.parquet")
    if year:
        rows = _run(f"""
            SELECT offense_group, crime_against, SUM(count) AS count
            FROM '{path}'
            WHERE year = $1
            GROUP BY offense_group, crime_against
            ORDER BY count DESC
        """, [year])
    else:
        rows = _run(f"""
            SELECT offense_group, crime_against, SUM(count) AS count
            FROM '{path}'
            WHERE year = (SELECT MAX(year) FROM '{path}')
            GROUP BY offense_group, crime_against
            ORDER BY count DESC
        """)
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def get_crime_temporal(year: int | None = None) -> list[dict]:
    """Get day-of-week x month crime patterns (city-wide)."""
    path = _q("data/aggregated/civic_crime_temporal.parquet")
    if year:
        rows = _run(f"""
            SELECT dow, month, crime_against, SUM(count) AS count
            FROM '{path}'
            WHERE year = $1
            GROUP BY dow, month, crime_against
            ORDER BY dow, month
        """, [year])
    else:
        rows = _run(f"""
            SELECT dow, month, crime_against, SUM(count) AS count
            FROM '{path}'
            WHERE year = (SELECT MAX(year) FROM '{path}')
            GROUP BY dow, month, crime_against
            ORDER BY dow, month
        """)
    return [{k: _clean(v) for k, v in r.items()} for r in rows]
```

**Step 6: Verify queries work**

Run: `uv run python3 -c "from api.queries import get_city_trends, get_crime_detail, get_crime_temporal, get_zip_centroids, get_competitors, get_map_points; print('city trends keys:', list(get_city_trends().keys())); print('crime detail:', len(get_crime_detail())); print('crime temporal:', len(get_crime_temporal())); print('centroids:', len(get_zip_centroids())); c = get_competitors('Restaurants/Bars', '92101'); print('competitors:', c['count'], 'density:', c['density'], 'nearby:', len(c['nearby_zips'])); m = get_map_points('311', '92101', 2023, 2025, 100); print('map points:', len(m))"`

Expected: all functions return data without errors. Nearby count should be < 20 (geographically filtered, not all 82 zips).

**Step 7: Commit**

```bash
git add api/queries.py
git commit -m "feat: add query functions for map points, city trends, competitors, crime detail"
```

---

### Task 4: Add API endpoints and MCP tools

**Files:**
- Modify: `api/main.py`
- Modify: `api/mcp_server.py`

**Step 1: Add 5 new endpoints to `api/main.py`**

Add after the `/311-services` endpoint:

```python
@app.get("/map-points")
def map_points(
    layer: str = Query(..., description="Layer: 311, permits, crime, or solar"),
    zip_code: str | None = Query(None, description="Center zip code for spatial filter"),
    year_min: int | None = Query(None, description="Minimum year"),
    year_max: int | None = Query(None, description="Maximum year"),
    limit: int = Query(50000, ge=1, le=100000, description="Max points"),
):
    """Get lat/lng points for a map layer, filtered by location and time."""
    return queries.get_map_points(layer, zip_code, year_min, year_max, limit)


@app.get("/city-trends")
def city_trends():
    """City-wide per-zip average time-series for all trend metrics."""
    return queries.get_city_trends()


@app.get("/competitors")
def competitors(
    category: str = Query(..., description="Business category"),
    zip_code: str = Query(..., description="Center zip code"),
):
    """Competitor analysis: businesses in category for a zip + geographically nearby context."""
    return queries.get_competitors(category, zip_code)


@app.get("/crime-detail")
def crime_detail(year: int | None = Query(None, description="Year (default: latest)")):
    """City-wide offense group breakdown."""
    return queries.get_crime_detail(year)


@app.get("/crime-temporal")
def crime_temporal(year: int | None = Query(None, description="Year (default: latest)")):
    """Day-of-week x month crime patterns (city-wide)."""
    return queries.get_crime_temporal(year)
```

**Step 2: Add 5 new MCP tools to `api/mcp_server.py`**

Add before the `main()` function:

```python
@mcp.tool()
def get_map_points(
    layer: str,
    zip_code: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 50000,
) -> list[dict]:
    """Get lat/lng map points for a civic data layer.

    Layers: 311 (service requests), permits (construction), crime, solar.
    Optionally filter by zip_code (spatial bounding box) and year range."""
    return queries.get_map_points(layer, zip_code, year_min, year_max, limit)


@mcp.tool()
def get_city_trends() -> dict:
    """Get city-wide per-zip average time-series trends.

    Returns average-per-zip values for business formation, permits, crime,
    and solar. Use for chart comparison lines against individual zips."""
    return queries.get_city_trends()


@mcp.tool()
def find_competitors(category: str, zip_code: str) -> dict:
    """Find competitors in a business category near a zip code.

    Returns matching businesses, density per 1k residents, city average
    density, and geographically nearby zip codes with same-category counts."""
    return queries.get_competitors(category, zip_code)


@mcp.tool()
def get_crime_detail(year: int | None = None) -> list[dict]:
    """Get city-wide crime breakdown by offense group.

    Returns 36 offense types (larceny, assault, vandalism, etc.) with
    counts and crime_against category. Defaults to latest year."""
    return queries.get_crime_detail(year)


@mcp.tool()
def get_crime_temporal(year: int | None = None) -> list[dict]:
    """Get city-wide crime temporal patterns.

    Returns day-of-week x month breakdown showing when crime occurs.
    Useful for assessing risk during business operating hours."""
    return queries.get_crime_temporal(year)
```

**Step 3: Verify API starts**

Run: `timeout 5 uv run uvicorn api.main:app --port 8099 2>&1 || true`
Expected: "Uvicorn running on http://127.0.0.1:8099" (no import errors)

**Step 4: Commit**

```bash
git add api/main.py api/mcp_server.py
git commit -m "feat: add Phase 3 API endpoints and MCP tools"
```

---

### Task 5: Dashboard — replace ZIP_COORDS with centroid parquet + add map layers

**Files:**
- Modify: `dashboard/app.py`

This is the largest dashboard change. Replace the hardcoded `ZIP_COORDS` dict with a dynamic lookup from `zip_centroids.parquet`, and add multi-layer map rendering with toggles. The map supports both zip-level and area-level views.

**Step 1: Replace ZIP_COORDS dict with cached centroid loader**

Remove lines 503-537 (the `ZIP_COORDS` dict) and replace with:

```python
@st.cache_data(ttl=3600)
def _load_zip_coords() -> dict[str, tuple[float, float]]:
    """Load zip centroids from parquet (replaces hardcoded ZIP_COORDS)."""
    coords = queries.get_zip_centroids()
    if not coords:
        # Fallback to a few key zips if parquet missing
        return {"92101": (32.7194, -117.1628)}
    return coords

ZIP_COORDS = _load_zip_coords()
```

**Step 2: Add map data loaders**

Add after the `_load_311_services` cached function (around line 234):

```python
@st.cache_data(ttl=3600)
def _load_map_layer(layer: str, zip_code: str | None = None,
                     year_min: int | None = None, year_max: int | None = None):
    """Load map points for a layer, filtered by location and time."""
    rows = queries.get_map_points(layer, zip_code, year_min, year_max, limit=80000)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
```

**Step 3: Create a `_render_map` helper function**

This supports both zip-level and area-level views. When an area is selected, it computes the center from constituent zip centroids and uses wider zoom.

Add after the `_latest_yoy` function:

```python
def _render_map(zip_code: str | None = None, area: str | None = None,
                key_prefix: str = "map"):
    """Render multi-layer interactive map for a zip code or area.

    Area mode: centers on average of constituent zip centroids, zoom 11.
    Zip mode: centers on zip centroid, zoom 13.
    """
    # Determine center and zoom
    if area and not zip_code:
        # Area mode: average constituent zip centroids
        area_zips_data = queries.get_area_zips(area)
        area_zip_codes = [z["zip_code"] for z in area_zips_data] if area_zips_data else []
        lats, lngs = [], []
        for zc in area_zip_codes:
            if zc in ZIP_COORDS:
                lat, lng = ZIP_COORDS[zc]
                lats.append(lat)
                lngs.append(lng)
        if lats:
            center_lat = sum(lats) / len(lats)
            center_lng = sum(lngs) / len(lngs)
        else:
            center_lat, center_lng = 32.7157, -117.1611
        zoom = 11
        # Use first constituent zip for spatial filtering (wider bbox in query)
        filter_zip = area_zip_codes[0] if area_zip_codes else None
    else:
        center_lat, center_lng = ZIP_COORDS.get(zip_code, (32.7157, -117.1611))
        zoom = 13
        filter_zip = zip_code

    # Unique key suffix to avoid widget conflicts
    key_id = zip_code or area or "default"

    # Layer toggles
    toggle_cols = st.columns(4)
    show_311 = toggle_cols[0].checkbox("311 requests", value=True,
                                        key=f"map_311_{key_prefix}_{key_id}")
    show_permits = toggle_cols[1].checkbox("permits", value=False,
                                            key=f"map_permits_{key_prefix}_{key_id}")
    show_crime = toggle_cols[2].checkbox("crime", value=False,
                                          key=f"map_crime_{key_prefix}_{key_id}")
    show_solar = toggle_cols[3].checkbox("solar", value=False,
                                          key=f"map_solar_{key_prefix}_{key_id}")

    # Year range
    yr_min, yr_max = st.slider(
        "year range", min_value=2019, max_value=2025, value=(2022, 2025),
        key=f"map_year_{key_prefix}_{key_id}",
    )

    layers = []

    if show_311:
        df_311 = _load_map_layer("311", filter_zip, yr_min, yr_max)
        if not df_311.empty:
            layers.append(pdk.Layer(
                "HexagonLayer",
                data=df_311,
                get_position=["lng", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 500],
                extruded=False,
                pickable=True,
                color_range=[
                    [65, 182, 196],
                    [127, 205, 187],
                    [199, 233, 180],
                    [237, 248, 177],
                    [255, 255, 204],
                    [255, 237, 160],
                ],
            ))

    if show_permits:
        df_permits = _load_map_layer("permits", filter_zip, yr_min, yr_max)
        if not df_permits.empty:
            # Color by type: solar=green, other=blue
            df_permits = df_permits.copy()
            df_permits["color_r"] = df_permits.apply(
                lambda r: 76 if r.get("is_solar") else 131, axis=1
            )
            df_permits["color_g"] = df_permits.apply(
                lambda r: 175 if r.get("is_solar") else 201, axis=1
            )
            df_permits["color_b"] = df_permits.apply(
                lambda r: 80 if r.get("is_solar") else 255, axis=1
            )
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_permits,
                get_position=["lng", "lat"],
                get_fill_color=["color_r", "color_g", "color_b", 160],
                get_radius=40,
                pickable=True,
            ))

    if show_crime:
        df_crime = _load_map_layer("crime", filter_zip, yr_min, yr_max)
        if not df_crime.empty:
            layers.append(pdk.Layer(
                "HexagonLayer",
                data=df_crime,
                get_position=["lng", "lat"],
                radius=100,
                elevation_scale=4,
                extruded=False,
                pickable=True,
                color_range=[
                    [255, 255, 178],
                    [254, 204, 92],
                    [253, 141, 60],
                    [240, 59, 32],
                    [189, 0, 38],
                    [128, 0, 38],
                ],
            ))

    if show_solar:
        df_solar = _load_map_layer("solar", filter_zip, yr_min, yr_max)
        if not df_solar.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_solar,
                get_position=["lng", "lat"],
                get_fill_color=[76, 175, 80, 180],
                get_radius=50,
                pickable=True,
            ))

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=center_lat, longitude=center_lng, zoom=zoom, pitch=0,
        ),
        layers=layers,
        map_style="mapbox://styles/mapbox/light-v11",
    ))
```

**Step 4: Replace the empty map in `_render_zip_explorer()`**

Find the map section in `_render_zip_explorer()` (lines 813-827) and replace:

```python
    # Map
    st.subheader("map")
    _render_map(zip_code=zip_code, key_prefix=key_prefix)
```

This replaces the empty `pdk.Deck(layers=[])` with the multi-layer map.

**Step 5: Add map to area overview**

In the area overview section (inside `if level == "area" and selected_area and not drilldown_zip:`), add a map section. Insert after the constituent zips table (after `st.info("no zip code data available for this area")` block, around line 980), before the business categories section:

```python
            st.divider()
            st.subheader("map")
            _render_map(area=selected_area, key_prefix=f"area_{selected_area}")
```

**Step 6: Verify the dashboard loads**

Run: `uv run streamlit run dashboard/app.py`
Expected: Explorer tab shows map with 311 layer toggled on by default, layer checkboxes visible. Area mode shows map centered wider with zoom 11.

**Step 7: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add multi-layer interactive map to explorer and area views"
```

---

### Task 6: Dashboard — add trend visualizations to explorer

**Files:**
- Modify: `dashboard/app.py`

**Step 1: Add city trends cached loader**

Add after `_load_map_layer`:

```python
@st.cache_data(ttl=3600)
def _load_city_trends() -> dict:
    """Load city-wide per-zip average trends for chart comparison lines."""
    return queries.get_city_trends()
```

**Step 2: Create a `_render_trend_charts` helper**

Both the zip/area line and city average line are plotted on the SAME y-axis (same scale) for direct visual comparison. The city average is per-zip average, not city total.

Add after `_render_map`:

```python
def _render_trend_charts(zip_code: str | None = None, area: str | None = None,
                          key_prefix: str = "trends"):
    """Render trend line charts with city per-zip average comparison."""
    trends = _load_trends(zip_code=zip_code, area=area)
    city_trends = _load_city_trends()

    if not trends:
        return

    label = zip_code or area or "selected"

    chart_configs = [
        ("business_formation", "business formation"),
        ("permits", "construction permits"),
        ("crime", "crime incidents"),
        ("solar", "solar installations"),
    ]

    first = True
    for series_key, series_label in chart_configs:
        data = trends.get(series_key, [])
        if not data:
            continue

        city_data = city_trends.get(series_key, [])

        years = [d["year"] for d in data]
        counts = [d["count"] for d in data]
        year_range = f"{min(years)}-{max(years)}" if years else ""

        with st.expander(f"{series_label} ({year_range})", expanded=first):
            fig = go.Figure()

            # Primary line: selected zip/area
            fig.add_trace(go.Scatter(
                x=years, y=counts,
                mode="lines+markers",
                name=label,
                line=dict(color=CHART_COLOR, width=2),
                hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
            ))

            # City per-zip average line (same y-axis for direct comparison)
            if city_data:
                city_years = [d["year"] for d in city_data]
                city_counts = [d["count"] for d in city_data]
                fig.add_trace(go.Scatter(
                    x=city_years, y=city_counts,
                    mode="lines",
                    name="city avg per zip",
                    line=dict(color="#999", width=1, dash="dash"),
                    hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
                ))

            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="year",
                yaxis_title="count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"trend_{series_key}_{key_prefix}")

        first = False
```

**Step 3: Add trend charts to `_render_zip_explorer()`**

Insert after the map section (after `_render_map(zip_code=zip_code, key_prefix=key_prefix)`):

```python
    st.divider()
    st.subheader("trends")
    _render_trend_charts(zip_code=zip_code, key_prefix=f"zip_{key_prefix}_{zip_code}")
```

**Step 4: Add trend charts to area overview**

In the area overview section, insert after the area map section (after `_render_map(area=selected_area, ...)`):

```python
            st.divider()
            st.subheader("trends")
            _render_trend_charts(area=selected_area, key_prefix=f"area_{selected_area}")
```

**Step 5: Verify**

Run: `uv run streamlit run dashboard/app.py`
Expected: Trend expanders show after the map in both zip and area views. First chart expanded by default. Each shows two lines on the same y-axis: the selected zip/area line and the city avg per zip (dashed gray).

**Step 6: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add trend line charts to explorer profiles"
```

---

### Task 7: Dashboard — add competitors tab

**Files:**
- Modify: `dashboard/app.py`

**Step 1: Add the competitors tab to the tab bar**

Change the tab creation line (line 833):

```python
tab_explorer, tab_compare, tab_rankings, tab_competitors = st.tabs(
    ["explorer", "compare", "rankings", "competitors"]
)
```

**Step 2: Add cached competitor loader**

```python
@st.cache_data(ttl=3600)
def _load_competitors(category: str, zip_code: str) -> dict:
    """Load competitor analysis via the query layer."""
    return queries.get_competitors(category, zip_code)
```

**Step 3: Add the competitors tab content**

Add after the rankings tab section (before the footer):

```python
# ── COMPETITORS TAB ──
with tab_competitors:
    st.subheader("competitive landscape")
    st.caption("find competitors in any business category")

    comp_cols = st.columns(2)
    with comp_cols[0]:
        categories = _category_options()
        comp_cat = st.selectbox(
            "business category",
            categories if categories else ["No categories available"],
            key="comp_category",
        )
    with comp_cols[1]:
        comp_zip_labels = [f"{z} — {n}" if n else z for z, n in zip_options]
        comp_zip_sel = st.selectbox(
            "zip code",
            comp_zip_labels,
            index=next((i for i, (z, _) in enumerate(zip_options) if z == (selected_zip or "92101")), 0),
            key="comp_zip",
        )
        comp_zip_idx = comp_zip_labels.index(comp_zip_sel)
        comp_zip = zip_options[comp_zip_idx][0]

    if comp_cat and comp_zip:
        comp_data = _load_competitors(comp_cat, comp_zip)

        # Summary metrics
        r1 = st.columns(3)
        r1[0].metric(f"{comp_cat.lower()} in {comp_zip}", comp_data.get("count", 0))
        density = comp_data.get("density")
        city_avg = comp_data.get("city_avg_density")
        if density is not None:
            delta = None
            if city_avg:
                diff = density - city_avg
                delta = f"{diff:+.1f} vs city avg ({city_avg}/1k)"
            r1[1].metric("per 1,000 residents", f"{density:.1f}", delta, delta_color="off")
        if city_avg is not None:
            r1[2].metric("city avg density", f"{city_avg:.1f}/1k")

        st.divider()

        # Nearby zips comparison (geographically filtered)
        nearby = comp_data.get("nearby_zips", [])
        if nearby:
            st.subheader(f"nearby zip codes with {comp_cat.lower()}")

            # Map: scatter dots sized by competitor count
            nearby_with_coords = []
            for n in nearby:
                zc = n["zip_code"]
                if zc in ZIP_COORDS:
                    lat, lng = ZIP_COORDS[zc]
                    nearby_with_coords.append({
                        "zip_code": zc,
                        "neighborhood": n.get("neighborhood", ""),
                        "count": n.get("active_count", 0),
                        "per_1k": n.get("per_1k", 0),
                        "lat": lat,
                        "lng": lng,
                        "is_selected": zc == comp_zip,
                    })

            if nearby_with_coords:
                map_df = pd.DataFrame(nearby_with_coords)
                map_df["color_r"] = map_df["is_selected"].apply(lambda x: 255 if x else 131)
                map_df["color_g"] = map_df["is_selected"].apply(lambda x: 107 if x else 201)
                map_df["color_b"] = map_df["is_selected"].apply(lambda x: 107 if x else 255)
                map_df["radius"] = map_df["count"].apply(
                    lambda c: max(200, min(1500, c * 30))
                )

                center_lat, center_lng = ZIP_COORDS.get(comp_zip, (32.7157, -117.1611))

                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=center_lat, longitude=center_lng,
                        zoom=11, pitch=0,
                    ),
                    layers=[pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position=["lng", "lat"],
                        get_fill_color=["color_r", "color_g", "color_b", 180],
                        get_radius="radius",
                        pickable=True,
                    )],
                    tooltip={
                        "text": "{zip_code} ({neighborhood})\n{count} businesses\n{per_1k}/1k residents"
                    },
                    map_style="mapbox://styles/mapbox/light-v11",
                ))

            # Table
            nearby_df = pd.DataFrame(nearby)
            rename = {
                "zip_code": "zip code",
                "neighborhood": "neighborhood",
                "active_count": "count",
                "per_1k": "per 1k residents",
            }
            display_cols = [c for c in rename if c in nearby_df.columns]
            nearby_display = nearby_df[display_cols].rename(columns=rename)

            # Add vs city avg column
            if city_avg and city_avg > 0:
                nearby_display["vs city avg"] = nearby_display["per 1k residents"].apply(
                    lambda x: f"{((x / city_avg) - 1) * 100:+.0f}%" if x else "\u2014"
                )

            st.dataframe(nearby_display, use_container_width=True, hide_index=True,
                          height=min(400, 35 * len(nearby_display) + 38))

        st.divider()

        # Business directory
        businesses = comp_data.get("businesses", [])
        if businesses:
            st.subheader(f"business directory ({len(businesses)} results)")
            biz_df = pd.DataFrame(businesses)
            display_biz_cols = ["business_name", "address", "start_date"]
            display_biz = biz_df[[c for c in display_biz_cols if c in biz_df.columns]]
            display_biz = display_biz.rename(columns={
                "business_name": "name",
                "start_date": "since",
            })
            st.dataframe(display_biz, use_container_width=True, hide_index=True,
                          height=min(400, 35 * len(display_biz) + 38))
        else:
            st.info(f"no {comp_cat.lower()} businesses found in {comp_zip}")
```

**Step 4: Verify**

Run: `uv run streamlit run dashboard/app.py`
Expected: 4 tabs visible, competitors tab shows category/zip selectors, density metrics, nearby zip map + table (geographically filtered), business directory

**Step 5: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add competitors tab with density analysis and business directory"
```

---

### Task 8: Dashboard — enhanced crime detail section

**Files:**
- Modify: `dashboard/app.py`

**Step 1: Add cached loaders for crime detail data**

```python
@st.cache_data(ttl=3600)
def _load_crime_detail(year: int | None = None) -> list[dict]:
    """Load city-wide crime detail by offense group."""
    return queries.get_crime_detail(year)


@st.cache_data(ttl=3600)
def _load_crime_temporal(year: int | None = None) -> list[dict]:
    """Load crime temporal patterns (day x month)."""
    return queries.get_crime_temporal(year)
```

**Step 2: Create a `_render_crime_detail` helper**

```python
def _render_crime_detail(key_prefix: str):
    """Render city-wide crime detail and temporal pattern expanders."""
    # City-wide crime types
    with st.expander("city-wide crime types (detailed)", expanded=False):
        detail = _load_crime_detail()
        if detail:
            det_df = pd.DataFrame(detail)
            fig_det = go.Figure(go.Bar(
                x=det_df["count"],
                y=det_df["offense_group"],
                orientation="h",
                marker_color=[
                    "#ff6b6b" if r.get("crime_against") == "Person"
                    else CHART_COLOR if r.get("crime_against") == "Property"
                    else "#ffa500"
                    for r in detail
                ],
                text=det_df.apply(
                    lambda r: f"{int(r['count']):,}  ({r['crime_against']})", axis=1
                ),
                textposition="outside",
            ))
            fig_det.update_layout(
                height=max(400, len(det_df) * 22),
                margin=dict(l=0, r=100, t=0, b=0),
                yaxis=dict(autorange="reversed"),
                xaxis_title="incidents",
            )
            st.plotly_chart(fig_det, use_container_width=True,
                            key=f"crime_det_{key_prefix}")
        else:
            st.info("city-wide crime detail not available")

    # Temporal patterns
    with st.expander("when does crime happen? (city-wide)", expanded=False):
        temporal = _load_crime_temporal()
        if temporal:
            t_df = pd.DataFrame(temporal)
            # Aggregate across crime_against for the heatmap
            heatmap_df = t_df.groupby(["dow", "month"])["count"].sum().reset_index()

            # Pivot for heatmap
            pivot = heatmap_df.pivot(index="dow", columns="month", values="count").fillna(0)

            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            # Ensure all dow/month values present
            pivot = pivot.reindex(index=range(0, 7), columns=range(1, 13), fill_value=0)

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=month_labels,
                y=dow_labels,
                colorscale="YlOrRd",
                hovertemplate="%{y}, %{x}: %{z:,} incidents<extra></extra>",
            ))
            fig_heat.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_heat, use_container_width=True,
                            key=f"crime_temp_{key_prefix}")
        else:
            st.info("crime temporal data not available")
```

**Step 3: Add crime detail to `_render_zip_explorer()`**

In `_render_zip_explorer()`, after the existing crime breakdown expander and before the permit timelines section, add:

```python
    _render_crime_detail(f"{key_prefix}_{zip_code}")
```

**Step 4: Add crime detail to area overview**

In the area overview section, after the existing crime breakdown expander, add:

```python
            _render_crime_detail(f"area_{selected_area}")
```

**Step 5: Verify**

Run: `uv run streamlit run dashboard/app.py`
Expected: Crime section shows zip-level 3-category breakdown + city-wide detail expander + temporal heatmap expander

**Step 6: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add enhanced crime detail with offense types and temporal patterns"
```

---

### Task 9: Update models, health check, and memory

**Files:**
- Modify: `api/models.py`
- Modify: `api/queries.py` (health check)

**Step 1: Add new response models to `api/models.py`**

Add after the existing models. Use `Field(default_factory=list)` to avoid mutable default arguments:

```python
from pydantic import Field


class CompetitorResult(BaseModel):
    zip_code: str
    category: str
    count: int
    businesses: list[dict] = Field(default_factory=list)
    density: float | None = None
    city_avg_density: float | None = None
    nearby_zips: list[dict] = Field(default_factory=list)


class CrimeDetail(BaseModel):
    offense_group: str
    crime_against: str
    count: int


class CrimeTemporal(BaseModel):
    dow: int
    month: int
    crime_against: str
    count: int
```

**Step 2: Add new map parquets to health check in `queries.py`**

In the `get_health()` function, add to the `files` dict:

```python
        "map_311": _AGG / "map_311.parquet",
        "map_permits": _AGG / "map_permits.parquet",
        "map_crime": _AGG / "map_crime.parquet",
        "map_solar": _AGG / "map_solar.parquet",
        "zip_centroids": _AGG / "zip_centroids.parquet",
        "civic_crime_detail": _AGG / "civic_crime_detail.parquet",
        "civic_crime_temporal": _AGG / "civic_crime_temporal.parquet",
```

**Step 3: Commit**

```bash
git add api/models.py api/queries.py
git commit -m "feat: add Phase 3 response models and health check entries"
```

---

### Task 10: Smoke test and integration verification

**Files:**
- Create: `tests/test_phase3_smoke.py`

**Step 1: Write automated smoke tests for new query functions and edge cases**

```python
"""Smoke tests for Phase 3 query functions."""

from api import queries


def test_get_zip_centroids():
    coords = queries.get_zip_centroids()
    assert len(coords) > 50, f"Expected 50+ zip centroids, got {len(coords)}"
    assert "92101" in coords
    lat, lng = coords["92101"]
    assert 32.0 < lat < 33.5
    assert -118.0 < lng < -116.5


def test_get_map_points_basic():
    """Map points return data for each layer."""
    for layer in ("311", "permits", "crime", "solar"):
        pts = queries.get_map_points(layer, limit=10)
        assert len(pts) > 0, f"No map points for layer {layer}"
        assert "lat" in pts[0]


def test_get_map_points_with_filters():
    """Map points filtered by zip + year range don't error."""
    pts = queries.get_map_points("311", zip_code="92101", year_min=2023, year_max=2025, limit=100)
    assert isinstance(pts, list)
    # With spatial + year filter, should have fewer points than unfiltered
    all_pts = queries.get_map_points("311", limit=100)
    assert len(pts) <= len(all_pts)


def test_get_map_points_invalid_layer():
    pts = queries.get_map_points("invalid_layer")
    assert pts == []


def test_get_city_trends():
    trends = queries.get_city_trends()
    assert "business_formation" in trends
    # City avg should be per-zip average, not city total
    biz = trends["business_formation"]
    assert len(biz) > 0
    # Per-zip average should be reasonable (< 500 per zip per year)
    for row in biz:
        if row["count"] is not None:
            assert row["count"] < 500, f"City avg {row['count']} seems like total, not per-zip avg"


def test_get_competitors():
    result = queries.get_competitors("Restaurants/Bars", "92101")
    assert result["zip_code"] == "92101"
    assert result["category"] == "Restaurants/Bars"
    assert isinstance(result["businesses"], list)
    assert isinstance(result["nearby_zips"], list)

    # Nearby should be geographically close, not all 82 zips
    if result["nearby_zips"]:
        assert len(result["nearby_zips"]) <= 20


def test_get_competitors_nearby_geographic():
    """Verify nearby zips are actually geographically close."""
    result = queries.get_competitors("Restaurants/Bars", "92101")
    coords = queries.get_zip_centroids()
    center_lat, center_lng = coords.get("92101", (0, 0))

    for nz in result["nearby_zips"]:
        zc = nz["zip_code"]
        if zc in coords:
            lat, lng = coords[zc]
            assert abs(lat - center_lat) <= 0.11, f"{zc} too far north/south"
            assert abs(lng - center_lng) <= 0.11, f"{zc} too far east/west"


def test_get_crime_detail():
    detail = queries.get_crime_detail()
    assert len(detail) > 0
    assert "offense_group" in detail[0]
    assert "crime_against" in detail[0]
    assert "count" in detail[0]


def test_get_crime_temporal():
    temporal = queries.get_crime_temporal()
    assert len(temporal) > 0
    assert "dow" in temporal[0]
    assert "month" in temporal[0]
```

**Step 2: Run the tests**

Run: `uv run pytest tests/test_phase3_smoke.py -v`
Expected: All tests pass

**Step 3: Manual verification checklist**

Run: `uv run streamlit run dashboard/app.py`

Verify:
- [ ] Explorer tab (zip mode): map shows 311 heatmap by default, toggle layers on/off, year slider works
- [ ] Explorer tab (zip mode): trend charts show after map with city avg per zip comparison
- [ ] Explorer tab (zip mode): crime detail expanders show offense types and temporal heatmap
- [ ] Explorer tab (area mode): map centered on area with zoom 11
- [ ] Explorer tab (area mode): trends and crime detail work at area level
- [ ] Competitors tab: category + zip selectors work, density metrics display
- [ ] Competitors tab: nearby map shows geographically close zips only
- [ ] Competitors tab: business directory shows matching businesses
- [ ] Compare tab: still works correctly (no regression)
- [ ] Rankings tab: still works correctly

**Step 4: Commit**

```bash
git add tests/test_phase3_smoke.py
git commit -m "test: add Phase 3 smoke tests for new query functions"
```

**Step 5: Final commit if any remaining changes**

```bash
git add -A
git commit -m "feat: complete Phase 3 — spatial intelligence and competitive analysis"
```

---

## Summary

| Task | What | Files | Key changes |
|------|------|-------|-------------|
| 1 | Pipeline: map points + crime data | `ingest_civic.py` | 6 new CIVIC_SOURCES entries |
| 2 | Pipeline: zip centroids | `ingest_civic.py` | `build_zip_centroids()` from Census Gazetteer |
| 3 | Query layer | `queries.py` | 7 new functions (centroids, map points, city trends, competitors, crime detail/temporal) |
| 4 | API + MCP | `main.py`, `mcp_server.py` | 5 endpoints, 5 tools |
| 5 | Dashboard: maps | `app.py` | `_render_map()` with zip + area support, layer toggles |
| 6 | Dashboard: trends | `app.py` | `_render_trend_charts()` with same-axis city avg |
| 7 | Dashboard: competitors | `app.py` | new tab with geographic nearby filter |
| 8 | Dashboard: crime detail | `app.py` | `_render_crime_detail()` with offense types + heatmap |
| 9 | Models + health | `models.py`, `queries.py` | models with Field(default_factory), health entries |
| 10 | Smoke tests + verification | `tests/test_phase3_smoke.py` | automated regression tests |

**Cumulative totals after Phase 3:** ~31 parquets, 24 API endpoints, 23 MCP tools
