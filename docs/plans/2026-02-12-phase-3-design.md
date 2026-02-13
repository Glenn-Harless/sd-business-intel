# Phase 3: Spatial Intelligence & Competitive Analysis

## Overview

Phase 3 transforms SD Business Intel from a data lookup tool into a spatial intelligence platform. The core thesis: entrepreneurs don't think in zip codes and percentile rankings — they think in maps, competitors, and trends over time. This phase adds the visual and spatial layers that make the data actionable.

**Four sub-phases:**
- **3a** — Geospatial visualization (multi-layer interactive maps)
- **3b** — Competitive intelligence (competitor mapping + proximity search)
- **3c** — Trend visualizations (time-series charts for all metrics)
- **3d** — Enhanced crime context (detailed taxonomy + temporal patterns)

---

## Phase 3a: Geospatial Visualization

### Problem

The dashboard has pydeck integrated and a hardcoded `ZIP_COORDS` dict with 32 of 82 zip centroids, but the maps are empty placeholders. The explorer tab renders a blank basemap centered on a zip centroid. The compare tab shows two colored dots. Meanwhile, 4 sibling projects contain **4.86 million geocoded civic data points** sitting unused:

| Source | Parquet | Rows | Columns |
|--------|---------|------|---------|
| sd-get-it-done | map_points.parquet | 2,976,516 | lat, lng, service_name, request_year, comm_plan_name, council_district |
| sd-housing-permits | map_points.parquet | 1,176,239 | lat, lng, approval_type_clean, approval_year, valuation, total_du, is_housing, is_solar, zip_code |
| sd-public-safety | map_points.parquet | 597,040 | lat, lng, offense_group, crime_against, agency, year, city |
| sd-climate-action | solar_map_points.parquet | 115,284 | lat, lng, year, valuation, zip_code, approval_days, policy_era |

An entrepreneur evaluating a neighborhood should be able to *see* where crime clusters, where construction is booming, where 311 complaints concentrate, and where solar adoption is spreading — not just read percentile rankings.

### What It Does

Adds a multi-layer interactive map to the explorer tab that visualizes civic activity around any selected zip code or area. Users toggle layers on/off and see spatial patterns that aggregate statistics can't reveal.

### How It Should Look

**Explorer tab — map section (replaces current empty map):**

```
[map fills full width, ~400px height]

Layer toggles (horizontal chips below map):
  [x] 311 requests  [ ] permits  [ ] crime  [ ] solar

Year range slider: [2021 -------|--- 2025]
```

**Map layers:**

1. **311 Requests** — `HexagonLayer` heatmap showing request density. Hexagons colored by volume (blue → yellow → red). Hovering shows hex count + top service type. Default ON for the selected zip, showing surrounding context.

2. **Permits** — `ScatterplotLayer` with dots colored by type. Building permits in blue, solar permits in green. Dot size scaled by valuation when available. Hovering shows approval type + year.

3. **Crime** — `HexagonLayer` heatmap in red/orange tones. Separate from 311 to avoid visual confusion. Hovering shows hex count + dominant crime category (Person/Property/Society).

4. **Solar** — `ScatterplotLayer` with green dots. Dot size scaled by system valuation when available. Shows solar adoption spread patterns.

**View behavior:**
- Single zip selected: zoom 13, centered on zip centroid
- Area selected: zoom 11-12, centered on area centroid (average of constituent zip centroids)
- All layers filter to points within the visible region (not just the selected zip — show surrounding context)

**Zip centroid expansion:**
- Replace hardcoded 32-entry `ZIP_COORDS` with Census Gazetteer ZCTA centroids for all 82 profiled zips
- Download from Census Bureau: `https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.txt`
- Pipe-delimited file with INTPTLAT and INTPTLONG columns

### Data Pipeline

**Ingestion:**
- Add 4 entries to `CIVIC_SOURCES` in `pipeline/ingest_civic.py`:
  - `sd-get-it-done/data/aggregated/map_points.parquet` → `map_311.parquet`
  - `sd-housing-permits/data/aggregated/map_points.parquet` → `map_permits.parquet`
  - `sd-public-safety/data/aggregated/map_points.parquet` → `map_crime.parquet`
  - `sd-climate-action/data/aggregated/solar_map_points.parquet` → `map_solar.parquet`

**Aggregation:**
- These files are already aggregated by sibling projects — no further transformation needed
- Copy directly to `data/aggregated/` as-is
- Filter to San Diego zips only during dashboard queries (not in pipeline)

**Zip centroids:**
- Download Census Gazetteer file once during pipeline build
- Parse ZCTA, INTPTLAT, INTPTLONG columns
- Save as `data/aggregated/zip_centroids.parquet`
- Replace hardcoded `ZIP_COORDS` dict in dashboard with parquet lookup

### Dashboard Implementation

**Map rendering approach:**
- Load map points lazily — only when layer is toggled ON
- Use `@st.cache_data(ttl=3600)` for each layer's data loader
- Filter by year range before passing to pydeck (DuckDB WHERE clause)
- For 311 (3M points): always use HexagonLayer (aggregates client-side), limit to recent 3 years by default
- For crime (600K points): HexagonLayer, limit to recent 3 years
- For permits (1.2M points): ScatterplotLayer with sampling if needed, or HexagonLayer for density view
- For solar (115K points): ScatterplotLayer (small enough for individual dots)

**Performance consideration:**
- Pydeck HexagonLayer handles aggregation client-side — sending 100K+ points is fine
- For very large datasets (311), filter to selected zip + surrounding zips (± 0.05 lat/lng) to reduce payload
- Use DuckDB spatial filtering: `WHERE lat BETWEEN ? AND ? AND lng BETWEEN ? AND ?`

### API / MCP

**New endpoints:**
- `GET /map-points?layer=311&zip_code=92101&year_min=2021&year_max=2025&limit=50000` — returns lat/lng points for a layer, filtered by location and time
- MCP tool: `get_map_points(layer, zip_code, year_min, year_max)` — same data for Claude queries

**Query layer:**
- `get_map_points(layer: str, zip_code: str | None, year_min: int, year_max: int, limit: int) -> list[dict]`

### What Problem It Solves

"I'm looking at opening a restaurant in North Park. The stats say crime is moderate and permits are growing, but *where* exactly? Is the crime concentrated on one street? Are the permits residential or commercial? Is the construction happening near my target block or across the neighborhood?"

Aggregate statistics hide spatial patterns. Maps reveal them.

---

## Phase 3b: Competitive Intelligence

### Problem

The #1 question every entrepreneur asks is: **"Who are my competitors, and where are they?"**

Currently, the platform shows business counts per category per zip (e.g., "92101 has 847 restaurants"). But it can't answer:
- "How many coffee shops are within 2 miles of this location?"
- "Where exactly are they? What are their names?"
- "Is there a cluster I should avoid, or a gap I should fill?"

The business dataset has 51,261 active businesses with names, addresses, zip codes, and categories — but no map visualization and no proximity analysis.

### What It Does

Adds a new **"competitors"** tab to the dashboard. Users pick a business category and a center point (zip code), and see all matching businesses listed and mapped. Shows density rings and identifies potential gaps in coverage.

### How It Should Look

```
Tab bar: [explorer] [compare] [rankings] [competitors]

╔══════════════════════════════════════════════════════════════╗
║  category: [Coffee Shops ▼]    center: [92103 - Hillcrest ▼] ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  34 coffee shops in 92103                                    ║
║  density: 2.8 per 1,000 residents                           ║
║  city average: 1.2 per 1,000 residents                      ║
║  area (Hillcrest / Bankers Hill): 41 total                  ║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │              [map: zip boundary shaded]               │    ║
║  │              [dots for each business]                 │    ║
║  │              [surrounding zips faded]                 │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  business directory (34 results)                            ║
║  ┌─────────────────────────┬───────────┬────────────────┐   ║
║  │ name                    │ address   │ since          │   ║
║  ├─────────────────────────┼───────────┼────────────────┤   ║
║  │ bird rock coffee roast  │ 1234 uni… │ 2015           │   ║
║  │ dark horse coffee       │ 5678 ada… │ 2018           │   ║
║  └─────────────────────────┴───────────┴────────────────┘   ║
║                                                              ║
║  nearby zips with same category:                            ║
║  ┌──────────┬───────┬─────────────┬──────────────────────┐  ║
║  │ zip      │ count │ per 1k      │ vs city avg          │  ║
║  ├──────────┼───────┼─────────────┼──────────────────────┤  ║
║  │ 92104    │ 28    │ 2.1         │ +75%                 │  ║
║  │ 92116    │ 15    │ 1.5         │ +25%                 │  ║
║  └──────────┴───────┴─────────────┴──────────────────────┘  ║
╚══════════════════════════════════════════════════════════════╝
```

**Key interactions:**
- Selecting a category + zip immediately updates all views
- Business directory is sortable by name or start date
- Map shows all businesses in that category for the selected zip, with neighboring zip businesses in a lighter shade
- "Nearby zips" table shows same-category counts in adjacent zips for comparison

### Data Source

Uses existing `data/processed/businesses.parquet` (51,261 rows):
- `business_name`, `address`, `zip_code`, `category`, `start_date`, `status`

No geocoding needed for MVP — businesses are plotted at zip centroid with jitter, or listed in a table. Point-level mapping of individual businesses is a Phase 4 enhancement (requires address geocoding, which is challenging given data quality — addresses like `"384505THAVE"` need parsing).

**MVP approach:** Map shows zip-level density (choropleth of surrounding zips colored by competitor count), not individual business dots. The business directory table provides the detail. This avoids the geocoding problem entirely while still delivering spatial competitive intelligence.

### API / MCP

**New endpoints:**
- `GET /competitors?category=Coffee+Shops&zip_code=92103` — returns businesses in that category/zip + nearby zip summary
- MCP tool: `find_competitors(category, zip_code)` — "how many coffee shops are near Hillcrest?"

**Query layer:**
- `get_competitors(category: str, zip_code: str) -> dict` — returns `{ businesses: [...], nearby_zips: [...], density: float, city_avg_density: float }`

### What Problem It Solves

"I want to open a yoga studio. Where in San Diego has demand but not too much competition? If I pick North Park, who exactly am I competing against, and how long have they been around?"

This is the most direct business decision the platform can support. Every other feature provides context; this one provides the competitive answer.

---

## Phase 3c: Trend Visualizations

### Problem

The platform has four time-series datasets (business formation 2015+, permits 2002+, crime 2021+, solar 2012+) but only surfaces them as single YoY percentage changes in the profile view. A user sees "business formation: +12.3% YoY" but can't see:
- Is this part of a 5-year growth trend, or a one-year bounce-back?
- Was there a COVID dip and recovery?
- How does this zip's trajectory compare to the city overall?

Numbers without context are noise. Charts provide context.

### What It Does

Adds interactive Plotly line charts for each time-series metric inside the explorer profile view. Charts show the selected zip/area alongside the city average, with YoY annotations at each data point.

### How It Should Look

**Inside explorer tab, below the momentum score section:**

```
trends
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ business formation (2015-2025)          [expanded]
  ┌────────────────────────────────────────────┐
  │     ╱╲                                     │
  │    ╱  ╲        ╱─── 92103                  │
  │   ╱    ╲──────╱                            │
  │  ╱                   ╱─── city avg         │
  │ ╱      ─────────────╱                      │
  │ 2015  2017  2019  2021  2023  2025         │
  └────────────────────────────────────────────┘

▸ construction permits (2002-2025)        [collapsed]
▸ crime incidents (2021-2025)             [collapsed]
▸ solar installations (2012-2025)         [collapsed]
```

**Chart details:**
- Primary line: selected zip/area data (blue, `CHART_COLOR`)
- Secondary line: city-wide average (gray, dashed)
- Hover tooltips: year, count, YoY % change
- Y-axis: count (not percentage)
- X-axis: year
- Each chart in its own `st.expander`, first one expanded by default

**Area mode:**
- Primary line shows area aggregate (sum of constituent zips)
- City average line still shown for comparison

### Data Source

Already built — `get_zip_trends(zip_code)` and `get_area_trends(area)` in `api/queries.py` return:
```python
{
    "business_formation": [{"year": 2015, "count": 45, "yoy_pct": null}, ...],
    "permits": [{"year": 2002, "count": 120, "yoy_pct": null}, ...],
    "crime": [{"year": 2021, "count": 300, "yoy_pct": null}, ...],
    "solar": [{"year": 2012, "count": 15, "yoy_pct": null}, ...]
}
```

**City average trend:** Needs a new query function that aggregates across all zips for each metric — `get_city_trends() -> dict`. Same structure as zip trends but summed across all SD zips.

### Dashboard Implementation

- Use `plotly.graph_objects.Figure` with `go.Scatter` traces
- Two traces per chart: zip/area line + city average line
- `st.plotly_chart(fig, use_container_width=True)`
- Wrap each chart in `st.expander("metric name (year range)", expanded=False)`
- First expander expanded by default

### API / MCP

**New endpoint:**
- `GET /city-trends` — city-wide aggregated time series (for chart comparison line)
- MCP tool: `get_city_trends()` — "what's the city-wide trend for business formation?"

### What Problem It Solves

"The momentum score for La Jolla is 62/100 and permits are up 15% YoY. But is this a new trend or has it been growing for 5 years? Did COVID impact it? Am I catching a wave or arriving late?"

A line chart answers these questions instantly. A single percentage number cannot.

---

## Phase 3d: Enhanced Crime Context

### Problem

The current crime breakdown shows three broad categories — Person, Property, Society — at the zip level. This is too coarse to be actionable. An entrepreneur cares about *what kind* of crime:
- A restaurant owner worries about car break-ins (property) and assault (person)
- A retail shop worries about shoplifting (property) and robbery (person)
- A nightlife business worries about DUI and disorderly conduct (society)

"Property crime: 450 incidents" doesn't distinguish between car theft and vandalism.

### What It Does

Replaces the 3-category crime breakdown with a more detailed view using the 36-type offense group taxonomy available in `sd-public-safety`, and adds temporal patterns showing *when* crime happens (day of week and time of year).

### Available Data

**City-wide detailed crime (sd-public-safety):**

| Parquet | Columns | Rows |
|---------|---------|------|
| crime_by_type.parquet | offense_group, offense_description, crime_against, year, count | ~1,800 |
| temporal_patterns.parquet | dow, month, year, crime_against, count | ~1,200 |

**Already ingested zip-level (sd-business-intel):**
| Parquet | Columns | Rows |
|---------|---------|------|
| civic_crime.parquet | zip_code, crime_against, year, count | ~3,000 |

**Key constraint:** The 36-type offense group detail is only available city-wide, not per zip. Zip-level data only has the 3-category breakdown (Person/Property/Society). This is a limitation of the source data from sd-public-safety.

### How It Should Look

**Crime section in explorer profile (replaces current crime breakdown expander):**

```
crime context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

zip-level breakdown (92103, 2024)
  person:   145  ████████░░░░  38%
  property: 198  ███████████░  52%
  society:   38  ███░░░░░░░░░  10%

▸ city-wide crime types (2024)              [collapsed]
  ┌───────────────────────────┬───────┬────────────┐
  │ offense                   │ count │ category   │
  ├───────────────────────────┼───────┼────────────┤
  │ larceny/theft             │ 42150 │ property   │
  │ simple assault            │ 18210 │ person     │
  │ destruction/vandalism     │ 15300 │ property   │
  │ motor vehicle theft       │ 12800 │ property   │
  │ drug/narcotic violations  │  9400 │ society    │
  │ ...                       │       │            │
  └───────────────────────────┴───────┴────────────┘

▸ when does crime happen? (city-wide)       [collapsed]
  [heatmap: day of week × month, colored by incident count]
```

### Data Pipeline

**Ingestion — add 2 entries to `CIVIC_SOURCES`:**
- `sd-public-safety/data/aggregated/crime_by_type.parquet` → `civic_crime_detail.parquet`
- `sd-public-safety/data/aggregated/temporal_patterns.parquet` → `civic_crime_temporal.parquet`

**No further aggregation needed** — these are already aggregated by the sibling project.

### Dashboard Implementation

- Keep existing zip-level 3-category bar chart (it's zip-specific and useful)
- Add city-wide detail expander below: `st.expander("city-wide crime types")` with a dataframe of top 15 offense groups
- Add temporal expander: `st.expander("when does crime happen?")` with a plotly heatmap (day-of-week rows × month columns, colored by count)
- Year filter: show most recent complete year by default

### API / MCP

**New endpoints:**
- `GET /crime-detail?year=2024` — city-wide offense group breakdown
- `GET /crime-temporal?year=2024` — day/month patterns
- MCP tools: `get_crime_detail(year)`, `get_crime_temporal(year)`

### What Problem It Solves

"Crime count is 381 in this zip. Should I be worried? Is it mostly petty theft, or violent crime? Does it happen at night when my restaurant would be open, or during the day?"

Detailed crime types and temporal patterns turn a scary number into an actionable risk assessment.

---

## Summary

| Sub-phase | Core deliverable | New parquets | New endpoints | New MCP tools |
|-----------|-----------------|-------------|---------------|---------------|
| 3a | Multi-layer interactive maps | 4 (map points) + 1 (zip centroids) | 1 | 1 |
| 3b | Competitor analysis tab | 0 (uses existing businesses.parquet) | 1 | 1 |
| 3c | Time-series trend charts | 0 (uses existing trend data) | 1 | 1 |
| 3d | Detailed crime context | 2 (crime detail + temporal) | 2 | 2 |
| **Total** | | **7** | **5** | **5** |

**Cumulative platform totals after Phase 3:** ~31 parquets, 24 API endpoints, 23 MCP tools

### Build Order

1. **3a first** — the map infrastructure (zip centroids, map point ingestion) is a prerequisite for 3b
2. **3b second** — depends on zip centroids from 3a and is the highest-value user feature
3. **3c third** — independent of maps, adds depth to existing profile views
4. **3d fourth** — independent, adds new data sources and detail to crime section

### Phase 4 Preview (Not In Scope)

These are natural follow-ons that were considered but deferred:
- **Business geocoding** — parse and geocode 51K business addresses for point-level competitor mapping (address quality is poor, needs cleaning pipeline)
- **Multi-criteria search** — filter zips by multiple metrics simultaneously ("income > $75K AND crime < 300 AND 10+ restaurants per 1K")
- **Similar neighborhoods finder** — k-NN clustering to answer "what's like North Park but cheaper?"
- **Export & sharing** — PDF reports, CSV downloads, shareable profile URLs
- **Budget integration** — city spending by area from sd-city-budget
- **Demographic deep-dive** — age pyramids, education breakdowns, commute data
