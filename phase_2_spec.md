# Phase 2: Deeper Local Insights — Implementation Spec

## Context

Phase 1 + 1.5 are complete: 82-zip neighborhood profiles, compare, rankings, API, MCP, three-tab dashboard. Phase 2 adds the features that make this tool genuinely useful for business owners — area-level thinking, temporal trends, and richer civic context. The independent AI ranked the options 2A → 2C → 2B; the user agrees.

**Key discovery during planning:** The sibling civic parquets already contain rich time-series data (permits: 25 years, crime: 6 years, solar: 18 years, 311: 10 years monthly). The current pipeline only uses the latest year. Also, business tax cert dates are silently lost in the pipeline (all NULL) due to a `TRY_CAST` parsing issue with `all_varchar=true` — fixing this unlocks 50 years of business formation data.

**Pre-implementation verification (Feb 2026):** All assumptions verified against actual codebase and data. Cached business tax cert CSVs exist locally (51,512 rows in `data/raw/btax_active_*.csv`) with dates in `MM/DD/YYYY` format — no re-download needed despite seshat.datasd.org still returning 403. Civic parquet year ranges confirmed: permits 2002–2026, crime 2021–2026, solar 2012–2026, 311 monthly 2016-05 to 2026-02.

---

## Phase 2.1 — Pipeline Foundations

### 2.1.1 Fix business date parsing bug

**File:** `pipeline/transform.py` (~line 212, `_build_businesses`)

The pipeline reads CSVs with `all_varchar=true`, then uses `TRY_CAST("START DT" AS DATE)`. The date strings in the CSVs (e.g., `"10/11/2019"`) don't match DuckDB's default DATE format (ISO YYYY-MM-DD), so `TRY_CAST` silently returns NULL for all rows. Verified: all 51,511 rows in `businesses.parquet` have NULL `start_date` and `created_date`.

**Fix:** Use `TRY_CAST(STRPTIME("START DT", '%m/%d/%Y') AS DATE)` (and same for CREATION DT). Apply to all three date columns (START DT, CREATION DT, EXP DT). Cached raw CSVs exist locally (`data/raw/btax_active_1.csv`, `btax_active_2.csv`, 51,512 rows) — the pipeline's download step will skip gracefully since the source is still 403, and the existing CSVs will be re-parsed with correct date handling.

**Note:** Line numbers in this spec are approximate — they shift as the file evolves.

### 2.1.2 Add `ZIP_TO_AREA` mapping

**File:** `pipeline/transform.py` (new dict, near `ZIP_TO_NEIGHBORHOOD`)

Curated mapping of the **82 zips that have full neighborhood profiles** → ~20 areas. Business owners think in areas, not zips. Only profiled zips are included to ensure consistent data quality — no gaps in area metrics.

```
Proposed areas (~20):
  Downtown                          → 92101
  Uptown / North Park               → 92103, 92104, 92116
  Golden Hill / City Heights        → 92102, 92105
  Barrio Logan / Logan Heights      → 92113
  Pacific Beach                     → 92109
  Ocean Beach / Point Loma          → 92106, 92107
  Mission Valley / Linda Vista      → 92108, 92110, 92111
  Clairemont                        → 92117
  La Jolla / University City        → 92037, 92122
  Sorrento Valley / Mira Mesa       → 92121, 92126
  Rancho Bernardo / Scripps Ranch   → 92127, 92128, 92131
  Carmel Valley / Rancho Penasquitos → 92129, 92130
  Carlsbad                          → 92008, 92009, 92010, 92011
  Oceanside                         → 92054, 92056, 92057, 92058
  Encinitas / Del Mar / Solana Beach → 92007, 92014, 92024, 92075
  Escondido                         → 92025, 92026, 92027, 92029
  Vista / San Marcos                → 92069, 92078, 92081, 92083, 92084
  Chula Vista / National City       → 91910, 91911, 91913, 91914, 91915, 91950
  La Mesa / Lemon Grove             → 91941, 91942, 91945
  El Cajon                          → 92019, 92020, 92021
  South Bay / San Ysidro            → 92154, 92173, 91932, 92139
  East County                       → 91901, 91977, 91978, 92040, 92065, 92071
  Poway / Rancho Santa Fe           → 92064, 92067, 92091
  Other (Coronado, Fallbrook, etc.) → remaining profiled zips
```

Final list to be refined during implementation. During implementation, cross-reference against the 82 zips in `neighborhood_profile.parquet` — any zip in the proposed mapping that isn't in the profile set should be dropped. `ZIP_TO_NEIGHBORHOOD` currently has 58 entries; the mapping may produce fewer areas than 20 depending on which zips have full profiles.

### 2.1.3 Build area-level aggregated parquets

**File:** `pipeline/transform.py` (new function `_build_area_profiles`)

**New parquets:**

1. **`area_profile.parquet`** (~20 rows)
   - Columns: area, zip_codes (list), zip_count, population, median_age, median_income, median_home_value, median_rent, pct_bachelors_plus, active_count, total_count, category_count, businesses_per_1k, new_permits, permit_valuation, solar_installs, crime_count, median_311_days, total_311_requests
   - Aggregation rules:
     - SUM: population, active_count, total_count, new_permits, crime_count, solar_installs, total_311_requests, permit_valuation
     - Population-weighted AVG: median_income, median_age, median_rent, median_home_value, pct_bachelors_plus → `SUM(metric * population) / SUM(population)`
     - Computed: businesses_per_1k = `1000 * SUM(active) / SUM(population)`, category_count = `COUNT(DISTINCT category)` from business_by_zip

2. **`area_business_by_category.parquet`** (~500 rows)
   - Columns: area, category, active_count, total_count, per_1k
   - Aggregated from business_by_zip using ZIP_TO_AREA mapping

3. ~~`zip_to_area.parquet`~~ **Not needed as a parquet.** The `ZIP_TO_AREA` dict in `pipeline/transform.py` serves as the lookup table. The pipeline can inject the area column directly into `neighborhood_profile.parquet` and `area_profile.parquet` already contains the zip list. No separate file needed — keeps file count manageable (already 13 parquets).

### 2.1.4 Build trend parquets

**File:** `pipeline/transform.py` (new function `_build_trends`)

**New parquets:**

1. **`trend_business_formation.parquet`**
   - Columns: zip_code, year, new_businesses
   - Source: processed/businesses.parquet `start_date` (once date bug is fixed)
   - Filter: year >= 2015 (meaningful recent data)
   - GROUP BY zip_code, EXTRACT(YEAR FROM start_date)

2. **`trend_permits.parquet`** — already exists as civic_permits.parquet (year x zip)
   - No new parquet needed — just query civic_permits directly for trends

3. **`trend_crime.parquet`** — already exists as civic_crime.parquet (year x zip)
   - No new parquet needed

4. **`trend_solar.parquet`** — already exists as civic_solar.parquet (year x zip)
   - No new parquet needed

5. **`trend_311_monthly.parquet`** — NEW ingestion from sibling project
   - Source: `sd-get-it-done/data/aggregated/monthly_trends.parquet`
   - Actual source columns: `request_month_start (DATE), total_requests (BIGINT), closed_requests (BIGINT), avg_resolution_days (DOUBLE), median_resolution_days (DOUBLE)`
   - Extract `year` and `month` from `request_month_start` during ingestion: `EXTRACT(YEAR FROM request_month_start) AS year, EXTRACT(MONTH FROM request_month_start) AS month`
   - City-wide (no zip granularity available), but useful for overall trend context
   - Data range: 2016-05 to 2026-02 (118 rows)

**Net new parquets:** area_profile, area_business_by_category, trend_business_formation, trend_311_monthly (4 new files — zip_to_area replaced by dict + column in neighborhood_profile)

---

## Phase 2.2 — Query Layer + API + MCP

### 2.2.1 New area query functions

**File:** `api/queries.py`

1. **`get_areas() -> list[dict]`**
   - Returns all areas with: area name, zip_count, population, active_count, businesses_per_1k, median_income
   - Source: area_profile.parquet
   - Used for area picker dropdown

2. **`get_area_profile(area: str) -> dict`**
   - Full area profile: demographics, business landscape (with top categories), civic signals
   - Comparison to city averages (reuse city_averages.parquet — compare area totals/weighted-avgs to city avgs)
   - Narrative via `_build_area_narrative()` (similar to `_build_narrative()` but area-adapted language)
   - Include constituent zip list with per-zip highlights (best/worst within area)
   - Source: area_profile + area_business_by_category

3. **`compare_areas(area_a: str, area_b: str) -> dict`**
   - Head-to-head like compare_zips but at area level
   - Source: area_profile

4. **`get_area_rankings(sort_by, sort_desc, category, limit) -> list[dict]`**
   - Rank areas by any metric (like get_rankings but for areas)
   - Source: area_profile + area_business_by_category

5. **`get_area_zips(area: str) -> list[dict]`**
   - Return all zips in an area with their key metrics
   - Source: neighborhood_profile filtered to area's zips
   - Enables drill-down from area → zip

### 2.2.2 Trend query functions

**File:** `api/queries.py`

6. **`get_zip_trends(zip_code: str) -> dict`**
   - Returns year-over-year data for a zip: business formation, permits, crime, solar
   - Computes YoY change % for latest year
   - Source: trend_business_formation + civic_permits + civic_crime + civic_solar
   - Returns: `{ "business_formation": [{year, count, yoy_pct}, ...], "permits": [...], ... }`

7. **`get_area_trends(area: str) -> dict`**
   - Same as zip trends but aggregated across area's zips
   - SUM counts per year, then compute YoY

### 2.2.3 Update existing functions

- **`get_filters()`** — add `areas` key with list of area names
- **`get_neighborhood_profile()`** — add `area` field to response (which area this zip belongs to) and `trends` summary (latest YoY changes)

### 2.2.4 API endpoints

**File:** `api/main.py`

New endpoints:
- `GET /areas` → list of areas with summary metrics
- `GET /area-profile?area=Carlsbad` → full area profile
- `GET /compare-areas?area_a=Carlsbad&area_b=Downtown` → head-to-head
- `GET /area-rankings?sort_by=population&sort_desc=true` → ranked areas
- `GET /area-zips?area=Carlsbad` → zips within area
- `GET /zip-trends?zip=92101` → time-series for a zip
- `GET /area-trends?area=Carlsbad` → time-series for an area

### 2.2.5 Pydantic models

**File:** `api/models.py`

New models: `AreaSummary`, `AreaProfile`, `AreaComparison`, `AreaRankingRow`, `TrendPoint`, `TrendSeries`, `ZipTrends`

**Bug fix:** The existing `NeighborhoodProfile` model (line 66) is missing the `narrative` field, but `queries.get_neighborhood_profile()` returns it. Add `narrative: str` to the model. Also add `area: str | None` field for the new area association.

### 2.2.6 MCP tools

**File:** `api/mcp_server.py`

New tools mirroring each new query function (7 new tools):
- `get_areas()`, `get_area_profile()`, `compare_areas()`, `get_area_rankings()`, `get_area_zips()`, `get_zip_trends()`, `get_area_trends()`

---

## Phase 2.3 — Dashboard

### 2.3.1 Area-first navigation with drill-down

**File:** `dashboard/app.py`

**New sidebar flow:**
1. **Level picker**: "explore by: area | zip code" (default: area)
2. **Area mode**:
   - Area dropdown (20 areas, sorted by population)
   - Below it: optional "drill into zip" dropdown showing zips within selected area
   - When an area is selected (no zip drill): show area-level profile
   - When a zip is drilled into: show zip-level profile (existing behavior) with breadcrumb "Carlsbad > 92009"
3. **Zip mode**: existing zip dropdown (unchanged, for power users)

### 2.3.2 Area explorer tab

When area is selected (no zip drill-down):
- **Area summary card**: population, total businesses, businesses per 1k, median income — with delta vs city avg
- **Constituent zips mini-table**: ranked by active businesses, clickable to drill down
- **Business categories chart**: aggregated across all zips in area (from area_business_by_category)
- **Trend sparklines**: small plotly line charts for business formation, permits, crime (from area_trends)
- **Area narrative**: `_build_area_narrative()` — similar structure to zip narrative but area-appropriate language ("carlsbad has 4 zip codes with a combined population of...")

### 2.3.3 Trend indicators in profile cards

For both zip and area profiles:
- Add YoY trend arrows to key metric cards (permits ↑23%, crime ↓12%)
- Use `st.metric(delta=...)` which already supports this
- Only show delta when trend data is available

### 2.3.4 Area compare tab

When "area" mode is active in compare tab:
- Two area dropdowns instead of two zip dropdowns
- Same comparison layout: narrative, metrics table with winner highlighting, category comparison chart
- Reuse `_build_comparison_narrative()` pattern adapted for areas

### 2.3.5 Area rankings tab

When "area" mode is active in rankings tab:
- Same controls (metric dropdown, category filter, sort toggle, limit)
- Ranks ~20 areas instead of 82 zips
- Default limit adjusted (10 → shows half of all areas)

### 2.3.6 Refactor: dashboard calls query layer directly

**Key change for Phase 2:** The dashboard currently duplicates all SQL and narrative logic from `api/queries.py` in `dashboard/app.py` (~200 lines of duplicated code, including `_build_narrative()` at app.py:301 and queries.py:335). With 7+ new functions being added, maintaining two copies is unsustainable.

**Refactor:** Have `dashboard/app.py` import and call functions from `api/queries.py` directly instead of duplicating SQL. The query layer already returns `list[dict]` / `dict` which converts trivially to pandas DataFrames. This eliminates the "must update in TWO places" constraint going forward.

**Migration approach:**
1. Replace dashboard's inline SQL + `_build_narrative()` with calls to `api.queries.*`
2. Convert dict results to DataFrames where needed: `pd.DataFrame(result)` or `pd.Series(result)`
3. Keep dashboard-specific presentation logic (chart configs, layout) in app.py
4. Remove the duplicated `_build_narrative()` and `_build_comparison_narrative()` from app.py

### 2.3.7 Narrative functions

**Now only in `api/queries.py`** (single source of truth after 2.3.6 refactor):
- `_build_area_narrative()` (new) — area-adapted language
- Update `_build_narrative()` to include trends

Narrative enhancements:
- Area narrative: "carlsbad (4 zip codes, pop 95k) has far more businesses per capita than the avg sd area..."
- Trend language in zip narrative: "...permits up 23% year-over-year, crime trending down"
- Keep pure f-string logic, NO AI/LLM

---

## Phase 2.4 — Polish & Verification

### 2.4.1 Pipeline rebuild
- Run `uv run python -m pipeline.build --force` to regenerate all parquets
- Verify row counts: area_profile (~20 rows), area_business_by_category (~500), trend_business_formation (~800+)
- Verify business dates are no longer NULL

### 2.4.2 API testing
- `uvicorn api.main:app --reload`
- Hit each new endpoint in browser at `/docs`
- Verify area profiles return correct aggregated metrics
- Verify trends return multi-year data with YoY percentages

### 2.4.3 MCP testing
- Use each new MCP tool via Claude Code
- `get_areas()` → list of ~20 areas
- `get_area_profile("Carlsbad")` → full profile with categories and narrative
- `compare_areas("Downtown", "Carlsbad")` → head-to-head
- `get_zip_trends("92101")` → multi-year trends

### 2.4.4 Dashboard testing
- `streamlit run dashboard/app.py`
- Test area-first navigation flow
- Test drill-down from area to zip
- Test area compare and rankings tabs
- Verify trend indicators appear on profile cards
- Test toggle between area mode and zip mode

---

## Implementation Order

The work is structured in dependency order:

1. **Fix business date bug** (2.1.1) — prerequisite for trend data
2. **Add ZIP_TO_AREA mapping** (2.1.2) — prerequisite for area parquets
3. **Build area parquets** (2.1.3) — prerequisite for area queries
4. **Build trend parquets** (2.1.4) — prerequisite for trend queries
5. **Run pipeline** to generate all new parquets
6. **Query layer** (2.2.1–2.2.3) — area + trend functions
7. **API endpoints + models** (2.2.4–2.2.5) — includes fixing NeighborhoodProfile model
8. **MCP tools** (2.2.6)
9. **Refactor dashboard to call query layer** (2.3.6) — do this BEFORE adding new dashboard features
10. **Dashboard: area navigation + explorer** (2.3.1–2.3.2)
11. **Dashboard: trend indicators** (2.3.3)
12. **Dashboard: area compare + rankings** (2.3.4–2.3.5)
13. **Narratives** (2.3.7) — now single source in queries.py
14. **Polish + verification** (2.4)

---

## Files Modified

| File | Changes |
|------|---------|
| `pipeline/transform.py` | Fix date bug (all 3 date columns), add ZIP_TO_AREA dict, add `area` column to neighborhood_profile, new `_build_area_profiles()`, new `_build_trends()` |
| `pipeline/ingest_civic.py` | Add 311 monthly trends ingestion (extract year/month from request_month_start DATE) |
| `api/queries.py` | 7 new query functions, update get_filters, update get_neighborhood_profile, new `_build_area_narrative()`, update `_build_narrative()` with trends |
| `api/models.py` | ~7 new Pydantic models + fix existing `NeighborhoodProfile` (add `narrative`, `area` fields) |
| `api/main.py` | 7 new endpoints |
| `api/mcp_server.py` | 7 new MCP tools |
| `dashboard/app.py` | **Refactor:** remove duplicated SQL/narrative logic, import from `api.queries`. Then: area-first nav, area explorer/compare/rankings, trend indicators |

---

## Available Temporal Data Inventory

| Dataset | Granularity | Time Span | Zip-Level? | Source |
|---------|------------|-----------|------------|--------|
| Business starts | individual dates | 1974–2026 (52 yrs) | Yes | businesses.parquet (once date bug fixed) |
| Housing permits | year × zip | 2002–2026 (gaps) | Yes | civic_permits.parquet (already ingested) |
| Solar installs | year × zip | 2012–2026 (gaps) | Yes | civic_solar.parquet (already ingested) |
| Crime | year × zip | 2021–2026 (6 yrs) | Yes | civic_crime.parquet (already ingested) |
| 311 requests | monthly | 2016–2026 (10 yrs) | No (city-wide) | sd-get-it-done monthly_trends.parquet (new ingestion) |
| Energy consumption | year × zip | 2012–2025 (14 yrs) | Yes | sd-climate-action (not yet ingested, possible future) |

---

## Not in Scope (deferred)

- **Momentum score** composite — interesting but premature; ship raw trends first, score later
- **2B deep cross-project context** (top 3 permits, 311 service breakdown) — defer to Phase 2.5
- **Map polygons for areas** — would need GeoJSON boundaries, defer
- **Homelessness integration** — data too coarse (5 regions × 2 years), not zip-level
- **Energy consumption trends** — available in sd-climate-action but not yet ingested; good Phase 2.5 candidate
