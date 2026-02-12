# Task 5: Distribution Context

## Full Build Order (for context)

This is task 5 from a 6-task "Phase 1.5" sprint for the SD Business Intel project — a San Diego small business intelligence tool that helps entrepreneurs evaluate where to open a business by cross-referencing business tax certificates, census demographics, and civic data.

```
1. Narrative summary          ✅ DONE — auto-generated sentence comparing zip to city avg
2. Business density per capita ✅ DONE — businesses_per_1k metric in pipeline, API, dashboard, MCP
3. Zip comparison             ✅ DONE — side-by-side compare tab, head-to-head table, grouped bar chart, dual-marker map
4. Rankings table             ✅ DONE — sortable table of all 82 zips by any metric, category filter adds context + dropdown option
5. Distribution context       ← YOU ARE HERE
6. Category deep-dive         — click a category bar to drill into businesses
```

Tasks 1-4 are complete and working. The dashboard now has three tabs: explorer (single zip), compare (two zips head-to-head), rankings (find best zip for X). After task 5, every metric in the explorer tab will show where this zip falls relative to all others.

---

## What to Build

**Percentile ranks and distribution context** for the explorer tab. When a user looks at a zip code and sees "median income: $86,403" they currently have no idea if that's high or low among SD zips — the city avg comparison helps, but doesn't tell you "this is the 15th richest out of 82 zips" or "this is in the top 20%."

Add percentile rank badges and optional spark histograms (or mini distribution charts) to the explorer tab's metric cards, so users instantly understand where a zip falls in the full distribution.

### User stories
- "Is $86k income high or low for San Diego?" → badge: "top 25% (rank 18 of 82)"
- "How does this zip's crime compare to the range across all zips?" → spark histogram showing this zip's position in the distribution
- "Are there many zips with similar business density, or is this an outlier?" → distribution shape gives that context

---

## Architecture

**Read CLAUDE.md first** — it defines all conventions. Key rules:
- ALL SQL lives in `api/queries.py` — dashboard and API both call these functions
- Fresh `duckdb.connect()` per query, no persistent connections
- Dashboard uses DuckDB directly via its own `query()` helper (returns pandas DataFrame)
- All text/labels lowercase in the dashboard
- Use `@st.cache_data(ttl=3600)` for data loaders
- Use plotly for charts

The project follows a 4-layer pattern: **pipeline** (parquet files) → **query layer** (`api/queries.py`) → **API** (`api/main.py`) + **MCP** (`api/mcp_server.py`) → **dashboard** (`dashboard/app.py`).

---

## Files to Modify

### 1. `api/queries.py` — add `get_distribution()` or enhance `get_neighborhood_profile()`

Two approaches (pick whichever is cleaner):

**Option A — New `get_distribution()` function:**
```python
def get_distribution(metric: str) -> list[dict]:
    """Return all 82 zip codes' values for a single metric, for building distributions."""
```
Returns: `[{"zip_code": "92101", "value": 86403}, ...]` — the dashboard/API can compute percentiles client-side.

**Option B — Add percentiles directly to `get_neighborhood_profile()` response:**
Add a `percentiles` key to the profile response with precomputed ranks for each metric:
```python
"percentiles": {
    "population": {"rank": 45, "of": 82, "percentile": 45},
    "median_income": {"rank": 18, "of": 82, "percentile": 78},
    ...
}
```

Option B is likely better — it's a single call and the profile already returns comparison_to_avg, so percentiles are a natural extension. Compute the rank using DuckDB window functions:
```sql
RANK() OVER (ORDER BY median_income DESC NULLS LAST) AS median_income_rank
```

Validate the metric whitelist — don't pass arbitrary column names to SQL.

### 2. `api/models.py` — add percentile model if needed

If adding to the profile response, add a small model:
```python
class PercentileInfo(BaseModel):
    rank: int
    of: int
    percentile: int  # 0-100, higher = "better" position
```

### 3. `api/main.py` — update if adding new endpoint

If you go with Option A (new function), add a `/distribution` endpoint. If Option B, the existing `/neighborhood-profile` endpoint already returns the enhanced data.

### 4. `api/mcp_server.py` — update if adding new tool

Same logic as above.

### 5. `dashboard/app.py` — add distribution context to explorer tab

This is the main visual change. The explorer tab currently shows metric cards in 4 rows of 3 (lines 452-505). Each card uses `_fmt_metric()` (line 378) to show value + delta vs city avg.

**Add to each metric card:**
- A percentile badge: small text like "top 25% · rank 18 of 82" or "78th percentile"
- Keep it subtle — this is supporting context, not the main value

**Optionally add a spark distribution:**
- A tiny plotly histogram (or st.bar_chart) showing the distribution of all 82 values for that metric, with this zip's position highlighted
- This could be a toggle ("show distributions") to avoid visual clutter
- Or a single summary chart at the bottom of the metrics section

**UX guidelines:**
- Don't overwhelm the metric cards — the current layout is clean
- Percentile text should be small/muted (use st.caption or small markdown)
- "Higher is better" vs "lower is better" matters for how you phrase the percentile (crime rank 5 of 82 should say "5th lowest" not "top 6%")
- Consider using the `_fmt_metric()` helper pattern — extend it or add a sibling

---

## Current File Layout (accurate as of task 4 completion)

### `api/queries.py` (592 lines)
- `_run()` / `_run_one()` — SQL execution helpers (lines 24-35)
- `get_filters()` — line 41
- `get_health()` — line 75
- `get_neighborhood_profile()` — line 112 (returns full profile dict with demographics, business_landscape, civic_signals, comparison_to_avg, narrative)
- `get_businesses()` — line 237
- `_build_narrative()` — line 282
- `_build_comparison_narrative()` — line 349
- `_RANKING_METRICS` — line 435
- `get_rankings()` — line 444
- `compare_zips()` — line 530
- `_clean()` — line 578

### `api/models.py` (108 lines)
- FilterOptions, HealthResponse, Demographics, CategoryCount, BusinessLandscape, CivicSignals, ComparisonValue, NeighborhoodProfile, HeadToHeadMetric, ZipComparison, RankingRow, BusinessRecord

### `api/main.py` (92 lines)
- Endpoints: `/`, `/health`, `/filters`, `/neighborhood-profile`, `/compare`, `/rankings`, `/businesses`

### `api/mcp_server.py` (99 lines)
- Tools: get_filters, get_health, get_neighborhood_profile, get_businesses, get_rankings, compare_zips

### `dashboard/app.py` (858 lines)
- Imports + config (1-31)
- Helpers: `_valid()`, `query()` (33-49)
- Sidebar: zip selectors (52-93)
- Cached loaders: `_load_profile`, `_load_city_avg`, `_load_biz_by_zip`, `_load_businesses`, `_category_options`, `_load_rankings` (98-218)
- Data loading for selected zip (221-235)
- Helpers: `_build_narrative`, `_build_comparison_narrative`, `_fmt_metric`, `_fmt_compare` (238-406)
- ZIP_COORDS dict (409-443)
- **Tabs line 449**: `tab_explorer, tab_compare, tab_rankings = st.tabs([...])`
- **Explorer tab** (452-582): narrative, 4 rows of 3 metric cards, business categories chart, business directory, map
- **Compare tab** (584-758): narrative, head-to-head HTML table, category comparison chart, dual-marker map
- **Rankings tab** (760-847): metric dropdown, category filter, sort toggle, limit slider, dataframe table
- Footer (849-858)

### `neighborhood_profile.parquet` (82 rows, 18 columns)
```
zip_code: VARCHAR, neighborhood: VARCHAR
active_count: BIGINT, total_count: BIGINT, category_count: BIGINT
population: INTEGER, median_age: FLOAT, median_income: INTEGER
median_home_value: INTEGER, median_rent: INTEGER, pct_bachelors_plus: DOUBLE
businesses_per_1k: DOUBLE
new_permits: DOUBLE, permit_valuation: DOUBLE, solar_installs: DOUBLE
crime_count: DOUBLE, median_311_days: DOUBLE, total_311_requests: DOUBLE
```

---

## Existing Patterns to Follow

### Metric card pattern (explorer tab)
```python
r1 = st.columns(3)
l, v, d = _fmt_metric("population", pop, "avg_population")
r1[0].metric(l, v, d, delta_color="normal")
```

### `_fmt_metric()` helper (line 378)
```python
def _fmt_metric(label, val, avg_key=None, prefix="", decimals=0):
    if not _valid(val):
        return label, "—", None
    val_f = float(val)
    fmt = f"{prefix}{val_f:,.{decimals}f}"
    delta = None
    if avg_key and avg_row is not None:
        avg_val = avg_row.get(avg_key)
        if _valid(avg_val):
            diff = val_f - float(avg_val)
            sign = "+" if diff >= 0 else "-"
            delta = f"{sign}{prefix}{abs(diff):,.{decimals}f} vs avg"
    return label, fmt, delta
```

### Comparison to avg (profile response)
The profile already computes `comparison_to_avg` with value, city_avg, vs_avg_pct for key metrics. Percentiles are a natural companion to this.

---

## Data Gotchas (from prior experience)

- **82 zips in neighborhood_profile** — this is the canonical set. Some metrics may be NULL for some zips (e.g., military bases have no median_income). Use NULLS LAST and count only non-null values when computing rank.
- **"Higher is better" varies by metric** — population/income/businesses = higher is good; crime/rent/311 days = lower is good. The narrative already tracks this via its `higher_is` field. Reuse that mapping.
- **Extreme outliers**: 92121 Sorrento Valley has ~447 businesses per 1k (few residents + biotech campus). It will always be rank 1. That's correct.
- **Streamlit widget keys**: always use explicit `key=` params for widgets inside tabs to prevent state collisions. Don't use `range()` + `format_func` + `lambda` for selectboxes — use direct string labels.

---

## Verification

1. **Query layer**: Check that percentile data is correct — rank 1 should be the highest value for "higher is good" metrics and lowest for "lower is good" metrics
2. **Dashboard**: Explorer tab metric cards should show percentile context without breaking the existing layout
3. **Edge cases**: Zip with NULL metric (rank should be excluded), zip at rank 1 and 82, tie handling
4. **Tabs unchanged**: Compare and rankings tabs should be unaffected
5. **API/MCP**: If adding to profile response, verify the JSON structure is correct

---

## What NOT to Do

- Don't modify the pipeline or parquet files — all data needed already exists
- Don't change compare or rankings tab behavior
- Don't add a new tab — this enhances the existing explorer tab
- Don't add full-size histograms that overwhelm the metric cards — keep distribution context compact
- Don't use AI/LLM for any text generation — all labels and formatting are pure f-strings
- Don't add features beyond what's described here (save category deep-dive for task 6)
- Don't silently override any existing controls (lesson from task 4: controls must do what they say)
