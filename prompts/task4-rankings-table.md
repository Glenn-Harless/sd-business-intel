# Task 4: Rankings Table

## Full Build Order (for context)

This is task 4 from a 6-task "Phase 1.5" sprint for the SD Business Intel project — a San Diego small business intelligence tool that helps entrepreneurs evaluate where to open a business by cross-referencing business tax certificates, census demographics, and civic data.

```
1. Narrative summary          ✅ DONE — auto-generated sentence comparing zip to city avg
2. Business density per capita ✅ DONE — businesses_per_1k metric in pipeline, API, dashboard, MCP
3. Zip comparison             ✅ DONE — side-by-side compare tab, head-to-head table, grouped bar chart, dual-marker map
4. Rankings table             ← YOU ARE HERE
5. Distribution context       — percentile ranks / spark histograms
6. Category deep-dive         — click a category bar to drill into businesses
```

Tasks 1-3 are complete and working. After task 4, the dashboard will have three interaction modes: explore a single zip (explorer tab), compare two zips (compare tab), and find the best zip for a specific goal (rankings tab).

---

## What to Build

A "rankings" tab in the dashboard that answers: **"Which zip code is best for X?"** — the inverse of the explorer. Instead of "tell me about 92101", it's "show me the top 10 zips by income" or "least crime per capita" or "most restaurants per 1k residents".

This is a sortable table of all 82 SD zip codes ranked by a user-selected metric, with an optional category filter for per-category density rankings.

### User stories
- "Show me the 10 zip codes with the highest median income"
- "Which neighborhoods have the most restaurants per capita?"
- "Where are the safest neighborhoods with the most new permits?"

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

### 1. `api/queries.py` — add `get_rankings()`

Add a function that returns all 82 zip codes from `neighborhood_profile.parquet` ranked by a chosen metric, with optional per-category density.

```python
def get_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 82,
) -> list[dict]:
```

**Available columns in `neighborhood_profile.parquet` (82 rows):**
```
zip_code: VARCHAR
neighborhood: VARCHAR
active_count: BIGINT
total_count: BIGINT
category_count: BIGINT
population: INTEGER
median_age: FLOAT
median_income: INTEGER
median_home_value: INTEGER
median_rent: INTEGER
pct_bachelors_plus: DOUBLE
businesses_per_1k: DOUBLE
new_permits: DOUBLE
permit_valuation: DOUBLE
solar_installs: DOUBLE
crime_count: DOUBLE
median_311_days: DOUBLE
total_311_requests: DOUBLE
```

When `category` is provided, join with `business_by_zip.parquet` (columns: `zip_code, category, total_count, active_count`, 4954 rows) to compute `category_per_1k = 1000 * active_count / population` for that specific category, and sort by that density.

Return `list[dict]` with each row containing: zip_code, neighborhood, the sort metric value, and a few key context columns (population, median_income, active_count) so the table is useful without clicking into each zip.

**Validation**: whitelist the allowed `sort_by` values — don't let arbitrary column names through to SQL.

### 2. `api/models.py` — add `RankingRow` model

A Pydantic model for the ranking response rows.

### 3. `api/main.py` — add `/rankings` endpoint

```
GET /rankings?sort_by=median_income&sort_desc=true&category=restaurants&limit=20
```

Thin wrapper around `queries.get_rankings()`.

### 4. `api/mcp_server.py` — add `get_rankings` tool

So Claude can answer "which neighborhoods have the most X?" directly.

### 5. `dashboard/app.py` — add "rankings" tab

The dashboard currently has two tabs (line 378):
```python
tab_explorer, tab_compare = st.tabs(["explorer", "compare"])
```

Change to three:
```python
tab_explorer, tab_compare, tab_rankings = st.tabs(["explorer", "compare", "rankings"])
```

**Rankings tab contents:**
- **Metric selector** — dropdown to pick the sort column. Group them logically:
  - Demographics: population, median income, median age, median rent, median home value, % bachelor's+
  - Business: active businesses, businesses per 1k, category count
  - Civic: new permits, crime count, 311 median days, solar installs
- **Sort direction** — toggle ascending/descending (some metrics are "lower is better" like crime)
- **Category filter** (optional) — when selected, adds a `{category} per 1k` column computed from `business_by_zip.parquet` and sorts by that instead
- **Results table** — `st.dataframe` showing rank, zip code, neighborhood, the sorted metric (highlighted), and 2-3 context columns. Make it compact — this is a scanning/filtering tool, not a detail view.
- **Limit slider** — default 10, max 82

**UX detail**: When user clicks a row / selects a zip from this table, ideally it should link to the explorer tab for that zip. At minimum, make zip codes visually obvious so users can manually navigate. Don't over-engineer this — a clean table is the priority.

---

## Existing Patterns to Follow

Look at how `compare_zips` was added across all layers for the pattern:
- `api/queries.py` → `compare_zips()` function (line 435)
- `api/models.py` → `HeadToHeadMetric`, `ZipComparison` models (line 70)
- `api/main.py` → `GET /compare` endpoint (line 60)
- `api/mcp_server.py` → `compare_zips` tool (line 67)
- `dashboard/app.py` → `tab_compare` with content (line 510)

The dashboard tab structure starts at line 378. Explorer tab is `with tab_explorer:` (line 381), compare tab is `with tab_compare:` (line 510).

---

## Data Gotchas (from prior experience)

- **Business zip codes include out-of-area**: `business_by_zip.parquet` has ~1,218 unique zip codes (including out-of-state mailing addresses). When computing per-category density, MUST filter to zips that exist in `neighborhood_profile.parquet` (82 SD zips with enough data) or `demographics.parquet` (114 SD ZCTAs).
- **Per-category city avg density**: use total/total method (`SUM(category_count) / SUM(city_pop)`), NOT average of per-zip ratios — statistically correct for unequal populations.
- **Extreme outliers**: 92121 Sorrento Valley has ~447 businesses per 1k residents (few residents + biotech campus). 92101 Downtown ~63/1k. City avg ~20.3/1k. Rankings will naturally surface these — that's fine, it's the point.

---

## Verification

1. **Query layer**: `uv run python -c "from api.queries import get_rankings; import json; print(json.dumps(get_rankings('median_income', True, limit=5), indent=2))"`
2. **Category ranking**: `uv run python -c "from api.queries import get_rankings; import json; print(json.dumps(get_rankings('category_per_1k', True, 'restaurants', 5), indent=2))"`
3. **API**: `uv run uvicorn api.main:app --reload` → hit `/rankings?sort_by=median_income&limit=5`
4. **Dashboard**: `uv run streamlit run dashboard/app.py` → explorer tab unchanged, compare tab unchanged, new rankings tab works
5. **MCP**: use `get_rankings` tool from Claude Code
6. **Edge cases**: invalid sort_by value (return error, don't pass to SQL), category with no data, zip with zero population (handle division by zero)

---

## What NOT to Do

- Don't modify the pipeline or parquet files — all data needed already exists
- Don't change explorer or compare tab behavior
- Don't add pagination — 82 rows max, a simple limit slider is fine
- Don't over-engineer the UI — a clean sortable table with a metric picker is the goal
- Don't use AI/LLM for any text generation — all labels and formatting are pure f-strings
- Don't add features beyond what's described here (save distribution context for task 5)
