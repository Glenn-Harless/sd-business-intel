# Task 6: Category Deep-Dive

## Full Build Order (for context)

This is task 6 — the final task in a 6-task "Phase 1.5" sprint for the SD Business Intel project, a San Diego small business intelligence tool that helps entrepreneurs evaluate where to open a business by cross-referencing business tax certificates, census demographics, and civic data.

```
1. Narrative summary          ✅ DONE — auto-generated sentence comparing zip to city avg
2. Business density per capita ✅ DONE — businesses_per_1k metric in pipeline, API, dashboard, MCP
3. Zip comparison             ✅ DONE — side-by-side compare tab, head-to-head table, grouped bar chart, dual-marker map
4. Rankings table             ✅ DONE — sortable table of all 82 zips by any metric, category filter adds context + dropdown option
5. Distribution context       ✅ DONE — percentile rank badges on all 12 explorer tab metric cards
6. Category deep-dive         ← YOU ARE HERE
```

Tasks 1-5 are complete and working. The dashboard has three tabs: explorer (single zip with percentile badges), compare (two zips head-to-head), rankings (find best zip for X). Task 6 adds the final interactive feature: clicking a category bar in the explorer tab drills into the individual businesses in that category.

---

## What to Build

**Click a category bar → show the businesses in that category.** The explorer tab already has a horizontal bar chart of top business categories (lines 600-628) and a business directory table with a category filter dropdown (lines 632-654). But these are disconnected — clicking a bar does nothing, and the dropdown requires manual selection.

Connect them: when a user clicks a bar in the category chart, the business directory below should filter to show businesses in that category. This creates a natural drill-down workflow: see the landscape → click what's interesting → browse specific businesses.

### User stories
- "I see 249 food & accommodation businesses — who are they?" → click the bar → table below filters to food & accommodation
- "Which legal services firms are in this zip?" → click "legal services" bar → see the list
- Clicking again (or clicking a "show all" option) resets to all categories

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

### 1. `dashboard/app.py` — the main (probably only) change

This is primarily a dashboard UX task. The query layer and API already support everything needed — `_load_businesses(zip_code, category)` already accepts an optional category filter (line 132). The data is there; you just need to wire up the interaction.

**The challenge: Plotly click events in Streamlit.** Streamlit doesn't natively support Plotly click callbacks. There are a few approaches:

**Option A — `st.plotly_chart` with `on_select` (Streamlit 1.37+):**
Streamlit added `on_select="rerun"` support for plotly charts. When a user clicks a bar, the selection is stored and triggers a rerun:
```python
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="cat_chart")
# event.selection.points[0] gives the clicked bar info
```
This is the cleanest approach if available in the project's Streamlit version.

**Option B — Replace dropdown with bar-click simulation via `st.session_state`:**
Make the category bar chart's bars act as buttons by using `st.session_state` to track the selected category. Add clickable labels or use Streamlit's built-in column selection.

**Option C — Keep the dropdown but auto-sync:**
Make the existing category filter dropdown the source of truth, but add visual affordance (highlight the selected bar, show "click a bar to filter" hint). This is simpler but less magical.

**Recommendation:** Try Option A first — `on_select="rerun"` is the modern Streamlit pattern for chart interactivity. Fall back to Option C if `on_select` isn't available or doesn't work well with horizontal bar charts.

**Regardless of approach, the changes needed are:**

1. Connect the category chart click to the business directory filter
2. Highlight the selected category bar (different color for selected vs unselected)
3. Add a way to reset/clear the filter (show all businesses)
4. Move the business directory section to be visually tied to the chart (they're already adjacent — just make the connection obvious)

### 2. `api/queries.py` — likely no changes needed

`get_businesses()` (line 290) already supports filtering by zip_code + category. `_load_businesses()` in the dashboard (line 132) already wraps this. No new SQL should be needed.

### 3. `api/main.py` / `api/mcp_server.py` — no changes needed

The `/businesses` endpoint and `get_businesses` MCP tool already accept a `category` parameter.

---

## Current File Layout (accurate as of task 5 completion)

### `api/queries.py` (646 lines)
- `_PERCENTILE_METRICS` — metric direction dict (line 20)
- `_run()` / `_run_one()` — SQL execution helpers (lines 41-52)
- `get_filters()` — line 58
- `get_health()` — line 92
- `get_neighborhood_profile()` — line 129 (returns demographics, business_landscape, civic_signals, comparison_to_avg, percentiles, narrative)
- `get_businesses()` — line 290 (accepts zip_code, category, status, limit)
- `_build_narrative()` — line 335
- `_build_comparison_narrative()` — line 402
- `_RANKING_METRICS` — line 488
- `get_rankings()` — line 497
- `compare_zips()` — line 583
- `_clean()` — line 631

### `api/models.py` (116 lines)
- FilterOptions, HealthResponse, Demographics, CategoryCount, BusinessLandscape, CivicSignals, ComparisonValue, PercentileInfo, NeighborhoodProfile, HeadToHeadMetric, ZipComparison, RankingRow, BusinessRecord

### `api/main.py` (93 lines)
- Endpoints: `/`, `/health`, `/filters`, `/neighborhood-profile`, `/compare`, `/rankings`, `/businesses`

### `api/mcp_server.py` (100 lines)
- Tools: get_filters, get_health, get_neighborhood_profile, get_businesses, get_rankings, compare_zips

### `dashboard/app.py` (949 lines)
- Imports + config (1-31)
- Helpers: `_valid()`, `query()` (33-49)
- Sidebar: zip selectors (52-93)
- Cached loaders: `_load_profile`, `_load_city_avg`, `_load_biz_by_zip`, `_load_businesses`, `_category_options`, `_load_percentiles`, `_load_rankings` (98-272)
- Data loading for selected zip (275-287)
- Helpers: `_build_narrative`, `_build_comparison_narrative`, `_fmt_metric`, `_fmt_compare`, `_ordinal`, `_show_rank` (290-484)
- ZIP_COORDS dict (487-521)
- **Tabs line 527**: `tab_explorer, tab_compare, tab_rankings = st.tabs([...])`
- **Explorer tab** (530-672):
  - Title + narrative (531-536)
  - 4 rows of 3 metric cards with percentile badges (538-596)
  - Divider (598)
  - **Business categories chart** (600-628) — horizontal bar chart of top 15 categories
  - Divider (630)
  - **Business directory** (632-654) — category dropdown filter + dataframe table
  - Divider (656)
  - Map (658-672)
- **Compare tab** (674-848): narrative, head-to-head HTML table, category comparison chart, dual-marker map
- **Rankings tab** (850-944): metric dropdown, category filter, sort toggle, limit slider, dataframe table
- Footer (946-949)

### Data files used by this task
**`data/aggregated/business_by_zip.parquet`** — category counts per zip:
```
zip_code: VARCHAR, category: VARCHAR, total_count: BIGINT, active_count: BIGINT
```
Sample: `92101, "legal services", 398, 398`

**`data/processed/businesses.parquet`** — individual business records:
```
account_id: VARCHAR, business_name: VARCHAR, address: VARCHAR, zip_code: VARCHAR,
naics_code: VARCHAR, activity_description: VARCHAR, category: VARCHAR,
start_date: DATE, created_date: DATE, expiration_date: DATE, status: VARCHAR
```

---

## Key Code to Understand

### Current category chart (lines 600-628)
```python
if biz_by_zip is not None and not biz_by_zip.empty:
    population = row.get("population")
    if _valid(population) and float(population) > 0:
        biz_by_zip = biz_by_zip.copy()
        biz_by_zip["per_1k"] = (1000.0 * biz_by_zip["active_count"] / float(population)).round(1)
        bar_text = biz_by_zip.apply(lambda r: f"{int(r['active_count'])}  ({r['per_1k']}/1k)", axis=1)
    else:
        bar_text = biz_by_zip["active_count"]
    fig = go.Figure(go.Bar(
        x=biz_by_zip["active_count"],
        y=biz_by_zip["category"],
        orientation="h",
        marker_color=CHART_COLOR,
        text=bar_text,
        textposition="outside",
    ))
    fig.update_layout(
        height=max(300, len(biz_by_zip) * 28),
        margin=dict(l=0, r=40, t=0, b=0),
        yaxis=dict(autorange="reversed"),
        xaxis_title="active businesses",
    )
    st.plotly_chart(fig, use_container_width=True)
```

### Current business directory (lines 632-654)
```python
st.subheader("business directory")

biz_categories = []
if biz_by_zip is not None and not biz_by_zip.empty:
    biz_categories = biz_by_zip["category"].tolist()

cat_filter = st.selectbox(
    "filter by category",
    ["all"] + biz_categories,
    index=0,
)

businesses = _load_businesses(selected_zip, cat_filter if cat_filter != "all" else None)
if businesses is not None and not businesses.empty:
    st.dataframe(businesses, use_container_width=True, hide_index=True, height=400)
```

### `_load_businesses()` cached loader (line 132)
```python
@st.cache_data(ttl=3600)
def _load_businesses(zip_code: str, category: str | None = None):
    where = "WHERE zip_code = $1"
    params = [zip_code]
    if category:
        where += " AND category = $2"
        params.append(category)
    return query(f"""
        SELECT business_name, address, category, naics_code, start_date, status
        FROM '{_PROCESSED}/businesses.parquet'
        {where}
        ORDER BY business_name
        LIMIT 5000
    """, params)
```

---

## Existing Patterns to Follow

### Chart colors
```python
CHART_COLOR = "#83c9ff"       # blue — primary
COMPARE_COLOR = "#ff6b6b"     # red — secondary/compare
```

### Streamlit widget keys
Always use explicit `key=` params for widgets inside tabs to prevent state collisions across tabs. Example from rankings tab:
```python
cat_filter = st.selectbox("category filter", [...], key="rankings_category")
sort_label = st.selectbox("rank by", all_labels, key="rankings_sort_by")
```

### Streamlit selectbox gotcha
Do NOT use `range()` + `format_func` + `lambda` for selectboxes — causes widget state issues. Use direct string labels and map back via dict. (The sidebar zip selectors still use the old range() pattern — don't copy that.)

---

## Data Gotchas (from prior experience)

- **Business tax cert CSVs are currently 403** on seshat.datasd.org — if `businesses.parquet` doesn't exist, the business directory shows "no business records available." Handle this gracefully.
- **`_load_businesses` returns up to 5,000 rows** — some zip/category combos may have many businesses. The dataframe has `height=400` so it scrolls.
- **Category names come from `business_by_zip.parquet`** — they're the cleaned/mapped category strings (e.g., "food & accommodation", "legal services"), not raw NAICS codes.
- **The chart shows top 15 categories** (`LIMIT 15` in `_load_biz_by_zip`), but the dropdown shows the same list. If a user wants a category not in the top 15, they can still type in the selectbox.

---

## Implementation Notes

### If using `on_select="rerun"` (Option A):
```python
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="cat_chart")
selected_cat = None
if event and event.selection and event.selection.points:
    # The y-value of the clicked bar is the category name
    selected_cat = event.selection.points[0]["y"]
```

Then use `selected_cat` to:
1. Filter the business directory (replace or sync with the dropdown)
2. Highlight the selected bar in the chart (change its color)
3. Update the subheader (e.g., "business directory — food & accommodation")

### Highlighting the selected bar:
```python
colors = [CHART_COLOR if cat != selected_cat else "#ffa500"  # orange highlight
          for cat in biz_by_zip["category"]]
fig = go.Figure(go.Bar(
    ...
    marker_color=colors,
))
```

### Reset mechanism:
- Clicking the already-selected bar should deselect (toggle behavior)
- Or add a small "show all" button/link above the directory
- The dropdown could remain as an alternative way to filter

### Sync chart selection ↔ dropdown:
If you keep both the chart click and the dropdown, they need to stay in sync. Use `st.session_state` to store the selected category and have both controls read/write from it. Be careful with Streamlit's rerun model — setting session state and using it as a widget default in the same rerun can be tricky.

---

## Verification

1. **Click a bar** → business directory filters to that category, business count matches the bar value
2. **Click again** (or "show all") → directory resets to all businesses
3. **Switch zip codes** → selected category clears (or persists if the new zip also has that category — your call)
4. **No business data** → both chart and directory show graceful empty states (already handled)
5. **Compare and rankings tabs** — must be completely unaffected
6. **Chart appearance** — selected bar should be visually distinct (different color or opacity)
7. **Dropdown sync** — if keeping the dropdown, it should reflect the chart selection and vice versa

---

## What NOT to Do

- Don't modify the pipeline or parquet files — all data needed already exists
- Don't change compare or rankings tab behavior
- Don't add a new tab — this enhances the existing explorer tab
- Don't add new API endpoints or MCP tools — the existing `get_businesses` already supports category filtering
- Don't use AI/LLM for any text generation — all labels and formatting are pure f-strings
- Don't break the existing category dropdown — either enhance it or replace it cleanly
- Don't silently override existing controls (lesson from task 4: controls must do what they say)
- Don't add features beyond what's described here — this is the final task in the sprint
