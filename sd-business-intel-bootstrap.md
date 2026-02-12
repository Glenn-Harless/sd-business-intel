# SD Business Intel — Agent Bootstrap Prompt

Copy everything below the line and use it as the initial prompt for an AI agent in a new `sd-business-intel` repo.

---

## Project: San Diego Small Business Intelligence API

**For San Diego entrepreneurs evaluating where to open a business.** Aggregates public data to answer questions like "Is this neighborhood a good spot to open a coffee shop?" or "What does the competitive landscape look like for gyms in Hillcrest?"

This is the application layer on top of 7 existing civic data projects. It cross-references those datasets with new business-specific data sources to produce actionable location intelligence. The reference implementation for architecture is `sd-city-budget`. Follow the same patterns exactly.

### Who I Am

Glenn Harless — Lead Data Engineer in San Diego. MS Computational Analytics (Georgia Tech), BS Cognitive Science (UCSD). I've already built 7 interconnected San Diego civic data projects that share the architecture you'll use here. This is the layer on top that cross-references those datasets with new business-specific data sources.

### Architecture Pattern (MUST follow exactly)

Every project uses this three-layer stack:

1. **Pipeline** — downloads data from sources, transforms via DuckDB, outputs parquet files
2. **Dashboard** — Streamlit app querying parquets via DuckDB
3. **API** — FastAPI REST endpoints returning JSON
4. **MCP server** — lets Claude query the data directly as tools

The API and MCP server share a **single query layer** (`api/queries.py`). No SQL duplication. Both query parquets directly via DuckDB — no external database. The MCP server does NOT need the API running. The dashboard is also independent.

### File Structure (follow exactly)

```
sd-business-intel/
├── pipeline/
│   ├── __init__.py
│   ├── build.py              # Orchestrator: runs all ingest + transform steps
│   ├── ingest_btax.py        # Business tax certificates (data.sandiego.gov)
│   ├── ingest_census.py      # Census/ACS demographics (api.census.gov)
│   ├── ingest_places.py      # Google Places API (optional, requires key)
│   ├── ingest_civic.py       # Copy/refresh civic parquets from sibling projects
│   └── transform.py          # DuckDB transforms + aggregation exports
├── data/
│   ├── raw/                   # Source CSVs/JSONs (gitignored if >100MB)
│   ├── processed/             # Cleaned parquets
│   └── aggregated/            # Pre-aggregated parquets for dashboard/API
├── dashboard/
│   └── app.py                 # Streamlit app
├── api/
│   ├── __init__.py
│   ├── queries.py             # Shared query layer (ALL SQL lives here)
│   ├── models.py              # Pydantic response models
│   ├── main.py                # FastAPI app
│   └── mcp_server.py          # FastMCP server
├── .mcp.json                  # MCP registration for Claude Code
├── .github/
│   └── workflows/
│       └── refresh.yml        # Monthly data refresh cron
├── .gitignore
├── requirements.txt           # Streamlit Cloud deps
├── requirements-api.txt       # Render deps
├── pyproject.toml             # Full project deps (uv)
├── render.yaml                # Render deployment config
└── README.md
```

### Query Layer Pattern (`api/queries.py`)

This is the core file. Every query function:
- Creates a fresh `duckdb.connect()` (no persistent connection)
- Runs SQL against parquet files in `data/aggregated/`
- Returns `list[dict]` via `df.to_dict(orient="records")`
- Closes the connection

Always include:
- A `get_filters()` function returning valid parameter values
- Optional filter params with sensible defaults on every query function
- Small aggregated results (10-50 rows per query)

### Dependency Split

| File | Used by | Contains |
|------|---------|----------|
| `pyproject.toml` | Local dev (`uv sync`) | Everything |
| `requirements.txt` | Streamlit Cloud | streamlit, plotly, duckdb, pyarrow |
| `requirements-api.txt` | Render | fastapi, uvicorn, duckdb, pyarrow, pydantic |

### MCP Server Pattern

Uses `fastmcp` (FastMCP v2). Each tool mirrors a query function with identical signatures. Synchronous — DuckDB is fast in-process.

```python
from fastmcp import FastMCP
from api import queries

mcp = FastMCP(
    "SD Business Intel",
    instructions="San Diego small business intelligence. Call get_filters() first to see available parameter values.",
)

@mcp.tool()
def get_filters() -> dict:
    """Get valid filter values for all endpoints."""
    return queries.get_filters()
```

`.mcp.json` at project root:
```json
{
  "mcpServers": {
    "sd-business-intel": {
      "command": "uv",
      "args": ["run", "python", "-m", "api.mcp_server"]
    }
  }
}
```

### Deployment

| Layer | Platform | Config |
|-------|----------|--------|
| Dashboard | Streamlit Cloud | `requirements.txt`, main file `dashboard/app.py` |
| API | Render (free tier) | `requirements-api.txt`, start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| MCP | Local (Claude Code) | `.mcp.json`, stdio transport |

---

## Phased Build Plan

This project is significantly larger than any single civic data project. Build it in phases — each phase produces a working, deployable product.

### Phase 1: MVP — "What's in this neighborhood?"

**Goal**: Answer "what businesses are here, who lives here, and what's the area like?" for any San Diego zip code.

**Data sources**:
1. **SD Business Tax Certificates** (data.sandiego.gov) — the foundation dataset
2. **Census/ACS Demographics** (api.census.gov) — population, income, age, education, housing costs by zip/tract
3. **Civic cross-references** — copy aggregated parquets from the 7 sibling projects

**API endpoints**:
- `GET /filters` — available zip codes, neighborhoods, business categories, years
- `GET /health` — data file availability check
- `GET /neighborhood-profile?zip=92101` — demographics + business counts + civic signals (311 response time, crime rate, permit activity)
- `GET /businesses?zip=92101&category=restaurant` — businesses in an area by category

**Dashboard tabs**:
- **Explorer** — select a zip code, see full profile with map, key metrics, comparison to city average

**Deploy** MVP to Streamlit Cloud + Render before moving to Phase 2.

### Phase 2: Competition & Growth — "Is this market saturated? Is this area growing?"

**New data sources**:
4. **Google Places API** — ratings, review counts, price level, categories for active businesses (enriches tax cert data)
5. **County DEH Food Facility Inspections** — inspection scores, violations (restaurant/food businesses)
6. **ABC Liquor Licenses** (abc.ca.gov) — bar/restaurant density signal

**New API endpoints**:
- `GET /business-density?category=coffee` — count per capita by zip, saturation analysis
- `GET /competition?zip=92101&category=coffee` — competitive landscape for a specific business type + area
- `GET /growth-signals?zip=92101` — permits, new construction, population change, business openings/closings
- `GET /trends?zip=92101` — time-series of business openings, permit activity, 311 volume, crime

**New dashboard tabs**:
- **Competition** — business density heatmap, underserved area identification, ratings distribution
- **Growth** — where is SD growing? Permit heatmap, population change, new business openings

### Phase 3: Location Scorer — "Where should I open my business?"

**New API endpoints**:
- `GET /neighborhoods/ranked?category=gym` — all neighborhoods ranked by composite score
- `GET /best-locations?category=gym&budget=moderate` — top recommendations based on all signals
- `GET /safety-score?zip=92101` — composite from crime, calls for service, 311 patterns
- `GET /infrastructure?zip=92101` — 311 response times, capital project investment

**New dashboard tabs**:
- **Location Scorer** — interactive tool: select a business type, adjust priority weights (foot traffic vs. safety vs. growth vs. low competition), get ranked recommendations with per-factor breakdowns
- **Cross-Reference** — correlations across civic datasets (does faster 311 response predict higher business ratings? do areas with more permits have more openings?)

**Scoring methodology must be transparent**: show per-factor scores, not just a composite number. Let users adjust weights. Document the formula in the dashboard itself.

---

## Data Sources — Detail

### EXISTING civic data (already built, cross-reference via parquet joins)

7 existing projects with processed parquet files at `/Users/glennharless/dev-brain/sd-*/data/`.

| Project | Useful signals for business intel |
|---------|----------------------------------|
| `sd-get-it-done` | Neighborhood conditions — 311 request volume, resolution times, problem types. Fast resolution = responsive city services. |
| `sd-city-budget` | Infrastructure investment — where is the city spending on improvements? Capital projects signal growing neighborhoods. |
| `sd-housing-permits` | Growth indicators — new construction, dwelling units by zip, permit timelines. High permit activity = growing area. |
| `sd-traffic-transportation` | Foot traffic proxies — traffic volume by street, transit ridership, bike/ped counts. High traffic = high visibility. |
| `sd-climate-action` | Solar adoption, energy trends by zip — proxy for affluence and environmental consciousness. |
| `sd-public-safety` | Safety — crime patterns by area, calls for service density. Critical for location decisions. |
| `sd-homelessness` | Homelessness concentration by subregion — relevant for retail/food service location decisions. |

**Civic data refresh strategy**: `pipeline/ingest_civic.py` copies relevant aggregated parquets from sibling project directories into `data/aggregated/` with a `civic_` prefix. The monthly GitHub Action runs this step so cross-references stay current when sibling projects refresh. This script should:
- Check each sibling directory exists
- Copy only the specific parquets needed (not everything)
- Log which files were refreshed and their modification dates
- Fail gracefully if a sibling project doesn't exist (not everyone will have all 7 locally)

### NEW data sources

**Phase 1 (required — MVP)**:

1. **SD Business Tax Certificates** — https://data.sandiego.gov/datasets/business-listings/
   - CSV download, PDDL licensed, maintained by City Treasurer
   - Active + inactive certificates back to 1990
   - Active certs split into two subsets, inactive in 10-year increments
   - This is the core dataset — what businesses exist, where, what type, when opened/closed
   - Hosted on seshat.datasd.org (same CDN as Get It Done data)

2. **Census/ACS Demographics** — https://api.census.gov/
   - American Community Survey 5-year estimates by zip code / census tract
   - Key tables: B19013 (median income), B01003 (population), B01002 (median age), B15003 (education), B25077 (home values), B25064 (median rent)
   - Free, no rate limits, API key recommended but optional
   - Use the `census` Python package or raw HTTP requests

**Phase 2 (competitive landscape)**:

3. **Google Places API** — https://developers.google.com/maps/documentation/places/web-service/
   - Nearby Search (New): find businesses by type within a geographic area
   - Fields: name, address, rating, user_ratings_total, price_level, types, opening hours
   - Free tier: 10K calls/SKU/month at Essentials level (since March 2025)
   - Request only Basic-tier fields (name, address, types, rating, price_level) to minimize cost
   - **TOS-friendly for caching aggregate data** (unlike Yelp)
   - Requires API key via Google Cloud Console — the pipeline should skip this source gracefully if `GOOGLE_PLACES_API_KEY` env var is not set
   - Strategy: grid San Diego into ~50 zones, query each for target business types, deduplicate by place_id

4. **County DEH Food Facility Inspections** — check data.sandiegocounty.gov or sdcounty.gov
   - Inspection scores, violations, closure history for restaurants/food businesses
   - Research the actual download URL — may be Socrata API or CSV

5. **ABC Liquor Licenses** — https://www.abc.ca.gov/licensing/license-queries/
   - Public database of all California liquor license holders
   - Filter to San Diego County
   - Useful for bar/restaurant density analysis

**Phase 3 (nice to have)**:

6. **Walk Score API** — walkability, transit score, bike score by location. Free for limited use.
7. **SANDAG Economic Data** — employment by sector, commute patterns. Check opendata.sandag.org.
8. **SD Parking Meter Data** — if available on data.sandiego.gov, direct foot traffic proxy.

### Data processing approach

For each new source:
1. Download raw data to `data/raw/` (CSV, JSON, or API response)
2. Clean and transform with DuckDB in `pipeline/transform.py`
3. Save processed data to `data/processed/` as parquet
4. Create aggregated views in `data/aggregated/` optimized for the query layer (pre-joined, pre-grouped)

For civic cross-references:
1. `pipeline/ingest_civic.py` copies specific aggregated parquets from sibling directories
2. Prefixed as `civic_311.parquet`, `civic_permits.parquet`, etc.
3. Join on zip code or community plan name in the query layer
4. Monthly refresh keeps them current

---

## API Design

All endpoints return JSON with Pydantic models. Include `data_as_of` in responses for transparency.

### Phase 1 endpoints

```
GET /filters
GET /health
GET /neighborhood-profile?zip=92101
GET /businesses?zip=92101&category=restaurant&status=active
```

### Phase 2 endpoints

```
GET /business-density?category=coffee
GET /competition?zip=92101&category=coffee
GET /growth-signals?zip=92101
GET /trends?zip=92101
```

### Phase 3 endpoints

```
GET /neighborhoods/ranked?category=gym
GET /best-locations?category=gym&budget=moderate
GET /safety-score?zip=92101
GET /infrastructure?zip=92101
```

### Example response (`/neighborhood-profile`)

```json
{
  "zip_code": "92101",
  "neighborhood": "Downtown",
  "demographics": {
    "population": 38000,
    "median_income": 72000,
    "median_age": 34
  },
  "business_landscape": {
    "total_active": 2400,
    "top_categories": [
      {"category": "restaurant", "count": 380},
      {"category": "professional_services", "count": 290}
    ]
  },
  "civic_signals": {
    "median_311_resolution_days": 4.2,
    "crime_rate_per_1k": 45.3,
    "new_permits_last_12mo": 187
  },
  "data_as_of": "2026-02-01"
}
```

---

## Dashboard Design

Streamlit app with tabs added per phase.

**Phase 1**: Explorer tab only
**Phase 2**: + Competition, + Growth
**Phase 3**: + Location Scorer, + Cross-Reference

### Location Scorer (Phase 3) — the killer feature

This is the tab that makes the project genuinely useful vs. just another dashboard. Design requirements:

- User selects a business type (dropdown)
- User adjusts priority weights via sliders: foot traffic, safety, growth potential, low competition, demographics fit, infrastructure quality
- Results show ranked neighborhoods with **per-factor breakdown** (not just a composite number)
- Each factor shows: raw value, percentile rank within SD, weighted contribution to final score
- Formula documented directly in the dashboard with an expander ("How is this calculated?")
- Default weights should produce reasonable results without user adjustment

Use plotly for charts and pydeck for maps (consistent with existing projects).

---

## GitHub Actions (`.github/workflows/refresh.yml`)

Monthly cron (1st of month). Must also refresh civic cross-references.

```yaml
name: Monthly Data Refresh
on:
  schedule:
    - cron: '0 6 1 * *'
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run python -m pipeline.build
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "data: monthly refresh"
```

Note: Civic cross-references will only refresh in CI if the sibling repos are checked out alongside this one (e.g., in a monorepo or with a checkout step per repo). For the Streamlit Cloud deployment, the civic parquets committed to this repo are the source of truth — they update when the pipeline runs locally or when a manual workflow dispatch is triggered after sibling projects refresh.

---

## Build Order (within each phase)

1. **Pipeline** — get data downloaded and into parquets
2. **Query layer** — `api/queries.py`, test with plain Python
3. **API** — `api/main.py`, verify with `/docs`
4. **MCP server** — `api/mcp_server.py`, mirror the API
5. **Dashboard** — `dashboard/app.py`
6. **Deploy** — Streamlit Cloud + Render
7. **Move to next phase**

---

## Important Notes

- Use `uv` as the package manager (not pip, not poetry)
- All text/labels in the dashboard should be lowercase (matches my aesthetic)
- Keep the README concise — project description, setup instructions, data sources with attribution, API docs link
- Parquet files committed to repo if under 100MB, gitignored if larger
- No `.env` files committed — API keys loaded from environment variables
- Google Places API key: loaded from `GOOGLE_PLACES_API_KEY` env var. Pipeline must work without it (skip Places ingestion, log a warning). Census API key: `CENSUS_API_KEY`, also optional (Census works without a key, just rate-limited).
- Each phase should be fully functional and deployed before starting the next
