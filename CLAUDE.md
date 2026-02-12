# SD Business Intel — Small Business Location Intelligence

## Project Overview
San Diego small business intelligence for entrepreneurs evaluating where to open a business. Cross-references business tax certificates, census demographics, and 7 civic data projects to produce neighborhood profiles and competitive landscape analysis.

## Architecture — Follow sd-city-budget Pattern

### Project Structure
```
pipeline/       # Data ingestion + transformation
data/raw/       # Raw source data (gitignored)
data/processed/ # Cleaned parquets (businesses.parquet, demographics.parquet)
data/aggregated/# Pre-aggregated parquets for dashboard/API + civic cross-references
api/            # Shared query layer + FastAPI + MCP server
dashboard/      # Streamlit app
```

### Query Layer (`api/queries.py`)
- ALL SQL lives here — both FastAPI and MCP call these functions
- Fresh `duckdb.connect()` per query, no persistent connection
- Returns `list[dict]` via `df.to_dict(orient="records")`

### Dashboard Rules
- Use DuckDB for all data access — no loading full datasets into pandas
- `query()` helper: fresh `duckdb.connect()` per call, returns pandas DataFrame
- All text/labels lowercase
- `@st.cache_data(ttl=3600)` for filter options
- Use plotly for charts, pydeck for maps

### Pipeline
- `uv run python -m pipeline.build [--force]` runs full pipeline
- DuckDB for all transforms
- Data sources: seshat.datasd.org (business tax certs), api.census.gov (demographics), sibling project parquets (civic signals)

### Deployment
- Dashboard: Streamlit Cloud (`requirements.txt`, main file `dashboard/app.py`)
- API: Render (`requirements-api.txt`, `render.yaml`)
- MCP: Local Claude Code (`.mcp.json`, stdio transport)

## Data Sources
- **Business tax certificates**: seshat.datasd.org/sd_businesses_active/
- **Census ACS**: api.census.gov/data/2022/acs/acs5
- **Civic cross-references**: sibling sd-* project aggregated parquets

## Related Projects
- `sd-city-budget/` — municipal spending (architecture reference)
- `sd-get-it-done/` — 311 service requests
- `sd-housing-permits/` — construction permits
- `sd-climate-action/` — solar installations
- `sd-public-safety/` — crime data
- `sd-homelessness/` — point-in-time counts
