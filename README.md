# sd business intel

San Diego small business intelligence — location analytics for entrepreneurs evaluating where to open a business.

Aggregates public data to answer: "what businesses are here, who lives here, and what's the area like?" for any SD zip code.

## setup

```bash
uv sync
uv run python -m pipeline.build
```

## run

```bash
# dashboard
streamlit run dashboard/app.py

# api
uvicorn api.main:app --reload

# api docs
open http://localhost:8000/docs
```

## api endpoints

| endpoint | description |
|----------|-------------|
| `GET /filters` | available zip codes, categories, statuses |
| `GET /health` | data file availability check |
| `GET /neighborhood-profile?zip=92101` | demographics + business counts + civic signals |
| `GET /businesses?zip=92101&category=restaurant` | individual business records |

## data sources

- **business tax certificates** — [data.sandiego.gov](https://data.sandiego.gov/datasets/business-listings/) (City of San Diego, PDDL)
- **census/acs demographics** — [api.census.gov](https://api.census.gov/) (US Census Bureau)
- **civic cross-references** — aggregated parquets from sibling sd-* projects:
  - sd-housing-permits (construction permits by zip)
  - sd-climate-action (solar installations by zip)
  - sd-public-safety (crime by zip)
  - sd-get-it-done (311 response by neighborhood)
  - sd-homelessness (point-in-time counts by region)

## architecture

```
pipeline/       → downloads + transforms data into parquets
api/queries.py  → shared query layer (all SQL)
api/main.py     → FastAPI REST endpoints
api/mcp_server.py → MCP tools for Claude
dashboard/app.py  → Streamlit explorer
```

## deployment

| layer | platform | config |
|-------|----------|--------|
| dashboard | Streamlit Cloud | `requirements.txt` |
| api | Render | `requirements-api.txt`, `render.yaml` |
| mcp | local (Claude Code) | `.mcp.json` |
