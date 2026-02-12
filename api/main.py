"""FastAPI app for SD Business Intel API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import queries
from api.models import (
    AreaRankingRow,
    AreaSummary,
    AreaZipSummary,
    BusinessRecord,
    FilterOptions,
    HealthResponse,
    NeighborhoodProfile,
    RankingRow,
    TrendSeries,
)

app = FastAPI(
    title="SD Business Intel",
    description="San Diego small business intelligence — location analytics for entrepreneurs",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "SD Business Intel API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Check data file availability and freshness."""
    return queries.get_health()


@app.get("/filters", response_model=FilterOptions)
def filters():
    """Get available filter values for all endpoints."""
    return queries.get_filters()


@app.get("/neighborhood-profile")
def neighborhood_profile(
    zip: str = Query(..., description="5-digit zip code (e.g. 92101)"),
):
    """Full neighborhood profile: demographics + business counts + civic signals."""
    return queries.get_neighborhood_profile(zip)


@app.get("/compare")
def compare(
    zip_a: str = Query(..., description="First zip code (e.g. 92101)"),
    zip_b: str = Query(..., description="Second zip code (e.g. 92103)"),
):
    """Compare two zip codes head-to-head: demographics, business landscape, civic signals."""
    return queries.compare_zips(zip_a, zip_b)


@app.get("/rankings", response_model=list[RankingRow])
def rankings(
    sort_by: str = Query("population", description="Metric to sort by"),
    sort_desc: bool = Query(True, description="Sort descending (highest first)"),
    category: str | None = Query(None, description="Business category for per-1k density ranking"),
    limit: int = Query(20, ge=1, le=82, description="Number of results"),
):
    """Rank zip codes by a chosen metric, optionally by category density per 1k residents."""
    result = queries.get_rankings(sort_by, sort_desc, category, limit)
    if result and "error" in result[0]:
        raise HTTPException(status_code=400, detail=result[0]["error"])
    return result


@app.get("/businesses", response_model=list[BusinessRecord])
def businesses(
    zip: str | None = Query(None, description="Filter by zip code"),
    category: str | None = Query(None, description="Filter by business category"),
    status: str | None = Query(None, description="Filter by status (active/inactive)"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
):
    """Individual business records, filterable by zip, category, status."""
    return queries.get_businesses(zip, category, status, limit)


@app.get("/areas", response_model=list[AreaSummary])
def areas():
    """List all San Diego areas with summary stats."""
    return queries.get_areas()


@app.get("/area-profile")
def area_profile(area: str = Query(..., description="Area name")):
    """Full area profile: aggregated demographics, business counts, civic signals across member zips."""
    result = queries.get_area_profile(area)
    if not result:
        raise HTTPException(404, f"Area '{area}' not found")
    return result


@app.get("/compare-areas")
def compare_areas_endpoint(
    area_a: str = Query(..., description="First area"),
    area_b: str = Query(..., description="Second area"),
):
    """Compare two areas head-to-head: demographics, business landscape, civic signals."""
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
    """Rank areas by a chosen metric, optionally by category density per 1k residents."""
    return queries.get_area_rankings(sort_by, sort_desc, category, limit)


@app.get("/area-zips", response_model=list[AreaZipSummary])
def area_zips(area: str = Query(..., description="Area name")):
    """List individual zip codes within an area with their key metrics."""
    return queries.get_area_zips(area)


@app.get("/zip-trends")
def zip_trends(zip: str = Query(..., description="Zip code")):
    """Year-over-year trend data for a single zip: business formation, permits, crime, solar."""
    return queries.get_zip_trends(zip)


@app.get("/area-trends")
def area_trends(area: str = Query(..., description="Area name")):
    """Year-over-year trend data for an area: business formation, permits, crime, solar."""
    return queries.get_area_trends(area)
