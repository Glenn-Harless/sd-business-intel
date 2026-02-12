"""FastAPI app for SD Business Intel API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import queries
from api.models import (
    BusinessRecord,
    FilterOptions,
    HealthResponse,
    NeighborhoodProfile,
    RankingRow,
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
