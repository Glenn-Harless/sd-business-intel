"""MCP server for SD Business Intel.

Exposes 18 tools that let Claude query business intelligence parquets directly.
Uses FastMCP (v2) with stdio transport — spawned by Claude Code as a subprocess.
"""

from __future__ import annotations

from fastmcp import FastMCP

from api import queries

mcp = FastMCP(
    "SD Business Intel",
    instructions=(
        "San Diego small business intelligence — location analytics for "
        "entrepreneurs. Call get_filters() first to see available zip codes, "
        "business categories, and statuses."
    ),
)


@mcp.tool()
def get_filters() -> dict:
    """Get available filter values: zip codes, business categories, and statuses.

    Call this first to see what values are valid for other tools.
    """
    return queries.get_filters()


@mcp.tool()
def get_health() -> dict:
    """Check data file availability and freshness.

    Returns which data files exist and when data was last updated.
    """
    return queries.get_health()


@mcp.tool()
def get_neighborhood_profile(zip_code: str) -> dict:
    """Get full neighborhood profile for a San Diego zip code.

    Returns demographics (population, income, age, education, housing costs),
    business landscape (counts, top categories), civic signals (permits,
    solar installs, crime, 311 response times), and comparison to city averages.
    """
    return queries.get_neighborhood_profile(zip_code)


@mcp.tool()
def get_businesses(
    zip_code: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get individual business records.

    Filter by zip_code, category, and/or status. Use get_filters() to see
    valid values. Returns up to `limit` records (max 500).
    """
    return queries.get_businesses(zip_code, category, status, limit)


@mcp.tool()
def get_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Rank San Diego zip codes by a chosen metric.

    Sort by demographics (population, median_income, median_age, etc.),
    business metrics (active_count, businesses_per_1k), or civic signals
    (new_permits, crime_count, etc.). Optionally specify a business category
    to rank by that category's density per 1k residents.
    """
    return queries.get_rankings(sort_by, sort_desc, category, limit)


@mcp.tool()
def compare_zips(zip_a: str, zip_b: str) -> dict:
    """Compare two San Diego zip codes head-to-head.

    Returns both neighborhood profiles plus a head-to-head comparison
    showing differences in demographics, business landscape, and civic signals.
    """
    return queries.compare_zips(zip_a, zip_b)


@mcp.tool()
def get_areas() -> list[dict]:
    """List all San Diego areas with summary metrics.

    Returns area names, zip counts, population, business counts, and income.
    Use this to see what areas are available for deeper analysis."""
    return queries.get_areas()


@mcp.tool()
def get_area_profile(area: str) -> dict:
    """Get full area profile for a San Diego area.

    Returns demographics, business landscape, civic signals, comparison
    to city averages, and a narrative summary. Areas group nearby zip codes."""
    return queries.get_area_profile(area)


@mcp.tool()
def compare_areas(area_a: str, area_b: str) -> dict:
    """Compare two San Diego areas head-to-head.

    Returns both area profiles plus a comparison showing differences
    in demographics, business landscape, and civic signals."""
    return queries.compare_areas(area_a, area_b)


@mcp.tool()
def get_area_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Rank San Diego areas by a chosen metric.

    Sort by demographics, business metrics, or civic signals.
    Optionally filter by business category for category-specific density."""
    return queries.get_area_rankings(sort_by, sort_desc, category, limit)


@mcp.tool()
def get_area_zips(area: str) -> list[dict]:
    """Get all zip codes within a San Diego area with key metrics.

    Enables drill-down from area-level to zip-level analysis."""
    return queries.get_area_zips(area)


@mcp.tool()
def get_zip_trends(zip_code: str) -> dict:
    """Get year-over-year trend data for a San Diego zip code.

    Returns time-series data for business formation, permits, crime,
    and solar with YoY change percentages."""
    return queries.get_zip_trends(zip_code)


@mcp.tool()
def get_area_trends(area: str) -> dict:
    """Get year-over-year trend data for a San Diego area.

    Returns aggregated time-series across the area's zip codes
    for business formation, permits, crime, and solar."""
    return queries.get_area_trends(area)


@mcp.tool()
def get_momentum_scores(limit: int = 20) -> list[dict]:
    """Rank San Diego zip codes by momentum score (0-100).

    Composite score combining business formation, permit activity,
    crime trends, and solar adoption year-over-year changes."""
    return queries.get_momentum_scores(limit)


@mcp.tool()
def get_area_momentum(limit: int = 20) -> list[dict]:
    """Rank San Diego areas by momentum score (0-100).

    Population-weighted composite of zip-level momentum scores."""
    return queries.get_area_momentum(limit)


@mcp.tool()
def get_business_age(zip_code: str) -> list[dict]:
    """Get business age statistics by category for a zip code.

    Shows median age, avg age, pct under 2 years, pct over 10 years."""
    return queries.get_business_age(zip_code)


@mcp.tool()
def get_area_business_age(area: str) -> list[dict]:
    """Get business age statistics by category for an area.

    Shows median age, avg age, pct under 2 years, pct over 10 years."""
    return queries.get_area_business_age(area)


@mcp.tool()
def get_311_services() -> list[dict]:
    """Get city-wide 311 service type breakdown.

    Returns 47 service types with total requests, resolution times,
    and close rates. City-wide data (not zip-level)."""
    return queries.get_311_services()


@mcp.tool()
def get_map_points(
    layer: str,
    zip_code: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = 50000,
) -> list[dict]:
    """Get lat/lng map points for a civic data layer.

    Layers: 311 (service requests), permits (construction), crime, solar.
    Optionally filter by zip_code (spatial bounding box) and year range."""
    return queries.get_map_points(layer, zip_code, year_min, year_max, limit)


@mcp.tool()
def get_city_trends() -> dict:
    """Get city-wide per-zip average time-series trends.

    Returns average-per-zip values for business formation, permits, crime,
    and solar. Use for chart comparison lines against individual zips."""
    return queries.get_city_trends()


@mcp.tool()
def find_competitors(category: str, zip_code: str) -> dict:
    """Find competitors in a business category near a zip code.

    Returns matching businesses, density per 1k residents, city average
    density, and geographically nearby zip codes with same-category counts."""
    return queries.get_competitors(category, zip_code)


@mcp.tool()
def get_crime_detail(year: int | None = None) -> list[dict]:
    """Get city-wide crime breakdown by offense group.

    Returns 36 offense types (larceny, assault, vandalism, etc.) with
    counts and crime_against category. Defaults to latest year."""
    return queries.get_crime_detail(year)


@mcp.tool()
def get_crime_temporal(year: int | None = None) -> list[dict]:
    """Get city-wide crime temporal patterns.

    Returns day-of-week x month breakdown showing when crime occurs.
    Useful for assessing risk during business operating hours."""
    return queries.get_crime_temporal(year)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
