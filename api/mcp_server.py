"""MCP server for SD Business Intel.

Exposes 4 tools that let Claude query business intelligence parquets directly.
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
