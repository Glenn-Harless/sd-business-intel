"""Smoke tests for Phase 3 query functions."""

from api import queries


def test_get_zip_centroids():
    coords = queries.get_zip_centroids()
    assert len(coords) > 50, f"Expected 50+ zip centroids, got {len(coords)}"
    assert "92101" in coords
    lat, lng = coords["92101"]
    assert 32.0 < lat < 33.5
    assert -118.0 < lng < -116.5


def test_get_map_points_basic():
    """Map points return data for each layer."""
    for layer in ("311", "permits", "crime", "solar"):
        pts = queries.get_map_points(layer, limit=10)
        assert len(pts) > 0, f"No map points for layer {layer}"
        assert "lat" in pts[0]


def test_get_map_points_with_filters():
    """Map points filtered by zip + year range don't error."""
    pts = queries.get_map_points("311", zip_code="92101", year_min=2023, year_max=2025, limit=100)
    assert isinstance(pts, list)
    # With spatial + year filter, should have fewer points than unfiltered
    all_pts = queries.get_map_points("311", limit=100)
    assert len(pts) <= len(all_pts)


def test_get_map_points_invalid_layer():
    pts = queries.get_map_points("invalid_layer")
    assert pts == []


def test_get_city_trends():
    trends = queries.get_city_trends()
    assert "business_formation" in trends
    # City avg should be per-zip average, not city total
    biz = trends["business_formation"]
    assert len(biz) > 0
    # Per-zip average should be reasonable (< 500 per zip per year)
    for row in biz:
        if row["count"] is not None:
            assert row["count"] < 500, f"City avg {row['count']} seems like total, not per-zip avg"


def test_get_competitors():
    result = queries.get_competitors("drinking places", "92101")
    assert result["zip_code"] == "92101"
    assert result["category"] == "drinking places"
    assert isinstance(result["businesses"], list)
    assert isinstance(result["nearby_zips"], list)

    # Nearby should be geographically close, not all 82 zips
    if result["nearby_zips"]:
        assert len(result["nearby_zips"]) <= 20


def test_get_competitors_nearby_geographic():
    """Verify nearby zips are actually geographically close."""
    result = queries.get_competitors("drinking places", "92101")
    coords = queries.get_zip_centroids()
    center_lat, center_lng = coords.get("92101", (0, 0))

    for nz in result["nearby_zips"]:
        zc = nz["zip_code"]
        if zc in coords:
            lat, lng = coords[zc]
            assert abs(lat - center_lat) <= 0.11, f"{zc} too far north/south"
            assert abs(lng - center_lng) <= 0.11, f"{zc} too far east/west"


def test_get_crime_detail():
    detail = queries.get_crime_detail()
    assert len(detail) > 0
    assert "offense_group" in detail[0]
    assert "crime_against" in detail[0]
    assert "count" in detail[0]


def test_get_crime_temporal():
    temporal = queries.get_crime_temporal()
    assert len(temporal) > 0
    assert "dow" in temporal[0]
    assert "month" in temporal[0]
