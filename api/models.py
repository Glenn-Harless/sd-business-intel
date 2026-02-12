"""Pydantic response models for FastAPI's auto-generated OpenAPI docs."""

from __future__ import annotations

from pydantic import BaseModel


class FilterOptions(BaseModel):
    zip_codes: list[str]
    categories: list[str]
    statuses: list[str]


class HealthResponse(BaseModel):
    status: str
    files: dict[str, bool]
    data_as_of: str | None


class Demographics(BaseModel):
    population: int | None = None
    median_age: float | None = None
    median_income: int | None = None
    median_home_value: int | None = None
    median_rent: int | None = None
    pct_bachelors_plus: float | None = None


class CategoryCount(BaseModel):
    category: str
    active_count: int
    total_count: int
    per_1k: float | None = None
    city_avg_per_1k: float | None = None


class BusinessLandscape(BaseModel):
    active_count: int | None = None
    total_count: int | None = None
    category_count: int | None = None
    businesses_per_1k: float | None = None
    top_categories: list[CategoryCount] = []


class CivicSignals(BaseModel):
    new_permits: int | None = None
    permit_valuation: int | None = None
    solar_installs: int | None = None
    crime_count: int | None = None
    median_311_days: float | None = None
    total_311_requests: int | None = None


class ComparisonValue(BaseModel):
    value: float | int | None = None
    city_avg: float | None = None
    vs_avg_pct: float | None = None


class PercentileInfo(BaseModel):
    rank: int
    of: int
    percentile: int  # 0-100, higher = better position


class NeighborhoodProfile(BaseModel):
    zip_code: str
    neighborhood: str | None = None
    demographics: Demographics
    business_landscape: BusinessLandscape
    civic_signals: CivicSignals
    comparison_to_avg: dict[str, ComparisonValue] = {}
    percentiles: dict[str, PercentileInfo] = {}
    data_as_of: str | None = None


class HeadToHeadMetric(BaseModel):
    zip_a: float | int | None = None
    zip_b: float | int | None = None
    difference: float | None = None


class ZipComparison(BaseModel):
    zip_a: NeighborhoodProfile
    zip_b: NeighborhoodProfile
    head_to_head: dict[str, HeadToHeadMetric] = {}
    narrative: str = ""


class RankingRow(BaseModel):
    rank: int
    zip_code: str
    neighborhood: str | None = None
    sort_metric: str
    sort_value: float | int | None = None
    category: str | None = None
    category_active: int | None = None
    category_per_1k: float | None = None
    population: int | None = None
    median_income: int | None = None
    active_count: int | None = None


class BusinessRecord(BaseModel):
    account_id: str | None = None
    business_name: str | None = None
    address: str | None = None
    zip_code: str | None = None
    naics_code: str | None = None
    activity_description: str | None = None
    category: str | None = None
    start_date: str | None = None
    created_date: str | None = None
    expiration_date: str | None = None
    status: str | None = None
