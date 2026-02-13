"""Pydantic response models for FastAPI's auto-generated OpenAPI docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilterOptions(BaseModel):
    zip_codes: list[str]
    categories: list[str]
    statuses: list[str]
    areas: list[str] = []


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
    crime_breakdown: list = []
    energy: dict | None = None
    permit_timelines: list = []


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
    narrative: str = ""
    area: str | None = None
    momentum: dict | None = None
    business_age: list = []


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


class AreaSummary(BaseModel):
    area: str
    zip_count: int
    population: int | None = None
    active_count: int | None = None
    businesses_per_1k: float | None = None
    median_income: int | None = None


class AreaProfile(BaseModel):
    area: str
    zip_codes: list[str] = []
    zip_count: int
    demographics: Demographics
    business_landscape: BusinessLandscape
    civic_signals: CivicSignals
    comparison_to_avg: dict[str, ComparisonValue] = {}
    narrative: str = ""
    momentum: dict | None = None
    business_age: list = []


class AreaComparison(BaseModel):
    area_a: AreaProfile
    area_b: AreaProfile
    head_to_head: dict[str, HeadToHeadMetric] = {}
    narrative: str = ""


class AreaRankingRow(BaseModel):
    rank: int
    area: str
    sort_metric: str
    sort_value: float | int | None = None
    category: str | None = None
    category_active: int | None = None
    category_per_1k: float | None = None
    population: int | None = None
    median_income: int | None = None
    active_count: int | None = None


class TrendPoint(BaseModel):
    year: int
    count: int | None = None
    yoy_pct: float | None = None


class TrendSeries(BaseModel):
    business_formation: list[TrendPoint] = []
    permits: list[TrendPoint] = []
    crime: list[TrendPoint] = []
    solar: list[TrendPoint] = []


class CrimeBreakdown(BaseModel):
    crime_against: str = ""
    count: int = 0


class EnergyBenchmark(BaseModel):
    avg_kwh_per_customer: float | None = None
    total_kwh: float | None = None
    elec_customers: int | None = None


class PermitTimeline(BaseModel):
    permit_type: str = ""
    permit_count: int = 0
    median_days: float | None = None


class MomentumScore(BaseModel):
    zip_code: str | None = None
    area: str | None = None
    momentum_score: float | None = None
    biz_formation_yoy: float | None = None
    permit_yoy: float | None = None
    crime_yoy: float | None = None
    solar_yoy: float | None = None


class BusinessAge(BaseModel):
    category: str = ""
    business_count: int = 0
    median_age_years: float | None = None
    avg_age_years: float | None = None
    pct_under_2yr: float | None = None
    pct_over_10yr: float | None = None


class ServiceType(BaseModel):
    service_name: str = ""
    total_requests: int = 0
    closed_requests: int = 0
    avg_resolution_days: float | None = None
    median_resolution_days: float | None = None
    close_rate_pct: float | None = None


class AreaZipSummary(BaseModel):
    zip_code: str
    neighborhood: str | None = None
    population: int | None = None
    active_count: int | None = None
    businesses_per_1k: float | None = None
    median_income: int | None = None
    crime_count: int | None = None
    new_permits: int | None = None


class CompetitorResult(BaseModel):
    zip_code: str
    category: str
    count: int
    businesses: list[dict] = Field(default_factory=list)
    density: float | None = None
    city_avg_density: float | None = None
    nearby_zips: list[dict] = Field(default_factory=list)


class CrimeDetail(BaseModel):
    offense_group: str
    crime_against: str
    count: int


class CrimeTemporal(BaseModel):
    dow: int
    month: int
    crime_against: str
    count: int
