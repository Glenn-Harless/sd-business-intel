"""Shared query layer — ALL SQL lives here.

Both FastAPI and MCP call these functions. Each function creates a fresh
DuckDB connection, queries parquet files, and returns list[dict] or dict.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

_ROOT = Path(__file__).resolve().parent.parent
_AGG = _ROOT / "data" / "aggregated"
_PROCESSED = _ROOT / "data" / "processed"

# Metric → SQL sort direction for "best" ranking (rank 1 = best).
# DESC = higher is better, ASC = lower is better.
_PERCENTILE_METRICS = {
    "population": "DESC",
    "median_income": "DESC",
    "median_age": "DESC",
    "median_home_value": "DESC",
    "median_rent": "ASC",
    "pct_bachelors_plus": "DESC",
    "active_count": "DESC",
    "businesses_per_1k": "DESC",
    "new_permits": "DESC",
    "crime_count": "ASC",
    "median_311_days": "ASC",
    "solar_installs": "DESC",
}


def _q(path: str) -> str:
    """Resolve a parquet path relative to project root."""
    return str(_ROOT / path)


def _run(sql: str, params: list | None = None) -> list[dict]:
    """Execute SQL and return list of dicts."""
    con = duckdb.connect()
    df = con.execute(sql, params or []).fetchdf()
    con.close()
    return df.to_dict(orient="records")


def _run_one(sql: str, params: list | None = None) -> dict:
    """Execute SQL and return single dict (first row)."""
    rows = _run(sql, params)
    return rows[0] if rows else {}


# ── Public query functions ──


def get_filters() -> dict:
    """Get available filter values for all endpoints."""
    con = duckdb.connect()

    zip_codes = []
    categories = []
    statuses = ["active", "inactive"]

    # zip codes from neighborhood profile (always available)
    profile_path = _AGG / "neighborhood_profile.parquet"
    if profile_path.exists():
        zip_codes = sorted(
            con.execute(
                f"SELECT DISTINCT zip_code FROM '{profile_path}' ORDER BY zip_code"
            ).fetchdf()["zip_code"].tolist()
        )

    # categories from business_by_zip if available
    biz_path = _AGG / "business_by_zip.parquet"
    if biz_path.exists():
        categories = sorted(
            con.execute(
                f"SELECT DISTINCT category FROM '{biz_path}' ORDER BY category"
            ).fetchdf()["category"].tolist()
        )

    # areas from area_profile
    areas = []
    area_path = _AGG / "area_profile.parquet"
    if area_path.exists():
        areas = sorted(
            con.execute(
                f"SELECT DISTINCT area FROM '{area_path}' ORDER BY area"
            ).fetchdf()["area"].tolist()
        )

    con.close()
    return {
        "zip_codes": zip_codes,
        "categories": categories,
        "statuses": statuses,
        "areas": areas,
    }


def get_health() -> dict:
    """Check data file availability and freshness."""
    files = {
        "neighborhood_profile": _AGG / "neighborhood_profile.parquet",
        "demographics": _PROCESSED / "demographics.parquet",
        "businesses": _PROCESSED / "businesses.parquet",
        "business_by_zip": _AGG / "business_by_zip.parquet",
        "business_summary": _AGG / "business_summary.parquet",
        "city_averages": _AGG / "city_averages.parquet",
        "civic_permits": _AGG / "civic_permits.parquet",
        "civic_solar": _AGG / "civic_solar.parquet",
        "civic_crime": _AGG / "civic_crime.parquet",
        "civic_311": _AGG / "civic_311.parquet",
        "civic_homelessness": _AGG / "civic_homelessness.parquet",
    }

    status = {}
    latest_mtime = None
    for name, path in files.items():
        exists = path.exists()
        status[name] = exists
        if exists:
            mtime = path.stat().st_mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime

    data_as_of = None
    if latest_mtime:
        data_as_of = date.fromtimestamp(latest_mtime).isoformat()

    return {
        "status": "ok" if any(status.values()) else "no_data",
        "files": status,
        "data_as_of": data_as_of,
    }


def get_neighborhood_profile(zip_code: str) -> dict:
    """Get full neighborhood profile: demographics + business counts + civic signals."""
    con = duckdb.connect()

    profile_path = _AGG / "neighborhood_profile.parquet"
    if not profile_path.exists():
        con.close()
        return {"error": "neighborhood profile data not available"}

    rows = con.execute(
        f"SELECT * FROM '{profile_path}' WHERE zip_code = $1", [zip_code]
    ).fetchdf()

    if rows.empty:
        con.close()
        return {"error": f"no data for zip code {zip_code}"}

    row = rows.iloc[0].to_dict()

    # get top business categories for this zip
    top_categories = []
    biz_by_zip_path = _AGG / "business_by_zip.parquet"
    if biz_by_zip_path.exists():
        top_categories = con.execute(
            f"""
            SELECT category, active_count, total_count
            FROM '{biz_by_zip_path}'
            WHERE zip_code = $1
            ORDER BY active_count DESC
            LIMIT 10
            """,
            [zip_code],
        ).fetchdf().to_dict(orient="records")

    # per-category density (per 1k residents)
    population = _clean(row.get("population"))
    if population and population > 0:
        for cat in top_categories:
            cat["per_1k"] = round(1000.0 * cat["active_count"] / population, 1)
    else:
        for cat in top_categories:
            cat["per_1k"] = None

    # city avg per-category density
    np_path = _AGG / "neighborhood_profile.parquet"
    if biz_by_zip_path.exists() and np_path.exists():
        city_cat_density = con.execute(
            f"""
            WITH sd_zips AS (
                SELECT zip_code, population FROM '{np_path}' WHERE population > 0
            ),
            total_pop AS (SELECT SUM(population) AS city_pop FROM sd_zips)
            SELECT bz.category, ROUND(1000.0 * SUM(bz.active_count) / tp.city_pop, 2) AS city_per_1k
            FROM '{biz_by_zip_path}' bz
            JOIN sd_zips sz ON bz.zip_code = sz.zip_code
            CROSS JOIN total_pop tp
            GROUP BY bz.category, tp.city_pop
            """
        ).fetchdf().set_index("category")["city_per_1k"].to_dict()
        for cat in top_categories:
            cat["city_avg_per_1k"] = _clean(city_cat_density.get(cat["category"]))

    # get city averages for comparison
    avg = {}
    avg_path = _AGG / "city_averages.parquet"
    if avg_path.exists():
        avg_rows = con.execute(f"SELECT * FROM '{avg_path}'").fetchdf()
        if not avg_rows.empty:
            avg = avg_rows.iloc[0].to_dict()

    # compute percentile ranks across all zips
    rank_cols = []
    for metric, direction in _PERCENTILE_METRICS.items():
        rank_cols.append(
            f"RANK() OVER (ORDER BY {metric} {direction} NULLS LAST) AS {metric}_rank"
        )
        rank_cols.append(f"COUNT({metric}) OVER () AS {metric}_of")

    rank_df = con.execute(
        f"""
        WITH ranked AS (
            SELECT zip_code, {', '.join(rank_cols)}
            FROM '{profile_path}'
        )
        SELECT * FROM ranked WHERE zip_code = $1
        """,
        [zip_code],
    ).fetchdf()

    percentiles = {}
    if not rank_df.empty:
        rr = rank_df.iloc[0]
        for metric in _PERCENTILE_METRICS:
            if _clean(row.get(metric)) is None:
                continue
            rank = int(rr[f"{metric}_rank"])
            of = int(rr[f"{metric}_of"])
            if rank <= of:
                pctile = round(100 * (of - rank) / max(of - 1, 1))
                percentiles[metric] = {
                    "rank": rank,
                    "of": of,
                    "percentile": pctile,
                }

    con.close()

    # build comparison
    comparison = {}
    compare_fields = [
        ("population", "avg_population"),
        ("median_income", "avg_median_income"),
        ("median_home_value", "avg_median_home_value"),
        ("median_rent", "avg_median_rent"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus"),
        ("active_count", "avg_active_businesses"),
        ("businesses_per_1k", "avg_businesses_per_1k"),
    ]
    for local_key, avg_key in compare_fields:
        local_val = row.get(local_key)
        avg_val = avg.get(avg_key)
        if local_val is not None and avg_val is not None and avg_val != 0:
            comparison[local_key] = {
                "value": _clean(local_val),
                "city_avg": round(float(avg_val), 1),
                "vs_avg_pct": round(100 * (float(local_val) - float(avg_val)) / float(avg_val), 1),
            }

    return {
        "zip_code": zip_code,
        "neighborhood": row.get("neighborhood"),
        "area": _clean(row.get("area")),
        "demographics": {
            "population": _clean(row.get("population")),
            "median_age": _clean(row.get("median_age")),
            "median_income": _clean(row.get("median_income")),
            "median_home_value": _clean(row.get("median_home_value")),
            "median_rent": _clean(row.get("median_rent")),
            "pct_bachelors_plus": _clean(row.get("pct_bachelors_plus")),
        },
        "business_landscape": {
            "active_count": _clean(row.get("active_count")),
            "total_count": _clean(row.get("total_count")),
            "category_count": _clean(row.get("category_count")),
            "businesses_per_1k": _clean(row.get("businesses_per_1k")),
            "top_categories": top_categories,
        },
        "civic_signals": {
            "new_permits": _clean(row.get("new_permits")),
            "permit_valuation": _clean(row.get("permit_valuation")),
            "solar_installs": _clean(row.get("solar_installs")),
            "crime_count": _clean(row.get("crime_count")),
            "median_311_days": _clean(row.get("median_311_days")),
            "total_311_requests": _clean(row.get("total_311_requests")),
        },
        "comparison_to_avg": comparison,
        "percentiles": percentiles,
        "narrative": _build_narrative(row, avg),
        "data_as_of": get_health().get("data_as_of"),
    }


def get_businesses(
    zip_code: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get individual business records from processed parquet."""
    biz_path = _PROCESSED / "businesses.parquet"
    if not biz_path.exists():
        return []

    clauses = []
    params = []
    idx = 1

    if zip_code:
        clauses.append(f"zip_code = ${idx}")
        params.append(zip_code)
        idx += 1
    if category:
        clauses.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    if status:
        clauses.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT account_id, business_name, address, zip_code, naics_code,
               activity_description, category, start_date, created_date,
               expiration_date, status
        FROM '{biz_path}'
        {where}
        ORDER BY business_name
        LIMIT {min(limit, 5000)}
    """

    con = duckdb.connect()
    df = con.execute(sql, params).fetchdf()
    con.close()
    return df.to_dict(orient="records")


def _build_narrative(row: dict, avg: dict) -> str:
    """Generate a one-sentence narrative comparing a zip to city averages."""
    if not avg:
        return ""

    # (row_key, avg_key, label, higher_is, less_word)
    metrics = [
        ("population", "avg_population", "people", "good", "fewer"),
        ("median_income", "avg_median_income", "income", "good", "lower"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus", "college grads", "good", "fewer"),
        ("active_count", "avg_active_businesses", "businesses", "good", "fewer"),
        ("businesses_per_1k", "avg_businesses_per_1k", "businesses per capita", "good", "fewer"),
        ("crime_count", "avg_crime_count", "crime", "bad", "less"),
        ("median_311_days", "avg_median_311_days", "311 response time", "bad", "less"),
        ("new_permits", "avg_new_permits", "new permits", "good", "fewer"),
        ("median_rent", "avg_median_rent", "rent", "bad", "lower"),
    ]

    scored = []

    for row_key, avg_key, label, higher_is, less_word in metrics:
        val = _clean(row.get(row_key))
        avg_val = _clean(avg.get(avg_key))
        if val is None or avg_val is None or float(avg_val) == 0:
            continue

        ratio = float(val) / float(avg_val)

        if 0.9 <= ratio <= 1.1:
            continue

        if ratio > 10.0:
            clause = f"far more {label}"
        elif ratio >= 2.0:
            clause = f"{ratio:.0f}x the {label}"
        elif ratio > 1.1:
            clause = f"{(ratio - 1) * 100:.0f}% more {label}"
        elif ratio < 0.1:
            clause = f"far {less_word} {label}"
        elif ratio < 0.5:
            inv = 1 / ratio
            clause = f"1/{inv:.0f} the {label}"
        else:
            clause = f"{(1 - ratio) * 100:.0f}% {less_word} {label}"

        is_higher = ratio > 1.0
        is_positive = (is_higher and higher_is == "good") or (not is_higher and higher_is == "bad")
        scored.append((abs(ratio - 1), clause, is_positive))

    if not scored:
        return "close to city average across most metrics"

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:5]

    positives = [clause for _, clause, pos in top if pos]
    negatives = [clause for _, clause, pos in top if not pos]

    parts = []
    if positives:
        parts.append(", ".join(positives))
    if negatives:
        parts.append(", ".join(negatives))

    return "compared to the avg sd zip code: " + " — but ".join(parts)


def _build_comparison_narrative(profile_a: dict, profile_b: dict) -> str:
    """Generate a head-to-head narrative comparing two zip codes."""
    name_a = profile_a.get("neighborhood") or profile_a["zip_code"]
    name_b = profile_b.get("neighborhood") or profile_b["zip_code"]

    def _flat(p):
        flat = {}
        for section in ("demographics", "business_landscape", "civic_signals"):
            if section in p:
                for k, v in p[section].items():
                    if k != "top_categories":
                        flat[k] = v
        return flat

    a = _flat(profile_a)
    b = _flat(profile_b)

    # (key, label, higher_is, less_word)
    metrics = [
        ("population", "people", "good", "fewer"),
        ("median_income", "income", "good", "lower"),
        ("pct_bachelors_plus", "college grads", "good", "fewer"),
        ("active_count", "businesses", "good", "fewer"),
        ("businesses_per_1k", "businesses per capita", "good", "fewer"),
        ("crime_count", "crime", "bad", "less"),
        ("median_311_days", "311 response time", "bad", "less"),
        ("new_permits", "new permits", "good", "fewer"),
        ("median_rent", "rent", "bad", "lower"),
    ]

    a_wins = []
    b_wins = []

    for key, label, higher_is, less_word in metrics:
        val_a = _clean(a.get(key))
        val_b = _clean(b.get(key))
        if val_a is None or val_b is None:
            continue
        fa, fb = float(val_a), float(val_b)
        if fa == fb:
            continue

        a_higher = fa > fb
        bigger, smaller = (fa, fb) if a_higher else (fb, fa)

        if smaller > 0:
            ratio = bigger / smaller
            if ratio > 10.0:
                mag = "far"
            elif ratio >= 2.0:
                mag = f"{ratio:.0f}x"
            else:
                mag = f"{(ratio - 1) * 100:.0f}%"
        else:
            mag = "far"

        winner_is_a = a_higher if higher_is == "good" else not a_higher
        # clause from winner's perspective: winner always has the "good" side
        if higher_is == "good":
            clause = f"{mag} more {label}"
        else:
            clause = f"{mag} {less_word} {label}"

        score = abs(fa - fb) / max(abs(fa), abs(fb), 1)
        if winner_is_a:
            a_wins.append((score, clause))
        else:
            b_wins.append((score, clause))

    a_wins.sort(key=lambda t: t[0], reverse=True)
    b_wins.sort(key=lambda t: t[0], reverse=True)

    parts = []
    top_a = [c for _, c in a_wins[:3]]
    top_b = [c for _, c in b_wins[:3]]
    if top_a:
        parts.append(f"{name_a} has {', '.join(top_a)}")
    if top_b:
        parts.append(f"{name_b} has {', '.join(top_b)}")

    if not parts:
        return f"{name_a} and {name_b} are similar across most metrics"

    return " — ".join(parts)


def _build_area_narrative(row: dict, avg: dict) -> str:
    """Generate a one-sentence narrative comparing an area to city averages."""
    if not avg:
        return ""

    area = row.get("area", "this area")
    zip_count = row.get("zip_count", 0)
    pop = row.get("population")
    pop_str = f"pop {pop:,.0f}" if pop else ""

    metrics = [
        ("active_count", "avg_active_businesses", "businesses", "good", "fewer"),
        ("businesses_per_1k", "avg_businesses_per_1k", "businesses per 1k residents", "good", "fewer"),
        ("median_income", "avg_median_income", "median income", "good", "lower"),
        ("median_rent", "avg_median_rent", "median rent", "bad", "lower"),
        ("median_home_value", "avg_median_home_value", "home values", "bad", "lower"),
        ("crime_count", "avg_crime_count", "crime incidents", "bad", "fewer"),
        ("new_permits", "avg_new_permits", "new permits", "good", "fewer"),
        ("solar_installs", "avg_solar_installs", "solar installs", "good", "fewer"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus", "college-educated residents", "good", "fewer"),
    ]

    scored = []
    for row_key, avg_key, label, higher_is, less_word in metrics:
        local = row.get(row_key)
        city = avg.get(avg_key)
        if local is None or city is None or city == 0:
            continue
        ratio = local / city
        if 0.9 <= ratio <= 1.1:
            continue
        magnitude = abs(ratio - 1)
        is_higher = ratio > 1
        if ratio > 10 or (ratio > 0 and ratio < 0.1):
            mag_str = f"{ratio:.0f}x" if is_higher else f"1/{1/ratio:.0f}th the"
        elif ratio >= 2:
            mag_str = f"{ratio:.1f}x"
        elif ratio <= 0.5:
            mag_str = f"{100*(1-ratio):.0f}% {less_word}"
        elif is_higher:
            mag_str = f"{100*(ratio-1):.0f}% more"
        else:
            mag_str = f"{100*(1-ratio):.0f}% {less_word}"
        is_positive = (is_higher and higher_is == "good") or (not is_higher and higher_is == "bad")
        scored.append((magnitude, is_positive, f"{mag_str} {label}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]
    positives = [s[2] for s in top if s[1]]
    negatives = [s[2] for s in top if not s[1]]

    prefix = f"{area.lower()} ({zip_count} zip codes, {pop_str})" if pop_str else f"{area.lower()} ({zip_count} zip codes)"

    if not positives and not negatives:
        return f"{prefix} is close to city average across most metrics."

    parts = []
    if positives:
        parts.append(", ".join(positives))
    if negatives:
        neg_str = ", ".join(negatives)
        parts.append(f"but {neg_str}" if positives else neg_str)

    joined = " — ".join(parts)
    return f"compared to the avg sd area: {prefix} has {joined}."


def _build_area_comparison_narrative(profile_a: dict, profile_b: dict) -> str:
    """Generate a comparison narrative for two areas."""
    name_a = profile_a.get("area", "area a")
    name_b = profile_b.get("area", "area b")

    metrics = [
        ("population", "demographics", "population", "neutral"),
        ("median_income", "demographics", "median income", "good"),
        ("businesses_per_1k", "business_landscape", "businesses per capita", "good"),
        ("active_count", "business_landscape", "total businesses", "good"),
        ("crime_count", "civic_signals", "crime incidents", "bad"),
        ("new_permits", "civic_signals", "new permits", "good"),
        ("median_311_days", "civic_signals", "311 response time", "bad"),
    ]

    a_wins = []
    b_wins = []
    for metric, section, label, higher_is in metrics:
        val_a = profile_a.get(section, {}).get(metric)
        val_b = profile_b.get(section, {}).get(metric)
        if val_a is None or val_b is None or val_b == 0:
            continue
        ratio = val_a / val_b
        if 0.9 <= ratio <= 1.1:
            continue
        pct = abs(ratio - 1) * 100
        desc = f"{pct:.0f}% {'more' if ratio > 1 else 'fewer'} {label}"
        if higher_is == "neutral":
            continue
        a_better = (ratio > 1 and higher_is == "good") or (ratio < 1 and higher_is == "bad")
        if a_better:
            a_wins.append((pct, desc))
        else:
            b_wins.append((pct, desc))

    a_wins.sort(reverse=True)
    b_wins.sort(reverse=True)
    a_top = [w[1] for w in a_wins[:3]]
    b_top = [w[1] for w in b_wins[:3]]

    if not a_top and not b_top:
        return f"{name_a.lower()} and {name_b.lower()} are similar across most metrics."

    parts = []
    if a_top:
        parts.append(f"{name_a.lower()} has {', '.join(a_top)}")
    if b_top:
        parts.append(f"{name_b.lower()} has {', '.join(b_top)}")

    return " — ".join(parts) + "."


_RANKING_METRICS = frozenset({
    "population", "median_income", "median_age", "median_rent",
    "median_home_value", "pct_bachelors_plus", "active_count",
    "businesses_per_1k", "category_count", "new_permits",
    "crime_count", "median_311_days", "solar_installs",
    "total_311_requests", "permit_valuation",
})


def get_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 82,
) -> list[dict]:
    """Rank all 82 SD zip codes by a chosen metric.

    sort_by always controls the sort order. When category is provided, adds
    category_active and category_per_1k columns as context. Use
    sort_by="category_per_1k" to explicitly sort by category density.
    """
    profile_path = _AGG / "neighborhood_profile.parquet"
    if not profile_path.exists():
        return []

    limit = min(limit, 82)
    direction = "DESC" if sort_desc else "ASC"

    if sort_by == "category_per_1k" and not category:
        return [{"error": "category_per_1k requires a category parameter"}]

    if sort_by != "category_per_1k" and sort_by not in _RANKING_METRICS:
        return [{"error": f"invalid sort_by: {sort_by}. allowed: {sorted(_RANKING_METRICS)}"}]

    if category:
        biz_path = _AGG / "business_by_zip.parquet"
        if not biz_path.exists():
            return []

        select = ["np.zip_code", "np.neighborhood"]
        if sort_by != "category_per_1k":
            select.append(f"np.{sort_by} AS sort_value")
        select.extend([
            "COALESCE(bz.active_count, 0) AS category_active",
            """CASE WHEN np.population > 0
                 THEN ROUND(1000.0 * COALESCE(bz.active_count, 0) / np.population, 2)
                 ELSE NULL END AS category_per_1k""",
        ])
        for c in ("population", "median_income", "active_count"):
            if c != sort_by:
                select.append(f"np.{c}")

        order = "category_per_1k" if sort_by == "category_per_1k" else f"np.{sort_by}"
        where = "" if sort_by == "category_per_1k" else f"WHERE np.{sort_by} IS NOT NULL"

        sql = f"""
            SELECT {', '.join(select)}
            FROM '{profile_path}' np
            LEFT JOIN '{biz_path}' bz
                ON np.zip_code = bz.zip_code AND bz.category = $1
            {where}
            ORDER BY {order} {direction} NULLS LAST
            LIMIT {limit}
        """
        rows = _run(sql, [category])
        for i, r in enumerate(rows, 1):
            r["rank"] = i
            r["sort_metric"] = sort_by
            r["category"] = category
            for k in list(r):
                r[k] = _clean(r[k])
        return rows

    # No category — standard metric ranking
    context = ["population", "median_income", "active_count"]
    extras = [c for c in context if c != sort_by]
    cols = f"{sort_by} AS sort_value, " + ", ".join(extras)

    sql = f"""
        SELECT
            zip_code, neighborhood, {cols}
        FROM '{profile_path}'
        WHERE {sort_by} IS NOT NULL
        ORDER BY {sort_by} {direction}
        LIMIT {limit}
    """
    rows = _run(sql)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["sort_metric"] = sort_by
        for k in list(r):
            r[k] = _clean(r[k])
    return rows


def compare_zips(zip_a: str, zip_b: str) -> dict:
    """Compare two zip codes head-to-head."""
    profile_a = get_neighborhood_profile(zip_a)
    profile_b = get_neighborhood_profile(zip_b)

    if "error" in profile_a or "error" in profile_b:
        return {
            "zip_a": profile_a,
            "zip_b": profile_b,
            "error": profile_a.get("error") or profile_b.get("error"),
        }

    compare_metrics = [
        ("population", "demographics"),
        ("median_age", "demographics"),
        ("median_income", "demographics"),
        ("median_home_value", "demographics"),
        ("median_rent", "demographics"),
        ("pct_bachelors_plus", "demographics"),
        ("active_count", "business_landscape"),
        ("businesses_per_1k", "business_landscape"),
        ("new_permits", "civic_signals"),
        ("crime_count", "civic_signals"),
        ("median_311_days", "civic_signals"),
        ("solar_installs", "civic_signals"),
    ]

    head_to_head = {}
    for key, section in compare_metrics:
        val_a = profile_a[section].get(key)
        val_b = profile_b[section].get(key)
        diff = None
        if val_a is not None and val_b is not None:
            diff = round(float(val_a) - float(val_b), 1)
        head_to_head[key] = {
            "zip_a": val_a,
            "zip_b": val_b,
            "difference": diff,
        }

    return {
        "zip_a": profile_a,
        "zip_b": profile_b,
        "head_to_head": head_to_head,
        "narrative": _build_comparison_narrative(profile_a, profile_b),
    }


# ── Area query functions ──


def get_areas() -> list[dict]:
    """Get all areas with summary metrics."""
    path = _q("data/aggregated/area_profile.parquet")
    return _run(f"""
        SELECT
            area,
            zip_count,
            population,
            active_count,
            businesses_per_1k,
            median_income
        FROM '{path}'
        ORDER BY population DESC
    """)


def get_area_profile(area: str) -> dict:
    """Get full area profile with demographics, businesses, civic signals."""
    ap_path = _q("data/aggregated/area_profile.parquet")
    row = _run_one(f"SELECT * FROM '{ap_path}' WHERE area = $1", [area])
    if not row:
        return {}

    # top categories for area
    abc_path = _q("data/aggregated/area_business_by_category.parquet")
    top_cats = _run(f"""
        SELECT category, active_count, total_count, per_1k
        FROM '{abc_path}'
        WHERE area = $1
        ORDER BY active_count DESC
        LIMIT 10
    """, [area])

    # city avg for comparison
    avg_path = _q("data/aggregated/city_averages.parquet")
    avg = _run_one(f"SELECT * FROM '{avg_path}'")

    # city avg per-1k for each category (total/total method)
    bz_path = _q("data/aggregated/business_by_zip.parquet")
    demo_path = _q("data/processed/demographics.parquet")
    for cat in top_cats:
        city_cat = _run_one(f"""
            SELECT ROUND(1000.0 * SUM(bz.active_count) / NULLIF(
                (SELECT SUM(population) FROM '{demo_path}'), 0
            ), 2) AS city_avg_per_1k
            FROM '{bz_path}' bz
            WHERE bz.category = $1
              AND bz.zip_code IN (SELECT zip_code FROM '{demo_path}')
        """, [cat["category"]])
        cat["city_avg_per_1k"] = _clean(city_cat.get("city_avg_per_1k")) if city_cat else None

    # comparison to avg
    comparison = {}
    compare_fields = [
        ("population", "avg_population"),
        ("median_income", "avg_median_income"),
        ("median_home_value", "avg_median_home_value"),
        ("median_rent", "avg_median_rent"),
        ("pct_bachelors_plus", "avg_pct_bachelors_plus"),
        ("active_count", "avg_active_businesses"),
        ("businesses_per_1k", "avg_businesses_per_1k"),
    ]
    for local_key, avg_key in compare_fields:
        local_val = row.get(local_key)
        avg_val = avg.get(avg_key) if avg else None
        if local_val is not None and avg_val is not None and avg_val != 0:
            comparison[local_key] = {
                "value": _clean(local_val),
                "city_avg": _clean(avg_val),
                "vs_avg_pct": _clean(round(100 * (local_val - avg_val) / abs(avg_val), 1)),
            }

    zip_codes = row.get("zip_codes", [])

    return {
        "area": row.get("area"),
        "zip_codes": zip_codes if isinstance(zip_codes, list) else [],
        "zip_count": _clean(row.get("zip_count")),
        "demographics": {
            "population": _clean(row.get("population")),
            "median_age": _clean(row.get("median_age")),
            "median_income": _clean(row.get("median_income")),
            "median_home_value": _clean(row.get("median_home_value")),
            "median_rent": _clean(row.get("median_rent")),
            "pct_bachelors_plus": _clean(row.get("pct_bachelors_plus")),
        },
        "business_landscape": {
            "active_count": _clean(row.get("active_count")),
            "total_count": _clean(row.get("total_count")),
            "businesses_per_1k": _clean(row.get("businesses_per_1k")),
            "top_categories": [{k: _clean(v) for k, v in c.items()} for c in top_cats],
        },
        "civic_signals": {
            "new_permits": _clean(row.get("new_permits")),
            "permit_valuation": _clean(row.get("permit_valuation")),
            "solar_installs": _clean(row.get("solar_installs")),
            "crime_count": _clean(row.get("crime_count")),
            "median_311_days": _clean(row.get("median_311_days")),
            "total_311_requests": _clean(row.get("total_311_requests")),
        },
        "comparison_to_avg": comparison,
        "narrative": _build_area_narrative(row, avg),
    }


def compare_areas(area_a: str, area_b: str) -> dict:
    """Compare two areas head-to-head."""
    profile_a = get_area_profile(area_a)
    profile_b = get_area_profile(area_b)

    if not profile_a or not profile_b:
        return {"error": "One or both areas not found"}

    compare_metrics = [
        ("population", "demographics"),
        ("median_income", "demographics"),
        ("median_home_value", "demographics"),
        ("median_rent", "demographics"),
        ("pct_bachelors_plus", "demographics"),
        ("active_count", "business_landscape"),
        ("businesses_per_1k", "business_landscape"),
        ("new_permits", "civic_signals"),
        ("crime_count", "civic_signals"),
        ("solar_installs", "civic_signals"),
        ("median_311_days", "civic_signals"),
    ]

    head_to_head = {}
    for metric, section in compare_metrics:
        val_a = profile_a.get(section, {}).get(metric)
        val_b = profile_b.get(section, {}).get(metric)
        diff = None
        if val_a is not None and val_b is not None:
            diff = _clean(round(100 * (val_a - val_b) / abs(val_b), 1)) if val_b != 0 else None
        head_to_head[metric] = {
            "area_a": _clean(val_a),
            "area_b": _clean(val_b),
            "difference": diff,
        }

    return {
        "area_a": profile_a,
        "area_b": profile_b,
        "head_to_head": head_to_head,
        "narrative": _build_area_comparison_narrative(profile_a, profile_b),
    }


def get_area_rankings(
    sort_by: str = "population",
    sort_desc: bool = True,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Rank areas by a chosen metric."""
    ap_path = _q("data/aggregated/area_profile.parquet")
    direction = "DESC" if sort_desc else "ASC"

    if category:
        abc_path = _q("data/aggregated/area_business_by_category.parquet")
        if sort_by == "category_per_1k":
            order_col = "abc.per_1k"
        else:
            order_col = f"ap.{sort_by}"
        rows = _run(f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {order_col} {direction} NULLS LAST) AS rank,
                ap.area,
                ap.{sort_by} AS sort_value,
                '{sort_by}' AS sort_metric,
                abc.active_count AS category_active,
                abc.per_1k AS category_per_1k,
                '{category}' AS category,
                ap.population,
                ap.median_income,
                ap.active_count
            FROM '{ap_path}' ap
            LEFT JOIN '{abc_path}' abc ON ap.area = abc.area AND abc.category = $1
            ORDER BY {order_col} {direction} NULLS LAST
            LIMIT {limit}
        """, [category])
    else:
        rows = _run(f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {sort_by} {direction} NULLS LAST) AS rank,
                area,
                {sort_by} AS sort_value,
                '{sort_by}' AS sort_metric,
                NULL AS category,
                NULL AS category_active,
                NULL AS category_per_1k,
                population,
                median_income,
                active_count
            FROM '{ap_path}'
            ORDER BY {sort_by} {direction} NULLS LAST
            LIMIT {limit}
        """)

    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def get_area_zips(area: str) -> list[dict]:
    """Get all zips in an area with key metrics."""
    np_path = _q("data/aggregated/neighborhood_profile.parquet")
    return _run(f"""
        SELECT
            zip_code,
            neighborhood,
            population,
            active_count,
            businesses_per_1k,
            median_income,
            crime_count,
            new_permits
        FROM '{np_path}'
        WHERE area = $1
        ORDER BY active_count DESC
    """, [area])


# ── Trend query functions ──


def get_zip_trends(zip_code: str) -> dict:
    """Get year-over-year trend data for a zip."""
    trends = {}

    # Business formation
    tbf_path = _AGG / "trend_business_formation.parquet"
    if tbf_path.exists():
        rows = _run(f"""
            SELECT year, new_businesses AS count
            FROM '{tbf_path}'
            WHERE zip_code = $1
            ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["business_formation"] = rows

    # Permits
    cp_path = _AGG / "civic_permits.parquet"
    if cp_path.exists():
        rows = _run(f"""
            SELECT year, SUM(permit_count) AS count
            FROM '{cp_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["permits"] = rows

    # Crime
    cc_path = _AGG / "civic_crime.parquet"
    if cc_path.exists():
        rows = _run(f"""
            SELECT year, SUM(count) AS count
            FROM '{cc_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["crime"] = rows

    # Solar
    cs_path = _AGG / "civic_solar.parquet"
    if cs_path.exists():
        rows = _run(f"""
            SELECT year, SUM(solar_count) AS count
            FROM '{cs_path}'
            WHERE zip_code = $1
            GROUP BY year ORDER BY year
        """, [zip_code])
        _add_yoy(rows)
        trends["solar"] = rows

    return trends


def get_area_trends(area: str) -> dict:
    """Get year-over-year trend data aggregated across an area's zips."""
    from pipeline.transform import ZIP_TO_AREA
    zips = [z for z, a in ZIP_TO_AREA.items() if a == area]
    if not zips:
        return {}

    placeholders = ", ".join(f"'{z}'" for z in zips)
    trends = {}

    tbf_path = _AGG / "trend_business_formation.parquet"
    if tbf_path.exists():
        rows = _run(f"""
            SELECT year, SUM(new_businesses) AS count
            FROM '{tbf_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["business_formation"] = rows

    cp_path = _AGG / "civic_permits.parquet"
    if cp_path.exists():
        rows = _run(f"""
            SELECT year, SUM(permit_count) AS count
            FROM '{cp_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["permits"] = rows

    cc_path = _AGG / "civic_crime.parquet"
    if cc_path.exists():
        rows = _run(f"""
            SELECT year, SUM(count) AS count
            FROM '{cc_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["crime"] = rows

    cs_path = _AGG / "civic_solar.parquet"
    if cs_path.exists():
        rows = _run(f"""
            SELECT year, SUM(solar_count) AS count
            FROM '{cs_path}'
            WHERE zip_code IN ({placeholders})
            GROUP BY year ORDER BY year
        """)
        _add_yoy(rows)
        trends["solar"] = rows

    return trends


def _clean(val):
    """Convert numpy/pandas types to native Python, handle NaN."""
    if val is None:
        return None
    import math
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, float):
        if val == int(val):
            return int(val)
        return round(float(val), 1)
    return val


def _add_yoy(rows: list[dict]) -> None:
    """Add yoy_pct to each row (in-place)."""
    for i, row in enumerate(rows):
        if i == 0 or rows[i - 1]["count"] is None or rows[i - 1]["count"] == 0:
            row["yoy_pct"] = None
        else:
            row["yoy_pct"] = _clean(
                round(100 * (row["count"] - rows[i - 1]["count"]) / abs(rows[i - 1]["count"]), 1)
            )
        row["count"] = _clean(row["count"])
        row["year"] = _clean(row["year"])
