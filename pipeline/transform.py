"""DuckDB transforms: raw CSVs → processed parquets → aggregated parquets."""

from __future__ import annotations

from pathlib import Path

import duckdb

from pipeline.naics_map import map_naics

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
AGG_DIR = Path(__file__).resolve().parent.parent / "data" / "aggregated"

# hardcoded zip → neighborhood name mapping for ~40 SD zips
ZIP_TO_NEIGHBORHOOD: dict[str, str] = {
    "91901": "Alpine",
    "91902": "Bonita",
    "91910": "Chula Vista",
    "91911": "Chula Vista",
    "91913": "Chula Vista",
    "91914": "Chula Vista",
    "91915": "Chula Vista",
    "91932": "Imperial Beach",
    "91941": "La Mesa",
    "91942": "La Mesa",
    "91945": "Lemon Grove",
    "91950": "National City",
    "91977": "Spring Valley",
    "91978": "Spring Valley",
    "92007": "Cardiff-by-the-Sea",
    "92008": "Carlsbad",
    "92009": "Carlsbad",
    "92010": "Carlsbad",
    "92011": "Carlsbad",
    "92014": "Del Mar",
    "92019": "El Cajon",
    "92020": "El Cajon",
    "92021": "El Cajon",
    "92024": "Encinitas",
    "92025": "Escondido",
    "92026": "Escondido",
    "92027": "Escondido",
    "92028": "Fallbrook",
    "92029": "Escondido",
    "92037": "La Jolla",
    "92040": "Lakeside",
    "92054": "Oceanside",
    "92056": "Oceanside",
    "92057": "Oceanside",
    "92058": "Oceanside",
    "92064": "Poway",
    "92065": "Ramona",
    "92067": "Rancho Santa Fe",
    "92069": "San Marcos",
    "92071": "Santee",
    "92075": "Solana Beach",
    "92078": "San Marcos",
    "92081": "Vista",
    "92083": "Vista",
    "92084": "Vista",
    "92091": "Rancho Santa Fe",
    "92101": "Downtown",
    "92102": "Golden Hill / South Park",
    "92103": "Hillcrest / Mission Hills",
    "92104": "North Park / University Heights",
    "92105": "City Heights",
    "92106": "Point Loma",
    "92107": "Ocean Beach",
    "92108": "Mission Valley",
    "92109": "Pacific Beach / Mission Beach",
    "92110": "Morena / Bay Park",
    "92111": "Linda Vista / Clairemont",
    "92113": "Barrio Logan / Logan Heights",
    "92114": "Encanto / Lomita",
    "92115": "College Area",
    "92116": "Normal Heights / Kensington",
    "92117": "Clairemont",
    "92118": "Coronado",
    "92119": "San Carlos",
    "92120": "Del Cerro / Allied Gardens",
    "92121": "Sorrento Valley / Torrey Pines",
    "92122": "University City",
    "92123": "Serra Mesa / Kearny Mesa",
    "92124": "Tierrasanta",
    "92126": "Mira Mesa",
    "92127": "Rancho Bernardo",
    "92128": "Rancho Bernardo",
    "92129": "Rancho Penasquitos",
    "92130": "Carmel Valley",
    "92131": "Scripps Ranch",
    "92132": "Naval Base",
    "92134": "Naval Base",
    "92139": "Paradise Hills",
    "92140": "Naval Base",
    "92145": "Miramar",
    "92154": "Otay Mesa / San Ysidro",
    "92173": "San Ysidro",
}

# approximate zip → community plan name (for 311 data join)
ZIP_TO_COMM_PLAN: dict[str, str] = {
    "92101": "Downtown",
    "92102": "Southeastern San Diego",
    "92103": "Uptown",
    "92104": "Mid-City:Normal Heights",
    "92105": "Mid-City:City Heights",
    "92106": "Peninsula",
    "92107": "Ocean Beach",
    "92108": "Mission Valley",
    "92109": "Pacific Beach",
    "92110": "Linda Vista",
    "92111": "Linda Vista",
    "92113": "Barrio Logan",
    "92114": "Encanto Neighborhoods",
    "92115": "College Area",
    "92116": "Mid-City:Normal Heights",
    "92117": "Clairemont Mesa",
    "92119": "Navajo",
    "92120": "Navajo",
    "92121": "Torrey Pines",
    "92122": "University",
    "92123": "Kearny Mesa",
    "92124": "Tierrasanta",
    "92126": "Mira Mesa",
    "92127": "Rancho Bernardo",
    "92128": "Rancho Bernardo",
    "92129": "Rancho Penasquitos",
    "92130": "Carmel Valley",
    "92131": "Scripps Miramar Ranch",
    "92139": "Otay Mesa-Nestor",
    "92154": "Otay Mesa",
    "92173": "San Ysidro",
}

# zip → area mapping for area-level aggregation (only profiled zips)
ZIP_TO_AREA: dict[str, str] = {
    # Downtown
    "92101": "Downtown",
    # Uptown / North Park
    "92103": "Uptown / North Park",
    "92104": "Uptown / North Park",
    "92116": "Uptown / North Park",
    # Golden Hill / City Heights
    "92102": "Golden Hill / City Heights",
    "92105": "Golden Hill / City Heights",
    # Barrio Logan / Logan Heights
    "92113": "Barrio Logan / Logan Heights",
    # Pacific Beach
    "92109": "Pacific Beach",
    # Ocean Beach / Point Loma
    "92106": "Ocean Beach / Point Loma",
    "92107": "Ocean Beach / Point Loma",
    # Mission Valley / Linda Vista
    "92108": "Mission Valley / Linda Vista",
    "92110": "Mission Valley / Linda Vista",
    "92111": "Mission Valley / Linda Vista",
    # Clairemont
    "92117": "Clairemont",
    # La Jolla / University City
    "92037": "La Jolla / University City",
    "92122": "La Jolla / University City",
    # Sorrento Valley / Mira Mesa
    "92121": "Sorrento Valley / Mira Mesa",
    "92126": "Sorrento Valley / Mira Mesa",
    # Rancho Bernardo / Scripps Ranch
    "92127": "Rancho Bernardo / Scripps Ranch",
    "92128": "Rancho Bernardo / Scripps Ranch",
    "92131": "Rancho Bernardo / Scripps Ranch",
    # Carmel Valley / Rancho Penasquitos
    "92129": "Carmel Valley / Rancho Penasquitos",
    "92130": "Carmel Valley / Rancho Penasquitos",
    # Carlsbad
    "92008": "Carlsbad",
    "92009": "Carlsbad",
    "92010": "Carlsbad",
    "92011": "Carlsbad",
    # Oceanside
    "92054": "Oceanside",
    "92056": "Oceanside",
    "92057": "Oceanside",
    "92058": "Oceanside",
    # Encinitas / Del Mar / Solana Beach
    "92007": "Encinitas / Del Mar / Solana Beach",
    "92014": "Encinitas / Del Mar / Solana Beach",
    "92024": "Encinitas / Del Mar / Solana Beach",
    "92075": "Encinitas / Del Mar / Solana Beach",
    # Escondido
    "92025": "Escondido",
    "92026": "Escondido",
    "92027": "Escondido",
    "92029": "Escondido",
    # Vista / San Marcos
    "92069": "Vista / San Marcos",
    "92078": "Vista / San Marcos",
    "92081": "Vista / San Marcos",
    "92083": "Vista / San Marcos",
    "92084": "Vista / San Marcos",
    # Chula Vista / National City
    "91910": "Chula Vista / National City",
    "91911": "Chula Vista / National City",
    "91913": "Chula Vista / National City",
    "91914": "Chula Vista / National City",
    "91915": "Chula Vista / National City",
    "91950": "Chula Vista / National City",
    # La Mesa / Lemon Grove
    "91941": "La Mesa / Lemon Grove",
    "91942": "La Mesa / Lemon Grove",
    "91945": "La Mesa / Lemon Grove",
    # El Cajon
    "92019": "El Cajon",
    "92020": "El Cajon",
    "92021": "El Cajon",
    # South Bay / San Ysidro
    "92154": "South Bay / San Ysidro",
    "92173": "South Bay / San Ysidro",
    "91932": "South Bay / San Ysidro",
    "92139": "South Bay / San Ysidro",
    # East County
    "91901": "East County",
    "91977": "East County",
    "91978": "East County",
    "92040": "East County",
    "92065": "East County",
    "92071": "East County",
    # Poway / Rancho Santa Fe
    "92064": "Poway / Rancho Santa Fe",
    "92067": "Poway / Rancho Santa Fe",
    "92091": "Poway / Rancho Santa Fe",
    # Unmapped profiled zips (will default to "Other" at runtime):
    # 91902 — Bonita
    # 92028 — Fallbrook
    # 92114 — Encanto / Lomita
    # 92115 — College Area
    # 92118 — Coronado
    # 92119 — San Carlos
    # 92120 — Del Cerro / Allied Gardens
    # 92123 — Serra Mesa / Kearny Mesa
    # 92124 — Tierrasanta
    # 92132 — Naval Base
    # 92134 — Naval Base
    # 92140 — Naval Base
    # 92145 — Miramar
}


def _register_naics_udf(con: duckdb.DuckDBPyConnection) -> None:
    """Register map_naics as a scalar UDF in DuckDB."""
    con.create_function("map_naics", map_naics, ["VARCHAR", "VARCHAR"], "VARCHAR")


def transform() -> None:
    """Run all transforms: raw → processed → aggregated."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _register_naics_udf(con)

    _build_businesses(con)
    _build_demographics(con)
    _build_aggregates(con)

    con.close()
    print("  [done] all transforms complete")


def _build_businesses(con: duckdb.DuckDBPyConnection) -> None:
    """Union btax CSVs → processed/businesses.parquet."""
    raw_files = list(RAW_DIR.glob("btax_active_*.csv"))
    if not raw_files:
        print("  [warn] no btax CSV files found, skipping businesses")
        return

    union_parts = " UNION ALL ".join(
        f"SELECT * FROM read_csv_auto('{f}', all_varchar=true)" for f in raw_files
    )

    con.execute(f"""
        CREATE OR REPLACE TABLE businesses_raw AS
        SELECT * FROM ({union_parts})
    """)

    # check what columns we have (case-insensitive matching)
    cols_raw = [row[0] for row in con.execute("DESCRIBE businesses_raw").fetchall()]
    print(f"  [info] btax columns: {cols_raw}")
    cols_lower = {c.lower(): c for c in cols_raw}

    def _find(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    zip_col = _find("ZIP", "zip", "zip_code")
    naics_col = _find("NAICS", "naics", "naics_code")
    desc_col = _find("ACTIVITY DESC", "activity_desc", "naics_description") or _find("DBA NAME", "dba_name")
    name_col = _find("DBA NAME", "dba_name", "business_name")
    acct_col = _find("BUSINESS ACCT#", "business_acct", "account_key")
    start_col = _find("START DT", "start_dt", "business_start_date")
    create_col = _find("CREATION DT", "creation_dt", "date_cert_created")
    exp_col = _find("EXP DT", "exp_dt", "date_cert_expired")
    addr_col = _find("ADDRESS", "address", "business_address")

    if zip_col is None:
        print("  [error] cannot find zip column in btax data")
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE businesses AS
        SELECT
            {f'"{acct_col}" AS account_id,' if acct_col else "'unknown' AS account_id,"}
            {f'"{name_col}" AS business_name,' if name_col else "'unknown' AS business_name,"}
            {f'"{addr_col}" AS address,' if addr_col else "NULL AS address,"}
            LEFT(REGEXP_REPLACE("{zip_col}", '[^0-9]', '', 'g'), 5) AS zip_code,
            {f'"{naics_col}" AS naics_code,' if naics_col else "NULL AS naics_code,"}
            {f'"{desc_col}" AS activity_description,' if desc_col else "NULL AS activity_description,"}
            map_naics(
                {f'"{naics_col}"' if naics_col else "NULL"},
                {f'"{desc_col}"' if desc_col else "''"}
            ) AS category,
            {f'''TRY_CAST(STRPTIME("{start_col}", '%m/%d/%Y') AS DATE) AS start_date,''' if start_col else "NULL AS start_date,"}
            {f'''TRY_CAST(STRPTIME("{create_col}", '%m/%d/%Y') AS DATE) AS created_date,''' if create_col else "NULL AS created_date,"}
            {f'''TRY_CAST(STRPTIME("{exp_col}", '%m/%d/%Y') AS DATE) AS expiration_date,''' if exp_col else "NULL AS expiration_date,"}
            CASE
                WHEN {f'''TRY_CAST(STRPTIME("{exp_col}", '%m/%d/%Y') AS DATE) IS NULL OR TRY_CAST(STRPTIME("{exp_col}", '%m/%d/%Y') AS DATE) >= CURRENT_DATE''' if exp_col else "TRUE"}
                THEN 'active' ELSE 'inactive'
            END AS status
        FROM businesses_raw
        WHERE "{zip_col}" IS NOT NULL
          AND LENGTH(REGEXP_REPLACE("{zip_col}", '[^0-9]', '', 'g')) >= 5
    """)

    # deduplicate
    con.execute("""
        CREATE OR REPLACE TABLE businesses AS
        SELECT DISTINCT * FROM businesses
    """)

    out = PROCESSED_DIR / "businesses.parquet"
    con.execute(f"COPY businesses TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    count = con.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
    print(f"  [done] businesses.parquet -> {count:,} rows")


def _build_demographics(con: duckdb.DuckDBPyConnection) -> None:
    """Census ACS CSV → processed/demographics.parquet."""
    census_file = RAW_DIR / "census_acs.csv"
    if not census_file.exists():
        print("  [warn] census_acs.csv not found, skipping demographics")
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE demographics AS
        SELECT
            "zip code tabulation area" AS zip_code,
            NULLIF(CAST(B01003_001E AS INTEGER), -666666666) AS population,
            NULLIF(CAST(B01002_001E AS FLOAT), -666666666) AS median_age,
            NULLIF(CAST(B19013_001E AS INTEGER), -666666666) AS median_income,
            NULLIF(CAST(B25077_001E AS INTEGER), -666666666) AS median_home_value,
            NULLIF(CAST(B25064_001E AS INTEGER), -666666666) AS median_rent,
            NULLIF(CAST(B15003_001E AS INTEGER), -666666666) AS education_total,
            (
                COALESCE(NULLIF(CAST(B15003_022E AS INTEGER), -666666666), 0) +
                COALESCE(NULLIF(CAST(B15003_023E AS INTEGER), -666666666), 0) +
                COALESCE(NULLIF(CAST(B15003_024E AS INTEGER), -666666666), 0) +
                COALESCE(NULLIF(CAST(B15003_025E AS INTEGER), -666666666), 0)
            ) AS bachelors_plus,
            CASE
                WHEN NULLIF(CAST(B15003_001E AS INTEGER), -666666666) > 0
                THEN ROUND(100.0 * (
                    COALESCE(NULLIF(CAST(B15003_022E AS INTEGER), -666666666), 0) +
                    COALESCE(NULLIF(CAST(B15003_023E AS INTEGER), -666666666), 0) +
                    COALESCE(NULLIF(CAST(B15003_024E AS INTEGER), -666666666), 0) +
                    COALESCE(NULLIF(CAST(B15003_025E AS INTEGER), -666666666), 0)
                ) / NULLIF(CAST(B15003_001E AS INTEGER), -666666666), 1)
                ELSE NULL
            END AS pct_bachelors_plus
        FROM read_csv_auto('{census_file}', all_varchar=true)
    """)

    out = PROCESSED_DIR / "demographics.parquet"
    con.execute(f"COPY demographics TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    count = con.execute("SELECT COUNT(*) FROM demographics").fetchone()[0]
    print(f"  [done] demographics.parquet -> {count:,} rows")


def _build_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    """Build all aggregated parquets from processed data."""
    biz_file = PROCESSED_DIR / "businesses.parquet"
    demo_file = PROCESSED_DIR / "demographics.parquet"

    has_biz = biz_file.exists()
    has_demo = demo_file.exists()

    if has_biz:
        con.execute(f"CREATE OR REPLACE VIEW biz AS SELECT * FROM '{biz_file}'")
    if has_demo:
        con.execute(f"CREATE OR REPLACE VIEW demo AS SELECT * FROM '{demo_file}'")

    # 1. business_by_zip — GROUP BY zip_code, category
    if has_biz:
        con.execute(f"""
            COPY (
                SELECT
                    zip_code,
                    category,
                    COUNT(*) AS total_count,
                    COUNT(*) FILTER (WHERE status = 'active') AS active_count
                FROM biz
                GROUP BY zip_code, category
                ORDER BY zip_code, active_count DESC
            ) TO '{AGG_DIR}/business_by_zip.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] business_by_zip.parquet")

    # 2. business_summary — per-zip totals
    if has_biz:
        con.execute(f"""
            COPY (
                SELECT
                    zip_code,
                    COUNT(*) AS total_count,
                    COUNT(*) FILTER (WHERE status = 'active') AS active_count,
                    COUNT(DISTINCT category) AS category_count,
                    MIN(start_date) AS earliest_start,
                    MAX(start_date) AS latest_start
                FROM biz
                GROUP BY zip_code
                ORDER BY zip_code
            ) TO '{AGG_DIR}/business_summary.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] business_summary.parquet")

    # 3. demographics_by_zip — demographics filtered to zips that have businesses
    if has_demo and has_biz:
        con.execute(f"""
            COPY (
                SELECT d.*
                FROM demo d
                WHERE d.zip_code IN (SELECT DISTINCT zip_code FROM biz)
                ORDER BY d.zip_code
            ) TO '{AGG_DIR}/demographics_by_zip.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] demographics_by_zip.parquet")
    elif has_demo:
        con.execute(f"""
            COPY (SELECT * FROM demo ORDER BY zip_code)
            TO '{AGG_DIR}/demographics_by_zip.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] demographics_by_zip.parquet (all zips, no business filter)")

    # 4. neighborhood_profile — pre-joined: business summary + demographics + civic signals
    _build_neighborhood_profile(con, has_biz, has_demo)

    # 5. city_averages — city-wide averages for comparison
    _build_city_averages(con, has_biz, has_demo)

    # 6. area profiles — area-level aggregation
    _build_area_profiles(con, has_biz, has_demo)

    # 7. zip_to_neighborhood — mapping table
    rows = [(z, n) for z, n in ZIP_TO_NEIGHBORHOOD.items()]
    con.execute("CREATE OR REPLACE TABLE zip_neighborhood(zip_code VARCHAR, neighborhood VARCHAR)")
    con.executemany("INSERT INTO zip_neighborhood VALUES (?, ?)", rows)
    con.execute(f"""
        COPY zip_neighborhood TO '{AGG_DIR}/zip_to_neighborhood.parquet'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print("  [done] zip_to_neighborhood.parquet")

    # 8. trend parquets
    _build_trends(con)

    # 9. business age analysis
    _build_business_age(con)

    # 10. momentum scores
    _build_momentum(con)


def _build_neighborhood_profile(
    con: duckdb.DuckDBPyConnection, has_biz: bool, has_demo: bool
) -> None:
    """Build neighborhood_profile.parquet — the main pre-joined table."""
    # start with zip → neighborhood mapping
    rows = [(z, n) for z, n in ZIP_TO_NEIGHBORHOOD.items()]
    con.execute("CREATE OR REPLACE TABLE zn(zip_code VARCHAR, neighborhood VARCHAR)")
    con.executemany("INSERT INTO zn VALUES (?, ?)", rows)

    # build the profile from whatever data we have
    select_parts = ["zn.zip_code", "zn.neighborhood"]
    from_parts = ["zn"]

    # area mapping
    area_rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE za_lookup(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO za_lookup VALUES (?, ?)", area_rows)
    select_parts.append("COALESCE(za_lookup.area, 'Other') AS area")
    from_parts.append("LEFT JOIN za_lookup ON zn.zip_code = za_lookup.zip_code")

    if has_biz:
        select_parts.extend([
            "bs.active_count",
            "bs.total_count",
            "bs.category_count",
        ])
        from_parts.append(
            f"LEFT JOIN '{AGG_DIR}/business_summary.parquet' bs ON zn.zip_code = bs.zip_code"
        )
    else:
        select_parts.extend(["NULL AS active_count", "NULL AS total_count", "NULL AS category_count"])

    if has_demo:
        select_parts.extend([
            "d.population",
            "d.median_age",
            "d.median_income",
            "d.median_home_value",
            "d.median_rent",
            "d.pct_bachelors_plus",
        ])
        from_parts.append(
            f"LEFT JOIN '{PROCESSED_DIR}/demographics.parquet' d ON zn.zip_code = d.zip_code"
        )
    else:
        select_parts.extend([
            "NULL AS population", "NULL AS median_age", "NULL AS median_income",
            "NULL AS median_home_value", "NULL AS median_rent", "NULL AS pct_bachelors_plus",
        ])

    # business density per 1k residents
    if has_biz and has_demo:
        select_parts.append(
            "ROUND(1000.0 * bs.active_count / NULLIF(d.population, 0), 1) AS businesses_per_1k"
        )
    else:
        select_parts.append("NULL AS businesses_per_1k")

    # civic signals — permits
    permits_file = AGG_DIR / "civic_permits.parquet"
    if permits_file.exists():
        select_parts.extend([
            "cp.permit_count AS new_permits",
            "cp.total_valuation AS permit_valuation",
        ])
        from_parts.append(f"""
            LEFT JOIN (
                SELECT zip_code, SUM(permit_count) AS permit_count, SUM(total_valuation) AS total_valuation
                FROM '{permits_file}'
                WHERE year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1
                GROUP BY zip_code
            ) cp ON zn.zip_code = cp.zip_code
        """)
    else:
        select_parts.extend(["NULL AS new_permits", "NULL AS permit_valuation"])

    # civic signals — solar
    solar_file = AGG_DIR / "civic_solar.parquet"
    if solar_file.exists():
        select_parts.append("cs.solar_count AS solar_installs")
        from_parts.append(f"""
            LEFT JOIN (
                SELECT zip_code, SUM(solar_count) AS solar_count
                FROM '{solar_file}'
                WHERE year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1
                GROUP BY zip_code
            ) cs ON zn.zip_code = cs.zip_code
        """)
    else:
        select_parts.append("NULL AS solar_installs")

    # civic signals — crime
    crime_file = AGG_DIR / "civic_crime.parquet"
    if crime_file.exists():
        select_parts.append("cc.crime_count")
        from_parts.append(f"""
            LEFT JOIN (
                SELECT zip_code, SUM(count) AS crime_count
                FROM '{crime_file}'
                WHERE year >= EXTRACT(YEAR FROM CURRENT_DATE) - 1
                GROUP BY zip_code
            ) cc ON zn.zip_code = cc.zip_code
        """)
    else:
        select_parts.append("NULL AS crime_count")

    # civic signals — 311 (join via comm_plan_name, approximate)
    civic_311_file = AGG_DIR / "civic_311.parquet"
    if civic_311_file.exists():
        # create zip → comm_plan mapping
        cp_rows = [(z, c) for z, c in ZIP_TO_COMM_PLAN.items()]
        con.execute("CREATE OR REPLACE TABLE zcp(zip_code VARCHAR, comm_plan_name VARCHAR)")
        con.executemany("INSERT INTO zcp VALUES (?, ?)", cp_rows)

        select_parts.extend([
            "c311_agg.median_resolution_days AS median_311_days",
            "c311_agg.total_requests AS total_311_requests",
        ])
        # aggregate 311 data by comm_plan_name first to avoid duplicates
        from_parts.append(f"""
            LEFT JOIN zcp ON zn.zip_code = zcp.zip_code
            LEFT JOIN (
                SELECT
                    comm_plan_name,
                    SUM(total_requests) AS total_requests,
                    ROUND(SUM(median_resolution_days * total_requests) / NULLIF(SUM(total_requests), 0), 1) AS median_resolution_days
                FROM '{civic_311_file}'
                GROUP BY comm_plan_name
            ) c311_agg ON zcp.comm_plan_name = c311_agg.comm_plan_name
        """)
    else:
        select_parts.extend(["NULL AS median_311_days", "NULL AS total_311_requests"])

    sql = f"""
        COPY (
            SELECT {', '.join(select_parts)}
            FROM {' '.join(from_parts)}
            ORDER BY zn.zip_code
        ) TO '{AGG_DIR}/neighborhood_profile.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    print("  [done] neighborhood_profile.parquet")


def _build_area_profiles(
    con: duckdb.DuckDBPyConnection, has_biz: bool, has_demo: bool
) -> None:
    """Build area_profile.parquet and area_business_by_category.parquet."""
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    if not np_path.exists():
        print("  [skip] area parquets (no neighborhood_profile)")
        return

    # Load ZIP_TO_AREA into a temp table
    rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE zip_area(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO zip_area VALUES (?, ?)", rows)

    # Handle unmapped profiled zips → "Other"
    con.execute(f"""
        INSERT INTO zip_area
        SELECT np.zip_code, 'Other'
        FROM '{np_path}' np
        WHERE np.zip_code NOT IN (SELECT zip_code FROM zip_area)
    """)

    # 1. area_profile.parquet
    con.execute(f"""
        COPY (
            SELECT
                za.area,
                LIST(DISTINCT za.zip_code ORDER BY za.zip_code) AS zip_codes,
                COUNT(DISTINCT za.zip_code) AS zip_count,
                SUM(np.population) AS population,
                ROUND(SUM(CAST(np.median_age AS DOUBLE) * np.population) / NULLIF(SUM(np.population), 0), 1) AS median_age,
                ROUND(SUM(CAST(np.median_income AS BIGINT) * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_income,
                ROUND(SUM(CAST(np.median_home_value AS BIGINT) * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_home_value,
                ROUND(SUM(CAST(np.median_rent AS BIGINT) * np.population) / NULLIF(SUM(np.population), 0), 0) AS median_rent,
                ROUND(SUM(CAST(np.pct_bachelors_plus AS DOUBLE) * np.population) / NULLIF(SUM(np.population), 0), 1) AS pct_bachelors_plus,
                SUM(np.active_count) AS active_count,
                SUM(np.total_count) AS total_count,
                ROUND(1000.0 * SUM(np.active_count) / NULLIF(SUM(np.population), 0), 1) AS businesses_per_1k,
                SUM(np.new_permits) AS new_permits,
                SUM(np.permit_valuation) AS permit_valuation,
                SUM(np.solar_installs) AS solar_installs,
                SUM(np.crime_count) AS crime_count,
                ROUND(SUM(np.median_311_days * np.total_311_requests) / NULLIF(SUM(np.total_311_requests), 0), 1) AS median_311_days,
                SUM(np.total_311_requests) AS total_311_requests
            FROM zip_area za
            JOIN '{np_path}' np ON za.zip_code = np.zip_code
            GROUP BY za.area
            ORDER BY SUM(np.population) DESC
        ) TO '{AGG_DIR}/area_profile.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print("  [done] area_profile.parquet")

    # 2. area_business_by_category.parquet
    biz_by_zip_path = AGG_DIR / "business_by_zip.parquet"
    if biz_by_zip_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    za.area,
                    bz.category,
                    SUM(bz.active_count) AS active_count,
                    SUM(bz.total_count) AS total_count,
                    ROUND(1000.0 * SUM(bz.active_count) / NULLIF(ap.population, 0), 2) AS per_1k
                FROM zip_area za
                JOIN '{biz_by_zip_path}' bz ON za.zip_code = bz.zip_code
                JOIN '{AGG_DIR}/area_profile.parquet' ap ON za.area = ap.area
                GROUP BY za.area, bz.category, ap.population
                ORDER BY za.area, SUM(bz.active_count) DESC
            ) TO '{AGG_DIR}/area_business_by_category.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print("  [done] area_business_by_category.parquet")


def _build_trends(con: duckdb.DuckDBPyConnection) -> None:
    """Build trend parquets for temporal analysis."""
    # 1. trend_business_formation — new businesses per zip per year
    biz_path = PROCESSED_DIR / "businesses.parquet"
    if biz_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    zip_code,
                    EXTRACT(YEAR FROM start_date)::INTEGER AS year,
                    COUNT(*) AS new_businesses
                FROM '{biz_path}'
                WHERE start_date IS NOT NULL
                  AND EXTRACT(YEAR FROM start_date) >= 2015
                GROUP BY zip_code, EXTRACT(YEAR FROM start_date)
                ORDER BY zip_code, year
            ) TO '{AGG_DIR}/trend_business_formation.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        count = con.execute(f"""
            SELECT COUNT(*) FROM '{AGG_DIR}/trend_business_formation.parquet'
        """).fetchone()[0]
        print(f"  [done] trend_business_formation.parquet -> {count:,} rows")
    else:
        print("  [skip] trend_business_formation.parquet (no businesses)")

    # 2. trend_311_monthly — extract year/month from DATE column
    monthly_path = AGG_DIR / "civic_311_monthly.parquet"
    if monthly_path.exists():
        con.execute(f"""
            COPY (
                SELECT
                    EXTRACT(YEAR FROM request_month_start)::INTEGER AS year,
                    EXTRACT(MONTH FROM request_month_start)::INTEGER AS month,
                    total_requests,
                    closed_requests,
                    avg_resolution_days,
                    median_resolution_days
                FROM '{monthly_path}'
                ORDER BY year, month
            ) TO '{AGG_DIR}/trend_311_monthly.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        count = con.execute(f"""
            SELECT COUNT(*) FROM '{AGG_DIR}/trend_311_monthly.parquet'
        """).fetchone()[0]
        print(f"  [done] trend_311_monthly.parquet -> {count:,} rows")
    else:
        print("  [skip] trend_311_monthly.parquet (no 311 monthly data)")


def _build_business_age(con: duckdb.DuckDBPyConnection) -> None:
    """Build business_age_stats.parquet and business_age_by_area.parquet."""
    biz_path = PROCESSED_DIR / "businesses.parquet"
    if not biz_path.exists():
        print("  [skip] business age parquets (no businesses)")
        return

    # 1. business_age_stats.parquet — per zip_code + category
    con.execute(f"""
        COPY (
            SELECT
                zip_code,
                category,
                COUNT(*) AS business_count,
                ROUND(MEDIAN(DATEDIFF('day', start_date, CURRENT_DATE) / 365.25), 1) AS median_age_years,
                ROUND(AVG(DATEDIFF('day', start_date, CURRENT_DATE) / 365.25), 1) AS avg_age_years,
                ROUND(100.0 * SUM(CASE WHEN DATEDIFF('day', start_date, CURRENT_DATE) < 730 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_under_2yr,
                ROUND(100.0 * SUM(CASE WHEN DATEDIFF('day', start_date, CURRENT_DATE) > 3652 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_over_10yr
            FROM '{biz_path}'
            WHERE start_date IS NOT NULL AND status = 'active'
            GROUP BY zip_code, category
            HAVING COUNT(*) >= 3
            ORDER BY zip_code, business_count DESC
        ) TO '{AGG_DIR}/business_age_stats.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = con.execute(f"""
        SELECT COUNT(*) FROM '{AGG_DIR}/business_age_stats.parquet'
    """).fetchone()[0]
    print(f"  [done] business_age_stats.parquet -> {count:,} rows")

    # 2. business_age_by_area.parquet — per area + category
    # Build ZIP_TO_AREA as VALUES-based temp table (same pattern as _build_area_profiles)
    rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE za_age(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO za_age VALUES (?, ?)", rows)

    # Include unmapped profiled zips as "Other"
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    if np_path.exists():
        con.execute(f"""
            INSERT INTO za_age
            SELECT np.zip_code, 'Other'
            FROM '{np_path}' np
            WHERE np.zip_code NOT IN (SELECT zip_code FROM za_age)
        """)

    con.execute(f"""
        COPY (
            SELECT
                COALESCE(za.area, 'Other') AS area,
                b.category,
                COUNT(*) AS business_count,
                ROUND(MEDIAN(DATEDIFF('day', b.start_date, CURRENT_DATE) / 365.25), 1) AS median_age_years,
                ROUND(AVG(DATEDIFF('day', b.start_date, CURRENT_DATE) / 365.25), 1) AS avg_age_years,
                ROUND(100.0 * SUM(CASE WHEN DATEDIFF('day', b.start_date, CURRENT_DATE) < 730 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_under_2yr,
                ROUND(100.0 * SUM(CASE WHEN DATEDIFF('day', b.start_date, CURRENT_DATE) > 3652 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_over_10yr
            FROM '{biz_path}' b
            LEFT JOIN za_age za ON b.zip_code = za.zip_code
            WHERE b.start_date IS NOT NULL AND b.status = 'active'
              AND b.zip_code IN (SELECT zip_code FROM za_age)
            GROUP BY COALESCE(za.area, 'Other'), b.category
            HAVING COUNT(*) >= 3
            ORDER BY area, business_count DESC
        ) TO '{AGG_DIR}/business_age_by_area.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = con.execute(f"""
        SELECT COUNT(*) FROM '{AGG_DIR}/business_age_by_area.parquet'
    """).fetchone()[0]
    print(f"  [done] business_age_by_area.parquet -> {count:,} rows")


def _build_momentum(con: duckdb.DuckDBPyConnection) -> None:
    """Build momentum_scores.parquet and momentum_by_area.parquet."""
    # We need at least trend_business_formation to proceed
    biz_trend_path = AGG_DIR / "trend_business_formation.parquet"
    if not biz_trend_path.exists():
        print("  [skip] momentum parquets (no trend data)")
        return

    permits_path = AGG_DIR / "civic_permits.parquet"
    crime_path = AGG_DIR / "civic_crime.parquet"
    solar_path = AGG_DIR / "civic_solar.parquet"

    # Build CTEs for each YoY metric
    # Business formation YoY
    biz_cte = f"""
        biz_yoy AS (
            SELECT
                zip_code,
                SUM(CASE WHEN year = 2025 THEN new_businesses ELSE 0 END) AS val_2025,
                SUM(CASE WHEN year = 2024 THEN new_businesses ELSE 0 END) AS val_2024,
                CASE
                    WHEN SUM(CASE WHEN year = 2024 THEN new_businesses ELSE 0 END) > 0
                    THEN ROUND(100.0 * (
                        SUM(CASE WHEN year = 2025 THEN new_businesses ELSE 0 END) -
                        SUM(CASE WHEN year = 2024 THEN new_businesses ELSE 0 END)
                    ) / SUM(CASE WHEN year = 2024 THEN new_businesses ELSE 0 END), 1)
                    ELSE NULL
                END AS biz_formation_yoy
            FROM '{biz_trend_path}'
            WHERE year IN (2024, 2025)
            GROUP BY zip_code
            HAVING SUM(CASE WHEN year = 2024 THEN new_businesses ELSE 0 END) > 0
               OR SUM(CASE WHEN year = 2025 THEN new_businesses ELSE 0 END) > 0
        )
    """

    # Permits YoY
    if permits_path.exists():
        permit_cte = f"""
        permit_yoy AS (
            SELECT
                zip_code,
                CASE
                    WHEN SUM(CASE WHEN year = 2024 THEN permit_count ELSE 0 END) > 0
                    THEN ROUND(100.0 * (
                        SUM(CASE WHEN year = 2025 THEN permit_count ELSE 0 END) -
                        SUM(CASE WHEN year = 2024 THEN permit_count ELSE 0 END)
                    ) / SUM(CASE WHEN year = 2024 THEN permit_count ELSE 0 END), 1)
                    ELSE NULL
                END AS permit_yoy
            FROM '{permits_path}'
            WHERE year IN (2024, 2025)
            GROUP BY zip_code
            HAVING SUM(CASE WHEN year = 2024 THEN permit_count ELSE 0 END) > 0
               OR SUM(CASE WHEN year = 2025 THEN permit_count ELSE 0 END) > 0
        )"""
    else:
        permit_cte = """
        permit_yoy AS (
            SELECT NULL::VARCHAR AS zip_code, NULL::DOUBLE AS permit_yoy
            WHERE FALSE
        )"""

    # Crime YoY (inverted: decrease = positive)
    if crime_path.exists():
        crime_cte = f"""
        crime_yoy AS (
            SELECT
                zip_code,
                CASE
                    WHEN SUM(CASE WHEN year = 2024 THEN count ELSE 0 END) > 0
                    THEN ROUND(-100.0 * (
                        SUM(CASE WHEN year = 2025 THEN count ELSE 0 END) -
                        SUM(CASE WHEN year = 2024 THEN count ELSE 0 END)
                    ) / SUM(CASE WHEN year = 2024 THEN count ELSE 0 END), 1)
                    ELSE NULL
                END AS crime_yoy
            FROM '{crime_path}'
            WHERE year IN (2024, 2025)
            GROUP BY zip_code
            HAVING SUM(CASE WHEN year = 2024 THEN count ELSE 0 END) > 0
               OR SUM(CASE WHEN year = 2025 THEN count ELSE 0 END) > 0
        )"""
    else:
        crime_cte = """
        crime_yoy AS (
            SELECT NULL::VARCHAR AS zip_code, NULL::DOUBLE AS crime_yoy
            WHERE FALSE
        )"""

    # Solar YoY
    if solar_path.exists():
        solar_cte = f"""
        solar_yoy AS (
            SELECT
                zip_code,
                CASE
                    WHEN SUM(CASE WHEN year = 2024 THEN solar_count ELSE 0 END) > 0
                    THEN ROUND(100.0 * (
                        SUM(CASE WHEN year = 2025 THEN solar_count ELSE 0 END) -
                        SUM(CASE WHEN year = 2024 THEN solar_count ELSE 0 END)
                    ) / SUM(CASE WHEN year = 2024 THEN solar_count ELSE 0 END), 1)
                    ELSE NULL
                END AS solar_yoy
            FROM '{solar_path}'
            WHERE year IN (2024, 2025)
            GROUP BY zip_code
            HAVING SUM(CASE WHEN year = 2024 THEN solar_count ELSE 0 END) > 0
               OR SUM(CASE WHEN year = 2025 THEN solar_count ELSE 0 END) > 0
        )"""
    else:
        solar_cte = """
        solar_yoy AS (
            SELECT NULL::VARCHAR AS zip_code, NULL::DOUBLE AS solar_yoy
            WHERE FALSE
        )"""

    # Combine: PERCENT_RANK each YoY, multiply by 25, COALESCE nulls to 12.5
    # Filter to SD zips only (those in demographics_by_zip)
    demo_path = AGG_DIR / "demographics_by_zip.parquet"
    con.execute(f"""
        COPY (
            WITH
                {biz_cte},
                {permit_cte},
                {crime_cte},
                {solar_cte},
            sd_zips AS (
                SELECT DISTINCT zip_code FROM '{demo_path}'
            ),
            all_zips AS (
                SELECT zip_code FROM sd_zips
                WHERE zip_code IN (SELECT zip_code FROM biz_yoy)
                   OR zip_code IN (SELECT zip_code FROM permit_yoy)
                   OR zip_code IN (SELECT zip_code FROM crime_yoy)
                   OR zip_code IN (SELECT zip_code FROM solar_yoy)
            ),
            combined AS (
                SELECT
                    az.zip_code,
                    b.biz_formation_yoy,
                    p.permit_yoy,
                    c.crime_yoy,
                    s.solar_yoy
                FROM all_zips az
                LEFT JOIN biz_yoy b ON az.zip_code = b.zip_code
                LEFT JOIN permit_yoy p ON az.zip_code = p.zip_code
                LEFT JOIN crime_yoy c ON az.zip_code = c.zip_code
                LEFT JOIN solar_yoy s ON az.zip_code = s.zip_code
            ),
            ranked AS (
                SELECT
                    zip_code,
                    biz_formation_yoy,
                    permit_yoy,
                    crime_yoy,
                    solar_yoy,
                    COALESCE(PERCENT_RANK() OVER (ORDER BY biz_formation_yoy) * 25, 12.5) AS biz_score,
                    COALESCE(PERCENT_RANK() OVER (ORDER BY permit_yoy) * 25, 12.5) AS permit_score,
                    COALESCE(PERCENT_RANK() OVER (ORDER BY crime_yoy) * 25, 12.5) AS crime_score,
                    COALESCE(PERCENT_RANK() OVER (ORDER BY solar_yoy) * 25, 12.5) AS solar_score
                FROM combined
            )
            SELECT
                zip_code,
                ROUND(biz_score + permit_score + crime_score + solar_score, 1) AS momentum_score,
                biz_formation_yoy,
                permit_yoy,
                crime_yoy,
                solar_yoy
            FROM ranked
            ORDER BY momentum_score DESC
        ) TO '{AGG_DIR}/momentum_scores.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = con.execute(f"""
        SELECT COUNT(*) FROM '{AGG_DIR}/momentum_scores.parquet'
    """).fetchone()[0]
    print(f"  [done] momentum_scores.parquet -> {count:,} rows")

    # 2. momentum_by_area.parquet — population-weighted average of zip scores
    demo_path = AGG_DIR / "demographics_by_zip.parquet"
    if not demo_path.exists():
        print("  [skip] momentum_by_area.parquet (no demographics)")
        return

    rows = [(z, a) for z, a in ZIP_TO_AREA.items()]
    con.execute("CREATE OR REPLACE TABLE za_momentum(zip_code VARCHAR, area VARCHAR)")
    con.executemany("INSERT INTO za_momentum VALUES (?, ?)", rows)

    # Include unmapped profiled zips as "Other"
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    if np_path.exists():
        con.execute(f"""
            INSERT INTO za_momentum
            SELECT np.zip_code, 'Other'
            FROM '{np_path}' np
            WHERE np.zip_code NOT IN (SELECT zip_code FROM za_momentum)
        """)

    con.execute(f"""
        COPY (
            SELECT
                za.area,
                ROUND(SUM(m.momentum_score * CAST(d.population AS DOUBLE)) / NULLIF(SUM(CAST(d.population AS DOUBLE)), 0), 1) AS momentum_score,
                ROUND(SUM(m.biz_formation_yoy * CAST(d.population AS DOUBLE)) / NULLIF(SUM(CASE WHEN m.biz_formation_yoy IS NOT NULL THEN CAST(d.population AS DOUBLE) ELSE 0 END), 0), 1) AS biz_formation_yoy,
                ROUND(SUM(m.permit_yoy * CAST(d.population AS DOUBLE)) / NULLIF(SUM(CASE WHEN m.permit_yoy IS NOT NULL THEN CAST(d.population AS DOUBLE) ELSE 0 END), 0), 1) AS permit_yoy,
                ROUND(SUM(m.crime_yoy * CAST(d.population AS DOUBLE)) / NULLIF(SUM(CASE WHEN m.crime_yoy IS NOT NULL THEN CAST(d.population AS DOUBLE) ELSE 0 END), 0), 1) AS crime_yoy,
                ROUND(SUM(m.solar_yoy * CAST(d.population AS DOUBLE)) / NULLIF(SUM(CASE WHEN m.solar_yoy IS NOT NULL THEN CAST(d.population AS DOUBLE) ELSE 0 END), 0), 1) AS solar_yoy
            FROM '{AGG_DIR}/momentum_scores.parquet' m
            JOIN '{demo_path}' d ON m.zip_code = d.zip_code
            LEFT JOIN za_momentum za ON m.zip_code = za.zip_code
            WHERE za.area IS NOT NULL
            GROUP BY za.area
            ORDER BY momentum_score DESC
        ) TO '{AGG_DIR}/momentum_by_area.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    count = con.execute(f"""
        SELECT COUNT(*) FROM '{AGG_DIR}/momentum_by_area.parquet'
    """).fetchone()[0]
    print(f"  [done] momentum_by_area.parquet -> {count:,} rows")


def _build_city_averages(
    con: duckdb.DuckDBPyConnection, has_biz: bool, has_demo: bool
) -> None:
    """Build city_averages.parquet — city-wide averages for comparison."""
    avg_parts = []

    if has_demo:
        avg_parts.append(f"""
            SELECT
                'demographics' AS source,
                AVG(population) AS avg_population,
                AVG(median_age) AS avg_median_age,
                AVG(median_income) AS avg_median_income,
                AVG(median_home_value) AS avg_median_home_value,
                AVG(median_rent) AS avg_median_rent,
                AVG(pct_bachelors_plus) AS avg_pct_bachelors_plus
            FROM '{PROCESSED_DIR}/demographics.parquet'
        """)

    if has_biz:
        # Filter to SD zips only — business_summary includes out-of-area mailing addresses
        biz_where = f"WHERE zip_code IN (SELECT zip_code FROM '{PROCESSED_DIR}/demographics.parquet')" if has_demo else ""
        avg_parts.append(f"""
            SELECT
                'business' AS source,
                AVG(active_count) AS avg_active_businesses,
                AVG(total_count) AS avg_total_businesses,
                AVG(category_count) AS avg_category_count,
                NULL AS placeholder1,
                NULL AS placeholder2,
                NULL AS placeholder3
            FROM '{AGG_DIR}/business_summary.parquet'
            {biz_where}
        """)

    if not avg_parts:
        print("  [skip] city_averages.parquet (no data)")
        return

    # store each as a separate parquet for simplicity — single row per source
    # civic signal averages from neighborhood_profile (already has all signals joined)
    np_path = AGG_DIR / "neighborhood_profile.parquet"
    has_np = np_path.exists()

    civic_subqueries = ""
    if has_np:
        civic_subqueries = f"""
            (SELECT AVG(crime_count) FROM '{np_path}' WHERE crime_count IS NOT NULL) AS avg_crime_count,
            (SELECT AVG(median_311_days) FROM '{np_path}' WHERE median_311_days IS NOT NULL) AS avg_median_311_days,
            (SELECT AVG(new_permits) FROM '{np_path}' WHERE new_permits IS NOT NULL) AS avg_new_permits,
            (SELECT AVG(solar_installs) FROM '{np_path}' WHERE solar_installs IS NOT NULL) AS avg_solar_installs
        """
    else:
        civic_subqueries = """
            NULL AS avg_crime_count,
            NULL AS avg_median_311_days,
            NULL AS avg_new_permits,
            NULL AS avg_solar_installs
        """

    if has_demo:
        con.execute(f"""
            COPY (
                SELECT
                    AVG(population) AS avg_population,
                    AVG(median_age) AS avg_median_age,
                    AVG(median_income) AS avg_median_income,
                    AVG(median_home_value) AS avg_median_home_value,
                    AVG(median_rent) AS avg_median_rent,
                    AVG(pct_bachelors_plus) AS avg_pct_bachelors_plus,
                    {f"(SELECT AVG(active_count) FROM '{AGG_DIR}/business_summary.parquet' WHERE zip_code IN (SELECT zip_code FROM '{PROCESSED_DIR}/demographics.parquet')) AS avg_active_businesses," if has_biz else "NULL AS avg_active_businesses,"}
                    {f"(SELECT AVG(total_count) FROM '{AGG_DIR}/business_summary.parquet' WHERE zip_code IN (SELECT zip_code FROM '{PROCESSED_DIR}/demographics.parquet')) AS avg_total_businesses," if has_biz else "NULL AS avg_total_businesses,"}
                    {f"(SELECT AVG(category_count) FROM '{AGG_DIR}/business_summary.parquet' WHERE zip_code IN (SELECT zip_code FROM '{PROCESSED_DIR}/demographics.parquet')) AS avg_category_count," if has_biz else "NULL AS avg_category_count,"}
                    {f"(SELECT ROUND(AVG(1000.0 * active_count / NULLIF(population, 0)), 1) FROM '{np_path}' WHERE population > 0 AND active_count IS NOT NULL) AS avg_businesses_per_1k," if has_np and has_biz else "NULL AS avg_businesses_per_1k,"}
                    {civic_subqueries}
                FROM '{PROCESSED_DIR}/demographics.parquet'
            ) TO '{AGG_DIR}/city_averages.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    elif has_biz:
        # No demographics to filter by — average includes all registered zips
        con.execute(f"""
            COPY (
                SELECT
                    NULL AS avg_population,
                    NULL AS avg_median_age,
                    NULL AS avg_median_income,
                    NULL AS avg_median_home_value,
                    NULL AS avg_median_rent,
                    NULL AS avg_pct_bachelors_plus,
                    AVG(active_count) AS avg_active_businesses,
                    AVG(total_count) AS avg_total_businesses,
                    AVG(category_count) AS avg_category_count,
                    NULL AS avg_businesses_per_1k,
                    {civic_subqueries}
                FROM '{AGG_DIR}/business_summary.parquet'
            ) TO '{AGG_DIR}/city_averages.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    print("  [done] city_averages.parquet")
