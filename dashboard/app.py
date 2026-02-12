"""Streamlit dashboard for SD Business Intel — neighborhood explorer."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ── Parquet paths ──
_AGG = "data/aggregated"
_PROCESSED = "data/processed"

_root = Path(__file__).resolve().parent.parent
if (_root / _AGG).exists():
    _AGG = str(_root / _AGG)
    _PROCESSED = str(_root / _PROCESSED)

st.set_page_config(
    page_title="sd business intel",
    page_icon="\U0001f4bc",
    layout="wide",
)

CHART_COLOR = "#83c9ff"
COMPARE_COLOR = "#ff6b6b"
HIGHLIGHT_COLOR = "#ffa500"


def _valid(val) -> bool:
    """Check if a value is non-null and non-NaN (safe for pandas NA types)."""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return True


def query(sql: str, params: list | None = None):
    """Run SQL against parquet files and return a pandas DataFrame."""
    con = duckdb.connect()
    return con.execute(sql, params or []).fetchdf()


# ── Sidebar ──
st.sidebar.title("sd business intel")
st.sidebar.caption("neighborhood explorer")


@st.cache_data(ttl=3600)
def _zip_options():
    try:
        df = query(f"""
            SELECT np.zip_code, np.neighborhood
            FROM '{_AGG}/neighborhood_profile.parquet' np
            ORDER BY np.zip_code
        """)
        return list(zip(df["zip_code"], df["neighborhood"]))
    except Exception:
        return []


zip_options = _zip_options()

if not zip_options:
    st.error("no data available. run the pipeline first: `uv run python -m pipeline.build`")
    st.stop()

zip_labels = [f"{z} — {n}" if n else z for z, n in zip_options]
selected_idx = st.sidebar.selectbox(
    "zip code 1",
    range(len(zip_labels)),
    format_func=lambda i: zip_labels[i],
    index=next((i for i, (z, _) in enumerate(zip_options) if z == "92101"), 0),
)
selected_zip = zip_options[selected_idx][0]
selected_neighborhood = zip_options[selected_idx][1]

compare_labels = ["-- none --"] + zip_labels
compare_idx = st.sidebar.selectbox(
    "zip code 2 (compare)",
    range(len(compare_labels)),
    format_func=lambda i: compare_labels[i],
    index=0,
)
compare_zip = zip_options[compare_idx - 1][0] if compare_idx > 0 else None
compare_neighborhood = zip_options[compare_idx - 1][1] if compare_idx > 0 else None

# ── Category deep-dive session state ──
if "explorer_cat" not in st.session_state:
    st.session_state.explorer_cat = None
if st.session_state.get("_explorer_zip") != selected_zip:
    st.session_state._explorer_zip = selected_zip
    st.session_state.explorer_cat = None

# ── Cached data loaders ──


@st.cache_data(ttl=3600)
def _load_profile(zip_code: str):
    return query(
        f"SELECT * FROM '{_AGG}/neighborhood_profile.parquet' WHERE zip_code = $1",
        [zip_code],
    )


@st.cache_data(ttl=3600)
def _load_city_avg():
    try:
        return query(f"SELECT * FROM '{_AGG}/city_averages.parquet'")
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_biz_by_zip(zip_code: str):
    try:
        return query(
            f"""
            SELECT category, active_count, total_count
            FROM '{_AGG}/business_by_zip.parquet'
            WHERE zip_code = $1
            ORDER BY active_count DESC
            LIMIT 15
            """,
            [zip_code],
        )
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_businesses(zip_code: str, category: str | None = None):
    try:
        where = "WHERE zip_code = $1"
        params = [zip_code]
        if category:
            where += " AND category = $2"
            params.append(category)
        return query(
            f"""
            SELECT business_name, address, category, naics_code, start_date, status
            FROM '{_PROCESSED}/businesses.parquet'
            {where}
            ORDER BY business_name
            LIMIT 5000
            """,
            params,
        )
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _category_options():
    try:
        df = query(f"""
            SELECT DISTINCT category
            FROM '{_AGG}/business_by_zip.parquet'
            ORDER BY category
        """)
        return df["category"].tolist()
    except Exception:
        return []


# Metric → (SQL direction for "best" rank, sense for badge phrasing).
# "high" = higher is good, "low" = lower is good, "neutral" = no judgement.
_RANK_METRICS = {
    "population": ("DESC", "high"),
    "median_income": ("DESC", "high"),
    "median_age": ("DESC", "neutral"),
    "median_home_value": ("DESC", "neutral"),
    "median_rent": ("ASC", "low"),
    "pct_bachelors_plus": ("DESC", "high"),
    "active_count": ("DESC", "high"),
    "businesses_per_1k": ("DESC", "high"),
    "new_permits": ("DESC", "high"),
    "crime_count": ("ASC", "low"),
    "median_311_days": ("ASC", "low"),
    "solar_installs": ("DESC", "high"),
}


@st.cache_data(ttl=3600)
def _load_percentiles(zip_code: str) -> dict:
    """Compute rank of each metric across all 82 zips for this zip code."""
    rank_cols = []
    for m, (d, _) in _RANK_METRICS.items():
        rank_cols.append(
            f"RANK() OVER (ORDER BY {m} {d} NULLS LAST) AS {m}_rank"
        )
        rank_cols.append(f"COUNT({m}) OVER () AS {m}_of")
    try:
        df = query(f"""
            WITH ranked AS (
                SELECT zip_code, {', '.join(rank_cols)}
                FROM '{_AGG}/neighborhood_profile.parquet'
            )
            SELECT * FROM ranked WHERE zip_code = $1
        """, [zip_code])
    except Exception:
        return {}
    if df.empty:
        return {}
    rr = df.iloc[0]
    result = {}
    for m, (_, sense) in _RANK_METRICS.items():
        r = rr.get(f"{m}_rank")
        o = rr.get(f"{m}_of")
        if r is not None and o is not None:
            r, o = int(r), int(o)
            if r <= o:  # skip NULLs (rank exceeds count)
                result[m] = {"rank": r, "of": o, "sense": sense}
    return result


_VALID_RANKING_SORT = frozenset({
    "population", "median_income", "median_age", "median_rent",
    "median_home_value", "pct_bachelors_plus", "active_count",
    "businesses_per_1k", "category_count", "new_permits",
    "crime_count", "median_311_days", "solar_installs",
})


@st.cache_data(ttl=3600)
def _load_rankings(sort_by, sort_desc, category, limit):
    direction = "DESC" if sort_desc else "ASC"
    profile = f"{_AGG}/neighborhood_profile.parquet"

    if sort_by == "category_per_1k" and not category:
        return pd.DataFrame()
    if sort_by != "category_per_1k" and sort_by not in _VALID_RANKING_SORT:
        return pd.DataFrame()

    if category:
        biz = f"{_AGG}/business_by_zip.parquet"
        select = ["np.zip_code", "np.neighborhood"]
        if sort_by != "category_per_1k":
            select.append(f"np.{sort_by}")
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

        return query(f"""
            SELECT {', '.join(select)}
            FROM '{profile}' np
            LEFT JOIN '{biz}' bz
                ON np.zip_code = bz.zip_code AND bz.category = $1
            {where}
            ORDER BY {order} {direction} NULLS LAST
            LIMIT {min(limit, 82)}
        """, [category])

    context = [c for c in ("population", "median_income", "active_count") if c != sort_by]
    cols = ", ".join(["zip_code", "neighborhood", sort_by] + context)

    return query(f"""
        SELECT {cols}
        FROM '{profile}'
        WHERE {sort_by} IS NOT NULL
        ORDER BY {sort_by} {direction}
        LIMIT {min(limit, 82)}
    """)


# ── Load data for selected zip ──
profile = _load_profile(selected_zip)
city_avg = _load_city_avg()
biz_by_zip = _load_biz_by_zip(selected_zip)
percentiles = _load_percentiles(selected_zip)

if profile.empty:
    st.warning(f"no data for zip code {selected_zip}")
    st.stop()

row = profile.iloc[0]
has_avg = city_avg is not None and not city_avg.empty
avg_row = city_avg.iloc[0] if has_avg else None


# ── Helpers ──


def _build_narrative(row, avg_row) -> str:
    """Generate a one-sentence narrative comparing this zip to city averages."""
    if avg_row is None:
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
        val = row.get(row_key)
        avg_val = avg_row.get(avg_key)
        if not _valid(val) or not _valid(avg_val) or float(avg_val) == 0:
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


def _build_comparison_narrative(row_a, row_b, name_a, name_b) -> str:
    """Generate a head-to-head narrative comparing two zip codes."""
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
        val_a = row_a.get(key)
        val_b = row_b.get(key)
        if not _valid(val_a) or not _valid(val_b):
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


def _fmt_metric(label, val, avg_key=None, prefix="", decimals=0):
    """Render a st.metric with optional delta vs city avg."""
    if not _valid(val):
        return label, "—", None
    val_f = float(val)
    fmt = f"{prefix}{val_f:,.{decimals}f}"
    delta = None
    if avg_key and avg_row is not None:
        avg_val = avg_row.get(avg_key)
        if _valid(avg_val):
            diff = val_f - float(avg_val)
            sign = "+" if diff >= 0 else "-"
            delta = f"{sign}{prefix}{abs(diff):,.{decimals}f} vs avg"
    return label, fmt, delta


def _fmt_compare(val, other, prefix="", decimals=0):
    """Format a value with delta against another value (for head-to-head comparison)."""
    if not _valid(val):
        return "—", None
    vf = float(val)
    formatted = f"{prefix}{vf:,.{decimals}f}"
    delta = None
    if _valid(other):
        diff = vf - float(other)
        sign = "+" if diff >= 0 else "-"
        delta = f"{sign}{prefix}{abs(diff):,.{decimals}f}"
    return formatted, delta


def _ordinal(n: int) -> str:
    """Return ordinal string: 1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, 21st, ..."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{({1: 'st', 2: 'nd', 3: 'rd'}).get(n % 10, 'th')}"


def _show_rank(col, metric_key, percentiles):
    """Show a subtle percentile badge caption under a metric card."""
    if not percentiles or metric_key not in percentiles:
        return
    p = percentiles[metric_key]
    rank, of, sense = p["rank"], p["of"], p["sense"]
    if sense == "low":
        text = f"{_ordinal(rank)} lowest of {of}"
    elif sense == "high":
        pct = max(1, round(100 * rank / of))
        if pct <= 50:
            text = f"top {pct}% · rank {rank} of {of}"
        else:
            text = f"rank {rank} of {of}"
    else:
        text = f"rank {rank} of {of}"
    col.caption(text)


# approximate zip code centroids for SD area
ZIP_COORDS: dict[str, tuple[float, float]] = {
    "92101": (32.7194, -117.1628),
    "92102": (32.7085, -117.1245),
    "92103": (32.7488, -117.1713),
    "92104": (32.7499, -117.1305),
    "92105": (32.7344, -117.1016),
    "92106": (32.7184, -117.2349),
    "92107": (32.7453, -117.2500),
    "92108": (32.7720, -117.1560),
    "92109": (32.7935, -117.2485),
    "92110": (32.7664, -117.2023),
    "92111": (32.7940, -117.1693),
    "92113": (32.6912, -117.1213),
    "92114": (32.6865, -117.0758),
    "92115": (32.7563, -117.0708),
    "92116": (32.7651, -117.1252),
    "92117": (32.8271, -117.2020),
    "92118": (32.6766, -117.1692),
    "92119": (32.7836, -117.0297),
    "92120": (32.7942, -117.0714),
    "92121": (32.8987, -117.2248),
    "92122": (32.8580, -117.2088),
    "92123": (32.8160, -117.1475),
    "92124": (32.8232, -117.0887),
    "92126": (32.9136, -117.1550),
    "92127": (33.0239, -117.0869),
    "92128": (32.9996, -117.0619),
    "92129": (32.9585, -117.1235),
    "92130": (32.9462, -117.2198),
    "92131": (32.9174, -117.1009),
    "92139": (32.6647, -117.0654),
    "92154": (32.5773, -117.0506),
    "92173": (32.5483, -117.0443),
}


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════
tab_explorer, tab_compare, tab_rankings = st.tabs(["explorer", "compare", "rankings"])

# ── EXPLORER TAB ──
with tab_explorer:
    st.title(f"{selected_neighborhood or selected_zip}")
    st.caption(f"zip code {selected_zip}")

    narrative = _build_narrative(row, avg_row)
    if narrative:
        st.markdown(f"*{narrative}*")

    # Row 1: demographics
    pop = row.get("population")
    income = row.get("median_income")
    active_biz = row.get("active_count")
    age = row.get("median_age")
    rent = row.get("median_rent")
    home_val = row.get("median_home_value")
    edu = row.get("pct_bachelors_plus")

    r1 = st.columns(3)
    l, v, d = _fmt_metric("population", pop, "avg_population")
    r1[0].metric(l, v, d, delta_color="normal")
    _show_rank(r1[0], "population", percentiles)
    l, v, d = _fmt_metric("median income", income, "avg_median_income", "$")
    r1[1].metric(l, v, d, delta_color="normal")
    _show_rank(r1[1], "median_income", percentiles)
    l, v, d = _fmt_metric("active businesses", active_biz, "avg_active_businesses")
    r1[2].metric(l, v, d, delta_color="normal")
    _show_rank(r1[2], "active_count", percentiles)

    r2 = st.columns(3)
    l, v, d = _fmt_metric("median age", age, "avg_median_age")
    r2[0].metric(l, v, d, delta_color="off")
    _show_rank(r2[0], "median_age", percentiles)
    l, v, d = _fmt_metric("median rent", rent, "avg_median_rent", "$")
    r2[1].metric(l, v, d, delta_color="inverse")
    _show_rank(r2[1], "median_rent", percentiles)
    l, v, d = _fmt_metric("median home value", home_val, "avg_median_home_value", "$")
    r2[2].metric(l, v, d, delta_color="off")
    _show_rank(r2[2], "median_home_value", percentiles)

    r3 = st.columns(3)
    l, v, d = _fmt_metric("% bachelor's+", edu, "avg_pct_bachelors_plus", "", 1)
    r3[0].metric(l, v, d, delta_color="normal")
    _show_rank(r3[0], "pct_bachelors_plus", percentiles)

    permits = row.get("new_permits")
    crime = row.get("crime_count")
    solar = row.get("solar_installs")
    median_311 = row.get("median_311_days")

    l, v, d = _fmt_metric("crime count", crime)
    r3[1].metric(l, v, d, delta_color="inverse")
    _show_rank(r3[1], "crime_count", percentiles)
    l, v, d = _fmt_metric("311 median days", median_311, decimals=1)
    r3[2].metric(l, v, d, delta_color="inverse")
    _show_rank(r3[2], "median_311_days", percentiles)

    r4 = st.columns(3)
    l, v, d = _fmt_metric("new permits", permits)
    r4[0].metric(l, v, d, delta_color="normal")
    _show_rank(r4[0], "new_permits", percentiles)
    l, v, d = _fmt_metric("solar installs", solar)
    r4[1].metric(l, v, d, delta_color="normal")
    _show_rank(r4[1], "solar_installs", percentiles)
    biz_per_1k = row.get("businesses_per_1k")
    l, v, d = _fmt_metric("businesses per 1k residents", biz_per_1k, "avg_businesses_per_1k", "", 1)
    r4[2].metric(l, v, d, delta_color="normal")
    _show_rank(r4[2], "businesses_per_1k", percentiles)

    st.divider()

    # Business categories chart
    st.subheader("top business categories")

    if biz_by_zip is not None and not biz_by_zip.empty:
        selected_cat = st.session_state.explorer_cat

        population = row.get("population")
        if _valid(population) and float(population) > 0:
            biz_by_zip = biz_by_zip.copy()
            biz_by_zip["per_1k"] = (1000.0 * biz_by_zip["active_count"] / float(population)).round(1)
            bar_text = biz_by_zip.apply(lambda r: f"{int(r['active_count'])}  ({r['per_1k']}/1k)", axis=1)
        else:
            bar_text = biz_by_zip["active_count"]

        colors = [HIGHLIGHT_COLOR if cat == selected_cat else CHART_COLOR
                  for cat in biz_by_zip["category"]]

        fig = go.Figure(go.Bar(
            x=biz_by_zip["active_count"],
            y=biz_by_zip["category"],
            orientation="h",
            marker_color=colors,
            text=bar_text,
            textposition="outside",
        ))
        fig.update_layout(
            height=max(300, len(biz_by_zip) * 28),
            margin=dict(l=0, r=40, t=0, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="active businesses",
        )

        event = st.plotly_chart(
            fig, use_container_width=True,
            on_select="rerun", selection_mode="points",
            key=f"cat_chart_{selected_zip}",
        )

        # process chart click → update selected category
        new_cat = None
        try:
            if event and event.selection and event.selection.points:
                new_cat = event.selection.points[0]["y"]
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        if new_cat != selected_cat:
            st.session_state.explorer_cat = new_cat
            st.rerun()

        if selected_cat:
            st.caption(f"showing: **{selected_cat}** · click the bar again to deselect")
        else:
            st.caption("click a bar to filter the business directory below")
    else:
        st.info("no business data available — business tax cert files may be 403. "
                "place CSVs manually in data/raw/ and re-run pipeline.")

    st.divider()

    # Business list table
    active_cat = st.session_state.explorer_cat
    if active_cat:
        st.subheader(f"business directory — {active_cat}")
        if st.button("show all categories", key="explorer_clear_cat"):
            st.session_state.explorer_cat = None
            st.rerun()
    else:
        st.subheader("business directory")

    businesses = _load_businesses(selected_zip, active_cat)
    if businesses is not None and not businesses.empty:
        st.dataframe(
            businesses,
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    else:
        st.info("no business records available for this zip code")

    st.divider()

    # Map
    st.subheader("map")

    lat, lon = ZIP_COORDS.get(selected_zip, (32.7157, -117.1611))

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=13,
            pitch=0,
        ),
        layers=[],
        map_style="mapbox://styles/mapbox/light-v11",
    ))

# ── COMPARE TAB ──
with tab_compare:
    if compare_zip is None:
        st.info("select a second zip code in the sidebar to compare neighborhoods")
    elif compare_zip == selected_zip:
        st.warning("select a different zip code to compare")
    else:
        profile_b = _load_profile(compare_zip)
        biz_by_zip_b = _load_biz_by_zip(compare_zip)

        if profile_b.empty:
            st.warning(f"no data for zip code {compare_zip}")
        else:
            row_b = profile_b.iloc[0]
            name_a = selected_neighborhood or selected_zip
            name_b = compare_neighborhood or compare_zip

            # Comparison narrative
            comp_narrative = _build_comparison_narrative(row, row_b, name_a, name_b)
            st.markdown(f"*{comp_narrative}*")

            # Head-to-head metrics table
            st.subheader(f"{name_a} vs {name_b}")

            # (label, key, prefix, decimals, higher_is)
            #   higher_is: "good" = green when higher, "bad" = green when lower, "neutral" = no color
            compare_metrics = [
                ("population", "population", "", 0, "good"),
                ("median income", "median_income", "$", 0, "good"),
                ("active businesses", "active_count", "", 0, "good"),
                ("businesses per 1k", "businesses_per_1k", "", 1, "good"),
                ("% bachelor's+", "pct_bachelors_plus", "", 1, "good"),
                ("median age", "median_age", "", 0, "neutral"),
                ("median rent", "median_rent", "$", 0, "bad"),
                ("median home value", "median_home_value", "$", 0, "neutral"),
                ("new permits", "new_permits", "", 0, "good"),
                ("crime count", "crime_count", "", 0, "bad"),
                ("311 median days", "median_311_days", "", 1, "bad"),
                ("solar installs", "solar_installs", "", 0, "good"),
            ]

            def _fmt_val(v, prefix, decimals):
                if not _valid(v):
                    return "—"
                return f"{prefix}{float(v):,.{decimals}f}"

            def _winner_class(va, vb, higher_is):
                """Return ('winner','loser') or ('','') css classes for a/b."""
                if not _valid(va) or not _valid(vb) or higher_is == "neutral":
                    return "", ""
                fa, fb = float(va), float(vb)
                if fa == fb:
                    return "", ""
                a_better = (fa > fb) if higher_is == "good" else (fa < fb)
                return ("winner", "loser") if a_better else ("loser", "winner")

            table_rows = []
            for label, key, prefix, decimals, higher_is in compare_metrics:
                va = row.get(key)
                vb = row_b.get(key)
                ca, cb = _winner_class(va, vb, higher_is)
                table_rows.append((label, _fmt_val(va, prefix, decimals), ca,
                                   _fmt_val(vb, prefix, decimals), cb))

            html = f"""
            <style>
            .cmp-table {{ width:100%; border-collapse:collapse; font-size:0.95rem; }}
            .cmp-table th {{ text-align:left; padding:8px 12px; border-bottom:2px solid #444;
                             font-weight:600; }}
            .cmp-table td {{ padding:6px 12px; border-bottom:1px solid #eee; }}
            .cmp-table .metric-label {{ font-weight:500; }}
            .cmp-table .val {{ text-align:right; font-variant-numeric:tabular-nums; }}
            .cmp-table .winner {{ color:#21ba45; font-weight:600; }}
            .cmp-table .loser {{ color:#999; }}
            </style>
            <table class="cmp-table">
            <tr>
              <th>metric</th>
              <th style="text-align:right">{name_a} ({selected_zip})</th>
              <th style="text-align:right">{name_b} ({compare_zip})</th>
            </tr>
            """
            for label, fmt_a, cls_a, fmt_b, cls_b in table_rows:
                html += f"""<tr>
                  <td class="metric-label">{label}</td>
                  <td class="val {cls_a}">{fmt_a}</td>
                  <td class="val {cls_b}">{fmt_b}</td>
                </tr>"""
            html += "</table>"

            st.markdown(html, unsafe_allow_html=True)

            st.divider()

            # Category comparison chart
            st.subheader("business categories comparison")

            if (biz_by_zip is not None and not biz_by_zip.empty
                    and biz_by_zip_b is not None and not biz_by_zip_b.empty):
                cats_a = biz_by_zip.set_index("category")["active_count"]
                cats_b = biz_by_zip_b.set_index("category")["active_count"]
                all_cats = sorted(
                    set(cats_a.index) | set(cats_b.index),
                    key=lambda c: cats_a.get(c, 0) + cats_b.get(c, 0),
                    reverse=True,
                )[:15]
                all_cats = list(reversed(all_cats))

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name=f"{name_a} ({selected_zip})",
                    y=all_cats,
                    x=[cats_a.get(c, 0) for c in all_cats],
                    orientation="h",
                    marker_color=CHART_COLOR,
                ))
                fig.add_trace(go.Bar(
                    name=f"{name_b} ({compare_zip})",
                    y=all_cats,
                    x=[cats_b.get(c, 0) for c in all_cats],
                    orientation="h",
                    marker_color=COMPARE_COLOR,
                ))
                fig.update_layout(
                    barmode="group",
                    height=max(400, len(all_cats) * 35),
                    margin=dict(l=0, r=40, t=0, b=0),
                    xaxis_title="active businesses",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("business category data not available for comparison")

            st.divider()

            # Map with both markers
            st.subheader("map")

            lat_a, lon_a = ZIP_COORDS.get(selected_zip, (32.7157, -117.1611))
            lat_b, lon_b = ZIP_COORDS.get(compare_zip, (32.7157, -117.1611))

            mid_lat = (lat_a + lat_b) / 2
            mid_lon = (lon_a + lon_b) / 2

            dist = math.sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)
            zoom = 13 if dist < 0.02 else 12 if dist < 0.05 else 11 if dist < 0.1 else 10

            marker_data = pd.DataFrame([
                {"lat": lat_a, "lon": lon_a, "label": f"{name_a} ({selected_zip})",
                 "color": [131, 201, 255, 200]},
                {"lat": lat_b, "lon": lon_b, "label": f"{name_b} ({compare_zip})",
                 "color": [255, 107, 107, 200]},
            ])

            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=mid_lat,
                    longitude=mid_lon,
                    zoom=zoom,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=marker_data,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=500,
                        pickable=True,
                    ),
                ],
                tooltip={"text": "{label}"},
                map_style="mapbox://styles/mapbox/light-v11",
            ))

# ── RANKINGS TAB ──
with tab_rankings:
    st.subheader("zip code rankings")
    st.caption("find the best zip code for a specific goal")

    _base_metric_labels = [
        "population", "median income", "median age", "median rent",
        "median home value", "% bachelor's+", "active businesses",
        "businesses per 1k", "category count", "new permits",
        "crime count", "311 median days", "solar installs",
    ]
    _base_metric_keys = [
        "population", "median_income", "median_age", "median_rent",
        "median_home_value", "pct_bachelors_plus", "active_count",
        "businesses_per_1k", "category_count", "new_permits",
        "crime_count", "median_311_days", "solar_installs",
    ]

    # category filter — rendered first so the rank-by dropdown can include
    # a dynamic "{category} per 1k" option when a category is selected
    categories = _category_options()
    cat_filter = st.selectbox(
        "category filter (adds per 1k column)",
        [None] + categories,
        format_func=lambda x: "-- none --" if x is None else x,
        key="rankings_category",
    )

    # build metric options — append category density when a category is active
    all_labels = list(_base_metric_labels)
    all_keys = list(_base_metric_keys)
    if cat_filter:
        all_labels.append(f"{cat_filter} per 1k")
        all_keys.append("category_per_1k")
    _label_to_key = dict(zip(all_labels, all_keys))

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        sort_label = st.selectbox(
            "rank by", all_labels, key="rankings_sort_by",
        )
        sort_key = _label_to_key.get(sort_label, "population")
    with ctrl2:
        sort_desc = st.toggle("highest first", value=True, key="rankings_sort_desc")

    rank_limit = st.slider("show top", min_value=5, max_value=82, value=10, key="rankings_limit")

    rankings_df = _load_rankings(sort_key, sort_desc, cat_filter, rank_limit)

    if rankings_df is not None and not rankings_df.empty:
        rankings_df = rankings_df.copy()
        rankings_df.insert(0, "rank", range(1, len(rankings_df) + 1))

        # build display column order
        display_cols = ["rank", "zip_code", "neighborhood"]

        # sort metric first (unless it's category_per_1k, added with category cols)
        if sort_key != "category_per_1k" and sort_key in rankings_df.columns:
            display_cols.append(sort_key)

        # category context columns when a category is selected
        if cat_filter:
            if "category_per_1k" in rankings_df.columns:
                display_cols.append("category_per_1k")
            if "category_active" in rankings_df.columns:
                display_cols.append("category_active")

        # standard context columns (skip duplicates)
        for c in ("population", "median_income", "active_count"):
            if c not in display_cols and c in rankings_df.columns:
                display_cols.append(c)

        display_df = rankings_df[display_cols].copy()

        # rename for display
        rename = {"zip_code": "zip code"}
        if sort_key != "category_per_1k":
            rename[sort_key] = sort_label
        if cat_filter:
            rename["category_per_1k"] = f"{cat_filter} per 1k"
            rename["category_active"] = f"{cat_filter} count"
        for col, label in [("population", "population"), ("median_income", "median income"),
                           ("active_count", "active businesses")]:
            if col in display_df.columns and col not in rename:
                rename[col] = label
        display_df.rename(columns=rename, inplace=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(display_df) + 38),
        )
    else:
        st.info("no ranking data available")

# ── Footer ──
st.divider()
st.caption("data sources: city of san diego open data, us census bureau, civic cross-references from sd-* projects")
