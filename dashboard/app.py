"""Streamlit dashboard for SD Business Intel — neighborhood explorer."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from api import queries

# ── Parquet paths (kept for dashboard-specific queries that have no query-layer equivalent) ──
_AGG = "data/aggregated"

_root = Path(__file__).resolve().parent.parent
if (_root / _AGG).exists():
    _AGG = str(_root / _AGG)

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


def _dashboard_query(sql: str, params: list | None = None):
    """Run SQL against parquet files and return a pandas DataFrame.

    Kept for the few dashboard-specific queries that have no query-layer
    equivalent: zip+neighborhood options and city averages for metric deltas.
    """
    con = duckdb.connect()
    return con.execute(sql, params or []).fetchdf()


# ── Sidebar ──
st.sidebar.title("sd business intel")
st.sidebar.caption("neighborhood explorer")

level = st.sidebar.radio(
    "explore by", ["area", "zip code"], index=0, horizontal=True, key="level_picker"
)


@st.cache_data(ttl=3600)
def _zip_options():
    """Get zip+neighborhood pairs for the sidebar selector.

    Uses a direct query because get_filters() only returns zip codes
    without neighborhood names.
    """
    try:
        df = _dashboard_query(f"""
            SELECT np.zip_code, np.neighborhood
            FROM '{_AGG}/neighborhood_profile.parquet' np
            ORDER BY np.zip_code
        """)
        return list(zip(df["zip_code"], df["neighborhood"]))
    except Exception:
        return []


@st.cache_data(ttl=3600)
def _area_options():
    """Get area summaries for the sidebar selector."""
    try:
        return queries.get_areas()
    except Exception:
        return []


zip_options = _zip_options()

if not zip_options:
    st.error("no data available. run the pipeline first: `uv run python -m pipeline.build`")
    st.stop()

# ── Sidebar: area mode vs zip mode ──
selected_area = None
selected_zip = None
selected_neighborhood = None
drilldown_zip = None
compare_zip = None
compare_neighborhood = None
compare_area = None

if level == "area":
    area_list = _area_options()
    if not area_list:
        st.sidebar.warning("no area data available")
        st.stop()

    area_names = [a["area"] for a in area_list]
    selected_area = st.sidebar.selectbox(
        "area",
        area_names,
        key="sidebar_area",
    )

    # drill-into-zip dropdown
    @st.cache_data(ttl=3600)
    def _area_zips(area: str):
        return queries.get_area_zips(area)

    area_zips = _area_zips(selected_area)
    if area_zips:
        zip_part_labels = [
            f"{z['zip_code']} \u2014 {z['neighborhood']}" if z.get("neighborhood") else z["zip_code"]
            for z in area_zips
        ]
        drilldown_labels = ["-- area overview --"] + zip_part_labels
        drilldown_sel = st.sidebar.selectbox(
            "drill into zip",
            drilldown_labels,
            index=0,
            key="sidebar_drilldown_zip",
        )
        if drilldown_sel != "-- area overview --":
            dd_idx = zip_part_labels.index(drilldown_sel)
            drilldown_zip = area_zips[dd_idx]["zip_code"]
            selected_zip = drilldown_zip
            selected_neighborhood = area_zips[dd_idx].get("neighborhood")

else:
    # zip code mode — same as original
    zip_labels = [f"{z} — {n}" if n else z for z, n in zip_options]
    selected_idx = st.sidebar.selectbox(
        "zip code",
        zip_labels,
        index=next((i for i, (z, _) in enumerate(zip_options) if z == "92101"), 0),
        key="sidebar_zip",
    )
    idx = zip_labels.index(selected_idx) if isinstance(selected_idx, str) else selected_idx
    selected_zip = zip_options[idx][0]
    selected_neighborhood = zip_options[idx][1]

    compare_labels = ["-- none --"] + zip_labels
    compare_sel = st.sidebar.selectbox(
        "zip code 2 (compare)",
        compare_labels,
        index=0,
        key="sidebar_compare_zip",
    )
    if compare_sel != "-- none --":
        cidx = zip_labels.index(compare_sel)
        compare_zip = zip_options[cidx][0]
        compare_neighborhood = zip_options[cidx][1]

# ── Category deep-dive session state ──
if "explorer_cat" not in st.session_state:
    st.session_state.explorer_cat = None
_tracking_key = selected_zip or selected_area or ""
if st.session_state.get("_explorer_target") != _tracking_key:
    st.session_state._explorer_target = _tracking_key
    st.session_state.explorer_cat = None

# ── Cached data loaders ──


@st.cache_data(ttl=3600)
def _load_profile(zip_code: str) -> dict:
    """Load a full neighborhood profile via the query layer."""
    return queries.get_neighborhood_profile(zip_code)


@st.cache_data(ttl=3600)
def _load_area_profile(area: str) -> dict:
    """Load a full area profile via the query layer."""
    return queries.get_area_profile(area)


@st.cache_data(ttl=3600)
def _load_city_avg():
    """Load city-wide averages for metric delta display.

    Uses a direct query because the query layer embeds comparison data
    inside each profile but doesn't expose a standalone city averages
    endpoint — and the dashboard needs avg values for metric deltas
    across all displayed metrics (including median_age which isn't in
    the profile's comparison_to_avg).
    """
    try:
        return _dashboard_query(f"SELECT * FROM '{_AGG}/city_averages.parquet'")
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_businesses(zip_code: str, category: str | None = None):
    """Load individual business records via the query layer."""
    rows = queries.get_businesses(zip_code=zip_code, category=category, limit=5000)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=3600)
def _category_options():
    """Get distinct business categories via the query layer."""
    filters = queries.get_filters()
    return filters.get("categories", [])


@st.cache_data(ttl=3600)
def _load_trends(zip_code: str | None = None, area: str | None = None) -> dict:
    """Load YoY trend data for a zip code or area."""
    if zip_code:
        return queries.get_zip_trends(zip_code)
    elif area:
        return queries.get_area_trends(area)
    return {}


# Metric -> sense for badge phrasing in _show_rank.
# "high" = higher is good, "low" = lower is good, "neutral" = no judgement.
_RANK_SENSE = {
    "population": "high",
    "median_income": "high",
    "median_age": "neutral",
    "median_home_value": "neutral",
    "median_rent": "low",
    "pct_bachelors_plus": "high",
    "active_count": "high",
    "businesses_per_1k": "high",
    "new_permits": "high",
    "crime_count": "low",
    "median_311_days": "low",
    "solar_installs": "high",
}


@st.cache_data(ttl=3600)
def _load_rankings(sort_by, sort_desc, category, limit):
    """Load zip code rankings via the query layer."""
    rows = queries.get_rankings(sort_by=sort_by, sort_desc=sort_desc,
                                category=category, limit=limit)
    if not rows or (len(rows) == 1 and "error" in rows[0]):
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # The query layer returns the sort metric as "sort_value". Rename to
    # the actual metric name so the display code can reference it directly.
    if "sort_value" in df.columns and sort_by != "category_per_1k":
        df.rename(columns={"sort_value": sort_by}, inplace=True)
    # Drop internal columns not needed for display
    for col in ("sort_metric", "rank", "category"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


@st.cache_data(ttl=3600)
def _load_area_rankings(sort_by, sort_desc, category, limit):
    """Load area rankings via the query layer."""
    rows = queries.get_area_rankings(sort_by=sort_by, sort_desc=sort_desc,
                                     category=category, limit=limit)
    if not rows or (len(rows) == 1 and "error" in rows[0]):
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "sort_value" in df.columns and sort_by != "category_per_1k":
        df.rename(columns={"sort_value": sort_by}, inplace=True)
    for col in ("sort_metric", "rank", "category"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


# ── Load city averages (always needed) ──
city_avg = _load_city_avg()
has_avg = city_avg is not None and not city_avg.empty
avg_row = city_avg.iloc[0] if has_avg else None


# ── Helpers ──


def _fmt_metric(label, val, avg_key=None, prefix="", decimals=0):
    """Render a st.metric with optional delta vs city avg."""
    if not _valid(val):
        return label, "\u2014", None
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
        return "\u2014", None
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
    rank, of = p["rank"], p["of"]
    sense = _RANK_SENSE.get(metric_key, "neutral")
    if sense == "low":
        text = f"{_ordinal(rank)} lowest of {of}"
    elif sense == "high":
        pct = max(1, round(100 * rank / of))
        if pct <= 50:
            text = f"top {pct}% \u00b7 rank {rank} of {of}"
        else:
            text = f"rank {rank} of {of}"
    else:
        text = f"rank {rank} of {of}"
    col.caption(text)


def _latest_yoy(trends: dict, series_name: str) -> str | None:
    """Get the latest YoY percentage from a trend series, formatted for st.metric delta."""
    series = trends.get(series_name, [])
    if not series:
        return None
    latest = series[-1]
    yoy = latest.get("yoy_pct")
    if yoy is None:
        return None
    return f"{yoy:+.0f}% yoy"


def _flat_profile(p):
    """Flatten a nested profile dict for metric access."""
    flat = {}
    for section in ("demographics", "business_landscape", "civic_signals"):
        if section in p:
            for k, v in p[section].items():
                if k != "top_categories":
                    flat[k] = v
    return flat


def _fmt_val(v, prefix, decimals):
    """Format a value for the comparison table."""
    if not _valid(v):
        return "\u2014"
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


# (label, key, prefix, decimals, higher_is)
#   higher_is: "good" = green when higher, "bad" = green when lower, "neutral" = no color
_COMPARE_METRICS = [
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

_CMP_TABLE_CSS = """
<style>
.cmp-table { width:100%; border-collapse:collapse; font-size:0.95rem; }
.cmp-table th { text-align:left; padding:8px 12px; border-bottom:2px solid #444;
                 font-weight:600; }
.cmp-table td { padding:6px 12px; border-bottom:1px solid #eee; }
.cmp-table .metric-label { font-weight:500; }
.cmp-table .val { text-align:right; font-variant-numeric:tabular-nums; }
.cmp-table .winner { color:#21ba45; font-weight:600; }
.cmp-table .loser { color:#999; }
</style>
"""


def _render_compare_table(flat_a, flat_b, name_a, name_b):
    """Render a head-to-head HTML comparison table."""
    table_rows = []
    for label, key, prefix, decimals, higher_is in _COMPARE_METRICS:
        va = flat_a.get(key)
        vb = flat_b.get(key)
        ca, cb = _winner_class(va, vb, higher_is)
        table_rows.append((label, _fmt_val(va, prefix, decimals), ca,
                           _fmt_val(vb, prefix, decimals), cb))

    html = _CMP_TABLE_CSS + f"""
    <table class="cmp-table">
    <tr>
      <th>metric</th>
      <th style="text-align:right">{name_a}</th>
      <th style="text-align:right">{name_b}</th>
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


def _render_category_comparison_chart(top_cats_a, top_cats_b, name_a, name_b):
    """Render a side-by-side bar chart comparing business categories."""
    if not top_cats_a or not top_cats_b:
        st.info("business category data not available for comparison")
        return

    cats_a = {c["category"]: c["active_count"] for c in top_cats_a}
    cats_b = {c["category"]: c["active_count"] for c in top_cats_b}
    all_cats = sorted(
        set(cats_a.keys()) | set(cats_b.keys()),
        key=lambda c: cats_a.get(c, 0) + cats_b.get(c, 0),
        reverse=True,
    )[:15]
    all_cats = list(reversed(all_cats))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name_a,
        y=all_cats,
        x=[cats_a.get(c, 0) for c in all_cats],
        orientation="h",
        marker_color=CHART_COLOR,
    ))
    fig.add_trace(go.Bar(
        name=name_b,
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


def _render_zip_explorer(profile, _demo, _biz, _civic, percentiles, zip_code,
                         neighborhood, key_prefix):
    """Render the full zip-level explorer content (metrics, categories, directory, map).

    Shared by both zip-mode and area-drilldown-mode to avoid duplication.
    """
    # Load YoY trend data for civic signal deltas
    trends = _load_trends(zip_code=zip_code)

    pop = _demo.get("population")
    income = _demo.get("median_income")
    active_biz = _biz.get("active_count")
    age = _demo.get("median_age")
    rent = _demo.get("median_rent")
    home_val = _demo.get("median_home_value")
    edu = _demo.get("pct_bachelors_plus")

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

    permits = _civic.get("new_permits")
    crime = _civic.get("crime_count")
    solar = _civic.get("solar_installs")
    median_311 = _civic.get("median_311_days")

    l, v, _ = _fmt_metric("crime count", crime)
    r3[1].metric(l, v, _latest_yoy(trends, "crime"), delta_color="inverse")
    _show_rank(r3[1], "crime_count", percentiles)
    l, v, d = _fmt_metric("311 median days", median_311, decimals=1)
    r3[2].metric(l, v, d, delta_color="inverse")
    _show_rank(r3[2], "median_311_days", percentiles)

    r4 = st.columns(3)
    l, v, _ = _fmt_metric("new permits", permits)
    r4[0].metric(l, v, _latest_yoy(trends, "permits"), delta_color="normal")
    _show_rank(r4[0], "new_permits", percentiles)
    l, v, _ = _fmt_metric("solar installs", solar)
    r4[1].metric(l, v, _latest_yoy(trends, "solar"), delta_color="normal")
    _show_rank(r4[1], "solar_installs", percentiles)
    biz_per_1k = _biz.get("businesses_per_1k")
    l, v, d = _fmt_metric("businesses per 1k residents", biz_per_1k, "avg_businesses_per_1k", "", 1)
    r4[2].metric(l, v, d, delta_color="normal")
    _show_rank(r4[2], "businesses_per_1k", percentiles)

    st.divider()

    # Business categories chart
    st.subheader("top business categories")

    top_categories = _biz.get("top_categories", [])
    if top_categories:
        biz_by_zip = pd.DataFrame(top_categories)
        selected_cat = st.session_state.explorer_cat

        population = _demo.get("population")
        if _valid(population) and float(population) > 0 and "per_1k" in biz_by_zip.columns:
            bar_text = biz_by_zip.apply(
                lambda r: f"{int(r['active_count'])}  ({r['per_1k']}/1k)", axis=1
            )
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
            key=f"cat_chart_{key_prefix}_{zip_code}",
        )

        # process chart click -> update selected category
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
            st.caption(f"showing: **{selected_cat}** -- click the bar again to deselect")
        else:
            st.caption("click a bar to filter the business directory below")
    else:
        st.info("no business data available -- business tax cert files may be 403. "
                "place CSVs manually in data/raw/ and re-run pipeline.")

    st.divider()

    # Business list table
    active_cat = st.session_state.explorer_cat
    if active_cat:
        st.subheader(f"business directory -- {active_cat}")
        if st.button("show all categories", key=f"clear_cat_{key_prefix}_{zip_code}"):
            st.session_state.explorer_cat = None
            st.rerun()
    else:
        st.subheader("business directory")

    businesses = _load_businesses(zip_code, active_cat)
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

    lat, lon = ZIP_COORDS.get(zip_code, (32.7157, -117.1611))

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


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════
tab_explorer, tab_compare, tab_rankings = st.tabs(["explorer", "compare", "rankings"])

# ── EXPLORER TAB ──
with tab_explorer:

    # ── Area mode: area overview (no zip drill-down) ──
    if level == "area" and selected_area and not drilldown_zip:
        area_profile = _load_area_profile(selected_area)
        if not area_profile:
            st.warning(f"no data for area: {selected_area}")
        else:
            st.title(selected_area.lower())
            narrative = area_profile.get("narrative", "")
            if narrative:
                st.markdown(f"*{narrative}*")

            _a_demo = area_profile.get("demographics", {})
            _a_biz = area_profile.get("business_landscape", {})
            _a_civic = area_profile.get("civic_signals", {})

            # Load YoY trend data for civic signal deltas
            area_trends = _load_trends(area=selected_area)

            # Summary metric cards
            r1 = st.columns(4)
            pop = _a_demo.get("population")
            l, v, d = _fmt_metric("population", pop, "avg_population")
            r1[0].metric(l, v, d, delta_color="normal")
            active_biz = _a_biz.get("active_count")
            l, v, d = _fmt_metric("active businesses", active_biz, "avg_active_businesses")
            r1[1].metric(l, v, d, delta_color="normal")
            biz_per_1k = _a_biz.get("businesses_per_1k")
            l, v, d = _fmt_metric("businesses per 1k", biz_per_1k, "avg_businesses_per_1k", "", 1)
            r1[2].metric(l, v, d, delta_color="normal")
            income = _a_demo.get("median_income")
            l, v, d = _fmt_metric("median income", income, "avg_median_income", "$")
            r1[3].metric(l, v, d, delta_color="normal")

            r2 = st.columns(4)
            rent = _a_demo.get("median_rent")
            l, v, d = _fmt_metric("median rent", rent, "avg_median_rent", "$")
            r2[0].metric(l, v, d, delta_color="inverse")
            home_val = _a_demo.get("median_home_value")
            l, v, d = _fmt_metric("median home value", home_val, "avg_median_home_value", "$")
            r2[1].metric(l, v, d, delta_color="off")
            crime = _a_civic.get("crime_count")
            l, v, _ = _fmt_metric("crime count", crime)
            r2[2].metric(l, v, _latest_yoy(area_trends, "crime"), delta_color="inverse")
            permits = _a_civic.get("new_permits")
            l, v, _ = _fmt_metric("new permits", permits)
            r2[3].metric(l, v, _latest_yoy(area_trends, "permits"), delta_color="normal")

            st.divider()

            # Constituent zips table
            st.subheader("zip codes in this area")
            area_zips_data = queries.get_area_zips(selected_area)
            if area_zips_data:
                zips_df = pd.DataFrame(area_zips_data)
                rename_map = {
                    "zip_code": "zip code",
                    "active_count": "active businesses",
                    "businesses_per_1k": "businesses per 1k",
                    "median_income": "median income",
                    "crime_count": "crime count",
                    "new_permits": "new permits",
                }
                display_cols = [c for c in rename_map if c in zips_df.columns]
                zips_display = zips_df[display_cols].rename(columns=rename_map)
                st.dataframe(
                    zips_display,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 35 * len(zips_display) + 38),
                )
            else:
                st.info("no zip code data available for this area")

            st.divider()

            # Business categories chart
            st.subheader("top business categories")
            top_categories = _a_biz.get("top_categories", [])
            if top_categories:
                biz_df = pd.DataFrame(top_categories)

                if _valid(pop) and float(pop) > 0 and "per_1k" in biz_df.columns:
                    bar_text = biz_df.apply(
                        lambda r: f"{int(r['active_count'])}  ({r['per_1k']}/1k)", axis=1
                    )
                else:
                    bar_text = biz_df["active_count"]

                fig = go.Figure(go.Bar(
                    x=biz_df["active_count"],
                    y=biz_df["category"],
                    orientation="h",
                    marker_color=CHART_COLOR,
                    text=bar_text,
                    textposition="outside",
                ))
                fig.update_layout(
                    height=max(300, len(biz_df) * 28),
                    margin=dict(l=0, r=40, t=0, b=0),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="active businesses",
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"area_cat_chart_{selected_area}")
            else:
                st.info("no business data available")

    # ── Area mode: drilled into a zip ──
    elif level == "area" and selected_area and drilldown_zip:
        profile = _load_profile(drilldown_zip)

        if "error" in profile:
            st.warning(f"no data for zip code {drilldown_zip}")
        else:
            # Breadcrumb
            st.caption(f"{selected_area.lower()} > {drilldown_zip}")
            st.title(selected_neighborhood or drilldown_zip)
            st.caption(f"zip code {drilldown_zip}")

            _demo = profile.get("demographics", {})
            _biz = profile.get("business_landscape", {})
            _civic = profile.get("civic_signals", {})
            percentiles = profile.get("percentiles", {})

            narrative = profile.get("narrative", "")
            if narrative:
                st.markdown(f"*{narrative}*")

            _render_zip_explorer(
                profile, _demo, _biz, _civic, percentiles, drilldown_zip,
                selected_neighborhood, "drilldown"
            )

    # ── Zip mode: standard zip explorer ──
    elif level == "zip code" and selected_zip:
        profile = _load_profile(selected_zip)

        if "error" in profile:
            st.warning(f"no data for zip code {selected_zip}")
            st.stop()

        _demo = profile.get("demographics", {})
        _biz = profile.get("business_landscape", {})
        _civic = profile.get("civic_signals", {})
        percentiles = profile.get("percentiles", {})

        st.title(f"{selected_neighborhood or selected_zip}")
        st.caption(f"zip code {selected_zip}")

        narrative = profile.get("narrative", "")
        if narrative:
            st.markdown(f"*{narrative}*")

        _render_zip_explorer(
            profile, _demo, _biz, _civic, percentiles, selected_zip,
            selected_neighborhood, "zipmode"
        )

    else:
        st.info("select an area or zip code in the sidebar to explore")


# ── COMPARE TAB ──
with tab_compare:
    if level == "area":
        # Area compare mode
        area_list_cmp = _area_options()
        if not area_list_cmp:
            st.info("no area data available for comparison")
        else:
            area_names_cmp = [a["area"] for a in area_list_cmp]
            c1, c2 = st.columns(2)
            with c1:
                cmp_area_a = st.selectbox(
                    "area 1", area_names_cmp, index=0, key="compare_area_a"
                )
            with c2:
                default_b = min(1, len(area_names_cmp) - 1)
                cmp_area_b = st.selectbox(
                    "area 2", area_names_cmp, index=default_b, key="compare_area_b"
                )

            if cmp_area_a == cmp_area_b:
                st.warning("select two different areas to compare")
            else:
                comparison_result = queries.compare_areas(cmp_area_a, cmp_area_b)
                if "error" in comparison_result:
                    st.warning(comparison_result["error"])
                else:
                    comp_narrative = comparison_result.get("narrative", "")
                    if comp_narrative:
                        st.markdown(f"*{comp_narrative}*")

                    profile_a = comparison_result["area_a"]
                    profile_b = comparison_result["area_b"]

                    st.subheader(f"{cmp_area_a.lower()} vs {cmp_area_b.lower()}")

                    flat_a = _flat_profile(profile_a)
                    flat_b = _flat_profile(profile_b)
                    _render_compare_table(flat_a, flat_b, cmp_area_a.lower(), cmp_area_b.lower())

                    st.divider()

                    # Category comparison chart
                    st.subheader("business categories comparison")
                    top_cats_a = profile_a.get("business_landscape", {}).get("top_categories", [])
                    top_cats_b = profile_b.get("business_landscape", {}).get("top_categories", [])
                    _render_category_comparison_chart(
                        top_cats_a, top_cats_b,
                        cmp_area_a.lower(), cmp_area_b.lower()
                    )

    else:
        # Zip compare mode — original behavior
        if compare_zip is None:
            st.info("select a second zip code in the sidebar to compare neighborhoods")
        elif compare_zip == selected_zip:
            st.warning("select a different zip code to compare")
        else:
            profile_a = _load_profile(selected_zip)
            profile_b = _load_profile(compare_zip)

            if "error" in profile_a:
                st.warning(f"no data for zip code {selected_zip}")
            elif "error" in profile_b:
                st.warning(f"no data for zip code {compare_zip}")
            else:
                name_a = selected_neighborhood or selected_zip
                name_b = compare_neighborhood or compare_zip

                comparison_result = queries.compare_zips(selected_zip, compare_zip)
                comp_narrative = comparison_result.get("narrative", "")
                st.markdown(f"*{comp_narrative}*")

                st.subheader(f"{name_a} vs {name_b}")

                flat_a = _flat_profile(profile_a)
                flat_b = _flat_profile(profile_b)
                _render_compare_table(
                    flat_a, flat_b,
                    f"{name_a} ({selected_zip})", f"{name_b} ({compare_zip})"
                )

                st.divider()

                st.subheader("business categories comparison")

                _biz_a = profile_a.get("business_landscape", {})
                _biz_b = profile_b.get("business_landscape", {})
                top_cats_a = _biz_a.get("top_categories", [])
                top_cats_b = _biz_b.get("top_categories", [])
                _render_category_comparison_chart(
                    top_cats_a, top_cats_b,
                    f"{name_a} ({selected_zip})", f"{name_b} ({compare_zip})"
                )

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
    _base_metric_labels = [
        "population", "median income", "median age", "median rent",
        "median home value", "% bachelor's+", "active businesses",
        "businesses per 1k", "new permits",
        "crime count", "311 median days", "solar installs",
    ]
    _base_metric_keys = [
        "population", "median_income", "median_age", "median_rent",
        "median_home_value", "pct_bachelors_plus", "active_count",
        "businesses_per_1k", "new_permits",
        "crime_count", "median_311_days", "solar_installs",
    ]

    if level == "area":
        st.subheader("area rankings")
        st.caption("find the best area for a specific goal")

        # category filter
        categories = _category_options()
        cat_filter = st.selectbox(
            "category filter (adds per 1k column)",
            [None] + categories,
            format_func=lambda x: "-- none --" if x is None else x,
            key="area_rankings_category",
        )

        all_labels = list(_base_metric_labels)
        all_keys = list(_base_metric_keys)
        if cat_filter:
            all_labels.append(f"{cat_filter} per 1k")
            all_keys.append("category_per_1k")
        _label_to_key = dict(zip(all_labels, all_keys))

        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            sort_label = st.selectbox(
                "rank by", all_labels, key="area_rankings_sort_by",
            )
            sort_key = _label_to_key.get(sort_label, "population")
        with ctrl2:
            sort_desc = st.toggle("highest first", value=True, key="area_rankings_sort_desc")

        rank_limit = st.slider("show top", min_value=5, max_value=30, value=10,
                               key="area_rankings_limit")

        rankings_df = _load_area_rankings(sort_key, sort_desc, cat_filter, rank_limit)

        if rankings_df is not None and not rankings_df.empty:
            rankings_df = rankings_df.copy()
            rankings_df.insert(0, "rank", range(1, len(rankings_df) + 1))

            display_cols = ["rank", "area"]

            if sort_key != "category_per_1k" and sort_key in rankings_df.columns:
                display_cols.append(sort_key)

            if cat_filter:
                if "category_per_1k" in rankings_df.columns:
                    display_cols.append("category_per_1k")
                if "category_active" in rankings_df.columns:
                    display_cols.append("category_active")

            for c in ("population", "median_income", "active_count"):
                if c not in display_cols and c in rankings_df.columns:
                    display_cols.append(c)

            display_df = rankings_df[display_cols].copy()

            rename = {}
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
            st.info("no area ranking data available")

    else:
        st.subheader("zip code rankings")
        st.caption("find the best zip code for a specific goal")

        # add category_count to zip-mode labels/keys
        zip_metric_labels = list(_base_metric_labels)
        zip_metric_keys = list(_base_metric_keys)
        # insert category count after businesses per 1k
        ins_idx = zip_metric_keys.index("businesses_per_1k") + 1
        zip_metric_labels.insert(ins_idx, "category count")
        zip_metric_keys.insert(ins_idx, "category_count")

        # category filter
        categories = _category_options()
        cat_filter = st.selectbox(
            "category filter (adds per 1k column)",
            [None] + categories,
            format_func=lambda x: "-- none --" if x is None else x,
            key="rankings_category",
        )

        all_labels = list(zip_metric_labels)
        all_keys = list(zip_metric_keys)
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

        rank_limit = st.slider("show top", min_value=5, max_value=82, value=10,
                               key="rankings_limit")

        rankings_df = _load_rankings(sort_key, sort_desc, cat_filter, rank_limit)

        if rankings_df is not None and not rankings_df.empty:
            rankings_df = rankings_df.copy()
            rankings_df.insert(0, "rank", range(1, len(rankings_df) + 1))

            display_cols = ["rank", "zip_code", "neighborhood"]

            if sort_key != "category_per_1k" and sort_key in rankings_df.columns:
                display_cols.append(sort_key)

            if cat_filter:
                if "category_per_1k" in rankings_df.columns:
                    display_cols.append("category_per_1k")
                if "category_active" in rankings_df.columns:
                    display_cols.append("category_active")

            for c in ("population", "median_income", "active_count"):
                if c not in display_cols and c in rankings_df.columns:
                    display_cols.append(c)

            display_df = rankings_df[display_cols].copy()

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
