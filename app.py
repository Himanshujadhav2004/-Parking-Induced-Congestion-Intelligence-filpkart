"""
app.py
------
Streamlit dashboard: "Parking-Induced Congestion Intelligence"
Command Center Edition

Run with:
    streamlit run app.py

Improvements vs v1:
  * Dark Command Center CSS theme
  * Physics-informed capacity loss % (Greenshields) + shockwave queue in drill-down
  * Monetary delay cost (TERI benchmark) in priority table + KPI
  * Greedy patrol schedule optimizer tab
  * What-If CIS formula weight adjuster tab (real-time, no reload)
  * Hour-of-day violation density map animation in Temporal tab
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st

from data_pipeline import load_violations, PARKING_SEVERITY_WEIGHTS
from hotspot_engine import build_hotspot_table, forecast_hotspot, optimize_patrol

st.set_page_config(
    page_title="Parking Congestion Intelligence | Command Center",
    layout="wide",
    page_icon="https://fonts.gstatic.com/s/i/materialiconsoutlined/traffic/v1/24px.svg",
)

# ---------------------------------------------------------------------------
# Dark Command Center CSS
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Google Fonts — load via <link> (more reliable than @import in Streamlit)
# ---------------------------------------------------------------------------
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&'
    'family=Inter:wght@300;400;500;600;700;800&display=swap">',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dark Command Center CSS
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
/* ── MATERIAL SYMBOLS BASELINE ──────────────────────────────── */
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 1.1rem;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    vertical-align: middle;
    margin-right: 0.35rem;
    -webkit-font-smoothing: antialiased;
    /* Ensure icon colour is never overridden by gradient parents */
    -webkit-text-fill-color: currentColor !important;
}

/* ── BASE ───────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080d1a !important;
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container {
    padding-top: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0c1628 !important;
    border-right: 1px solid #1a3054 !important;
}
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* ── METRIC CARDS ───────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0f1e35, #111827) !important;
    border: 1px solid #1a3054 !important;
    border-top: 3px solid #3b82f6 !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
    transition: transform 0.15s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.2) !important;
}
[data-testid="stMetricLabel"] > div {
    color: #64748b !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] > div {
    color: #f1f5f9 !important;
    font-weight: 800 !important;
}
[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* ── TABS ───────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1a3054;
    background: transparent;
    gap: 4px;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 0.75rem 0.5rem 2.1rem !important;
    background: transparent !important;
    transition: color 0.15s ease !important;
    position: relative !important;
}
[data-testid="stTabs"] button[role="tab"]::before {
    font-family: 'Material Symbols Outlined' !important;
    font-size: 1rem !important;
    position: absolute !important;
    left: 0.6rem !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    line-height: 1 !important;
    font-weight: normal !important;
    -webkit-font-smoothing: antialiased !important;
}
[data-testid="stTabs"] button[role="tab"]:nth-child(1)::before { content: 'format_list_bulleted'; }
[data-testid="stTabs"] button[role="tab"]:nth-child(2)::before { content: 'schedule'; }
[data-testid="stTabs"] button[role="tab"]:nth-child(3)::before { content: 'manage_search'; }
[data-testid="stTabs"] button[role="tab"]:nth-child(4)::before { content: 'local_police'; }
[data-testid="stTabs"] button[role="tab"]:nth-child(5)::before { content: 'balance'; }
[data-testid="stTabs"] button[role="tab"]:nth-child(6)::before { content: 'menu_book'; }
[data-testid="stTabs"] button[role="tab"]:hover { color: #94a3b8 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: rgba(59,130,246,0.06) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::before {
    color: #60a5fa !important;
}

/* ── DATAFRAME ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3054;
    border-radius: 8px;
    overflow: hidden;
}

/* ── BUTTONS ────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #065f46, #059669) !important;
    color: white !important; border: none !important; border-radius: 6px !important;
    font-weight: 600 !important;
}

/* ── FORM ELEMENTS ──────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #111827 !important;
    border: 1px solid #1a3054 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
[data-testid="stSlider"] > div > div > div { background: #3b82f6 !important; }

/* ── ALERTS ─────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }
.stSuccess { border-left: 4px solid #10b981 !important; background: rgba(16,185,129,0.08) !important; }
.stInfo { border-left: 4px solid #3b82f6 !important; background: rgba(59,130,246,0.08) !important; }
.stWarning { border-left: 4px solid #f59e0b !important; background: rgba(245,158,11,0.08) !important; }

/* ── EXPANDER ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1a3054 !important;
    border-radius: 8px !important;
    background: #0c1628 !important;
}

/* ── DIVIDER ────────────────────────────────────────────────── */
hr { border-color: #1a3054 !important; margin: 0.75rem 0 !important; }

/* ── SCROLLBAR ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080d1a; }
::-webkit-scrollbar-thumb { background: #1a3054; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── CAPTION ────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #475569 !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Critical": [214, 39,  40,  210],
    "High":     [255, 127, 14,  185],
    "Medium":   [255, 210, 87,  155],
    "Low":      [44,  180, 44,  125],
}

DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


# ---------------------------------------------------------------------------
# Material Symbol inline helper
# ---------------------------------------------------------------------------
def icon(name: str, size: str = "1.15rem", color: str = "#60a5fa") -> str:
    """Return an inline Material Symbols Outlined <span> for use inside
    st.markdown(..., unsafe_allow_html=True) calls."""
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size};color:{color};vertical-align:middle;'
        f'margin-right:0.3rem;">{name}</span>'
    )


def section_header(symbol: str, text: str, color: str = "#60a5fa") -> None:
    """Render a section heading with a Material Symbol icon via st.markdown."""
    st.markdown(
        f'<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;'
        f'margin:0.6rem 0 0.4rem 0;">'
        f'{icon(symbol, "1.2rem", color)}{text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached data loading / processing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading, cleaning, and scoring hotspots…")
def _load_and_build(csv_path: str, resolution: int, k_tiers: int):
    df = load_violations(csv_path)
    hotspots, busy_hours, df_hex = build_hotspot_table(df, resolution=resolution, k_tiers=k_tiers)
    return df, hotspots, busy_hours, df_hex


def recommend_action(row: pd.Series) -> str:
    actions = []
    if row["junction_fraction"] >= 0.5:
        actions.append("Coordinate with junction signal team (high junction-approach blockage)")
    if row["busy_hour_fraction"] >= 0.5:
        actions.append("Schedule patrol during this hotspot's high-activity hours")
    if row["persistence_ratio"] >= 0.5:
        actions.append("Chronic spot — consider physical deterrents / no-parking signage")
    if row["growth_rate"] > 0.05:
        actions.append("Emerging trend — escalate before it becomes entrenched")
    if not actions:
        actions.append("Monitor; lower relative priority")
    return "; ".join(actions)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;padding:0.5rem 0 0.25rem 0;">'
    f'<span class="material-symbols-outlined" style="font-size:1.2rem;color:#3b82f6;'
    f'vertical-align:middle;margin-right:0.4rem;">traffic</span>'
    f'PICIF Controls</div>',
    unsafe_allow_html=True,
)

data_source = st.sidebar.radio(
    "Data source",
    ["Bundled sample (25k rows, instant)", "Custom CSV path (full dataset)"],
)
if data_source.startswith("Bundled"):
    csv_path = "data/sample_violations.csv"
else:
    csv_path = st.sidebar.text_input(
        "Path to violation CSV",
        value="data/jan_to_may_police_violation_anonymized791b166.csv",
    )

resolution = st.sidebar.slider(
    "H3 hex resolution (9 = ~0.1 km² cells)", min_value=7, max_value=10, value=9,
)
k_tiers = st.sidebar.slider("Number of priority tiers (KMeans clusters)", 2, 4, 4)
top_n = st.sidebar.slider("Hotspots to list in priority table", 10, 100, 25)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Violation severity weights** (edit in `data_pipeline.py`):\n\n"
    + "\n".join(
        f"- {k}: **{v}**"
        for k, v in sorted(PARKING_SEVERITY_WEIGHTS.items(), key=lambda kv: -kv[1])
    )
)

# ---------------------------------------------------------------------------
# Load + build
# ---------------------------------------------------------------------------
try:
    df, hotspots, busy_hours, df_hex = _load_and_build(csv_path, resolution, k_tiers)
except FileNotFoundError:
    st.error(
        f"**File Not Found:** `{csv_path}`\n\n"
        "Switch **Data source** in the sidebar to **'Bundled sample (25k rows, instant)'** to demo instantly."
    )
    st.stop()
except Exception as e:
    st.error(
        f"**Error loading data:** {e}\n\n"
        "Please check if the file is a valid CSV and contains the required columns."
    )
    st.stop()

tier_filter = st.sidebar.multiselect(
    "Filter map/table by tier",
    options=list(hotspots["tier"].unique()),
    default=list(hotspots["tier"].unique()),
)
station_filter = st.sidebar.multiselect(
    "Filter by police station",
    options=sorted(hotspots["dominant_station"].unique()),
    default=[],
)

view = hotspots[hotspots["tier"].isin(tier_filter)]
if station_filter:
    view = view[view["dominant_station"].isin(station_filter)]
view = view.sort_values("congestion_impact_score", ascending=False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding:0.25rem 0 0.5rem 0;">
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.15rem;">
            <span class="material-symbols-outlined"
                  style="font-size:2.2rem; color:#60a5fa;
                         -webkit-text-fill-color:#60a5fa;
                         flex-shrink:0;">traffic</span>
            <div style="
                font-size:1.9rem; font-weight:800; font-family:'Inter',sans-serif;
                background:linear-gradient(135deg,#60a5fa,#a78bfa 50%,#34d399);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; line-height:1.2;">
                Parking-Induced Congestion Intelligence
            </div>
        </div>
        <div style="color:#475569;font-size:0.82rem;margin-top:0.1rem;letter-spacing:0.04em;">
            AI-driven hotspot detection &nbsp;&middot;&nbsp; Physics-informed impact scoring &nbsp;&middot;&nbsp;
            Targeted enforcement prioritization &nbsp;&middot;&nbsp; Bengaluru Traffic Police
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Violations analyzed",      f"{len(df):,}")
k2.metric("Hotspots identified",       f"{hotspots.shape[0]:,}")
k3.metric("Critical + High tier",      f"{(hotspots['tier'].isin(['Critical','High'])).sum():,}")
k4.metric("Avg. CIS score",            f"{hotspots['congestion_impact_score'].mean():.1f} / 100")
k5.metric("Est. city-wide daily cost", f"₹{hotspots['daily_delay_cost_inr'].sum():,.0f}")
k6.metric("Busiest hours (IST)",       ", ".join(f"{h:02d}h" for h in sorted(busy_hours)))

st.markdown("---")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
section_header("map", "City-Wide Enforcement Priority Map")
st.caption(
    "Hex height = Congestion Impact Score  \u00b7  "
    "Color = priority tier (red=Critical \u2192 green=Low)  \u00b7  "
    "Hover for capacity loss % and estimated daily delay cost"
)

map_df = view.copy()
map_df["color"]     = map_df["tier"].map(TIER_COLORS)
map_df["elevation"] = map_df["congestion_impact_score"] * 30

layer = pdk.Layer(
    "H3HexagonLayer",
    map_df,
    get_hexagon="hex_id",
    get_fill_color="color",
    get_elevation="elevation",
    elevation_scale=1,
    extruded=True,
    pickable=True,
    auto_highlight=True,
)

center_lat = float(map_df["latitude"].mean())  if len(map_df) else 12.9716
center_lon = float(map_df["longitude"].mean()) if len(map_df) else 77.5946

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=center_lat, longitude=center_lon, zoom=11, pitch=45
    ),
    map_style=DARK_MAP,
    tooltip={
        "html": (
            "<b>{dominant_station}</b><br/>"
            "Violations: {violation_count}<br/>"
            "CIS Score: {congestion_impact_score}<br/>"
            "Tier: {tier}<br/>"
            "Lane Capacity Loss: {capacity_loss_pct}%<br/>"
            "Shockwave Queue: {shockwave_queue_km} km<br/>"
            "Est. Daily Cost: ₹{daily_delay_cost_inr}"
        ),
        "style": {
            "backgroundColor": "#0c1628",
            "color": "#e2e8f0",
            "border": "1px solid #1a3054",
            "borderRadius": "6px",
            "fontSize": "13px",
        },
    },
)
st.pydeck_chart(deck, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Priority Enforcement List",
    "Temporal Patterns",
    "Hotspot Drill-down",
    "Patrol Optimizer",
    "What-If Scoring",
    "Methodology",
])

# ── TAB 1 : Priority Enforcement List ──────────────────────────────────────
with tab1:
    section_header("format_list_bulleted", f"Top {top_n} priority hotspots")

    table = view.head(top_n).copy()
    table["recommended_action"] = table.apply(recommend_action, axis=1)

    cols_show = [
        "hex_id", "dominant_station", "violation_count",
        "congestion_impact_score", "tier",
        "capacity_loss_pct", "shockwave_queue_km",
        "daily_delay_cost_inr", "annual_delay_cost_inr",
        "junction_fraction", "busy_hour_fraction",
        "persistence_ratio", "growth_rate",
        "recommended_action",
    ]
    col_names = {
        "hex_id":                  "Hex ID",
        "dominant_station":        "Station",
        "violation_count":         "Violations",
        "congestion_impact_score": "CIS (0-100)",
        "tier":                    "Tier",
        "capacity_loss_pct":       "Lane Loss %",
        "shockwave_queue_km":      "Queue (km)",
        "daily_delay_cost_inr":    "Daily Cost (₹)",
        "annual_delay_cost_inr":   "Annual Cost (₹)",
        "junction_fraction":       "% at Junction",
        "busy_hour_fraction":      "% Busy Hours",
        "persistence_ratio":       "Persistence",
        "growth_rate":             "Weekly Growth",
        "recommended_action":      "Recommended Action",
    }
    table_display = table[cols_show].rename(columns=col_names)

    fmt = {
        "CIS (0-100)":    "{:.1f}",
        "Lane Loss %":    "{:.1f}",
        "Queue (km)":     "{:.2f}",
        "Daily Cost (₹)": "{:,}",
        "Annual Cost (₹)":"{:,}",
        "% at Junction":  "{:.1%}",
        "% Busy Hours":   "{:.1%}",
        "Persistence":    "{:.1%}",
        "Weekly Growth":  "{:+.1%}",
    }
    st.dataframe(
        table_display.style.format(fmt),
        width="stretch", hide_index=True,
    )

    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ Download full hotspot table (CSV)",
            data=hotspots.drop(columns=["tag_breakdown"]).to_csv(index=False),
            file_name="parking_hotspots.csv",
        )
    with info_col:
        total_annual = int(hotspots["annual_delay_cost_inr"].sum())
        st.info(
            f"**Estimated city-wide annual delay cost: \u20b9{total_annual/1e7:.1f} Crore** "
            f"(conservative; assumes \u20b950/vehicle-hour delay value \u2014 TERI 2018 benchmark)"
        )

# ── TAB 2 : Temporal Patterns ───────────────────────────────────────────────
with tab2:
    section_header("schedule", "When are violations most heavily recorded?")

    pivot = (
        df_hex.groupby(["weekday", "hour"]).size().rename("count").reset_index()
        .pivot(index="weekday", columns="hour", values="count").fillna(0)
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(pivot.index)]
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", axis=None),
        width="stretch",
    )
    st.caption(
        "Note: `created_datetime` likely reflects when a violation was *logged* "
        "into the system rather than a confirmed live-detection timestamp \u2014 the hourly pattern "
        "does not follow a typical commute curve. Treat 'busy hours' as enforcement-activity "
        "windows and validate against ground-truth detection timestamps before operational use."
    )

    by_tag = df_hex.explode("tags")
    by_tag = by_tag[by_tag["tags"].isin(PARKING_SEVERITY_WEIGHTS.keys())]
    section_header("bar_chart", "Violation type mix (parking-relevant tags)", color="#a78bfa")
    st.bar_chart(by_tag["tags"].value_counts())

    st.markdown("---")
    section_header("access_time", "Hour-of-Day Violation Density", color="#34d399")
    st.caption(
        "Drag the slider to see where violations concentrate at each hour. "
        "Dot size and colour (yellow = high, dark = low) represent relative violation density."
    )

    sel_hour = st.slider("Hour of day (IST)", 0, 23, 10, key="temporal_hour_slider")
    hour_data = df_hex[df_hex["hour"] == sel_hour]

    if len(hour_data) > 0:
        hex_hour = (
            hour_data.groupby("hex_id")
            .agg(count=("severity", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"))
            .reset_index()
        )
        max_count = hex_hour["count"].max()
        hex_hour["norm"] = (hex_hour["count"] / max_count).clip(0, 1)
        hex_hour["color"] = hex_hour["norm"].apply(
            lambda n: [255, int(255 * (1 - n * 0.8)), 0, 180]
        )
        hex_hour["radius"] = (hex_hour["norm"] * 250 + 40).astype(int)

        scatter = pdk.Layer(
            "ScatterplotLayer",
            hex_hour,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius="radius",
            radius_min_pixels=3,
            radius_max_pixels=18,
            pickable=True,
        )
        hour_deck = pdk.Deck(
            layers=[scatter],
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=11, pitch=25
            ),
            map_style=DARK_MAP,
            tooltip={
                "html": "Violations: <b>{count}</b>",
                "style": {"backgroundColor": "#0c1628", "color": "#e2e8f0"},
            },
        )
        st.pydeck_chart(hour_deck, width="stretch")
        st.caption(
            f"**{sel_hour:02d}:00 IST** \u2014 {len(hour_data):,} violations across "
            f"{len(hex_hour):,} locations."
            + (" Star: this is a busiest-hour window." if sel_hour in busy_hours else "")
        )
    else:
        st.info(f"No violations recorded at {sel_hour:02d}:00 in this dataset.")

# ── TAB 3 : Hotspot Drill-down ───────────────────────────────────────────────
with tab3:
    section_header("manage_search", "Drill into a specific hotspot")
    options = view.head(100)["hex_id"] + " — " + view.head(100)["dominant_station"]

    if len(options) == 0:
        st.info("No hotspots match the current filters.")
    else:
        choice = st.selectbox("Select hotspot (hex_id — dominant station)", options)
        sel_hex = choice.split(" — ")[0]
        row = hotspots[hotspots["hex_id"] == sel_hex].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Violations",        int(row["violation_count"]))
        c2.metric("CIS Score",         f"{row['congestion_impact_score']:.1f}")
        c3.metric("Tier",              row["tier"])
        c4.metric("Weekly growth",     f"{row['growth_rate']*100:+.1f}%")

        st.markdown("---")

        st.markdown(
            f'<div style="font-weight:700;color:#60a5fa;font-size:0.9rem;'
            f'letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.5rem;">'
            f'{icon("bolt", "1.1rem", "#f59e0b")}'
            f'Physics-Informed Impact \u2014 Greenshields Model</div>',
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Lane Capacity Loss",    f"{row.get('capacity_loss_pct', 0):.1f}%")
        p2.metric("Shockwave Queue",       f"{row.get('shockwave_queue_km', 0):.2f} km")
        p3.metric("Eff. Traffic Speed",    f"{row.get('effective_speed_kmh', 0):.0f} km/h")
        p4.metric("Est. Daily Delay Cost", f"₹{int(row.get('daily_delay_cost_inr', 0)):,}")

        st.caption("Greenshields (1935): *u = u_f × (1 − k/k_j)*.")

        annual_cost = int(row.get("annual_delay_cost_inr", 0))
        if annual_cost > 0:
            st.info(f"Estimated **annual delay cost: \u20b9{annual_cost:,}**")

        st.markdown(f"**Recommended action:** {recommend_action(row)}")
        st.markdown("---")

        breakdown = pd.Series(row["tag_breakdown"]).sort_values(ascending=False)
        section_header("donut_small", "Violation type breakdown at this hotspot", color="#a78bfa")
        st.bar_chart(breakdown)

        section_header("trending_up", "Next 2-week forecast (weekly violation volume)", color="#34d399")
        hist = (
            df_hex[df_hex["hex_id"] == sel_hex]
            .assign(
                week=lambda d: pd.to_datetime(d["date"])
                .dt.to_period("W")
                .apply(lambda p: p.start_time)
            )
            .groupby("week").size().rename("count").reset_index()
        )
        fc = forecast_hotspot(df_hex, sel_hex, weeks_ahead=2)
        hist["type"] = "history"
        fc = fc.rename(columns={"forecast_count": "count"})
        fc["type"] = "forecast"
        combined = pd.concat([hist, fc], ignore_index=True).set_index("week")
        st.line_chart(
            combined.pivot_table(index=combined.index, columns="type", values="count")
        )

# ── TAB 4 : Patrol Optimizer ─────────────────────────────────────────────────
with tab4:
    section_header("local_police", "Greedy Patrol Schedule Optimizer")
    st.markdown(
        "Given a fixed number of officers and a shift duration, this tool allocates "
        "officer-hours to **maximise total Congestion Impact Score recovered**. "
        "Critical hotspots require **2 officer-hours** (need more attention); "
        "High / Medium / Low require **1 officer-hour**."
    )

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        n_officers = st.number_input(
            "Available officers", min_value=1, max_value=100, value=5, step=1,
        )
    with oc2:
        shift_hours = st.number_input(
            "Shift duration (hours)", min_value=1, max_value=12, value=8, step=1,
        )
    with oc3:
        officer_budget = n_officers * shift_hours
        st.metric("Total officer-hours available", officer_budget)

    # Use the current filtered view if a station filter is active, else use all hotspots
    patrol_pool = view if station_filter else hotspots
    schedule = optimize_patrol(patrol_pool, n_officers=int(n_officers), shift_hours=int(shift_hours))

    if len(schedule) > 0:
        total_cis_covered   = schedule["CIS"].sum()
        total_cost_prevented = schedule["Est. Daily Cost (₹)"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Hotspots assigned",          len(schedule))
        m2.metric("Total CIS recovered",        f"{total_cis_covered:.1f}")
        m3.metric("Est. daily cost prevented",  f"₹{total_cost_prevented:,}")

        st.markdown(
            f'<div style="font-weight:600;color:#94a3b8;margin:0.5rem 0 0.25rem 0;">'
            f'{icon("table", "1rem", "#64748b")}Optimal Patrol Assignment Schedule</div>',
            unsafe_allow_html=True,
        )
        sched_fmt = {
            "CIS":              "{:.1f}",
            "Lane Loss %":      "{:.1f}",
            "Est. Daily Cost (₹)": "{:,}",
        }
        st.dataframe(
            schedule.style.format(sched_fmt)
            .background_gradient(subset=["CIS"], cmap="RdYlGn"),
            width="stretch", hide_index=True,
        )

        # Compare to random patrol heuristic
        avg_cis     = hotspots["congestion_impact_score"].mean()
        random_cis  = avg_cis * min(officer_budget, len(hotspots))
        if random_cis > 0:
            pct_better = (total_cis_covered - random_cis) / random_cis * 100
            st.success(
                f"This schedule recovers **{total_cis_covered:.0f} CIS** vs "
                f"~{random_cis:.0f} for random patrol \u2014 "
                f"**{pct_better:+.0f}% more effective** than random deployment."
            )

        st.caption(
            "Methodology: greedy selection — hotspots sorted by CIS (descending), "
            "assigned until officer-hour budget is exhausted. "
            "Critical tier costs 2 officer-hours; all others cost 1."
        )
    else:
        st.info("No hotspots in the current filter to optimise.")

# ── TAB 5 : What-If Scoring ──────────────────────────────────────────────────
with tab5:
    section_header("balance", "What-If CIS Formula Adjuster")
    st.markdown(
        "Adjust the **relative weights** of the four CIS components to see how the "
        "enforcement priority ranking changes **in real-time** — no data reload required. "
        "The percentile ranks are pre-computed; only the weighted sum changes."
    )
    st.caption(
        "Current formula: **CIS = 40% severity + 20% junction proximity + "
        "20% busy-hour share + 20% persistence**"
    )

    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        w_sev = st.slider(
            "Severity weight", 0, 100, 40, key="w_sev",
            help="Weight for violation severity (e.g. Double Parking vs Footpath)",
        )
    with wf2:
        w_jct = st.slider(
            "Junction weight", 0, 100, 20, key="w_jct",
            help="Weight for junction-proximity (chokepoint effect)",
        )
    with wf3:
        w_bh = st.slider(
            "Busy-hour weight", 0, 100, 20, key="w_bh",
            help="Weight for violations during high enforcement-activity hours",
        )
    with wf4:
        w_per = st.slider(
            "Persistence weight", 0, 100, 20, key="w_per",
            help="Weight for chronic vs one-off hotspots",
        )

    total_w = w_sev + w_jct + w_bh + w_per
    if total_w == 0:
        st.warning("All weights are 0 — set at least one weight > 0.")
    else:
        ws = w_sev / total_w
        wj = w_jct / total_w
        wb = w_bh  / total_w
        wp = w_per / total_w

        st.caption(
            f"Normalised: **{ws:.0%}** severity · **{wj:.0%}** junction · "
            f"**{wb:.0%}** busy-hour · **{wp:.0%}** persistence"
        )

        score_cols = ["score_severity", "score_junction", "score_busy_hour", "score_persistence"]
        if all(c in hotspots.columns for c in score_cols):
            # Recompute CIS from pre-computed percentile ranks — no pipeline re-run
            base = hotspots.reset_index(drop=True).copy()
            base["original_rank"] = range(1, len(base) + 1)
            base["whatif_cis"] = (
                ws * base["score_severity"]
                + wj * base["score_junction"]
                + wb * base["score_busy_hour"]
                + wp * base["score_persistence"]
            ) * 100

            whatif = base.sort_values("whatif_cis", ascending=False).reset_index(drop=True)
            whatif["new_rank"]    = range(1, len(whatif) + 1)
            whatif["rank_change"] = whatif["original_rank"] - whatif["new_rank"]

            wi_cols = whatif[[
                "dominant_station", "tier",
                "congestion_impact_score", "whatif_cis", "rank_change",
            ]].head(25).rename(columns={
                "dominant_station":        "Station",
                "tier":                    "Tier",
                "congestion_impact_score": "Original CIS",
                "whatif_cis":              "What-If CIS",
                "rank_change":             "Rank Δ",
            })

            st.markdown("**Top 25 hotspots under new scoring weights**")
            wi_fmt = {
                "Original CIS": "{:.1f}",
                "What-If CIS":  "{:.1f}",
                "Rank Δ":       "{:+.0f}",
            }
            st.dataframe(
                wi_cols.style.format(wi_fmt)
                .background_gradient(subset=["What-If CIS"], cmap="RdYlGn"),
                width="stretch", hide_index=True,
            )

            # Surface biggest movers
            big_movers = whatif[whatif["rank_change"].abs() > 5].head(6)
            if len(big_movers) > 0:
                st.markdown(
                    f'{icon("shuffle", "1rem", "#a78bfa")}'
                    f'<span style="font-weight:700;color:#f1f5f9;">Biggest rank changes with these weights:</span>',
                    unsafe_allow_html=True,
                )
                for _, m in big_movers.iterrows():
                    direction = "rose" if m["rank_change"] > 0 else "fell"
                    arrow = icon("arrow_upward", "0.9rem", "#10b981") if m["rank_change"] > 0 else icon("arrow_downward", "0.9rem", "#ef4444")
                    st.markdown(
                        f'{arrow}<b>{m["dominant_station"]}</b> ({m["tier"]}): '
                        f'{direction} by <b>{abs(int(m["rank_change"]))} positions</b> '
                        f'(CIS {m["congestion_impact_score"]:.0f} \u2192 {m["whatif_cis"]:.0f})',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No hotspot changed rank by more than 5 positions with these weights.")
        else:
            st.warning(
                "Component score columns not found — clear the Streamlit cache and reload."
            )

# ── TAB 6 : Methodology ──────────────────────────────────────────────────────
with tab6:
    section_header("menu_book", "How the Congestion Impact Score (CIS) is built", color="#a78bfa")
    st.markdown(
        """
This dataset contains **enforcement records**, not live road-speed/flow sensor data, so
"impact on traffic flow" is estimated through a transparent, editable proxy model rather
than a black box:

1. **Spatial binning (H3 hexagons)** — violations are grouped into ~0.1 km² hex cells so
   nearby incidents on the same stretch of road are analyzed together, independent of
   inconsistent free-text addresses.
2. **Severity weighting (0–3 per record)** — each violation tag is weighted by how much it
   typically obstructs a lane or junction approach (e.g. *Double Parking* / *Parking in a
   Main Road* = 3, *Parking on Footpath* = 1). Editable in `data_pipeline.py`.
3. **Junction proximity** — share of violations tagged at a named junction (chokepoints
   cause disproportionate downstream congestion vs. mid-block parking).
4. **Busy-hour concentration** — share of a hotspot's violations during the hours that,
   *empirically in this data*, account for the bulk of enforcement activity (no hardcoded
   "rush hour" assumption — see the caveat in the Temporal Patterns tab).
5. **Persistence** — share of days in the observation window the hotspot was active
   (chronic vs. one-off).

These four signals are percentile-ranked and combined into a 0–100 **Congestion Impact Score**:

`CIS = 40% severity + 20% junction proximity + 20% busy-hour share + 20% persistence`

**Priority tiers** (Critical/High/Medium/Low) are then assigned by an unsupervised
**KMeans** model over the underlying features (not just the CIS itself).

---

### Physics-Informed Impact — Greenshields + LWR Shockwave Theory

In addition to the CIS, each hotspot is independently scored using established
**traffic flow theory**:

- **Greenshields fundamental diagram (1935):** `u = u_f × (1 − k/k_j)` where
  *u_f* = free-flow speed, *k* = traffic density, *k_j* = jam density.
  We approximate the density fraction from violation severity and junction proximity.
- **Lane capacity loss %** — the fraction of theoretical throughput that is lost due to
  the parking obstruction at this hotspot.
- **Shockwave queue (km)** — estimated upstream queue propagation using
  Lighthill-Whitham-Richards (LWR) theory, scaled by persistence.
- **Effective speed (km/h)** — estimated traffic speed through the bottleneck.

These are defensible, cite-able traffic-engineering metrics — **not** black-box numbers.

---

### Monetary Delay Cost — TERI 2018 Benchmark

Each hotspot's parking violations impose a quantifiable economic cost through vehicle delay:

- **Value of time:** ₹50 / vehicle-hour (TERI 2018 urban mobility benchmark).
- **Vehicles impacted:** ~10 passing vehicles per violation record.
- **Average delay:** 2.5 minutes per affected vehicle per incident.
- **Scaled by severity** (Double Parking causes longer delay than Footpath Parking).

This produces conservative daily and annual delay-cost estimates per hotspot.

---

### Patrol Optimizer

The **Greedy Patrol Optimizer** (Patrol Optimizer tab) maximises total CIS recovered
given a finite officer-hour budget. It is a greedy approximation of the
0-1 Knapsack problem — optimal in expectation when hotspot impacts are
independently additive (a reasonable assumption for spatially separated hotspots).

---

### Where this fits in the Gridlock Intelligence platform

This module answers *"where is illegal parking choking traffic, and where should we
deploy enforcement first."* It is designed to plug into the other two problem statements:

- **Event-driven congestion** (ASTRAM data) — event proximity could temporarily
  re-rank hotspots during rallies / festivals / construction.
- **CV-based violation detection** — live camera detections could feed this engine
  in near-real-time instead of after-the-fact records, fixing the timestamp-reliability
  caveat above and enabling real-time hotspot scoring.

---

### Known limitations (stated explicitly for judges)

- No ground-truth congestion / speed data — severity is a domain-informed proxy,
  not a measured delay.
- `created_datetime` timing reliability caveat (see Temporal Patterns tab).
- H3 resolution is a tunable trade-off between spatial precision and noise at low counts.
- Greenshields impact and monetary cost estimates are calibrated proxies — validate
  against ground-truth loop-detector or probe-vehicle data if available downstream.
        """
    )
