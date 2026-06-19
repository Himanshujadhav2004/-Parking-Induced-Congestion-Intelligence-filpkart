"""
app.py
------
Streamlit dashboard: "Parking-Induced Congestion Intelligence"

Run with:
    streamlit run app.py

By default it loads the bundled 25k-row sample (data/sample_violations.csv)
so the demo works instantly. Point it at the full export via the sidebar
for the real analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st

from data_pipeline import load_violations, PARKING_SEVERITY_WEIGHTS
from hotspot_engine import build_hotspot_table, forecast_hotspot

st.set_page_config(
    page_title="Parking Congestion Intelligence",
    layout="wide",
    page_icon="🚧",
)

TIER_COLORS = {
    "Critical": [214, 39, 40, 190],
    "High": [255, 127, 14, 170],
    "Medium": [255, 221, 87, 140],
    "Low": [44, 160, 44, 110],
}


# ---------------------------------------------------------------------------
# Cached data loading / processing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading, cleaning, and scoring hotspots...")
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
st.sidebar.title("🚧 Controls")

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
    "H3 hex resolution (9 = ~0.1 km² cells, finer; 8 = ~0.46 km², coarser)",
    min_value=7, max_value=10, value=9,
)
k_tiers = st.sidebar.slider("Number of priority tiers (KMeans clusters)", 2, 4, 4)
top_n = st.sidebar.slider("Hotspots to list", 10, 100, 25)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Severity weights (editable in data_pipeline.py):\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in sorted(PARKING_SEVERITY_WEIGHTS.items(), key=lambda kv: -kv[1]))
)

# ---------------------------------------------------------------------------
# Load + build
# ---------------------------------------------------------------------------
try:
    df, hotspots, busy_hours, df_hex = _load_and_build(csv_path, resolution, k_tiers)
except FileNotFoundError:
    st.error(
        f"📂 **File Not Found:** `{csv_path}`\n\n"
        "The selected dataset file could not be found. Please check the path and make sure the file exists.\n\n"
        "**To resolve this:**\n"
        "1. Switch the **Data source** in the sidebar to **'Bundled sample (25k rows, instant)'** to view the demo instantly.\n"
        "2. Or place your custom CSV file at the path shown in the sidebar."
    )
    st.stop()
except Exception as e:
    st.error(
        f"🚨 **Error loading data:** {e}\n\n"
        "Please check if the file is a valid CSV and contains the required columns."
    )
    st.stop()

tier_filter = st.sidebar.multiselect(
    "Filter map/table by tier", options=list(hotspots["tier"].unique()),
    default=list(hotspots["tier"].unique()),
)
station_filter = st.sidebar.multiselect(
    "Filter by police station", options=sorted(hotspots["dominant_station"].unique()),
    default=[],
)

view = hotspots[hotspots["tier"].isin(tier_filter)]
if station_filter:
    view = view[view["dominant_station"].isin(station_filter)]
view = view.sort_values("congestion_impact_score", ascending=False)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("🚧 Parking-Induced Congestion Intelligence")
st.caption(
    "AI-driven detection of illegal-parking hotspots and their estimated impact on traffic flow, "
    "for targeted enforcement prioritization."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Violations analyzed", f"{len(df):,}")
k2.metric("Hotspots identified", f"{hotspots.shape[0]:,}")
k3.metric("Critical + High tier", f"{(hotspots['tier'].isin(['Critical','High'])).sum():,}")
k4.metric("Avg. Congestion Impact Score", f"{hotspots['congestion_impact_score'].mean():.1f} / 100")
k5.metric("Busiest hours (IST, data-derived)", ", ".join(f"{h:02d}:00" for h in sorted(busy_hours)))

st.markdown("---")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
st.subheader("Hotspot Map")
st.caption("Hex height = Congestion Impact Score · Color = priority tier (red=Critical → blue=Low)")

map_df = view.copy()
map_df["color"] = map_df["tier"].map(TIER_COLORS)
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

center_lat = float(map_df["latitude"].mean()) if len(map_df) else 12.9716
center_lon = float(map_df["longitude"].mean()) if len(map_df) else 77.5946

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=45),
    tooltip={
        "html": "<b>{dominant_station}</b><br/>"
                "Violations: {violation_count}<br/>"
                "Score: {congestion_impact_score}<br/>"
                "Tier: {tier}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    },
)
st.pydeck_chart(deck, width='stretch')

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Priority Enforcement List", "⏱️ Temporal Patterns", "🔎 Hotspot Drill-down", "📖 Methodology"]
)

with tab1:
    st.subheader(f"Top {top_n} priority hotspots")
    table = view.head(top_n).copy()
    table["recommended_action"] = table.apply(recommend_action, axis=1)
    table_display = table[[
        "hex_id", "dominant_station", "violation_count", "congestion_impact_score",
        "tier", "junction_fraction", "busy_hour_fraction", "persistence_ratio",
        "growth_rate", "recommended_action",
    ]].rename(columns={
        "hex_id": "Hex ID", "dominant_station": "Station", "violation_count": "Violations",
        "congestion_impact_score": "CIS (0-100)", "tier": "Tier",
        "junction_fraction": "% at Junction", "busy_hour_fraction": "% in Busy Hours",
        "persistence_ratio": "Persistence", "growth_rate": "Weekly Growth Rate",
        "recommended_action": "Recommended Action",
    })
    st.dataframe(table_display, width='stretch', hide_index=True)
    st.download_button(
        "Download full hotspot table (CSV)",
        data=hotspots.drop(columns=["tag_breakdown"]).to_csv(index=False),
        file_name="parking_hotspots.csv",
    )

with tab2:
    st.subheader("When are violations most heavily recorded?")
    pivot = (
        df_hex.groupby(["weekday", "hour"]).size().rename("count").reset_index()
        .pivot(index="weekday", columns="hour", values="count").fillna(0)
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(pivot.index)]
    st.dataframe(pivot.style.background_gradient(cmap="OrRd", axis=None), width='stretch')
    st.caption(
        "⚠️ Data caveat: `created_datetime` likely reflects when a violation was logged into the "
        "system rather than a confirmed live-detection timestamp — the hourly pattern does not "
        "follow a typical commute curve. Treat 'busy hours' as **enforcement-activity windows** "
        "in this data, and validate against ground-truth detection timestamps before using this "
        "to schedule patrols operationally."
    )

    by_tag = df_hex.explode("tags")
    by_tag = by_tag[by_tag["tags"].isin(PARKING_SEVERITY_WEIGHTS.keys())]
    st.subheader("Violation type mix (parking-relevant tags)")
    st.bar_chart(by_tag["tags"].value_counts())

with tab3:
    st.subheader("Drill into a specific hotspot")
    options = view.head(100)["hex_id"] + " — " + view.head(100)["dominant_station"]
    if len(options) == 0:
        st.info("No hotspots match the current filters.")
    else:
        choice = st.selectbox("Select hotspot (hex_id — dominant station)", options)
        sel_hex = choice.split(" — ")[0]
        row = hotspots[hotspots["hex_id"] == sel_hex].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Violations", int(row["violation_count"]))
        c2.metric("CIS", f"{row['congestion_impact_score']:.1f}")
        c3.metric("Tier", row["tier"])
        c4.metric("Weekly growth", f"{row['growth_rate']*100:.1f}%")

        st.markdown(f"**Recommended action:** {recommend_action(row)}")

        breakdown = pd.Series(row["tag_breakdown"]).sort_values(ascending=False)
        st.bar_chart(breakdown)

        st.markdown("**Next 2-week forecast (weekly violation volume)**")
        hist = (
            df_hex[df_hex["hex_id"] == sel_hex]
            .assign(week=lambda d: pd.to_datetime(d["date"]).dt.to_period("W").apply(lambda p: p.start_time))
            .groupby("week").size().rename("count").reset_index()
        )
        fc = forecast_hotspot(df_hex, sel_hex, weeks_ahead=2)
        hist["type"] = "history"
        fc = fc.rename(columns={"forecast_count": "count"})
        fc["type"] = "forecast"
        combined = pd.concat([hist, fc], ignore_index=True).set_index("week")
        st.line_chart(combined.pivot_table(index=combined.index, columns="type", values="count"))

with tab4:
    st.subheader("How the Congestion Impact Score (CIS) is built")
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

These four signals are percentile-ranked and combined into a 0–100 **Congestion Impact
Score**:

`CIS = 40% severity + 20% junction proximity + 20% busy-hour share + 20% persistence`

**Priority tiers (Critical/High/Medium/Low)** are then assigned by an unsupervised
**KMeans** model over the underlying features (not just the CIS itself), so a hotspot can
be flagged Critical for, say, being a small-but-extremely-persistent chokepoint even if its
raw volume is modest — letting the AI surface patterns a simple top-N-by-count list would miss.

**Growth-rate trend** (linear slope of weekly counts) flags *emerging* hotspots before they
become entrenched, and a lightweight forecast projects each hotspot's next 2 weeks of
expected violations for patrol/resource planning.

---

### Where this fits in a fuller "Gridlock Intelligence" platform
This module answers *"where is illegal parking choking traffic, and where should we deploy
enforcement first."* It is designed to plug into the other two problem statements in this
hackathon:
- **Event-driven congestion** data (ASTRAM) could add a "scheduled event nearby" feature to
  temporarily re-rank hotspots during rallies/festivals/construction.
- **Computer-vision violation detection** could feed this engine *live* detections instead of
  after-the-fact enforcement records, fixing the timestamp-reliability caveat above and
  enabling near-real-time hotspot scoring.

### Known limitations (stated explicitly for judges)
- No ground-truth congestion/speed data — severity is a domain-informed proxy, not measured delay.
- `created_datetime` timing reliability caveat (see Temporal Patterns tab).
- H3 resolution is a tunable trade-off between spatial precision and noise at low counts.
        """
    )
