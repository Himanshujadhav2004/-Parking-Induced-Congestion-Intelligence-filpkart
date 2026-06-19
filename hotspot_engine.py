"""
hotspot_engine.py
-----------------
Turns the cleaned record-level violation data into ranked, scored
"hotspots" (H3 hexagon cells) with:

  * a multi-factor Congestion Impact Score (CIS)               -> WHY it matters
  * an unsupervised KMeans priority tier (Critical/High/...)    -> AI classification
  * an empirically-derived "busy hours" flag (no hardcoded assumption)
  * a simple growth-trend signal to flag emerging hotspots
  * a lightweight next-2-week forecast per hotspot for patrol planning
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TIER_ORDER = ["Critical", "High", "Medium", "Low"]


# ---------------------------------------------------------------------------
# 1. Empirically-derived "busy hours" (replaces a hardcoded rush-hour guess)
# ---------------------------------------------------------------------------
def compute_busy_hours(df: pd.DataFrame, coverage: float = 0.5) -> set[int]:
    """Return the smallest set of hours-of-day that together account for
    at least `coverage` (e.g. 50%) of all violations city-wide. This lets
    the rest of the pipeline reason about "high enforcement-activity
    hours" without assuming a textbook commute curve.
    """
    counts = df["hour"].value_counts().sort_values(ascending=False)
    cum = counts.cumsum() / counts.sum()
    busy = counts[: (cum < coverage).sum() + 1].index.tolist()
    return set(int(h) for h in busy)


# ---------------------------------------------------------------------------
# 2. H3 hexagon binning
# ---------------------------------------------------------------------------
def add_hex_id(df: pd.DataFrame, resolution: int = 9) -> pd.DataFrame:
    """Resolution 9 ~= 0.10 km^2 per hex (roughly a city block cluster).
    Use 8 (~0.46 km^2) for a coarser, faster view on very large cities.
    """
    df = df.copy()
    df["hex_id"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    return df


# ---------------------------------------------------------------------------
# 3. Aggregate per-hex features
# ---------------------------------------------------------------------------
def aggregate_hotspots(df: pd.DataFrame, busy_hours: set[int]) -> pd.DataFrame:
    n_days = max((pd.to_datetime(df["date"]).max() - pd.to_datetime(df["date"]).min()).days, 1)

    df = df.copy()
    df["is_busy_hour"] = df["hour"].isin(busy_hours)

    grouped = df.groupby("hex_id")
    agg = grouped.agg(
        violation_count=("severity", "size"),
        severity_weighted_count=("severity", "sum"),
        avg_severity=("severity", "mean"),
        junction_fraction=("at_junction", "mean"),
        busy_hour_fraction=("is_busy_hour", "mean"),
        active_days=("date", "nunique"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        dominant_station=("police_station", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        dominant_vehicle=("vehicle_type", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    ).reset_index()

    agg["persistence_ratio"] = (agg["active_days"] / n_days).clip(upper=1.0)

    # representative tag breakdown per hex, kept for drill-down in the UI.
    # Built via pivot rather than groupby().apply() to guarantee exactly
    # one row per hex_id (apply() with dict-valued returns can otherwise
    # silently broadcast into a hex_id x tag MultiIndex).
    exploded = df.explode("tags")
    tag_counts = exploded.groupby(["hex_id", "tags"]).size().unstack(fill_value=0)
    tag_breakdown = tag_counts.apply(lambda row: row[row > 0].to_dict(), axis=1)
    tag_breakdown.name = "tag_breakdown"
    agg = agg.merge(tag_breakdown, on="hex_id", how="left")

    return agg


# ---------------------------------------------------------------------------
# 4. Growth trend (weekly slope) per hex -> flags emerging hotspots
# ---------------------------------------------------------------------------
def compute_growth_trend(df: pd.DataFrame, min_weeks: int = 3) -> pd.DataFrame:
    weekly = (
        df.assign(week=pd.to_datetime(df["date"]).dt.to_period("W").apply(lambda p: p.start_time))
        .groupby(["hex_id", "week"])
        .size()
        .rename("count")
        .reset_index()
    )

    def slope_pct(g: pd.DataFrame) -> float:
        g = g.sort_values("week")
        if len(g) < min_weeks or g["count"].mean() == 0:
            return 0.0
        x = np.arange(len(g))
        y = g["count"].to_numpy(dtype=float)
        b1 = np.polyfit(x, y, 1)[0]
        return float(b1 / y.mean())  # normalized weekly growth rate

    trend = weekly.groupby("hex_id").apply(slope_pct).rename("growth_rate").reset_index()
    return trend


# ---------------------------------------------------------------------------
# 5. Composite Congestion Impact Score (interpretable, rule-based)
# ---------------------------------------------------------------------------
def score_hotspots(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()

    def pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    agg["score_severity"] = pct_rank(agg["severity_weighted_count"])
    agg["score_junction"] = pct_rank(agg["junction_fraction"])
    agg["score_busy_hour"] = pct_rank(agg["busy_hour_fraction"])
    agg["score_persistence"] = pct_rank(agg["persistence_ratio"])

    agg["congestion_impact_score"] = (
        0.40 * agg["score_severity"]
        + 0.20 * agg["score_junction"]
        + 0.20 * agg["score_busy_hour"]
        + 0.20 * agg["score_persistence"]
    ) * 100  # 0-100 scale for readability

    return agg


# ---------------------------------------------------------------------------
# 6. Unsupervised priority tiering (KMeans) -- the "AI" classification layer
#    on top of the interpretable composite score above.
# ---------------------------------------------------------------------------
def tier_hotspots(agg: pd.DataFrame, k: int = 4, random_state: int = 42) -> pd.DataFrame:
    agg = agg.copy()
    features = agg[[
        "severity_weighted_count", "junction_fraction",
        "busy_hour_fraction", "persistence_ratio", "growth_rate",
    ]].fillna(0.0)

    k_eff = min(k, max(agg["hex_id"].nunique(), 1))
    if k_eff < 2:
        agg["tier"] = "Critical"
        return agg

    X = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    agg["cluster"] = km.fit_predict(X)

    # Order clusters by mean composite score, best -> worst, and map onto
    # human-readable tier labels so the AI grouping stays interpretable.
    cluster_rank = (
        agg.groupby("cluster")["congestion_impact_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    labels = (TIER_ORDER + ["Low"] * (k_eff - len(TIER_ORDER)))[:k_eff]
    tier_map = dict(zip(cluster_rank, labels))
    agg["tier"] = agg["cluster"].map(tier_map)
    return agg


# ---------------------------------------------------------------------------
# 7. Lightweight forecast for the next N weeks (patrol planning aid)
# ---------------------------------------------------------------------------
def forecast_hotspot(df: pd.DataFrame, hex_id: str, weeks_ahead: int = 2) -> pd.DataFrame:
    sub = df[df["hex_id"] == hex_id]
    weekly = (
        sub.assign(week=pd.to_datetime(sub["date"]).dt.to_period("W").apply(lambda p: p.start_time))
        .groupby("week")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("week")
    )
    if len(weekly) < 3:
        last = weekly["count"].mean() if len(weekly) else 0.0
        future_weeks = pd.date_range(
            (weekly["week"].max() if len(weekly) else pd.Timestamp.today()),
            periods=weeks_ahead + 1, freq="W",
        )[1:]
        return pd.DataFrame({"week": future_weeks, "forecast_count": [max(last, 0)] * weeks_ahead})

    x = np.arange(len(weekly))
    y = weekly["count"].to_numpy(dtype=float)
    b1, b0 = np.polyfit(x, y, 1)
    future_x = np.arange(len(weekly), len(weekly) + weeks_ahead)
    future_y = np.clip(b1 * future_x + b0, 0, None)
    future_weeks = pd.date_range(weekly["week"].max(), periods=weeks_ahead + 1, freq="W")[1:]
    return pd.DataFrame({"week": future_weeks, "forecast_count": future_y})


# ---------------------------------------------------------------------------
# Orchestration helper used by the dashboard
# ---------------------------------------------------------------------------
def build_hotspot_table(df: pd.DataFrame, resolution: int = 9, k_tiers: int = 4) -> tuple[pd.DataFrame, set[int], pd.DataFrame]:
    busy_hours = compute_busy_hours(df)
    df_hex = add_hex_id(df, resolution=resolution)
    agg = aggregate_hotspots(df_hex, busy_hours)
    trend = compute_growth_trend(df_hex)
    agg = agg.merge(trend, on="hex_id", how="left")
    agg["growth_rate"] = agg["growth_rate"].fillna(0.0)
    agg = score_hotspots(agg)
    agg = tier_hotspots(agg, k=k_tiers)
    agg = agg.sort_values("congestion_impact_score", ascending=False).reset_index(drop=True)
    return agg, busy_hours, df_hex


if __name__ == "__main__":
    import sys
    import os
    from data_pipeline import load_violations

    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists("data/sample_violations.csv"):
        path = "data/sample_violations.csv"
    else:
        path = "jan_to_may_police_violation_anonymized791b166.csv"

    try:
        df = load_violations(path)
        hotspots, busy_hours, df_hex = build_hotspot_table(df)
        print(f"Processed data from {path}")
        print("Empirically-derived busy hours (IST):", sorted(busy_hours))
        print(hotspots[[
            "hex_id", "dominant_station", "violation_count",
            "congestion_impact_score", "tier", "growth_rate",
        ]].head(15).to_string(index=False))
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'. Please specify a valid file path as an argument.")
