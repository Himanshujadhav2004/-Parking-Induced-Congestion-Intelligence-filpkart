# Parking-Induced Congestion Intelligence

**Gridlock Hackathon 2.0 — Round 2 Prototype**
Problem statement: *Poor Visibility on Parking-Induced Congestion*

> How can AI-driven parking intelligence detect illegal parking hotspots and
> quantify their impact on traffic flow to enable targeted enforcement?

## What this is

A working prototype that turns ~298K raw illegal-parking enforcement
records into a ranked, explainable, map-based enforcement priority list —
moving from *reactive patrol* to *targeted, data-driven deployment*.

It does **not** claim to have ground-truth congestion/speed data (the
dataset doesn't include that). Instead it builds a transparent, editable
**Congestion Impact Score** from signals that are genuinely in the data
(severity of obstruction, junction proximity, timing concentration,
persistence), and layers an **unsupervised KMeans model** on top to assign
priority tiers — so this is real AI-driven classification, not just a
sorted count, while remaining fully explainable to a non-technical reviewer.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads instantly using a bundled 25,000-row random sample
(`data/sample_violations.csv`) so you can demo it immediately. To run the
full analysis:

1. Copy `jan_to_may_police_violation_anonymized791b166.csv` into the `data/` folder.
2. In the sidebar, switch **Data source** to "Custom CSV path" and point it
   at that file (default path already matches the suggested location).

First load of the full dataset takes ~20–25 seconds (H3 binning + KMeans
over ~300K rows); it's cached after that, including across sidebar
interactions that don't change the data source.

## Project structure

```
parking_intelligence/
├── app.py               # Streamlit dashboard (map, tables, drill-down, methodology)
├── data_pipeline.py      # Load/clean raw CSV, parse violation tags, severity weights, IST time features
├── hotspot_engine.py      # H3 binning, feature aggregation, scoring, KMeans tiering, trend & forecast
├── data/
│   └── sample_violations.csv   # 25k-row sample for instant demo
├── requirements.txt
└── README.md
```

## How the scoring works (also explained in the app's "Methodology" tab)

1. **H3 hex binning** (`hotspot_engine.add_hex_id`) — groups nearby violations
   into ~0.10 km² cells (resolution 9, adjustable via the sidebar), avoiding
   the inconsistency of free-text addresses.
2. **Severity weighting** (`data_pipeline.PARKING_SEVERITY_WEIGHTS`) — each
   violation tag gets a 0–3 weight for how much it typically blocks a lane
   or junction approach (e.g. *Double Parking* = 3, *Parking on Footpath* = 1).
   This is an explicit, editable assumption — recalibrate with a traffic
   engineer's input if available.
3. **Junction proximity** — share of a hotspot's violations recorded at a
   named junction (chokepoints cause disproportionate downstream effects).
4. **Busy-hour concentration** — share of violations during the hours that,
   *empirically in this dataset*, account for most enforcement activity.
   We deliberately did **not** hardcode a textbook rush-hour window (see
   caveat below).
5. **Persistence** — fraction of days in the observation window the
   hotspot was active (chronic vs. one-off).

These combine into a 0–100 **Congestion Impact Score**:
`CIS = 40% severity + 20% junction proximity + 20% busy-hour share + 20% persistence`

An unsupervised **KMeans** model (on the raw underlying features, not just
the CIS) then assigns each hotspot to a **Critical / High / Medium / Low**
tier, so a small but chronically obstructive spot can outrank a
high-volume-but-occasional one. A simple linear trend over weekly counts
flags **emerging hotspots** and produces a 2-week forward forecast for
patrol planning.

## Honest limitations (worth stating to judges)

- **No live traffic/speed data** in this dataset — "congestion impact" is a
  domain-informed proxy built from the enforcement record itself, not a
  measured delay. We say this explicitly rather than implying otherwise.
- **Timestamp caveat**: `created_datetime`'s hourly distribution doesn't
  follow a typical commute curve, suggesting it reflects when an officer/
  device logged the record rather than a confirmed live-detection time.
  We handle this by deriving "busy hours" empirically from the data instead
  of assuming rush hours, and we flag this directly in the dashboard.
- H3 resolution is a tunable precision/noise trade-off, exposed as a slider.

## Where this plugs into the bigger hackathon picture

This module is built to interoperate with the other two problem statements
in this hackathon (see the Methodology tab for the full pitch):
- **Event-driven congestion (ASTRAM data)** — event proximity could
  temporarily re-rank hotspots during rallies/festivals/construction.
- **CV-based violation detection** — live camera detections could feed this
  engine in near-real-time instead of after-the-fact records, removing the
  timestamp-reliability caveat above.
