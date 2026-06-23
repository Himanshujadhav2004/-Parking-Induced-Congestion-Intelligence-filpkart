# Parking-Induced Congestion Intelligence

**Gridlock Hackathon 2.0 — Round 2 Command Center Prototype**
Problem statement: *Poor Visibility on Parking-Induced Congestion*

> How can AI-driven parking intelligence detect illegal parking hotspots, quantify their physics and monetary impact on traffic flow, and optimize targeted enforcement?

---

## What this is

An interactive **Command Center Dashboard** that transforms ~298K raw illegal-parking enforcement records into a spatiotemporal, physics-informed, map-based enforcement priority platform. 

Rather than relying on reactive patrols or basic violation counts, this prototype layers an unsupervised **KMeans clustering model** with **traffic flow theory** and **economic delay benchmarks** to provide traffic commanders with clear, explainable, and actionable insights.

---

## Key Features & Improvements (v2 vs. v1)

### 1. 🌓 Dual Command Center Themes (Dark & Light)
- **Dark Theme (Default)**: A premium, dark-mode design system tailored for traffic control rooms with neon metric cards and elevation heatmaps.
- **Light Theme**: A clean, high-contrast, light-grey style with dark grey typography and an adaptive light map basemap for day-shift operators.
- **Material Symbols**: Integrated vector Google Material Symbols across headers, sidebars, tabs, and buttons, replacing basic emoji markers.

### 2. ⚡ Physics-Informed Capacity Loss (Greenshields + LWR Theory)
Calculates traffic flow impacts dynamically using classic traffic engineering formulas:
- **Greenshields fundamental diagram (1935)**: Estimates effective bottleneck speed: $u = u_f \times (1 - \frac{k}{k_j})$ where density ($k$) is derived from violation obstruction severity.
- **Lane Capacity Loss %**: Quantifies the percentage of structural carriageway throughput lost.
- **LWR Shockwave Queue (km)**: Models upstream shockwave queue propagation based on Lighthill-Whitham-Richards traffic flow theory.

### 3. 💰 Economic Delay Costing (TERI Benchmark)
Translates congestion delays into monetary terms:
- Applies the **TERI 2018 benchmark** (₹50 per vehicle-hour delay) to calculate daily and annual delay costs per hotspot.
- Displays city-wide daily delay costs in a dedicated KPI card to help prioritize enforcement based on economic losses.

### 4. 🚔 Greedy Patrol Schedule Optimizer
- Solve the resource allocation problem using a greedy approximation: Critical hotspots require **2 officer-hours**; all other tiers require **1 officer-hour**.
- Input the number of available officers and shift duration to instantly build the most efficient patrol schedule.
- Compares the results against a random patrol baseline to report percentage efficiency gains.

### 5. ⚖️ What-If CIS Formula Weight Adjuster
- Adjust the relative weights of the four Congestion Impact Score components (Severity, Junction Proximity, Busy-Hour Share, and Persistence) in real-time.
- Visualizes position shifts in the top 25 priority list (Rank $\Delta$) and flags "biggest movers" (shifting >5 spots) instantly without reloading the dataset.

### 6. 🕐 Hour-of-Day Violation Density Map
- Drag the hourly slider (0–23 IST) in the *Temporal Patterns* tab to animate spatiotemporal violation concentrations.
- Uses a yellow-to-red density gradient on the theme-aware map style.

---

## Quick Start

### Installation
Ensure you have Python 3.10+ installed, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Data Options
By default, the dashboard runs instantly using a bundled **25,000-row sample** (`data/sample_violations.csv`) for demonstration purposes.

To analyze the full dataset:
1. Place the full anonymized CSV (`jan_to_may_police_violation_anonymized791b166.csv`) inside the `data/` folder.
2. In the sidebar, select **Custom CSV path** and input the file path.
3. The initial load will take ~20 seconds to bin coordinates (H3) and fit KMeans, but all subsequent runs are cached.

---

## Project Structure

```
parking_intelligence/
├── app.py               # Streamlit Command Center (CSS styling, maps, components)
├── data_pipeline.py     # Clean raw CSV, parse tag arrays, compute severity weights
├── hotspot_engine.py    # H3 hex binning, Greenshields physics, TERI costing, KMeans, patrol optimizer
├── data/
│   └── sample_violations.csv   # 25k-row random sample for quick run
├── requirements.txt
└── README.md
```

---

## How the Congestion Impact Score (CIS) is Built

1. **H3 Spatial Binning**: Coordinates are binned into H3 hexagonal cells (~0.1 km², resolution 9 by default) to aggregate adjacent violations and bypass inconsistent text addresses.
2. **Obstruction Severity**: Weighted on a 0–3 scale based on obstruction potential (e.g. *Double Parking* = 3.0, *Footpath Parking* = 1.0).
3. **Junction Proximity**: Share of violations tagged at named junctions (chokepoints cause higher delay).
4. **Busy-Hour Share**: Portion of violations occurring during empirically busiest hours.
5. **Persistence**: Share of observation days with active violations.

$$\text{CIS} = 40\% \text{ Severity} + 20\% \text{ Junction} + 20\% \text{ Busy-Hour} + 20\% \text{ Persistence}$$

Unsupervised **KMeans Clustering** classifies the hotspots into **Critical / High / Medium / Low** priority tiers by examining the underlying features, rather than sorting by a simple count metric.

---

## Limitations (Candidly Stated)

- **Proxy-Based Congestion**: No live telemetry sensor data is present in the dataset. "Congestion impact" and queue propagation are computed proxies based on traffic flow theory.
- **Timestamp Caveat**: The violation logs represent when records were entered/saved by officers, rather than live-camera detection timestamps. Commute hours are derived empirically from the logs.
