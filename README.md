# Stories Coffee — Hackathon Submission

Business-focused data analysis for **Stories Coffee (Lebanon)**, built from raw POS exports covering:

- **Full year 2025**
- **January 2026 snapshot** (up to **22-Jan-2026**)

---

## Project Goal

Stories Coffee has strong operational data, but limited decision support.

This project answers a practical business question:

> **Which branches, products, and categories drive profit, where are the margin leaks, and what actions can increase profitability quickly?**

---

## What This Project Delivers

- ✅ Cleaned and normalized POS data from messy exports
- ✅ Branch-level and product-level profitability analysis
- ✅ Margin leakage detection (including modifiers)
- ✅ Seasonality and anomaly detection
- ✅ Jan 2026 vs Jan 2025 run-rate adjusted performance
- ✅ Branch segmentation (archetypes + playbook)
- ✅ ML forecasts for 2026
- ✅ Optimization outputs (bundle recommendations for target branches)
- ✅ Interactive Streamlit dashboard
- ✅ Executive-ready static visuals (PNG charts)

---

## Repository Structure

> Paths below are shown in both **relative repo format (recommended)** and the original local references where relevant.

### Core Files
- `data/raw/` — Raw CSV exports (provided by Stories)
- `src/run_analysis.py` — Reproducible ETL + KPI generation pipeline
- `dashboard.py` — Streamlit dashboard
- `src/make_visuals.py` — Static chart generation

### Outputs
- `outputs/tables/` — Cleaned datasets + KPI / EDA / ML / optimization tables
- `outputs/figures/` — Executive-ready PNG charts
- `reports/analysis_summary.md` — Technical findings summary
- `reports/EXECUTIVE_SUMMARY.md` — CEO-style executive brief draft

---

## Methodology

### 1) Data Cleaning & Parsing
- Parse **4 messy POS CSV files** with repeated headers/pages
- Remove repeated page headers and subtotal rows
- Standardize branch names (e.g. `Stories alay` → `Stories Aley`)

### 2) Revenue Correction
In some summary files, revenue is corrected using:

- `revenue_corrected = total_cost + total_profit`

### 3) KPI & Business Analysis
The pipeline generates KPIs for:

- **Branch performance** (annual + Jan YoY)
- **Run-rate adjusted Jan 2026 vs Jan 2025**
  - (Jan 2026 snapshot file ends on **22-Jan-2026**)
- **Product profit concentration**
- **High-volume / low-margin products**
- **Modifier margin leakage**
- **Category mix by branch**
- **Monthly seasonality index**
- **Branch-level + network-level anomaly detection**
- **Profit/revenue concentration (Pareto patterns)**
- **Location analysis segmentation** (branch archetypes + playbook)
- **Optimization / ML**
  - bundle offers for least-selling branches (bottom 40% by 2025 sales)
  - forecasting and branch performance prediction

---

## How to Run

### Run the full analysis pipeline
```bash
python3 src/run_analysis.py
```

### Generate static visuals
```bash
python3 src/make_visuals.py
```

### Launch dashboard
```bash
streamlit run dashboard.py
```
