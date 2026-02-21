# Stories Coffee Hackathon Submission

Business-focused data analysis for Stories Coffee (Lebanon), built from raw POS exports for **2025 full year + January 2026 snapshot**.

## Problem Statement
Stories has rich operational data but no decision framework.  
This project answers: **Which branches, products, and categories drive profit, where margin leaks exist, and what actions can increase profitability quickly?**

## Repository Structure

- `data/raw` — raw CSV exports (provided by Stories)
- `src/run_analysis.py` — reproducible ETL + KPI generation pipeline
- `outputs/tables` — cleaned datasets + KPI tables
- `outputs/figures` — static charts (from `make_visuals.py`)
- `reports/analysis_summary.md` — technical findings summary
- `reports/EXECUTIVE_SUMMARY.md` — CEO-style 2-page brief
- `dashboard.py` — Streamlit dashboard (run with `streamlit run dashboard.py`)

## Methodology

1. Parse four messy POS CSV files with repeated headers/pages.
2. Normalize branch names (`Stories alay` -> `Stories Aley`, etc.).
3. Correct revenue in summary files using:
   - `revenue_corrected = total_cost + total_profit`
4. Build business KPIs:
   - Branch performance (annual + Jan YoY)
   - Run-rate adjusted Jan 2026 vs Jan 2025 (since file snapshot date is 22-Jan-2026)
   - Product-level profit concentration
   - High-volume low-margin products
   - Modifier margin leakage
   - Category mix by branch
   - Monthly seasonality index
   - Branch-level and network-level anomaly detection
   - Profit/revenue concentration (Pareto patterns)
   - Location analysis segmentation (branch archetypes + playbook)
   - Optimization ML (bundle offers for low-sales/decline-risk branches)

## How To Run

**1. Clone and set up (first time)**

```bash
git clone https://github.com/YOUR_USERNAME/stories.git
cd stories
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the pipeline** (place raw POS CSVs in `data/raw` first)

```bash
python3 src/run_analysis.py
```

Outputs are written to `outputs/tables` and `reports/`.

## Key Findings (Highlights)

- Top 2025 branches by annual sales: **Ain El Mreisseh, Zalka, Khaldeh**.
- Category margins are strong overall (~70%+) but differ by branch mix.
- Frozen Yogurt and beverage groups dominate sales mix.
- Several modifier lines are structurally unprofitable (notably lactose-free replacements).
- June is the weakest month; August is the strongest (about **5.9x** June in total sales).
- Run-rate adjusted January 2026 shows mixed momentum:
  - Relative resilience in Le Mall and Centro Mall
  - Larger declines in Saida, LAU, and some legacy sites

## Reproducibility Notes

- No hardcoded absolute data source dependency inside logic; script expects files under `data/raw`.
- Pipeline tolerates repeated page headers and subtotal rows.
- Branch normalization and revenue correction are deterministic.

## Suggested Demo Flow (for judges)

1. Run the pipeline: `python3 src/run_analysis.py`
2. Open `reports/EXECUTIVE_SUMMARY.md`
3. Walk through key tables: `kpi_january_yoy_by_branch.csv`, `kpi_branch_category_mix_margin.csv`, `kpi_high_volume_low_margin_products.csv`, `kpi_modifiers_margin.csv`
4. Optional: run the dashboard: `streamlit run dashboard.py`

## Streamlit Dashboard

```bash
streamlit run dashboard.py
```

(Uses the project venv if activated; otherwise run with `python dashboard.py` and it will launch Streamlit.)

The dashboard reads from `outputs/tables`. If tables are missing, run `python3 src/run_analysis.py` first.

## Generate Static Visuals

```bash
python src/make_visuals.py
```

Charts are saved under `outputs/figures/` (seasonality, top_branches, top_products_profit, low_margin_products, branch_mix_vs_margin, network_anomalies, network_forecast_2026, branch_clusters, branch_archetype_mix, target_branches_priority, offer_uplift, etc.).

## Outputs Summary

- **KPIs:** `outputs/tables/kpi_*.csv` (January YoY, branch mix, top products, low margin, modifiers, etc.)
- **EDA:** `outputs/tables/eda_*.csv` (seasonality, anomalies, concentration)
- **ML:** `outputs/tables/ml_*.csv` (forecasts, performance prediction, clusters, archetypes, playbook)
- **Optimization:** `outputs/tables/opt_*.csv` (target branches, pair affinity, bundle recommendations)
- **Reports:** `reports/analysis_summary.md`, `reports/EXECUTIVE_SUMMARY.md`

Model note: product-level monthly values in `estimated_branch_product_month.csv` are **estimated** by distributing annual branch-product totals using each branch's monthly sales share.
