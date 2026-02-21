# Stories Coffee — Decision-Robust Retail Analytics

**Team:** Celine Sadaka, Raoul Saber, Zeina Hammound

---

A **retail analytics system** for Stories Coffee (Lebanon) built from raw POS data (2025 full year + January 2026 snapshot). The system is designed for **small-sample constraints**: limited branch count, high variance, and the need to prioritise actions rather than chase point-forecast accuracy. It emphasises **decision robustness**—ranking stability, precision targeting, scenario-based offer economics, and validation (LOOCV, baselines, bootstrap)—so that outputs support resource allocation and intervention design without overclaiming.

**What it does:** Cleans and canonicalises POS exports; builds branch and product KPIs, seasonality, and anomaly views; runs a validated ML layer (ridge regression with LOOCV, bounded predictions, reduced-feature variant selection by rank stability); and produces targeting and bundle recommendations with low/base/high scenario economics and similarity/support guardrails. An interactive dashboard and static reports surface KPIs, stability metrics, and decision precision. Tests and CI enforce schema and business invariants on every run.

**Audience:** Technical hackathon jurors, recruiters, or internal stakeholders who care about reproducible pipelines, honest validation, and decision-oriented design rather than headline accuracy.

---

## Executive summary (6 sentences)

This project is a **decision-robust retail analytics engine** built under small-sample constraints. It prioritises **risk prioritisation over point forecasting**: the model ranks branches and targets interventions rather than publishing a single forecast. **Stability diagnostics**—bootstrap ranking consistency, feature sign stability, and reduced-feature variant comparison—make signal strength explicit and drive model choice. The **optimization engine** uses scenario-based offer economics and minimum similarity/support guardrails so recommendations are conservative and auditable. The pipeline is **reproducible** and runs under **CI**; one command regenerates all outputs and tests enforce schema and invariants. The result is a technically honest, decision-focused system suitable for evaluation or portfolio use.

---

## Architecture (overview)

```
Raw POS → Cleaning & canonicalization → Feature engineering
    → ML validation (LOOCV, bootstrap, baselines, bounded predictions)
    → Ranking stability & variant selection → Optimization engine (scenario offers, guardrails)
    → Dashboard + Reports     Tests + CI → outputs/tables, reports
```

A single script (`src/run_analysis.py`) runs the full flow. Details, ASCII diagram, and narrative are in [`docs/NARRATIVE_AND_POSITIONING.md`](docs/NARRATIVE_AND_POSITIONING.md).

---

## What this project delivers

- Cleaned, canonicalised POS data and branch/product KPIs
- Margin leakage detection (products and modifiers), seasonality, and anomaly flags
- Validated ML layer: LOOCV, baseline comparison, ranking stability, variant selection, feature stability diagnostics, decision precision (top-K)
- Scenario-based bundle recommendations with similarity and support guardrails; affinity sensitivity table
- Streamlit dashboard (KPIs, stability, precision, offer economics) and static reports
- Automated tests and GitHub Actions CI

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
# Install dependencies first if needed: pip3 install -r requirements.txt
# Then run (use python3 -m if the streamlit command is not found):
python3 -m streamlit run dashboard.py
```

### Run automated checks
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Continuous Integration

A GitHub Actions workflow runs on each push/PR:

- Rebuild analysis outputs (`python src/run_analysis.py`)
- Execute integrity tests (`python -m unittest discover -s tests -p "test_*.py" -v`)

Workflow file: `.github/workflows/ci.yml`
