# Narrative and Positioning

## 1. Tight Narrative (Problem → Why Hard → Why Not Accuracy → Decision Robustness → What’s Different)

Retailers like Stories Coffee have rich POS data but often lack a clear way to turn it into prioritised decisions: which branches need attention first, which products are leaking margin, and which offers are worth rolling out. The real-world problem is **resource allocation under uncertainty**—not forecasting the future to two decimal places. Small-sample retail analytics is hard because branch count is limited, seasonality and one-off events dominate short-run variance, and any model trained on a dozen or so branches is inherently unstable. Prediction accuracy (RMSE, R²) is therefore not the main objective: point forecasts would be over-interpreted and are easily beaten by a naive baseline. This system instead focuses on **decision robustness**: whether the *ordering* of branches by risk is stable across resamples (ranking stability), whether acting on the top-K predicted-decline branches actually targets real decliners (precision/recall), and whether offer economics are expressed as low/base/high scenarios so that upside is never presented as a single number. What makes it different from “just running regression” is the full validation layer—LOOCV, bootstrap, baseline comparisons, reduced-feature variant selection by rank correlation—and the explicit guardrails: bounded predictions, affinity thresholds, and sensitivity tables so that both the model and the optimization engine are auditable and tunable without overclaiming.

---

## 2. Executive Positioning (6 Sentences)

This project is a **decision-robust retail analytics engine** built under small-sample constraints for Stories Coffee (Lebanon). It prioritises **risk prioritisation over point forecasting**: the model is used to rank branches and target interventions, not to publish a single “January YoY” number. **Stability diagnostics**—bootstrap ranking consistency, feature sign stability, and reduced-feature variant comparison—make it clear when the signal is weak and which model variant is chosen for decisions. The **optimization engine** uses scenario-based offer economics (low/base/high) and minimum similarity and support guardrails so that bundle recommendations are conservative and interpretable. The pipeline is **reproducible** and runs under **CI**: one command regenerates all outputs, and automated tests enforce schema and business invariants. The result is a technically honest, decision-oriented system suitable for hackathon evaluation or portfolio demonstration.

---

## 3. Architecture Diagram

### ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RAW POS INPUTS                                                             │
│  (4 CSV exports: 2025 full year + Jan 2026 snapshot)                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLEANING & CANONICALIZATION                                                 │
│  • Parse repeated headers / subtotals   • Branch name normalization          │
│  • Revenue correction (cost + profit)  • Clean tables → outputs/tables      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                                         │
│  • Branch mix (margin %, beverage share)   • Seasonality (CV, peak/trough)   │
│  • Anomaly counts   • Group/dominant category   • Takeaway vs table share     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ML VALIDATION LAYER                                                         │
│  • Ridge regression (bounded predictions)   • LOOCV RMSE & Spearman          │
│  • Naive baseline comparison                • Bootstrap coefficient variance │
│  • P90 error band → prediction intervals                                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RANKING STABILITY & VARIANT SELECTION                                       │
│  • Bootstrap rankings → mean pairwise Spearman, top-5 risk overlap          │
│  • Reduced-feature models (top-2, top-3) → LOOCV comparison                  │
│  • Model selection by Spearman (stability), not RMSE alone                   │
│  • Feature stability diagnostics (sign variance across bootstrap)            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OPTIMIZATION ENGINE                                                         │
│  • Target branches: bottom 40% by sales (configurable)                        │
│  • Product affinity: cosine similarity + min similarity & min support         │
│  • Scenario-based offers: low / base / high incremental profit per offer      │
│  • Affinity sensitivity table (pairs passing per threshold grid)             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                             ▼
┌──────────────────────────────┐   ┌──────────────────────────────────────────┐
│  DASHBOARD LAYER             │   │  TESTS + CI                                │
│  • Streamlit app             │   │  • Schema & invariant tests (unittest)     │
│  • KPIs, forecasts,          │   │  • Required tables/columns                 │
│    ranking stability,       │   │  • GitHub Actions: pipeline + tests on push  │
│    variant comparison,      │   │  • No new dependencies                     │
│    decision precision,      │   └──────────────────────────────────────────┘
│    offer scenario economics │   outputs/tables/*.csv  reports/*.md
└──────────────────────────────┘
```

### Short Explanation

Data flows from raw POS CSVs through a single pipeline: cleaning and canonicalization produce consistent branch and product tables; feature engineering builds mix, seasonality, and anomaly inputs. The ML layer fits a ridge model with LOOCV and baseline comparison and outputs bounded predictions and intervals. Ranking stability (bootstrap) and variant selection (reduced-feature LOOCV) decide which model is used for prioritisation; feature stability diagnostics flag unreliable coefficients. The optimization engine targets the bottom 40% of branches by sales, builds product pairs under similarity and support guardrails, and attaches scenario-based (low/base/high) economics to each offer. The dashboard surfaces KPIs, stability metrics, decision precision, and offer economics; tests and CI ensure reproducibility and schema integrity on every run.

---

## 4. Dashboard framing (titles + captions)

Used in the Streamlit dashboard for executive-ready presentation.

| Section | Title | Caption |
|--------|--------|--------|
| **Ranking stability** | Ranking stability | How consistent is the risk ordering across bootstrap resamples? Higher Spearman and top-5 overlap mean more stable prioritisation for resource allocation. |
| **Model variant comparison** | Model variant comparison | Full vs reduced-feature models. The selected variant is chosen by ranking stability (Spearman), not RMSE alone, to support more reliable prioritisation. |
| **Decision precision (Top-K)** | Decision precision (top-K decline) | If we act on the top 5 predicted-decline branches: what share actually declined (precision) and what share of all decliners did we capture (recall)? |
| **Offer scenario economics** | Optimization: Menu Engineering + Offer Engine | Scenario-based offer economics: each recommendation shows low / base / high incremental profit. Offers are restricted by similarity and support guardrails so upside is interpretable and conservative. |
