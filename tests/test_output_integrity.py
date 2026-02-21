import math
import re
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

sys.path.insert(0, str(ROOT / "src"))
from run_analysis import (  # noqa: E402
    TARGETING_BUDGET_PERCENTILE,
    YOY_CLIP_MAX,
    YOY_CLIP_MIN,
    canonical_branch,
)


REQUIRED_TABLES = {
    "clean_monthly_sales.csv",
    "clean_category_summary.csv",
    "clean_product_profit.csv",
    "clean_sales_group.csv",
    "kpi_january_yoy_by_branch.csv",
    "ml_branch_performance_prediction.csv",
    "ml_branch_performance_model_metrics.csv",
    "ml_branch_ranking_stability.csv",
    "ml_branch_model_variant_comparison.csv",
    "ml_feature_stability_diagnostics.csv",
    "ml_decision_precision_metrics.csv",
    "ml_forecast_backtest_metrics_2025.csv",
    "opt_target_branches.csv",
    "opt_branch_bundle_recommendations.csv",
    "opt_offer_scenario_summary.csv",
    "opt_targeting_sensitivity.csv",
    "opt_affinity_sensitivity.csv",
}

# Tables that may be empty (e.g. no training data)
MAY_BE_EMPTY_TABLES = {"ml_feature_stability_diagnostics.csv", "ml_decision_precision_metrics.csv"}


REQUIRED_COLUMNS = {
    "clean_monthly_sales.csv": {"year", "branch", "month", "month_num", "sales"},
    "clean_category_summary.csv": {
        "branch",
        "category",
        "qty",
        "revenue_corrected",
        "total_cost",
        "total_profit",
        "cost_pct",
        "profit_pct",
    },
    "clean_product_profit.csv": {
        "branch",
        "service_type",
        "category",
        "section",
        "product_desc",
        "qty",
        "revenue",
        "total_cost",
        "total_profit",
        "cost_pct",
        "profit_pct",
    },
    "clean_sales_group.csv": {"branch", "division", "group_name", "description", "qty", "sales"},
    "kpi_january_yoy_by_branch.csv": {
        "branch",
        "jan_2025",
        "jan_2026",
        "jan_2026_runrate",
        "jan_yoy_pct",
        "jan_runrate_yoy_pct",
    },
    "ml_branch_performance_prediction.csv": {
        "branch",
        "predicted_jan_runrate_yoy_pct",
        "prediction_interval_low",
        "prediction_interval_high",
    },
    "ml_branch_performance_model_metrics.csv": {
        "model",
        "training_samples",
        "r2_in_sample",
        "rmse_yoy_pct",
        "loocv_rmse_yoy_pct",
        "loocv_baseline_rmse_yoy_pct",
        "prediction_bounds",
    },
    "ml_forecast_backtest_metrics_2025.csv": {"model", "samples", "mae", "rmse", "smape_pct"},
    "opt_target_branches.csv": {
        "branch",
        "priority_score",
        "target_rank",
        "target_flag",
        "low_sales_intensity",
        "decline_intensity",
    },
    "opt_branch_bundle_recommendations.csv": {
        "branch",
        "anchor_product",
        "pair_product",
        "scenario_low_incremental_profit",
        "scenario_base_incremental_profit",
        "scenario_high_incremental_profit",
        "expected_incremental_profit",
        "uncertainty_ratio",
    },
    "opt_offer_scenario_summary.csv": {"scenario", "total_incremental_revenue", "total_incremental_profit"},
    "opt_targeting_sensitivity.csv": {
        "budget_percentile",
        "target_count",
        "sales_cutoff_min_selected",
        "avg_priority_score_selected",
    },
    "ml_branch_ranking_stability.csv": {
        "branch",
        "mean_rank",
        "rank_std",
        "in_top5_risk_count",
        "in_top5_risk_pct",
        "mean_pairwise_spearman",
        "mean_top5_overlap",
    },
    "ml_branch_model_variant_comparison.csv": {
        "model",
        "features",
        "loocv_rmse_yoy_pct",
        "loocv_spearman_rank_corr",
        "selected_by_stability",
    },
    "ml_feature_stability_diagnostics.csv": {
        "feature",
        "coefficient_mean",
        "coefficient_std",
        "sign_stable",
        "pct_positive_bootstrap",
    },
    "ml_decision_precision_metrics.csv": {
        "k",
        "n_targeted",
        "n_actually_declined_in_targeted",
        "total_actually_declined",
        "precision",
        "recall",
    },
    "opt_affinity_sensitivity.csv": {"min_similarity", "min_support", "pairs_passing"},
}


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing expected table: {path}")
    return pd.read_csv(path)


class TestCanonicalization(unittest.TestCase):
    def test_branch_aliases(self) -> None:
        self.assertEqual(canonical_branch("Stories alay"), "Stories Aley")
        self.assertEqual(canonical_branch("Stories sin el fil"), "Stories Sin El Fil")
        self.assertEqual(canonical_branch("Stories."), "Stories Unknown")

    def test_branch_non_branch_rows(self) -> None:
        self.assertIsNone(canonical_branch("total"))
        self.assertIsNone(canonical_branch("Total By Branch:"))
        self.assertIsNone(canonical_branch("Not Stories Branch"))


class TestOutputIntegrity(unittest.TestCase):
    def test_required_tables_exist_and_non_empty(self) -> None:
        for name in sorted(REQUIRED_TABLES):
            path = TABLES / name
            self.assertTrue(path.exists(), f"Missing table: {path}")
            df = pd.read_csv(path)
            if name not in MAY_BE_EMPTY_TABLES:
                self.assertGreater(len(df), 0, f"Empty table: {name}")

    def test_required_columns_present(self) -> None:
        for name, cols in REQUIRED_COLUMNS.items():
            path = TABLES / name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if len(df) == 0 and name in MAY_BE_EMPTY_TABLES:
                continue
            missing = cols.difference(df.columns)
            self.assertFalse(missing, f"{name} missing columns: {sorted(missing)}")

    def test_no_repeated_page_header_artifacts_in_clean_tables(self) -> None:
        pattern = re.compile(r"Page\s+\d+\s+of\s+\d+", flags=re.IGNORECASE)
        for path in sorted(TABLES.glob("clean_*.csv")):
            df = pd.read_csv(path)
            text_cols = [c for c in df.columns if df[c].dtype == object]
            for col in text_cols:
                hit = df[col].astype(str).str.contains(pattern, na=False).any()
                self.assertFalse(hit, f"Page artifact found in {path.name}:{col}")

    def test_monthly_sales_unique_branch_year_month(self) -> None:
        df = read_table("clean_monthly_sales.csv")
        dup = df.duplicated(subset=["year", "branch", "month_num"], keep=False)
        self.assertFalse(dup.any(), "Duplicate (year, branch, month_num) rows in clean_monthly_sales.csv")
        self.assertTrue(df["month_num"].between(1, 12).all(), "month_num values must be in [1, 12]")

    def test_category_revenue_identity(self) -> None:
        df = read_table("clean_category_summary.csv")
        diff = (df["revenue_corrected"] - (df["total_cost"] + df["total_profit"]))
        self.assertLessEqual(diff.abs().max(), 1e-6, "revenue_corrected must equal total_cost + total_profit")

    def test_predictions_are_bounded(self) -> None:
        df = read_table("ml_branch_performance_prediction.csv")
        self.assertTrue(df["predicted_jan_runrate_yoy_pct"].between(YOY_CLIP_MIN, YOY_CLIP_MAX).all())
        self.assertTrue(df["prediction_interval_low"].between(YOY_CLIP_MIN, YOY_CLIP_MAX).all())
        self.assertTrue(df["prediction_interval_high"].between(YOY_CLIP_MIN, YOY_CLIP_MAX).all())
        self.assertTrue((df["prediction_interval_low"] <= df["predicted_jan_runrate_yoy_pct"]).all())
        self.assertTrue((df["predicted_jan_runrate_yoy_pct"] <= df["prediction_interval_high"]).all())

    def test_model_metrics_schema_and_bounds(self) -> None:
        df = read_table("ml_branch_performance_model_metrics.csv")
        row = df.iloc[0]
        self.assertIn("loocv_rmse_yoy_pct", df.columns)
        self.assertIn("loocv_baseline_rmse_yoy_pct", df.columns)
        self.assertGreaterEqual(float(row["training_samples"]), 1)
        self.assertEqual(str(row["prediction_bounds"]), f"[{YOY_CLIP_MIN:.0f},{YOY_CLIP_MAX:.0f}]")

    def test_forecast_backtest_model_set(self) -> None:
        df = read_table("ml_forecast_backtest_metrics_2025.csv")
        models = set(df["model"].tolist())
        self.assertSetEqual(models, {"trend_walkforward", "naive_last_month", "moving_avg_3"})
        self.assertTrue((df["samples"] > 0).all())

    def test_offers_only_for_target_branches(self) -> None:
        target_df = read_table("opt_target_branches.csv")
        offer_df = read_table("opt_branch_bundle_recommendations.csv")

        target_set = set(target_df.loc[target_df["target_flag"] == 1, "branch"].dropna().unique())
        offer_set = set(offer_df["branch"].dropna().unique())

        self.assertGreater(len(offer_set), 0, "No offer branches found")
        unexpected = sorted(offer_set - target_set)
        self.assertFalse(unexpected, f"Offers found for non-target branches: {unexpected}")

    def test_target_selection_budget_and_rank_consistency(self) -> None:
        df = read_table("opt_target_branches.csv").copy()
        sales_df = df[df["sales_2025_total"] > 0]
        expected_count = max(1, int(math.ceil(len(sales_df) * TARGETING_BUDGET_PERCENTILE)))
        actual_count = int((df["target_flag"] == 1).sum())
        self.assertEqual(actual_count, expected_count)

        selected = df[df["target_flag"] == 1]
        non_selected = df[df["target_flag"] == 0]
        self.assertTrue((selected["target_rank"] <= expected_count).all())
        if not non_selected.empty:
            self.assertTrue((non_selected["target_rank"] > expected_count).all())

    def test_offer_scenarios_are_monotonic(self) -> None:
        df = read_table("opt_branch_bundle_recommendations.csv")
        self.assertTrue((df["scenario_low_incremental_revenue"] <= df["scenario_base_incremental_revenue"]).all())
        self.assertTrue((df["scenario_base_incremental_revenue"] <= df["scenario_high_incremental_revenue"]).all())
        self.assertTrue((df["scenario_low_incremental_profit"] <= df["scenario_base_incremental_profit"]).all())
        self.assertTrue((df["scenario_base_incremental_profit"] <= df["scenario_high_incremental_profit"]).all())
        self.assertTrue((df["scenario_low_incremental_profit"] <= df["expected_incremental_profit"]).all())
        self.assertTrue((df["expected_incremental_profit"] <= df["scenario_high_incremental_profit"]).all())
        self.assertTrue((df["uncertainty_ratio"] >= 0).all())

    def test_ranking_stability_has_summary(self) -> None:
        df = read_table("ml_branch_ranking_stability.csv")
        summary = df[df["branch"].astype(str).str.contains("summary", na=False)]
        self.assertGreater(len(summary), 0, "ml_branch_ranking_stability must contain (summary) row")
        if "mean_pairwise_spearman" in df.columns and summary.iloc[0].get("mean_pairwise_spearman") not in (None, ""):
            self.assertGreaterEqual(float(summary.iloc[0]["mean_pairwise_spearman"]), -1.0)
            self.assertLessEqual(float(summary.iloc[0]["mean_pairwise_spearman"]), 1.0)

    def test_model_variant_comparison_has_selected(self) -> None:
        df = read_table("ml_branch_model_variant_comparison.csv")
        self.assertIn("selected_by_stability", df.columns)
        selected_count = df["selected_by_stability"].astype(str).str.lower().eq("true").sum()
        self.assertEqual(selected_count, 1, "Exactly one model must be selected_by_stability")

    def test_affinity_sensitivity_grid(self) -> None:
        df = read_table("opt_affinity_sensitivity.csv")
        self.assertGreater(len(df), 0)
        self.assertIn("min_similarity", df.columns)
        self.assertIn("min_support", df.columns)
        self.assertIn("pairs_passing", df.columns)
        self.assertTrue((df["pairs_passing"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
