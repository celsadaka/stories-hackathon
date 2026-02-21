#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def ensure_runtime_or_reexec() -> None:
    try:
        import matplotlib.pyplot as plt_mod
        import pandas as pd_mod
        return plt_mod, pd_mod
    except ModuleNotFoundError as exc:
        # If user launched with system python (e.g. Code Runner), transparently rerun in project venv.
        if os.path.exists(VENV_PY) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PY):
            sys.stderr.write(
                f"Missing dependency in current interpreter ({sys.executable}): {exc.name}.\n"
                f"Re-running with project venv: {VENV_PY}\n"
            )
            os.execv(VENV_PY, [VENV_PY, __file__, *sys.argv[1:]])

        missing = exc.name or "required package"
        sys.stderr.write(
            f"Missing dependency: {missing}.\n"
            "Run with project venv: /Users/celinesadaka/Desktop/stories/.venv/bin/python src/make_visuals.py\n"
            "Or install deps: /Users/celinesadaka/Desktop/stories/.venv/bin/pip install -r requirements.txt\n"
        )
        raise


plt, pd = ensure_runtime_or_reexec()

TABLES = os.path.join(ROOT, "outputs", "tables")
FIGURES = os.path.join(ROOT, "outputs", "figures")

os.makedirs(FIGURES, exist_ok=True)


def save_seasonality() -> str:
    df = pd.read_csv(os.path.join(TABLES, "kpi_2025_monthly_seasonality.csv"))

    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], df["sales_2025"], marker="o")
    plt.xticks(rotation=45)
    plt.title("2025 Monthly Sales Seasonality")
    plt.tight_layout()

    out = os.path.join(FIGURES, "seasonality.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_top_branches() -> str:
    df = pd.read_csv(os.path.join(TABLES, "kpi_top_branches_2025_sales.csv"))

    plt.figure(figsize=(8, 5))
    plt.barh(df["branch"], df["total_by_year"])
    plt.title("Top Branches by 2025 Sales")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = os.path.join(FIGURES, "top_branches.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_top_products_profit() -> str:
    df = pd.read_csv(os.path.join(TABLES, "kpi_top_products_by_profit.csv")).head(10)

    plt.figure(figsize=(10, 5))
    plt.bar(df["product_desc"], df["profit"])
    plt.xticks(rotation=70)
    plt.title("Top Products by Profit")
    plt.tight_layout()

    out = os.path.join(FIGURES, "top_products_profit.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_low_margin_products() -> str:
    df = pd.read_csv(os.path.join(TABLES, "kpi_high_volume_low_margin_products.csv")).head(10).copy()
    df = df.sort_values("margin_pct")

    # Extreme outliers can flatten the rest of bars (e.g., <-1000%).
    # Clip display only, but annotate each bar with true margin.
    clip_floor = -100
    df["display_margin"] = df["margin_pct"].clip(lower=clip_floor)

    colors = ["#d73027" if v < 0 else "#f46d43" for v in df["margin_pct"]]

    plt.figure(figsize=(11, 6))
    bars = plt.barh(df["product_desc"], df["display_margin"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlim(clip_floor - 10, max(65, float(df["display_margin"].max()) + 5))
    plt.title("High Volume - Low Margin Products (display clipped at -100%)")
    plt.xlabel("Margin %")

    for bar, true_margin in zip(bars, df["margin_pct"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        label = f"{true_margin:.2f}%"
        plt.text(x + 1, y, label, va="center", fontsize=8)

    plt.tight_layout()

    out = os.path.join(FIGURES, "low_margin_products.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_branch_mix_vs_margin() -> str:
    df = pd.read_csv(os.path.join(TABLES, "kpi_branch_category_mix_margin.csv"))

    plt.figure(figsize=(10, 7))
    plt.scatter(df["beverage_share_pct"], df["overall_margin_pct"])

    plt.xlabel("Beverage Share %")
    plt.ylabel("Overall Margin %")
    plt.title("Branch Beverage Mix vs Margin")

    for _, row in df.iterrows():
        plt.text(row["beverage_share_pct"], row["overall_margin_pct"], row["branch"], fontsize=7)

    plt.tight_layout()

    out = os.path.join(FIGURES, "branch_mix_vs_margin.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out



def save_product_profit_pareto() -> str:
    df = pd.read_csv(os.path.join(TABLES, "eda_product_profit_concentration.csv")).head(60)

    plt.figure(figsize=(10, 5))
    plt.plot(df["rank"], df["cumulative_profit_share_pct"], marker="o")
    plt.axhline(80, linestyle="--", linewidth=1)
    plt.title("Product Profit Pareto (Cumulative Share)")
    plt.xlabel("Product Rank")
    plt.ylabel("Cumulative Profit Share %")
    plt.tight_layout()

    out = os.path.join(FIGURES, "product_profit_pareto.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_network_anomalies() -> str:
    df = pd.read_csv(os.path.join(TABLES, "eda_network_month_anomalies_2025.csv"))

    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], df["sales_2025"], marker="o", label="Sales")
    flagged = df[df["anomaly_flag"] == 1]
    if not flagged.empty:
        plt.scatter(flagged["month"], flagged["sales_2025"], color="red", s=60, label="Flagged")

    plt.xticks(rotation=45)
    plt.title("Network Monthly Sales with Anomaly Flags (2025)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIGURES, "network_anomalies.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_network_forecast() -> str:
    df = pd.read_csv(os.path.join(TABLES, "ml_network_monthly_forecast_2026.csv"))

    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], df["sales_2025"], marker="o", label="2025 Actual")
    plt.plot(df["month"], df["forecast_sales_2026"], marker="o", label="2026 Forecast")
    plt.xticks(rotation=45)
    plt.title("Network Monthly Sales: 2025 vs 2026 Forecast")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIGURES, "network_forecast_2026.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_branch_cluster_map() -> str:
    df = pd.read_csv(os.path.join(TABLES, "ml_branch_clusters.csv"))

    plt.figure(figsize=(10, 6))
    labels = sorted(df["cluster_label"].dropna().unique().tolist())
    for label in labels:
        d = df[df["cluster_label"] == label]
        plt.scatter(d["beverage_share_pct"], d["overall_margin_pct"], s=40, label=label)

    for _, row in df.iterrows():
        plt.annotate(
            str(row["branch"]),
            (row["beverage_share_pct"], row["overall_margin_pct"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
            alpha=0.85,
        )

    plt.xlabel("Beverage Share %")
    plt.ylabel("Overall Margin %")
    plt.title("Branch Segmentation (K-Means): Mix vs Margin")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = os.path.join(FIGURES, "branch_clusters.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_target_branches() -> str:
    df = pd.read_csv(os.path.join(TABLES, "opt_target_branches.csv"))
    df = df[df["target_flag"] == 1].sort_values("priority_score", ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    plt.barh(df["branch"], df["priority_score"])
    plt.title("Target Branch Priority (Low Sales + Decline Risk)")
    plt.xlabel("Priority Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = os.path.join(FIGURES, "target_branches_priority.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_optimization_offers() -> str:
    df = pd.read_csv(os.path.join(TABLES, "opt_branch_bundle_recommendations.csv"))
    if df.empty:
        plt.figure(figsize=(6, 4))
        plt.title("No Optimization Offers Generated")
        plt.tight_layout()
        out = os.path.join(FIGURES, "offer_uplift.png")
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    df = df.sort_values("estimated_incremental_profit", ascending=False).head(15)
    labels = df["branch"] + " | " + df["anchor_product"].str.slice(0, 18) + " + " + df["pair_product"].str.slice(0, 18)

    plt.figure(figsize=(11, 7))
    plt.barh(labels, df["estimated_incremental_profit"])
    plt.title("Top Bundle Offers by Estimated Incremental Profit")
    plt.xlabel("Estimated Incremental Profit")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = os.path.join(FIGURES, "offer_uplift.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_archetype_mix() -> str:
    df = pd.read_csv(os.path.join(TABLES, "ml_branch_archetypes.csv"))
    counts = df["cluster_label"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    plt.barh(counts.index, counts.values)
    plt.title("Branch Archetype Distribution")
    plt.xlabel("Number of Branches")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    out = os.path.join(FIGURES, "branch_archetype_mix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def main() -> None:
    created = [
        save_seasonality(),
        save_top_branches(),
        save_top_products_profit(),
        save_low_margin_products(),
        save_branch_mix_vs_margin(),
        save_product_profit_pareto(),
        save_network_anomalies(),
        save_network_forecast(),
        save_branch_cluster_map(),
        save_archetype_mix(),
        save_target_branches(),
        save_optimization_offers(),
    ]

    print("Saved visual files:")
    for p in created:
        print(f"- {p}")


if __name__ == "__main__":
    main()
