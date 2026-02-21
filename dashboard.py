from pathlib import Path
import os
import subprocess
import sys

# Streamlit apps must be run with `streamlit run dashboard.py`, not `python dashboard.py`.
# If launched with plain Python, re-launch via Streamlit and exit.
if __name__ == "__main__" and os.environ.get("STREAMLIT_DASHBOARD_LAUNCHED") != "1":
    env = {**os.environ, "STREAMLIT_DASHBOARD_LAUNCHED": "1"}
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]],
            env=env,
        ).returncode
    )

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"


def load_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_num(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return "-"


st.set_page_config(page_title="Stories Coffee Dashboard", layout="wide")
st.title("Stories Coffee Dashboard")
st.caption("Data window: 2025 full year + January 2026 snapshot")

jan = load_csv("kpi_january_yoy_by_branch.csv")
season = load_csv("kpi_2025_monthly_seasonality.csv")
top_products = load_csv("kpi_top_products_by_profit.csv")
low_margin = load_csv("kpi_high_volume_low_margin_products.csv")
mix = load_csv("kpi_branch_category_mix_margin.csv")
groups = load_csv("kpi_group_sales_performance.csv")
modifiers = load_csv("kpi_modifiers_margin.csv")
branch_product_month = load_csv("estimated_branch_product_month.csv")
eda_branch_seasonality_profile = load_csv("eda_branch_seasonality_profile_2025.csv")
eda_branch_seasonality_summary = load_csv("eda_branch_seasonality_summary_2025.csv")
eda_branch_anomalies = load_csv("eda_branch_month_anomalies_2025.csv")
eda_network_anomalies = load_csv("eda_network_month_anomalies_2025.csv")
eda_product_concentration = load_csv("eda_product_profit_concentration.csv")
eda_branch_concentration = load_csv("eda_branch_revenue_concentration_2025.csv")
ml_network_forecast = load_csv("ml_network_monthly_forecast_2026.csv")
ml_branch_forecast = load_csv("ml_branch_monthly_forecast_2026.csv")
ml_branch_perf = load_csv("ml_branch_performance_prediction.csv")
ml_cluster = load_csv("ml_branch_clusters.csv")
ml_metrics = load_csv("ml_branch_performance_model_metrics.csv")
ml_branch_archetypes = load_csv("ml_branch_archetypes.csv")
ml_archetype_playbook = load_csv("ml_archetype_playbook.csv")
opt_target_branches = load_csv("opt_target_branches.csv")
opt_bundle_offers = load_csv("opt_branch_bundle_recommendations.csv")
opt_pair_affinity = load_csv("opt_product_pair_affinity.csv")

required = [
    ("kpi_january_yoy_by_branch.csv", jan),
    ("kpi_2025_monthly_seasonality.csv", season),
    ("kpi_top_products_by_profit.csv", top_products),
]
missing = [name for name, df in required if df.empty]
if missing:
    st.error("Missing KPI tables. Run the pipeline first.")
    st.code("python3 src/run_analysis.py")
    st.write("Missing:", ", ".join(missing))
    st.stop()

st.subheader("Executive Snapshot")
c1, c2, c3, c4 = st.columns(4)
jan_2025_total = jan["jan_2025"].fillna(0).sum()
jan_2026_total = jan["jan_2026"].fillna(0).sum()
runrate_total = jan["jan_2026_runrate"].fillna(0).sum() if "jan_2026_runrate" in jan.columns else jan_2026_total
yoy_runrate = ((runrate_total - jan_2025_total) / jan_2025_total * 100) if jan_2025_total > 0 else 0
c1.metric("Jan 2025 Total", fmt_num(jan_2025_total))
c2.metric("Jan 2026 MTD", fmt_num(jan_2026_total))
c3.metric("Jan 2026 Run-Rate", fmt_num(runrate_total))
c4.metric("Run-Rate YoY", f"{yoy_runrate:.1f}%")

st.subheader("Branch January Performance")
col_l, col_r = st.columns([2, 1])
with col_l:
    jan_plot = jan.sort_values("jan_2026", ascending=False).copy()
    fig = px.bar(
        jan_plot,
        x="branch",
        y=["jan_2025", "jan_2026", "jan_2026_runrate"],
        barmode="group",
        title="Jan 2025 vs Jan 2026 MTD vs Jan 2026 Run-Rate",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Sales", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
with col_r:
    if "jan_runrate_yoy_pct" in jan.columns:
        ranked = jan.sort_values("jan_runrate_yoy_pct", ascending=False)
        st.markdown("**Top Movers (Run-Rate YoY)**")
        st.dataframe(ranked[["branch", "jan_runrate_yoy_pct"]].head(8), use_container_width=True)
        st.markdown("**Bottom Movers (Run-Rate YoY)**")
        st.dataframe(ranked[["branch", "jan_runrate_yoy_pct"]].tail(8), use_container_width=True)

st.subheader("Seasonality (2025)")
fig_s = px.line(
    season,
    x="month",
    y="sales_2025",
    markers=True,
    title="Monthly Sales Curve",
)
fig_s.update_layout(xaxis_title="", yaxis_title="Sales")
st.plotly_chart(fig_s, use_container_width=True)

st.subheader("Exploratory Analysis: Patterns, Seasonality, Anomalies")

eda_left, eda_right = st.columns(2)
with eda_left:
    if not eda_product_concentration.empty:
        pareto = eda_product_concentration.head(60).copy()
        fig_pareto = px.line(
            pareto,
            x="rank",
            y="cumulative_profit_share_pct",
            markers=True,
            title="Product Profit Pareto (Cumulative Share)",
        )
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="red")
        fig_pareto.update_layout(xaxis_title="Product Rank", yaxis_title="Cumulative Profit Share %")
        st.plotly_chart(fig_pareto, use_container_width=True)

        top80 = pareto[pareto["cumulative_profit_share_pct"] >= 80]
        if not top80.empty:
            st.caption(f"~80% of positive product profit comes from top {int(top80.iloc[0]['rank'])} products.")
with eda_right:
    if not eda_branch_concentration.empty:
        bconc = eda_branch_concentration.copy()
        fig_bconc = px.line(
            bconc,
            x="rank",
            y="cumulative_revenue_share_pct",
            markers=True,
            title="Branch Revenue Concentration (2025)",
        )
        fig_bconc.add_hline(y=80, line_dash="dash", line_color="red")
        fig_bconc.update_layout(xaxis_title="Branch Rank", yaxis_title="Cumulative Revenue Share %")
        st.plotly_chart(fig_bconc, use_container_width=True)

if not eda_branch_seasonality_profile.empty:
    st.markdown("**Branch Seasonality Heatmap (2025 Index)**")
    season_pivot = (
        eda_branch_seasonality_profile.pivot_table(
            index="branch", columns="month_num", values="seasonality_index", aggfunc="mean"
        )
        .reindex(columns=list(range(1, 13)), fill_value=0)
        .fillna(0)
    )
    season_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season_pivot = season_pivot.loc[season_pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_season_heat = px.imshow(
        season_pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Branch x Month Seasonality Index (1.0 = Branch Average Month)",
    )
    fig_season_heat.update_layout(xaxis_title="Month", yaxis_title="Branch")
    st.plotly_chart(fig_season_heat, use_container_width=True)

anom_l, anom_r = st.columns([2, 1])
with anom_l:
    if not eda_branch_anomalies.empty:
        flagged = eda_branch_anomalies[eda_branch_anomalies["anomaly_flag"] == 1].copy()
        if flagged.empty:
            st.info("No branch-month anomalies were flagged by the current thresholds.")
        else:
            flagged["severity"] = flagged["robust_zscore"].abs()
            flagged["month"] = pd.Categorical(flagged["month"], categories=MONTH_ORDER, ordered=True)
            flagged = flagged.sort_values(["branch", "month"])

            fig_anom = px.scatter(
                flagged,
                x="month",
                y="branch",
                size="severity",
                color="robust_zscore",
                color_continuous_scale="RdBu_r",
                hover_data=["sales", "mom_pct", "anomaly_reason"],
                title="Flagged Branch-Month Anomalies (2025)",
            )
            fig_anom.update_layout(xaxis_title="Month", yaxis_title="Branch")
            st.plotly_chart(fig_anom, use_container_width=True)

with anom_r:
    if not eda_network_anomalies.empty:
        net = eda_network_anomalies.copy()
        net["month"] = pd.Categorical(net["month"], categories=MONTH_ORDER, ordered=True)
        net = net.sort_values("month")

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=net["month"], y=net["sales_2025"], mode="lines+markers", name="Sales"))
        flagged_net = net[net["anomaly_flag"] == 1]
        if not flagged_net.empty:
            fig_net.add_trace(
                go.Scatter(
                    x=flagged_net["month"],
                    y=flagged_net["sales_2025"],
                    mode="markers",
                    marker=dict(size=12, color="red"),
                    name="Flagged",
                )
            )
        fig_net.update_layout(
            title="Network Monthly Sales with Anomaly Flags",
            xaxis_title="Month",
            yaxis_title="Sales",
            legend_title="",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        if not eda_branch_seasonality_summary.empty:
            st.markdown("**Highest Volatility Branches (CV%)**")
            st.dataframe(
                eda_branch_seasonality_summary[["branch", "cv_pct", "peak_month", "trough_month"]].head(8),
                use_container_width=True,
            )

st.subheader("Predictive Modeling (ML Layer)")

ml_left, ml_right = st.columns([2, 1])
with ml_left:
    if not ml_network_forecast.empty:
        net_fc = ml_network_forecast.copy()
        net_fc["month"] = pd.Categorical(net_fc["month"], categories=MONTH_ORDER, ordered=True)
        net_fc = net_fc.sort_values("month")

        fig_fc = px.line(
            net_fc,
            x="month",
            y=["sales_2025", "forecast_sales_2026"],
            markers=True,
            title="Network Monthly Sales: 2025 Actual vs 2026 Forecast",
        )
        fig_fc.update_layout(xaxis_title="Month", yaxis_title="Sales", legend_title="")
        st.plotly_chart(fig_fc, use_container_width=True)

with ml_right:
    if not ml_metrics.empty:
        mrow = ml_metrics.iloc[0]
        st.metric("Model R2 (in-sample)", f"{float(mrow['r2_in_sample']):.2f}")
        st.metric("Model RMSE (YoY pts)", f"{float(mrow['rmse_yoy_pct']):.2f}")
        st.metric("Training Branches", f"{int(mrow['training_samples'])}")

    if not ml_branch_perf.empty:
        st.markdown("**Predicted Top Growth**")
        st.dataframe(
            ml_branch_perf[["branch", "predicted_jan_runrate_yoy_pct", "growth_probability"]].head(6),
            use_container_width=True,
        )
        st.markdown("**Predicted Highest Risk**")
        st.dataframe(
            ml_branch_perf[["branch", "predicted_jan_runrate_yoy_pct", "growth_probability"]].tail(6),
            use_container_width=True,
        )

if not ml_cluster.empty:
    fig_cluster = px.scatter(
        ml_cluster,
        x="beverage_share_pct",
        y="overall_margin_pct",
        size="annual_sales_2025",
        color="cluster_label",
        text="branch",
        hover_name="branch",
        hover_data=["cv_pct", "anomaly_count_2025", "predicted_jan_runrate_yoy_pct"],
        title="Branch Archetypes (K-Means): Mix vs Margin",
    )
    fig_cluster.update_traces(textposition="top center", textfont_size=9)
    fig_cluster.update_layout(xaxis_title="Beverage Share %", yaxis_title="Overall Margin %")
    st.plotly_chart(fig_cluster, use_container_width=True)

if not ml_branch_archetypes.empty:
    st.markdown("**Branch Archetype Assignment**")
    st.dataframe(
        ml_branch_archetypes[
            [
                "branch",
                "cluster_label",
                "annual_sales_2025",
                "overall_margin_pct",
                "beverage_share_pct",
                "dominant_group",
                "takeaway_share_pct",
                "cv_pct",
                "predicted_jan_runrate_yoy_pct",
            ]
        ].sort_values(["cluster_label", "annual_sales_2025"], ascending=[True, False]),
        use_container_width=True,
    )

if not ml_archetype_playbook.empty:
    st.markdown("**Archetype Playbook**")
    st.dataframe(ml_archetype_playbook, use_container_width=True)

if not ml_branch_forecast.empty:
    st.markdown("**Branch Forecast Explorer**")
    b_options = sorted(ml_branch_forecast["branch"].dropna().unique().tolist())
    b_sel = st.selectbox("Forecast Branch", b_options, key="ml_branch_sel")
    bdf = ml_branch_forecast[ml_branch_forecast["branch"] == b_sel].copy()
    bdf["month"] = pd.Categorical(bdf["month"], categories=MONTH_ORDER, ordered=True)
    bdf = bdf.sort_values("month")

    fig_bf = px.line(
        bdf,
        x="month",
        y=["sales_2025", "forecast_final"],
        markers=True,
        title=f"{b_sel}: 2025 Monthly Sales vs 2026 Forecast",
    )
    fig_bf.update_layout(xaxis_title="Month", yaxis_title="Sales", legend_title="")
    st.plotly_chart(fig_bf, use_container_width=True)

st.subheader("Optimization: Menu Engineering + Offer Engine")

opt_l, opt_r = st.columns([1, 2])
with opt_l:
    if not opt_target_branches.empty:
        target_only = opt_target_branches[opt_target_branches["target_flag"] == 1].copy()
        st.metric("Target Branches", int(len(target_only)))
        if "estimated_incremental_profit" in opt_bundle_offers.columns and not opt_bundle_offers.empty:
            st.metric("Offer Engine Profit Upside", fmt_num(opt_bundle_offers["estimated_incremental_profit"].sum()))
        st.markdown("**Priority Branches**")
        st.dataframe(
            target_only[["branch", "sales_2025_total", "predicted_jan_runrate_yoy_pct", "priority_score"]].head(10),
            use_container_width=True,
        )

with opt_r:
    if not opt_target_branches.empty:
        target_plot = opt_target_branches[opt_target_branches["target_flag"] == 1].copy()
        if not target_plot.empty:
            fig_t = px.bar(
                target_plot.sort_values("priority_score", ascending=False).head(12),
                x="branch",
                y="priority_score",
                color="predicted_jan_runrate_yoy_pct",
                title="Target Branch Priority Score (Low Sales + Decline Risk)",
            )
            fig_t.update_layout(xaxis_title="", yaxis_title="Priority Score")
            st.plotly_chart(fig_t, use_container_width=True)

if not opt_bundle_offers.empty:
    st.markdown("**Branch Offer Recommendations**")
    branch_opts = sorted(opt_bundle_offers["branch"].dropna().unique().tolist())
    b_offer = st.selectbox("Offer Branch", branch_opts, key="opt_branch_sel")
    b_offers = opt_bundle_offers[opt_bundle_offers["branch"] == b_offer].copy()
    b_offers = b_offers.sort_values("offer_score", ascending=False)

    fig_offer = px.bar(
        b_offers,
        x="estimated_incremental_profit",
        y="anchor_product",
        color="pair_product",
        orientation="h",
        title=f"{b_offer}: Estimated Incremental Profit by Offer",
        hover_data=["offer_type", "attach_uplift_pct", "pair_similarity"],
    )
    fig_offer.update_layout(xaxis_title="Estimated Incremental Profit", yaxis_title="Anchor Product")
    st.plotly_chart(fig_offer, use_container_width=True)

    st.dataframe(
        b_offers[
            [
                "anchor_product",
                "pair_product",
                "offer_type",
                "pair_similarity",
                "attach_uplift_pct",
                "estimated_incremental_revenue",
                "estimated_incremental_profit",
                "rationale",
            ]
        ],
        use_container_width=True,
    )

if not opt_pair_affinity.empty:
    st.markdown("**Top Product Pair Affinities**")
    fig_pair = px.scatter(
        opt_pair_affinity.head(120),
        x="similarity",
        y="pair_margin_pct",
        size="pair_score",
        color="branch_overlap",
        hover_data=["product_a", "product_b", "category_a", "category_b"],
        title="Pair Affinity vs Margin (Top Candidates)",
    )
    fig_pair.update_layout(xaxis_title="Affinity Similarity", yaxis_title="Pair Margin %")
    st.plotly_chart(fig_pair, use_container_width=True)

st.subheader("Category Mix vs Margin")
if not mix.empty:
    fig_mix = px.scatter(
        mix,
        x="beverage_share_pct",
        y="overall_margin_pct",
        size="revenue_corrected",
        color="branch",
        hover_data=["food_share_pct", "profit"],
        title="Beverage Share vs Margin (Bubble size = Revenue)",
    )
    fig_mix.update_layout(xaxis_title="Beverage Share %", yaxis_title="Overall Margin %")
    st.plotly_chart(fig_mix, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Top Products by Profit")
    fig_tp = px.bar(
        top_products.head(15).sort_values("profit"),
        x="profit",
        y="product_desc",
        orientation="h",
        title="Top 15 Products",
    )
    fig_tp.update_layout(xaxis_title="Profit", yaxis_title="")
    st.plotly_chart(fig_tp, use_container_width=True)
with right:
    st.subheader("High-Volume Low-Margin")
    if not low_margin.empty:
        fig_lm = px.scatter(
            low_margin,
            x="qty",
            y="margin_pct",
            size="revenue",
            color="margin_pct",
            hover_name="product_desc",
            title="Margin Leakage Candidates",
        )
        fig_lm.update_layout(xaxis_title="Quantity", yaxis_title="Margin %")
        st.plotly_chart(fig_lm, use_container_width=True)

st.subheader("Sales by Group")
if not groups.empty:
    fig_g = px.treemap(groups.head(20), path=["group_name"], values="sales", title="Top 20 Groups by Sales")
    st.plotly_chart(fig_g, use_container_width=True)

st.subheader("Modifier Profitability")
if not modifiers.empty:
    fig_m = px.bar(
        modifiers.sort_values("profit").head(20),
        x="profit",
        y="product_desc",
        orientation="h",
        color="profit",
        title="Lowest-Profit Modifiers",
    )
    fig_m.update_layout(xaxis_title="Profit", yaxis_title="")
    st.plotly_chart(fig_m, use_container_width=True)

st.subheader("Branch x Product x Month (Estimated)")
if branch_product_month.empty:
    st.info("Missing estimated_branch_product_month.csv. Run: python3 src/run_analysis.py")
else:
    st.caption(
        "Allocation model: annual branch-product totals are distributed across months using each branch's monthly sales share."
    )

    bp = branch_product_month.copy()
    branch_options = sorted(bp["branch"].dropna().unique().tolist())
    branch_sel = st.selectbox("Branch", branch_options, key="bpm_branch")

    bp_branch = bp[bp["branch"] == branch_sel].copy()
    product_rank = (
        bp_branch.groupby("product_desc", as_index=False)["annual_product_revenue"]
        .max()
        .sort_values("annual_product_revenue", ascending=False)
    )

    top_n = st.slider("Top products in heatmap", min_value=5, max_value=30, value=12, key="bpm_topn")
    top_products_branch = product_rank["product_desc"].head(top_n).tolist()

    heat_df = bp_branch[bp_branch["product_desc"].isin(top_products_branch)].copy()
    heat = (
        heat_df.pivot_table(index="product_desc", columns="month_num", values="estimated_revenue", aggfunc="sum")
        .reindex(columns=list(range(1, 13)), fill_value=0)
        .fillna(0)
    )
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    heat.columns = [month_labels.get(c, str(c)) for c in heat.columns]
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]

    fig_heat = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title=f"{branch_sel}: Estimated Monthly Revenue Heatmap (Top {top_n} Products)",
    )
    fig_heat.update_layout(xaxis_title="Month", yaxis_title="Product")
    st.plotly_chart(fig_heat, use_container_width=True)

    product_options = product_rank["product_desc"].tolist()
    product_sel = st.selectbox("Product detail", product_options, key="bpm_product")
    bp_sel = bp_branch[bp_branch["product_desc"] == product_sel].sort_values("month_num")

    c1, c2, c3 = st.columns(3)
    c1.metric("Annual Product Revenue (Branch)", fmt_num(float(bp_sel["annual_product_revenue"].max())))
    c2.metric("Annual Product Profit (Branch)", fmt_num(float(bp_sel["annual_product_profit"].max())))
    c3.metric("Year", str(int(bp_sel["year"].max())))

    fig_bpm = px.line(
        bp_sel,
        x="month",
        y=["estimated_revenue", "estimated_profit"],
        markers=True,
        title=f"{branch_sel} - {product_sel}: Estimated Monthly Revenue/Profit",
    )
    fig_bpm.update_layout(xaxis_title="", yaxis_title="Estimated Value", legend_title="")
    st.plotly_chart(fig_bpm, use_container_width=True)

    month_pick = st.selectbox("Month Detail", bp_sel["month"].tolist(), key="bpm_month")
    month_row = bp_sel[bp_sel["month"] == month_pick].iloc[0]
    st.write(
        {
            "branch": month_row["branch"],
            "product": month_row["product_desc"],
            "month": month_row["month"],
            "estimated_revenue": float(month_row["estimated_revenue"]),
            "estimated_profit": float(month_row["estimated_profit"]),
            "month_share": float(month_row["month_share"]),
        }
    )

with st.expander("Raw KPI Tables"):
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
        [
            "January YoY",
            "Branch Mix",
            "Top Products",
            "Branch-Product-Month",
            "EDA Anomalies",
            "EDA Concentration",
            "ML Forecast",
            "ML Predictions",
            "Optimization",
        ]
    )
    with t1:
        st.dataframe(jan, use_container_width=True)
    with t2:
        st.dataframe(mix, use_container_width=True)
    with t3:
        st.dataframe(top_products, use_container_width=True)
    with t4:
        st.dataframe(branch_product_month, use_container_width=True)
    with t5:
        st.dataframe(eda_branch_anomalies, use_container_width=True)
    with t6:
        st.dataframe(eda_product_concentration, use_container_width=True)
    with t7:
        st.dataframe(ml_network_forecast, use_container_width=True)
        st.dataframe(ml_branch_forecast, use_container_width=True)
    with t8:
        st.dataframe(ml_branch_perf, use_container_width=True)
        st.dataframe(ml_cluster, use_container_width=True)
        st.dataframe(ml_branch_archetypes, use_container_width=True)
        st.dataframe(ml_archetype_playbook, use_container_width=True)
    with t9:
        st.dataframe(opt_target_branches, use_container_width=True)
        st.dataframe(opt_bundle_offers, use_container_width=True)
        st.dataframe(opt_pair_affinity, use_container_width=True)
