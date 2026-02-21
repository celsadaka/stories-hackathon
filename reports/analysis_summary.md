# Stories Coffee - Analysis Summary

## Scope
- Data sources: four raw POS exports (2025 full year + January 2026)
- Revenue correction applied for `rep_s_00673_SMRY.csv` and subtotal rows where needed: `revenue = total_cost + total_profit`
- Branch names normalized to consistent canonical labels
- January 2026 treated as month-to-date snapshot (day 22) with run-rate normalization

## Key Findings
1. **Mixed January 2026 momentum, with growth concentrated in a small subset of branches.**
   - Stories Le Mall: run-rate Jan YoY 6.8% (3881015 vs run-rate 4145303)
   - Stories Centro Mall: run-rate Jan YoY -3.5% (3264533 vs run-rate 3150905)
   - Stories Batroun: run-rate Jan YoY -6.2% (4266517 vs run-rate 4000976)
   - Stories Khaldeh: run-rate Jan YoY -12.1% (7468155 vs run-rate 6562719)
   - Stories Verdun: run-rate Jan YoY -13.8% (1747978 vs run-rate 1507424)
2. **Several legacy branches remain materially below prior-year pace and need local action plans.**
   - Stories Faqra: run-rate Jan YoY -100.0%
   - Stories Unknown: run-rate Jan YoY -100.0%
   - Stories Saida: run-rate Jan YoY -52.8%
   - Stories Lau: run-rate Jan YoY -43.5%
   - Stories Ain El Mreisseh: run-rate Jan YoY -29.3%
3. **Profit concentration is heavy in beverage/frozen-yogurt SKUs.**
   - MANGO YOGHURT COMBO SMALL: profit 25161778, margin 80.5%
   - ORIGINAL YOGHURT COMBO SMALL: profit 25139507, margin 77.3%
   - WATER: profit 20135344, margin 88.2%
   - CLASSIC CINNAMON ROLL LARGE: profit 20003166, margin 83.1%
   - BLUEBERRY YOGHURT COMBO SMALL: profit 17146591, margin 80.9%
4. **High-volume low-margin outliers are clear and actionable.**
   - VEGGIE SUB: qty 31562, margin -1309.7%
   - LABNEH SUB: qty 11399, margin -0.1%
   - PISTACHIO CRUNCH: qty 28294, margin 26.4%
   - FREEZE DROPS STRAWBERRY: qty 11033, margin 32.1%
   - FONDANT AU CHOCOLAT: qty 11019, margin 41.0%
5. **POS line-item view: replacement-milk modifiers can appear negative because revenue is recorded on the parent drink line, not on the modifier line.**
   - REPLACE LACTOSE FREE MEDIUM: profit -538124, margin 0.0%
   - REPLACE LACTOSE FREE SMALL: profit -283962, margin 0.0%
   - REPLACE LACTOSE FREE LARGE: profit -116219, margin 0.0%
   - REPLACE LACTOSE FREE: profit -14122, margin 0.0%
   - ADD FULL FAT MILK: profit -5195, margin 0.0%

## Top 2025 Branches by Annual Sales
- Stories Ain El Mreisseh: 119612862
- Stories Zalka: 107969194
- Stories Khaldeh: 84155612
- Stories Ramlet El Bayda: 56084690
- Stories Saida: 55006290

## Top Group Categories by Sales
- FROZEN YOGHURT: 183438028
- MIXED COLD BEVERAGES: 130257997
- MIXED HOT BEVERAGE: 128943178
- BLENDED BRINKS: 90971824
- BLACK COFFEE: 54951768

## Exploratory Analysis (Patterns, Seasonality, Anomalies)
- Hidden pattern: top 92 products account for ~80% of positive product profit (strong Pareto concentration).
- Hidden pattern: top 3 branches contribute 33.9% of total 2025 sales.
- Seasonality: network peak month is August (index 1.43) while trough month is June (index 0.24).
- Branches with highest monthly volatility (CV%): Stories Event Starco (325.2%), Stories Raouche (231.0%), Stories Kaslik (225.6%)
- Anomaly scan flagged 80 branch-month points for manual review (statistical outlier or sharp month-over-month shift).
  - Stories Mansourieh / October: z=41.02, mom=2.83
  - Stories Mansourieh / August: z=40.67, mom=2965.17
  - Stories Mansourieh / September: z=39.87, mom=-1.94
- Network-level anomaly months: May, June, July

## Predictive Modeling (ML Layer)
- Time-series forecast (trend x seasonality, January-anchored) projects 2026 network sales at 797891679 vs 920575395 in 2025 (YoY -13.3%).
- Walk-forward backtest on 2025 monthly data: trend RMSE=2165229.61, naive-last-month RMSE=1918057.21, moving-avg-3 RMSE=2049117.95.
- Branch performance ridge model (margin/mix/volatility/anomaly features): in-sample RMSE=14.18, LOOCV RMSE=26.10 vs LOOCV baseline RMSE=32.42; LOOCV Spearman=-0.10.
- Prediction bounds enforced at [-95%, 95%] with P90 absolute error band of ±43.2 YoY points.
- Model indicates broad contraction risk; least-negative branches are: Stories Bir Hasan (-15.2%), Stories Batroun (-15.7%), Stories Antelias (-16.0%)
- Predicted highest-risk branches: Stories Sin El Fil (-95.0%), Stories Sour 2 (-95.0%), Stories Unknown (-95.0%)
- K-means segmentation identified 5 branch operating archetypes: Seasonal/Tourist Branch: 1 branches, Commuter Grab-and-Go: 15 branches, Premium Beverage-Focused: 2 branches, Social/Dessert Branch: 3 branches, High Volume, Low Margin: 4 branches.

## Location Analysis (Branch Segmentation)
- Branches were segmented into 5 operational archetypes: Seasonal/Tourist Branch: 1 branches, Commuter Grab-and-Go: 15 branches, Premium Beverage-Focused: 2 branches, Social/Dessert Branch: 3 branches, High Volume, Low Margin: 4 branches.
- Largest archetype: Commuter Grab-and-Go (15 branches), with avg beverage share 55.3% and margin 71.1%.
- Archetype playbooks were generated for offer design, pricing, and operations by location type.

## Optimization (Menu Engineering + Offer Engine)
- ML prioritization selected 10 branches by continuous priority score (70% low-sales intensity + 30% decline risk), with execution budget set to top 40% of branches.
- Generated 50 branch-level bundle offers with expected upside of +74709 revenue and +61793 gross profit units (scenario profit range: +32858 to +92832).
- Most reusable product pair themes across branches: MANGO YOGHURT COMBO SMALL + MANGO YOGHURT COMBO MEDIUM (6 branches), WATER + DOUBLE ESPRESSO (4 branches), ORIGINAL YOGHURT COMBO SMALL + MANGO YOGHURT COMBO SMALL (4 branches).
- Offer logic: keep anchor products already strong in each branch, then attach high-affinity under-indexed pair items through combo pricing.

## Executive Insight
Stories operates economically as a beverage and frozen-yogurt driven business, where a small group of high-margin beverage SKUs generates most profitability.

However, selected food items and replacement-milk pricing/cost gaps can create incremental margin leakage even when drinks are not discounted.

The strategic opportunity is to scale the beverage-led growth engine while systematically removing loss-making products and pricing inconsistencies.

## Implementation Roadmap
- Scale the **beverage-led growth engine**: prioritize inventory, staffing, and promotion support for beverage-dominant branches/groups with strongest contribution.
- Run a **menu margin sprint**: target high-volume products with <60% margin for recipe, pricing, or bundle redesign.
- Review **replacement-milk economics** using incremental costing: keep base drink pricing, and add/adjust surcharges only where replacement ingredients cost more.
- Use **seasonality planning** for inventory/labor calibration around peak-demand months.

## Expected Impact
- Improving VEGGIE SUB to a minimum positive margin could recover approximately +1.79M gross profit.
- Converting the current replacement-milk line-loss pool into cost-neutral pricing (after costing validation) represents up to +0.96M gross profit.
- Top 3 branches contribute about 33.9% of 2025 sales, and top 3 product groups contribute about 51.9% of group sales; focused execution here maximizes ROI.
- August demand is approximately 5.9x June, supporting stronger seasonal staffing/inventory planning.

## Confidence and Limitations
Results reflect POS export structure and should be validated against master product costing and branch metadata for production deployment. Modifier profitability in these exports is line-item based and should be interpreted as incremental, not full-drink profitability.
