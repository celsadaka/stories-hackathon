# Executive Summary - Stories Coffee (2025 + Jan 2026)

> **3 Strategic Moves**
> 1. **Scale Beverage Engine**
> 2. **Remove Margin Leaks**
> 3. **Price Replacement Milks Correctly**

## 1) Business Overview
Stories Coffee operates a 25-branch network with strong multi-category sales but uneven branch momentum and hidden margin leakage in selected products/modifiers.

**Problem:** management has data, but no operating blueprint to convert it into faster profit growth.

## 2) Growth Trends (Branch Level)
- 2025 sales are concentrated: top 3 branches (Ain El Mreisseh, Zalka, Khaldeh) represent about **33.9%** of total 2025 network sales.
- January 2026 is a month-to-date snapshot (report timestamp: **22-Jan-2026**), so run-rate normalization was applied.
- On run-rate basis:
  - Relative resilience: **Le Mall (+6.8%)**
  - Near-flat: **Centro Mall (-3.5%)**
  - Deeper pressure: Saida, LAU, and selected legacy locations
- `Stories Unknown` appears in POS exports as a likely temporary/legacy branch code; it contributes **13.9M** (2025) and should be remapped to a valid branch ID in source systems.

## 3) Profit Drivers (What Works)
- Group mix is strongly beverage-led:
  - **Frozen Yogurt: 21.5%** of tracked group sales
  - **Mixed Cold Beverages: 15.3%**
  - **Mixed Hot Beverage: 15.1%**
  - Top 3 combined: **51.9%** of group sales
- High-profit SKUs are concentrated in beverages/frozen-yogurt lines (plus selected high-margin staples like water and cinnamon rolls).

## 4) Margin Leaks (What Hurts)
- Multiple high-volume products have structurally weak margins (notably subs/pastry outliers).
- `VEGGIE SUB` is the largest single outlier in this cut:
  - Revenue: **134,662**
  - Current gross profit: **-1,763,692**
- Replacement-milk substitution policy is leaking value:
  - POS line-item loss pool on ADD/REPLACE lines is approximately **964,220**; this is an incremental-cost signal because revenue is primarily booked on parent drink items
  - Largest drag: lactose-free replacement variants

## 5) Executive Insight
Stories operates economically as a beverage and frozen-yogurt profit engine. While core beverage categories drive strong margins and sales concentration, selected food items and replacement-milk pricing/cost gaps create disproportionate incremental margin leakage. The main opportunity is to scale the beverage-led model while eliminating hidden losses in low-margin products and substitutions.

## 6) Strategic Priorities
1. **Scale the Beverage Engine**
   Focus growth on high-margin beverage categories and replicate successful branch execution patterns.

2. **Fix Margin Leaks**
   Redesign or reprice high-volume low-margin items beginning with severe outliers.

3. **Price Replacement Milks Correctly**
   Keep base drink prices, but add/adjust surcharges where replacement milks have higher ingredient cost.

## 7) Expected Impact
- Bringing VEGGIE SUB to a minimum 20% margin could recover ~1.79M gross profit.
- Converting the current replacement-milk line-loss pool into cost-neutral pricing (after costing validation) represents up to ~0.96M gross-profit opportunity.
- Better seasonality planning reduces stockouts during peak months and excess inventory in slower months.

## 8) Implementation Roadmap
1. **Menu Margin Sprint (2 weeks)**
   - Target: convert severe outliers to minimum positive unit economics.
   - Example estimate: moving `VEGGIE SUB` to a 20% gross margin at current volume implies roughly **+1.79M** gross-profit recovery.

2. **Replacement-Milk Pricing Policy (immediate)**
   - Keep base drink pricing unchanged; review surcharge design where replacement milks have higher ingredient cost.
   - Estimated opportunity: up to **~0.96M** gross profit from the current replacement-milk line-loss pool (subject to costing validation).

3. **Branch Playbook (30 days)**
   - Clone top-branch execution patterns (mix, upsell, labor rhythm) into underperforming branches.

4. **Seasonality Operations Planning (quarterly)**
   - August demand is about **5.9x** June in this dataset; adjust inventory/labor/campaign calendars accordingly.

## 9) Final Note
Results reflect POS export structure and should be validated against master product costing and branch metadata for production deployment.

This analysis is implementation-ready: the pipeline is reproducible and can be rerun monthly with new POS exports in the same format.
