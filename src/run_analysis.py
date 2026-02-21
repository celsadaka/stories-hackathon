#!/usr/bin/env python3
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_TABLES = os.path.join(ROOT, "outputs", "tables")
OUT_REPORTS = os.path.join(ROOT, "reports")

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


def ensure_dirs() -> None:
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_REPORTS, exist_ok=True)


def parse_num(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def canonical_branch(raw: str) -> Optional[str]:
    name = clean_text(raw)
    if not name:
        return None
    lower = name.lower().replace("branch:", "").strip()
    if lower in {"total", "total by branch:", "stories"}:
        return None
    if not lower.startswith("stories"):
        return None
    lower = re.sub(r"^stories", "", lower).strip(" .:-")
    lower = re.sub(r"[^a-z0-9 ]+", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    if lower == "":
        return "Stories Unknown"
    alias = {
        "alay": "aley",
        "ain el mreisseh": "ain el mreisseh",
        "sin el fil": "sin el fil",
    }
    lower = alias.get(lower, lower)
    titled = " ".join(token.capitalize() if token not in {"el"} else "El" for token in lower.split())
    return f"Stories {titled}"


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_monthly_sales() -> Tuple[List[Dict], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], List[Dict]]:
    path = os.path.join(RAW_DIR, "REP_S_00134_SMRY.csv")
    branch_year_totals: List[Dict] = []
    jan_by_year_branch: Dict[int, Dict[str, float]] = defaultdict(dict)
    monthly_totals: Dict[int, Dict[str, float]] = defaultdict(dict)
    branch_monthly_map: Dict[Tuple[int, str, str], float] = {}
    current_year: Optional[int] = None
    section_mode = ""

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            col0 = clean_text(row[0] if len(row) > 0 else "")
            col1 = clean_text(row[1] if len(row) > 1 else "")
            row_l = ",".join([c.lower() for c in row])

            if col0 in {"2025", "2026"}:
                current_year = int(col0)

            if "total by year" in row_l:
                section_mode = "oct_dec_total"
            elif "january" in row_l and "total by year" not in row_l:
                section_mode = "jan_sep"

            # Monthly network totals row (Jan-Sep export section)
            if col1.lower() == "total":
                if len(row) >= 11 and current_year is not None and section_mode == "jan_sep":
                    for i, month in enumerate(MONTH_ORDER[:9]):
                        v = parse_num(row[2 + i] if len(row) > 2 + i else "")
                        if v is not None:
                            monthly_totals[current_year][month] = v
                continue

            branch = canonical_branch(col1)
            if not branch or current_year is None:
                continue

            # Jan-Sep branch rows: two layouts exist (11 columns and 14 columns)
            if len(row) >= 11 and section_mode == "jan_sep":
                start_idx = 3 if len(row) == 14 else 2
                for i, month in enumerate(MONTH_ORDER[:9]):
                    v = parse_num(row[start_idx + i] if len(row) > start_idx + i else "")
                    if v is None:
                        continue
                    branch_monthly_map[(current_year, branch, month)] = v
                    if month == "January":
                        jan_by_year_branch[current_year][branch] = v

            # Oct-Dec + total by year section
            total_by_year = None
            if len(row) >= 6 and section_mode == "oct_dec_total":
                total_by_year = parse_num(row[5])

            if total_by_year is not None:
                branch_year_totals.append(
                    {
                        "year": current_year,
                        "branch": branch,
                        "total_by_year": round(total_by_year, 2),
                    }
                )

                oct_v = parse_num(row[2] if len(row) > 2 else "")
                nov_v = parse_num(row[3] if len(row) > 3 else "")
                dec_v = parse_num(row[4] if len(row) > 4 else "")

                if oct_v is not None:
                    monthly_totals[current_year]["October"] = monthly_totals[current_year].get("October", 0.0) + oct_v
                    branch_monthly_map[(current_year, branch, "October")] = oct_v
                if nov_v is not None:
                    monthly_totals[current_year]["November"] = monthly_totals[current_year].get("November", 0.0) + nov_v
                    branch_monthly_map[(current_year, branch, "November")] = nov_v
                if dec_v is not None:
                    monthly_totals[current_year]["December"] = monthly_totals[current_year].get("December", 0.0) + dec_v
                    branch_monthly_map[(current_year, branch, "December")] = dec_v

    month_to_num = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    clean_monthly_sales = [
        {
            "year": y,
            "branch": b,
            "month": m,
            "month_num": month_to_num[m],
            "sales": round(v, 2),
        }
        for (y, b, m), v in branch_monthly_map.items()
    ]
    clean_monthly_sales.sort(key=lambda r: (r["year"], r["branch"], r["month_num"]))

    return branch_year_totals, jan_by_year_branch, monthly_totals, clean_monthly_sales



def infer_january_snapshot_day() -> Optional[int]:
    path = os.path.join(RAW_DIR, "REP_S_00134_SMRY.csv")
    date_re = re.compile(r"^(\d{1,2})-Jan-\d{4}$")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 20:
                break
            col0 = clean_text(row[0] if row else "")
            m = date_re.match(col0)
            if m:
                return int(m.group(1))
    return None


def infer_item_year(default_year: int = 2025) -> int:
    path = os.path.join(RAW_DIR, "rep_s_00014_SMRY.csv")
    year_re = re.compile(r"Years:(\d{4})")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 30:
                break
            joined = " ".join(row)
            m = year_re.search(joined)
            if m:
                return int(m.group(1))
    return default_year


def parse_category_summary() -> List[Dict]:
    path = os.path.join(RAW_DIR, "rep_s_00673_SMRY.csv")
    rows_out: List[Dict] = []
    current_branch: Optional[str] = None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            desc = clean_text(row[0])
            if not desc:
                continue
            if desc.startswith("22-") or desc in {"Category", "Stories", "Theoretical Profit By Category"}:
                continue
            if desc.startswith("REP_S_00673") or desc.startswith("Total By Branch:"):
                continue

            maybe_branch = canonical_branch(desc)
            if maybe_branch and len(row) > 1 and clean_text(row[1]) == "":
                current_branch = maybe_branch
                continue

            if desc in {"BEVERAGES", "FOOD"} and current_branch:
                qty = parse_num(row[1] if len(row) > 1 else "")
                total_cost = parse_num(row[4] if len(row) > 4 else "")
                total_profit = parse_num(row[6] if len(row) > 6 else "")
                cost_pct = parse_num(row[5] if len(row) > 5 else "")
                profit_pct = parse_num(row[8] if len(row) > 8 else "")
                if qty is None or total_cost is None or total_profit is None:
                    continue
                revenue_corrected = total_cost + total_profit
                rows_out.append(
                    {
                        "branch": current_branch,
                        "category": desc,
                        "qty": round(qty, 2),
                        "revenue_corrected": round(revenue_corrected, 2),
                        "total_cost": round(total_cost, 2),
                        "total_profit": round(total_profit, 2),
                        "cost_pct": round(cost_pct if cost_pct is not None else (100 * total_cost / revenue_corrected), 2),
                        "profit_pct": round(profit_pct if profit_pct is not None else (100 * total_profit / revenue_corrected), 2),
                    }
                )
    return rows_out


def parse_item_profitability() -> List[Dict]:
    path = os.path.join(RAW_DIR, "rep_s_00014_SMRY.csv")
    current_branch: Optional[str] = None
    current_service = ""
    current_category = ""
    current_section = ""
    out: List[Dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            desc = clean_text(row[0])
            if not desc:
                continue
            if desc.startswith("22-") or desc in {"Product Desc", "Stories", "Theoretical Profit By Item"}:
                continue
            if desc.startswith("Total By") or desc.startswith("REP_S_00014"):
                continue

            maybe_branch = canonical_branch(desc)
            if maybe_branch and len(row) > 1 and clean_text(row[1]) == "":
                current_branch = maybe_branch
                current_service = ""
                current_category = ""
                current_section = ""
                continue

            if desc in {"TAKE AWAY", "TABLE"}:
                current_service = desc
                continue
            if desc in {"BEVERAGES", "FOOD"}:
                current_category = desc
                continue
            if clean_text(row[1] if len(row) > 1 else "") == "":
                # Hierarchy labels (section/department/etc.)
                current_section = desc
                continue

            qty = parse_num(row[1] if len(row) > 1 else "")
            total_cost = parse_num(row[4] if len(row) > 4 else "")
            cost_pct = parse_num(row[5] if len(row) > 5 else "")
            total_profit = parse_num(row[6] if len(row) > 6 else "")
            profit_pct = parse_num(row[8] if len(row) > 8 else "")
            if qty is None or total_cost is None or total_profit is None:
                continue
            if not current_branch:
                continue
            revenue = total_cost + total_profit
            out.append(
                {
                    "branch": current_branch,
                    "service_type": current_service,
                    "category": current_category,
                    "section": current_section,
                    "product_desc": desc,
                    "qty": round(qty, 3),
                    "revenue": round(revenue, 2),
                    "total_cost": round(total_cost, 2),
                    "total_profit": round(total_profit, 2),
                    "cost_pct": round(cost_pct if cost_pct is not None else (100 * total_cost / revenue if revenue else 0), 2),
                    "profit_pct": round(profit_pct if profit_pct is not None else (100 * total_profit / revenue if revenue else 0), 2),
                }
            )
    return out


def parse_group_sales() -> List[Dict]:
    path = os.path.join(RAW_DIR, "rep_s_00191_SMRY-3.csv")
    out: List[Dict] = []
    current_branch = ""
    current_division = ""
    current_group = ""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            desc = clean_text(row[0])
            if not desc:
                continue
            if desc.startswith("19-") or desc in {"Description", "Stories", "Sales by Items By Group"}:
                continue
            if desc.startswith("Total by") or desc.startswith("REP_S_00191"):
                continue

            if desc.startswith("Branch:"):
                current_branch = canonical_branch(desc.replace("Branch:", "").strip()) or ""
                current_division = ""
                current_group = ""
                continue
            if desc.startswith("Division:"):
                current_division = clean_text(desc.replace("Division:", "").strip())
                continue
            if desc.startswith("Group:"):
                current_group = clean_text(desc.replace("Group:", "").strip())
                continue

            qty = parse_num(row[2] if len(row) > 2 else "")
            total_amount = parse_num(row[3] if len(row) > 3 else "")
            if qty is None or total_amount is None or not current_branch:
                continue
            out.append(
                {
                    "branch": current_branch,
                    "division": current_division,
                    "group_name": current_group,
                    "description": desc,
                    "qty": round(qty, 3),
                    "total_amount": round(total_amount, 2),
                }
            )
    return out


def rank_rows(rows: List[Dict], key: str, desc: bool = True, top_n: int = 10) -> List[Dict]:
    return sorted(rows, key=lambda r: r.get(key, 0), reverse=desc)[:top_n]


def pct_change(new: float, old: float) -> Optional[float]:
    if old is None or abs(old) < 1e-9:
        return None
    return (new - old) / old * 100.0



def safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def safe_std(values: List[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def fit_linear_trend(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    x = np.arange(1, len(values) + 1, dtype=float)
    y = np.array(values, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom else 0.0
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def logistic_prob(x: float, scale: float = 20.0) -> float:
    z = max(min(x / scale, 60.0), -60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def kmeans_fit(matrix: np.ndarray, k: int = 3, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=float)

    n_samples = matrix.shape[0]
    n_features = matrix.shape[1]
    k_eff = min(max(1, k), n_samples)

    order = np.argsort(matrix[:, 0])
    seed_idx = np.linspace(0, n_samples - 1, k_eff, dtype=int)
    centroids = matrix[order[seed_idx]].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for ci in range(k_eff):
            members = matrix[labels == ci]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(dists, axis=1)))
                centroids[ci] = matrix[farthest]
            else:
                centroids[ci] = members.mean(axis=0)

    if centroids.shape[1] != n_features:
        centroids = np.zeros((k_eff, n_features), dtype=float)

    return labels, centroids

def make_outputs() -> None:
    ensure_dirs()

    branch_year_totals, jan_by_year_branch, monthly_totals, clean_monthly_sales = parse_monthly_sales()
    category_summary = parse_category_summary()
    item_profit = parse_item_profitability()
    group_sales = parse_group_sales()
    item_year = infer_item_year()

    write_csv(
        os.path.join(OUT_TABLES, "clean_branch_year_totals.csv"),
        branch_year_totals,
        ["year", "branch", "total_by_year"],
    )
    write_csv(
        os.path.join(OUT_TABLES, "clean_category_summary.csv"),
        category_summary,
        ["branch", "category", "qty", "revenue_corrected", "total_cost", "total_profit", "cost_pct", "profit_pct"],
    )
    write_csv(
        os.path.join(OUT_TABLES, "clean_item_profitability.csv"),
        item_profit,
        ["branch", "service_type", "category", "section", "product_desc", "qty", "revenue", "total_cost", "total_profit", "cost_pct", "profit_pct"],
    )
    write_csv(
        os.path.join(OUT_TABLES, "clean_group_sales.csv"),
        group_sales,
        ["branch", "division", "group_name", "description", "qty", "total_amount"],
    )

    # Estimated branch x product x month table (allocation model)
    # Assumption: branch-product annual totals are distributed by each branch's monthly sales share.
    branch_product_annual = defaultdict(lambda: {"qty": 0.0, "revenue": 0.0, "profit": 0.0})
    for r in item_profit:
        key = (r["branch"], r["product_desc"])
        branch_product_annual[key]["qty"] += r["qty"]
        branch_product_annual[key]["revenue"] += r["revenue"]
        branch_product_annual[key]["profit"] += r["total_profit"]

    branch_annual_sales = {r["branch"]: r["total_by_year"] for r in branch_year_totals if r["year"] == item_year}

    branch_month_sales_map: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in clean_monthly_sales:
        if r["year"] == item_year:
            branch_month_sales_map[r["branch"]][r["month"]] = r["sales"]

    month_to_num = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    estimated_branch_product_month: List[Dict] = []
    for (branch, product_desc), vals in branch_product_annual.items():
        annual_branch_sales = branch_annual_sales.get(branch, 0.0)
        annual_product_revenue = vals["revenue"]
        annual_product_profit = vals["profit"]

        for month in MONTH_ORDER:
            monthly_branch_sales = branch_month_sales_map.get(branch, {}).get(month, 0.0)
            month_share = (monthly_branch_sales / annual_branch_sales) if annual_branch_sales else 0.0
            estimated_revenue = annual_product_revenue * month_share
            estimated_profit = annual_product_profit * month_share
            estimated_branch_product_month.append(
                {
                    "year": item_year,
                    "branch": branch,
                    "month": month,
                    "month_num": month_to_num[month],
                    "product_desc": product_desc,
                    "branch_annual_sales": round(annual_branch_sales, 2),
                    "monthly_branch_sales": round(monthly_branch_sales, 2),
                    "month_share": round(month_share, 6),
                    "annual_product_revenue": round(annual_product_revenue, 2),
                    "annual_product_profit": round(annual_product_profit, 2),
                    "estimated_revenue": round(estimated_revenue, 2),
                    "estimated_profit": round(estimated_profit, 2),
                }
            )

    estimated_branch_product_month.sort(key=lambda r: (r["branch"], r["product_desc"], r["month_num"]))
    write_csv(
        os.path.join(OUT_TABLES, "estimated_branch_product_month.csv"),
        estimated_branch_product_month,
        [
            "year",
            "branch",
            "month",
            "month_num",
            "product_desc",
            "branch_annual_sales",
            "monthly_branch_sales",
            "month_share",
            "annual_product_revenue",
            "annual_product_profit",
            "estimated_revenue",
            "estimated_profit",
        ],
    )

    # Explicit cleaning deliverables for submission section
    category_profit_table = [
        {
            "branch": r["branch"],
            "category": r["category"],
            "qty": r["qty"],
            "revenue": r["revenue_corrected"],
            "total_cost": r["total_cost"],
            "total_profit": r["total_profit"],
            "cost_pct": r["cost_pct"],
            "profit_pct": r["profit_pct"],
        }
        for r in category_summary
    ]
    write_csv(
        os.path.join(OUT_TABLES, "clean_category_profit.csv"),
        category_profit_table,
        ["branch", "category", "qty", "revenue", "total_cost", "total_profit", "cost_pct", "profit_pct"],
    )

    write_csv(
        os.path.join(OUT_TABLES, "clean_monthly_sales.csv"),
        clean_monthly_sales,
        ["year", "branch", "month", "month_num", "sales"],
    )

    product_profit_table = [
        {
            "branch": r["branch"],
            "service_type": r["service_type"],
            "category": r["category"],
            "section": r["section"],
            "product_desc": r["product_desc"],
            "qty": r["qty"],
            "revenue": r["revenue"],
            "total_cost": r["total_cost"],
            "total_profit": r["total_profit"],
            "cost_pct": r["cost_pct"],
            "profit_pct": r["profit_pct"],
        }
        for r in item_profit
    ]
    write_csv(
        os.path.join(OUT_TABLES, "clean_product_profit.csv"),
        product_profit_table,
        ["branch", "service_type", "category", "section", "product_desc", "qty", "revenue", "total_cost", "total_profit", "cost_pct", "profit_pct"],
    )

    sales_group_table = [
        {
            "branch": r["branch"],
            "division": r["division"],
            "group_name": r["group_name"],
            "description": r["description"],
            "qty": r["qty"],
            "sales": r["total_amount"],
        }
        for r in group_sales
    ]
    write_csv(
        os.path.join(OUT_TABLES, "clean_sales_group.csv"),
        sales_group_table,
        ["branch", "division", "group_name", "description", "qty", "sales"],
    )

    # KPI 1: January 2026 vs January 2025 by branch
    jan_growth: List[Dict] = []
    snapshot_day = infer_january_snapshot_day() or 31
    runrate_factor = 31.0 / float(snapshot_day) if snapshot_day > 0 else 1.0
    branches = sorted(set(jan_by_year_branch.get(2025, {}).keys()) | set(jan_by_year_branch.get(2026, {}).keys()))
    for b in branches:
        jan25 = jan_by_year_branch.get(2025, {}).get(b, 0.0)
        jan26 = jan_by_year_branch.get(2026, {}).get(b, 0.0)
        growth = pct_change(jan26, jan25) if jan25 > 0 else None
        jan26_runrate = jan26 * runrate_factor
        growth_runrate = pct_change(jan26_runrate, jan25) if jan25 > 0 else None
        jan_growth.append(
            {
                "branch": b,
                "jan_2025": round(jan25, 2),
                "jan_2026": round(jan26, 2),
                "jan_2026_runrate": round(jan26_runrate, 2),
                "jan_yoy_pct": "" if growth is None else round(growth, 2),
                "jan_runrate_yoy_pct": "" if growth_runrate is None else round(growth_runrate, 2),
            }
        )
    write_csv(
        os.path.join(OUT_TABLES, "kpi_january_yoy_by_branch.csv"),
        sorted(jan_growth, key=lambda r: r["jan_2026"], reverse=True),
        ["branch", "jan_2025", "jan_2026", "jan_2026_runrate", "jan_yoy_pct", "jan_runrate_yoy_pct"],
    )

    # KPI 2: 2025 annual leaders
    totals_2025 = [r for r in branch_year_totals if r["year"] == 2025]
    top_annual = rank_rows(totals_2025, "total_by_year", True, 10)
    write_csv(
        os.path.join(OUT_TABLES, "kpi_top_branches_2025_sales.csv"),
        top_annual,
        ["year", "branch", "total_by_year"],
    )

    # KPI 3: Category mix and margin by branch
    by_branch_cat = defaultdict(lambda: {"revenue": 0.0, "profit": 0.0, "beverage_rev": 0.0, "food_rev": 0.0})
    for r in category_summary:
        b = r["branch"]
        rev = r["revenue_corrected"]
        prof = r["total_profit"]
        by_branch_cat[b]["revenue"] += rev
        by_branch_cat[b]["profit"] += prof
        if r["category"] == "BEVERAGES":
            by_branch_cat[b]["beverage_rev"] += rev
        if r["category"] == "FOOD":
            by_branch_cat[b]["food_rev"] += rev
    branch_mix = []
    for b, v in by_branch_cat.items():
        total_rev = v["revenue"]
        bev_share = (v["beverage_rev"] / total_rev * 100.0) if total_rev else 0.0
        food_share = (v["food_rev"] / total_rev * 100.0) if total_rev else 0.0
        margin = (v["profit"] / total_rev * 100.0) if total_rev else 0.0
        branch_mix.append(
            {
                "branch": b,
                "revenue_corrected": round(total_rev, 2),
                "profit": round(v["profit"], 2),
                "overall_margin_pct": round(margin, 2),
                "beverage_share_pct": round(bev_share, 2),
                "food_share_pct": round(food_share, 2),
            }
        )
    write_csv(
        os.path.join(OUT_TABLES, "kpi_branch_category_mix_margin.csv"),
        sorted(branch_mix, key=lambda r: r["overall_margin_pct"], reverse=True),
        ["branch", "revenue_corrected", "profit", "overall_margin_pct", "beverage_share_pct", "food_share_pct"],
    )

    # KPI 4: Top products by gross profit
    prod_rollup = defaultdict(lambda: {"qty": 0.0, "revenue": 0.0, "profit": 0.0, "cost": 0.0})
    for r in item_profit:
        p = r["product_desc"]
        prod_rollup[p]["qty"] += r["qty"]
        prod_rollup[p]["revenue"] += r["revenue"]
        prod_rollup[p]["profit"] += r["total_profit"]
        prod_rollup[p]["cost"] += r["total_cost"]
    top_products = []
    for p, v in prod_rollup.items():
        rev = v["revenue"]
        margin = (v["profit"] / rev * 100.0) if rev else 0.0
        top_products.append(
            {
                "product_desc": p,
                "qty": round(v["qty"], 3),
                "revenue": round(rev, 2),
                "profit": round(v["profit"], 2),
                "margin_pct": round(margin, 2),
            }
        )
    write_csv(
        os.path.join(OUT_TABLES, "kpi_top_products_by_profit.csv"),
        rank_rows(top_products, "profit", True, 25),
        ["product_desc", "qty", "revenue", "profit", "margin_pct"],
    )

    # KPI 5: High-volume, low-margin products (actionable for price/recipe review)
    qty_values = sorted(v["qty"] for v in top_products)
    qty_threshold = qty_values[int(0.75 * len(qty_values))] if qty_values else 0.0
    low_margin_high_vol = [
        r
        for r in top_products
        if r["qty"] >= qty_threshold and r["margin_pct"] < 60 and r["revenue"] > 0
    ]
    write_csv(
        os.path.join(OUT_TABLES, "kpi_high_volume_low_margin_products.csv"),
        sorted(low_margin_high_vol, key=lambda r: (r["margin_pct"], -r["profit"]))[:40],
        ["product_desc", "qty", "revenue", "profit", "margin_pct"],
    )

    # KPI 6: Modifier diagnostics
    modifier_rows = [
        r
        for r in top_products
        if r["product_desc"].startswith("ADD ") or r["product_desc"].startswith("REPLACE ")
    ]
    write_csv(
        os.path.join(OUT_TABLES, "kpi_modifiers_margin.csv"),
        sorted(modifier_rows, key=lambda r: r["profit"])[:20] + rank_rows(modifier_rows, "profit", True, 20),
        ["product_desc", "qty", "revenue", "profit", "margin_pct"],
    )

    # KPI 7: Group performance from file 3
    group_rollup = defaultdict(lambda: {"qty": 0.0, "sales": 0.0})
    for r in group_sales:
        g = r["group_name"]
        group_rollup[g]["qty"] += r["qty"]
        group_rollup[g]["sales"] += r["total_amount"]
    group_perf = [{"group_name": g, "qty": round(v["qty"], 3), "sales": round(v["sales"], 2)} for g, v in group_rollup.items()]
    write_csv(
        os.path.join(OUT_TABLES, "kpi_group_sales_performance.csv"),
        rank_rows(group_perf, "sales", True, 30),
        ["group_name", "qty", "sales"],
    )

    # KPI 8: 2025 monthly seasonality index
    month_order = MONTH_ORDER
    month_vals = [monthly_totals.get(2025, {}).get(m, 0.0) for m in month_order]
    nonzero = [v for v in month_vals if v > 0]
    avg = sum(nonzero) / len(nonzero) if nonzero else 0.0
    seasonality = []
    for m in month_order:
        val = monthly_totals.get(2025, {}).get(m, 0.0)
        idx = (val / avg) if avg else 0.0
        seasonality.append({"month": m, "sales_2025": round(val, 2), "seasonality_index": round(idx, 3)})
    write_csv(
        os.path.join(OUT_TABLES, "kpi_2025_monthly_seasonality.csv"),
        seasonality,
        ["month", "sales_2025", "seasonality_index"],
    )


    # EDA 1: Branch seasonality profile and volatility (2025)
    sales_2025_by_branch: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in clean_monthly_sales:
        if r["year"] != 2025:
            continue
        sales_2025_by_branch[r["branch"]][r["month"]] = r["sales"]

    branch_seasonality_profile: List[Dict] = []
    branch_seasonality_summary: List[Dict] = []
    for branch, month_map in sales_2025_by_branch.items():
        vals = [month_map.get(m, 0.0) for m in MONTH_ORDER]
        nonzero_vals = [v for v in vals if v > 0]
        avg_branch = (sum(nonzero_vals) / len(nonzero_vals)) if nonzero_vals else 0.0
        mean_branch = (sum(vals) / len(vals)) if vals else 0.0
        std_branch = safe_std(vals, mean_branch)
        cv = (std_branch / mean_branch * 100.0) if mean_branch else 0.0

        peak_i = max(range(len(vals)), key=lambda i: vals[i]) if vals else 0
        trough_i = min(range(len(vals)), key=lambda i: vals[i]) if vals else 0
        peak_month = MONTH_ORDER[peak_i]
        trough_month = MONTH_ORDER[trough_i]
        peak_sales = vals[peak_i] if vals else 0.0
        trough_sales = vals[trough_i] if vals else 0.0

        branch_seasonality_summary.append(
            {
                "branch": branch,
                "avg_monthly_sales": round(mean_branch, 2),
                "std_monthly_sales": round(std_branch, 2),
                "cv_pct": round(cv, 2),
                "peak_month": peak_month,
                "peak_sales": round(peak_sales, 2),
                "trough_month": trough_month,
                "trough_sales": round(trough_sales, 2),
                "peak_to_trough_ratio": round((peak_sales / trough_sales) if trough_sales else 0.0, 3),
            }
        )

        for i, month in enumerate(MONTH_ORDER):
            sales_val = vals[i]
            idx = (sales_val / avg_branch) if avg_branch else 0.0
            branch_seasonality_profile.append(
                {
                    "branch": branch,
                    "month": month,
                    "month_num": i + 1,
                    "sales": round(sales_val, 2),
                    "branch_month_avg": round(avg_branch, 2),
                    "seasonality_index": round(idx, 3),
                    "is_peak_month": 1 if month == peak_month else 0,
                    "is_trough_month": 1 if month == trough_month else 0,
                }
            )

    write_csv(
        os.path.join(OUT_TABLES, "eda_branch_seasonality_profile_2025.csv"),
        sorted(branch_seasonality_profile, key=lambda r: (r["branch"], r["month_num"])),
        ["branch", "month", "month_num", "sales", "branch_month_avg", "seasonality_index", "is_peak_month", "is_trough_month"],
    )
    write_csv(
        os.path.join(OUT_TABLES, "eda_branch_seasonality_summary_2025.csv"),
        sorted(branch_seasonality_summary, key=lambda r: r["cv_pct"], reverse=True),
        ["branch", "avg_monthly_sales", "std_monthly_sales", "cv_pct", "peak_month", "peak_sales", "trough_month", "trough_sales", "peak_to_trough_ratio"],
    )

    # EDA 2: Branch-level anomalies (robust z-score + sharp MoM changes)
    branch_month_anomalies: List[Dict] = []
    for branch, month_map in sales_2025_by_branch.items():
        vals = [month_map.get(m, 0.0) for m in MONTH_ORDER]
        mean_val = (sum(vals) / len(vals)) if vals else 0.0
        std_val = safe_std(vals, mean_val)
        median_val = safe_median(vals)
        mad = safe_median([abs(v - median_val) for v in vals])

        prev_val: Optional[float] = None
        for i, month in enumerate(MONTH_ORDER):
            sales_val = vals[i]
            robust_z = (0.6745 * (sales_val - median_val) / mad) if mad > 0 else 0.0
            zscore = ((sales_val - mean_val) / std_val) if std_val > 0 else 0.0
            mom_pct = pct_change(sales_val, prev_val) if prev_val is not None else None

            reason = ""
            if abs(robust_z) >= 2.5:
                reason = "robust_z"
            elif abs(zscore) >= 2.0:
                reason = "std_z"
            elif mom_pct is not None and abs(mom_pct) >= 40.0:
                reason = "mom_spike"

            branch_month_anomalies.append(
                {
                    "branch": branch,
                    "month": month,
                    "month_num": i + 1,
                    "sales": round(sales_val, 2),
                    "mean_sales": round(mean_val, 2),
                    "median_sales": round(median_val, 2),
                    "std_sales": round(std_val, 2),
                    "robust_zscore": round(robust_z, 3),
                    "zscore": round(zscore, 3),
                    "mom_pct": "" if mom_pct is None else round(mom_pct, 2),
                    "anomaly_flag": 1 if reason else 0,
                    "anomaly_reason": reason,
                }
            )
            prev_val = sales_val

    write_csv(
        os.path.join(OUT_TABLES, "eda_branch_month_anomalies_2025.csv"),
        sorted(
            branch_month_anomalies,
            key=lambda r: (
                -r["anomaly_flag"],
                -abs(r["robust_zscore"]),
                r["branch"],
                r["month_num"],
            ),
        ),
        ["branch", "month", "month_num", "sales", "mean_sales", "median_sales", "std_sales", "robust_zscore", "zscore", "mom_pct", "anomaly_flag", "anomaly_reason"],
    )

    # EDA 3: Network month anomalies (2025 total sales)
    network_vals = [monthly_totals.get(2025, {}).get(m, 0.0) for m in MONTH_ORDER]
    network_mean = (sum(network_vals) / len(network_vals)) if network_vals else 0.0
    network_std = safe_std(network_vals, network_mean)
    network_median = safe_median(network_vals)
    network_mad = safe_median([abs(v - network_median) for v in network_vals])

    network_month_anomalies: List[Dict] = []
    prev_network_val: Optional[float] = None
    for i, month in enumerate(MONTH_ORDER):
        val = network_vals[i]
        robust_z = (0.6745 * (val - network_median) / network_mad) if network_mad > 0 else 0.0
        zscore = ((val - network_mean) / network_std) if network_std > 0 else 0.0
        mom_pct = pct_change(val, prev_network_val) if prev_network_val is not None else None
        is_anomaly = 1 if abs(robust_z) >= 2.5 or abs(zscore) >= 2.0 or (mom_pct is not None and abs(mom_pct) >= 30.0) else 0
        network_month_anomalies.append(
            {
                "month": month,
                "month_num": i + 1,
                "sales_2025": round(val, 2),
                "robust_zscore": round(robust_z, 3),
                "zscore": round(zscore, 3),
                "mom_pct": "" if mom_pct is None else round(mom_pct, 2),
                "anomaly_flag": is_anomaly,
            }
        )
        prev_network_val = val

    write_csv(
        os.path.join(OUT_TABLES, "eda_network_month_anomalies_2025.csv"),
        network_month_anomalies,
        ["month", "month_num", "sales_2025", "robust_zscore", "zscore", "mom_pct", "anomaly_flag"],
    )

    # EDA 4: Concentration patterns (branch sales + product profit)
    product_profit_rows = [r for r in top_products if r["profit"] > 0]
    total_positive_profit = sum(r["profit"] for r in product_profit_rows)
    product_profit_concentration: List[Dict] = []
    cumulative_profit_share = 0.0
    for rank, row in enumerate(sorted(product_profit_rows, key=lambda r: r["profit"], reverse=True), start=1):
        share = (row["profit"] / total_positive_profit * 100.0) if total_positive_profit else 0.0
        cumulative_profit_share += share
        product_profit_concentration.append(
            {
                "rank": rank,
                "product_desc": row["product_desc"],
                "profit": round(row["profit"], 2),
                "profit_share_pct": round(share, 3),
                "cumulative_profit_share_pct": round(cumulative_profit_share, 3),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "eda_product_profit_concentration.csv"),
        product_profit_concentration,
        ["rank", "product_desc", "profit", "profit_share_pct", "cumulative_profit_share_pct"],
    )

    branch_revenue_concentration: List[Dict] = []
    total_2025_sales = sum(r["total_by_year"] for r in totals_2025)
    cumulative_revenue_share = 0.0
    for rank, row in enumerate(sorted(totals_2025, key=lambda r: r["total_by_year"], reverse=True), start=1):
        share = (row["total_by_year"] / total_2025_sales * 100.0) if total_2025_sales else 0.0
        cumulative_revenue_share += share
        branch_revenue_concentration.append(
            {
                "rank": rank,
                "branch": row["branch"],
                "total_by_year": round(row["total_by_year"], 2),
                "revenue_share_pct": round(share, 3),
                "cumulative_revenue_share_pct": round(cumulative_revenue_share, 3),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "eda_branch_revenue_concentration_2025.csv"),
        branch_revenue_concentration,
        ["rank", "branch", "total_by_year", "revenue_share_pct", "cumulative_revenue_share_pct"],
    )



    # ML 1: Branch-level monthly sales forecast for 2026 (trend x seasonality)
    jan_runrate_by_branch = {r["branch"]: r["jan_2026_runrate"] for r in jan_growth}
    ml_branch_monthly_forecast: List[Dict] = []

    for branch, month_map in sales_2025_by_branch.items():
        sales_2025_seq = [month_map.get(m, 0.0) for m in MONTH_ORDER]
        avg_2025_branch = (sum(sales_2025_seq) / len(sales_2025_seq)) if sales_2025_seq else 0.0
        seasonality_idx_seq = [(v / avg_2025_branch) if avg_2025_branch else 1.0 for v in sales_2025_seq]

        first_q = float(np.mean(sales_2025_seq[:3])) if sales_2025_seq[:3] else 0.0
        last_q = float(np.mean(sales_2025_seq[-3:])) if sales_2025_seq[-3:] else 0.0
        trend_rate = ((last_q - first_q) / first_q) if first_q > 0 else 0.0
        trend_rate = float(np.clip(trend_rate, -0.30, 0.30))
        trend_multiplier = 1.0 + trend_rate

        raw_preds: List[float] = [max(v * trend_multiplier, 0.0) for v in sales_2025_seq]

        jan_raw_pred = raw_preds[0] if raw_preds else 0.0
        jan_actual_runrate = jan_runrate_by_branch.get(branch, 0.0)
        anchor_ratio = 1.0
        if jan_raw_pred > 0 and jan_actual_runrate > 0:
            anchor_ratio = jan_actual_runrate / jan_raw_pred
            anchor_ratio = min(max(anchor_ratio, 0.7), 1.3)

        for i, month in enumerate(MONTH_ORDER):
            sales_2025_month = sales_2025_seq[i]
            trend_component = max(sales_2025_month * trend_multiplier, 0.0)
            raw_pred = raw_preds[i]
            calibrated_pred = raw_pred * anchor_ratio
            final_pred = calibrated_pred

            ml_branch_monthly_forecast.append(
                {
                    "year": 2026,
                    "branch": branch,
                    "month": month,
                    "month_num": i + 1,
                    "sales_2025": round(sales_2025_month, 2),
                    "seasonality_index_2025": round(seasonality_idx_seq[i], 4),
                    "trend_component": round(trend_component, 2),
                    "forecast_raw": round(raw_pred, 2),
                    "jan_actual_runrate": round(jan_actual_runrate, 2),
                    "jan_anchor_ratio": round(anchor_ratio, 4),
                    "forecast_calibrated": round(calibrated_pred, 2),
                    "forecast_final": round(final_pred, 2),
                }
            )

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_monthly_forecast_2026.csv"),
        sorted(ml_branch_monthly_forecast, key=lambda r: (r["branch"], r["month_num"])),
        [
            "year",
            "branch",
            "month",
            "month_num",
            "sales_2025",
            "seasonality_index_2025",
            "trend_component",
            "forecast_raw",
            "jan_actual_runrate",
            "jan_anchor_ratio",
            "forecast_calibrated",
            "forecast_final",
        ],
    )

    ml_network_forecast: List[Dict] = []
    for i, month in enumerate(MONTH_ORDER):
        month_num = i + 1
        forecast_2026 = sum(r["forecast_final"] for r in ml_branch_monthly_forecast if r["month_num"] == month_num)
        sales_2025_month = monthly_totals.get(2025, {}).get(month, 0.0)
        yoy_pct = pct_change(forecast_2026, sales_2025_month) if sales_2025_month > 0 else None
        ml_network_forecast.append(
            {
                "year": 2026,
                "month": month,
                "month_num": month_num,
                "sales_2025": round(sales_2025_month, 2),
                "forecast_sales_2026": round(forecast_2026, 2),
                "forecast_yoy_pct": "" if yoy_pct is None else round(yoy_pct, 2),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_network_monthly_forecast_2026.csv"),
        ml_network_forecast,
        ["year", "month", "month_num", "sales_2025", "forecast_sales_2026", "forecast_yoy_pct"],
    )

    sales_2025_total_by_branch = {r["branch"]: r["total_by_year"] for r in totals_2025}
    forecast_2026_total_by_branch: Dict[str, float] = defaultdict(float)
    for r in ml_branch_monthly_forecast:
        forecast_2026_total_by_branch[r["branch"]] += r["forecast_final"]

    ml_branch_annual_forecast = []
    for branch, fc_total in forecast_2026_total_by_branch.items():
        s25 = sales_2025_total_by_branch.get(branch, 0.0)
        yoy_pct = pct_change(fc_total, s25) if s25 > 0 else None
        ml_branch_annual_forecast.append(
            {
                "branch": branch,
                "sales_2025_total": round(s25, 2),
                "forecast_2026_total": round(fc_total, 2),
                "forecast_yoy_pct": "" if yoy_pct is None else round(yoy_pct, 2),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_annual_forecast_2026.csv"),
        sorted(ml_branch_annual_forecast, key=lambda r: r["forecast_2026_total"], reverse=True),
        ["branch", "sales_2025_total", "forecast_2026_total", "forecast_yoy_pct"],
    )

    # ML 2: Branch performance prediction (regression + direction classification)
    branch_mix_map = {r["branch"]: r for r in branch_mix}
    cv_map = {r["branch"]: r["cv_pct"] for r in branch_seasonality_summary}
    anomaly_count_map: Dict[str, int] = defaultdict(int)
    for r in branch_month_anomalies:
        if r["anomaly_flag"] == 1:
            anomaly_count_map[r["branch"]] += 1

    target_map: Dict[str, float] = {
        r["branch"]: r["jan_runrate_yoy_pct"]
        for r in jan_growth
        if isinstance(r["jan_runrate_yoy_pct"], float)
    }

    branch_profit_2025_map: Dict[str, float] = defaultdict(float)
    for r in category_summary:
        branch_profit_2025_map[r["branch"]] += float(r["total_profit"])

    peak_to_trough_map = {r["branch"]: float(r.get("peak_to_trough_ratio", 0.0)) for r in branch_seasonality_summary}

    service_rev_map: Dict[str, Dict[str, float]] = defaultdict(lambda: {"takeaway": 0.0, "table": 0.0})
    for r in item_profit:
        branch = r["branch"]
        rev = float(r["revenue"])
        svc = clean_text(r.get("service_type", "")).upper()
        if svc == "TAKE AWAY":
            service_rev_map[branch]["takeaway"] += rev
        elif svc == "TABLE":
            service_rev_map[branch]["table"] += rev

    takeaway_share_map: Dict[str, float] = {}
    table_share_map: Dict[str, float] = {}
    for branch, v in service_rev_map.items():
        denom = v["takeaway"] + v["table"]
        takeaway_share_map[branch] = (v["takeaway"] / denom * 100.0) if denom > 0 else 0.0
        table_share_map[branch] = (v["table"] / denom * 100.0) if denom > 0 else 0.0

    group_sales_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in group_sales:
        branch = r["branch"]
        group_name = clean_text(r.get("group_name", ""))
        if not group_name:
            continue
        group_sales_map[branch][group_name] += float(r["total_amount"])

    dessert_keywords = ("YOGHURT", "DONUT", "DESSERT", "CAKE", "WAFFLE", "CREPE", "ROLL")
    dominant_group_map: Dict[str, str] = {}
    dominant_group_share_map: Dict[str, float] = {}
    dessert_share_map: Dict[str, float] = {}
    for branch, gmap in group_sales_map.items():
        total_sales = sum(gmap.values())
        if total_sales <= 0:
            dominant_group_map[branch] = ""
            dominant_group_share_map[branch] = 0.0
            dessert_share_map[branch] = 0.0
            continue
        dom_group, dom_sales = max(gmap.items(), key=lambda x: x[1])
        dessert_sales = sum(v for g, v in gmap.items() if any(k in g.upper() for k in dessert_keywords))
        dominant_group_map[branch] = dom_group
        dominant_group_share_map[branch] = dom_sales / total_sales * 100.0
        dessert_share_map[branch] = dessert_sales / total_sales * 100.0

    branches_model = sorted(set(sales_2025_total_by_branch.keys()) | set(branch_mix_map.keys()) | set(cv_map.keys()))
    feat_names = [
        "annual_sales_2025",
        "overall_margin_pct",
        "beverage_share_pct",
        "cv_pct",
        "anomaly_count_2025",
    ]

    feature_rows = []
    for branch in branches_model:
        mix_row = branch_mix_map.get(branch, {})
        feature_rows.append(
            {
                "branch": branch,
                "annual_sales_2025": float(sales_2025_total_by_branch.get(branch, 0.0)),
                "annual_profit_2025": float(branch_profit_2025_map.get(branch, 0.0)),
                "overall_margin_pct": float(mix_row.get("overall_margin_pct", 0.0)),
                "beverage_share_pct": float(mix_row.get("beverage_share_pct", 0.0)),
                "food_share_pct": float(mix_row.get("food_share_pct", 0.0)),
                "dominant_group": dominant_group_map.get(branch, ""),
                "dominant_group_share_pct": float(dominant_group_share_map.get(branch, 0.0)),
                "dessert_share_pct": float(dessert_share_map.get(branch, 0.0)),
                "takeaway_share_pct": float(takeaway_share_map.get(branch, 0.0)),
                "table_share_pct": float(table_share_map.get(branch, 0.0)),
                "peak_to_trough_ratio": float(peak_to_trough_map.get(branch, 0.0)),
                "cv_pct": float(cv_map.get(branch, 0.0)),
                "anomaly_count_2025": float(anomaly_count_map.get(branch, 0)),
                "actual_jan_runrate_yoy_pct": target_map.get(branch),
            }
        )

    train_rows = [r for r in feature_rows if r["actual_jan_runrate_yoy_pct"] is not None]
    if train_rows:
        x_train = np.array([[r[n] for n in feat_names] for r in train_rows], dtype=float)
        y_train = np.array([r["actual_jan_runrate_yoy_pct"] for r in train_rows], dtype=float)
        y_train = np.clip(y_train, -150.0, 150.0)
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std = np.where(x_std == 0, 1.0, x_std)
        x_train_z = (x_train - x_mean) / x_std
        x_design = np.column_stack([np.ones(len(x_train_z)), x_train_z])
        beta = np.linalg.lstsq(x_design, y_train, rcond=None)[0]
        y_hat = x_design @ beta
        sse = float(np.sum((y_train - y_hat) ** 2))
        sst = float(np.sum((y_train - y_train.mean()) ** 2))
        model_r2 = (1.0 - sse / sst) if sst > 0 else 0.0
        model_rmse = float(np.sqrt(np.mean((y_train - y_hat) ** 2)))
    else:
        x_mean = np.zeros(len(feat_names), dtype=float)
        x_std = np.ones(len(feat_names), dtype=float)
        beta = np.zeros(len(feat_names) + 1, dtype=float)
        model_r2 = 0.0
        model_rmse = 0.0

    ml_branch_performance_prediction: List[Dict] = []
    for r in feature_rows:
        x_vec = np.array([r[n] for n in feat_names], dtype=float)
        x_z = (x_vec - x_mean) / x_std
        pred_yoy_raw = float(beta[0] + np.dot(beta[1:], x_z))
        pred_yoy = float(np.clip(pred_yoy_raw, -150.0, 150.0))
        growth_prob = logistic_prob(pred_yoy, scale=20.0)
        actual = r["actual_jan_runrate_yoy_pct"]
        residual = (actual - pred_yoy) if actual is not None else None

        ml_branch_performance_prediction.append(
            {
                "branch": r["branch"],
                "annual_sales_2025": round(r["annual_sales_2025"], 2),
                "overall_margin_pct": round(r["overall_margin_pct"], 2),
                "beverage_share_pct": round(r["beverage_share_pct"], 2),
                "cv_pct": round(r["cv_pct"], 2),
                "anomaly_count_2025": int(r["anomaly_count_2025"]),
                "actual_jan_runrate_yoy_pct": "" if actual is None else round(actual, 2),
                "predicted_jan_runrate_yoy_pct": round(pred_yoy, 2),
                "growth_probability": round(growth_prob, 4),
                "predicted_direction": "Grow" if pred_yoy >= 0 else "Decline",
                "residual": "" if residual is None else round(residual, 2),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_performance_prediction.csv"),
        sorted(ml_branch_performance_prediction, key=lambda r: r["predicted_jan_runrate_yoy_pct"], reverse=True),
        [
            "branch",
            "annual_sales_2025",
            "overall_margin_pct",
            "beverage_share_pct",
            "cv_pct",
            "anomaly_count_2025",
            "actual_jan_runrate_yoy_pct",
            "predicted_jan_runrate_yoy_pct",
            "growth_probability",
            "predicted_direction",
            "residual",
        ],
    )

    coef_rows = [{"feature": "intercept", "coefficient": round(float(beta[0]), 6)}]
    for i, feat in enumerate(feat_names):
        coef_rows.append({"feature": feat, "coefficient": round(float(beta[i + 1]), 6)})
    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_performance_coefficients.csv"),
        coef_rows,
        ["feature", "coefficient"],
    )

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_performance_model_metrics.csv"),
        [
            {
                "model": "linear_regression_jan_runrate_yoy",
                "training_samples": len(train_rows),
                "features": ",".join(feat_names),
                "r2_in_sample": round(model_r2, 4),
                "rmse_yoy_pct": round(model_rmse, 4),
            }
        ],
        ["model", "training_samples", "features", "r2_in_sample", "rmse_yoy_pct"],
    )

    # ML 3: Branch segmentation (k-means clustering + archetype naming)
    seg_feat_names = [
        "annual_sales_2025",
        "annual_profit_2025",
        "overall_margin_pct",
        "beverage_share_pct",
        "dominant_group_share_pct",
        "dessert_share_pct",
        "cv_pct",
        "peak_to_trough_ratio",
        "takeaway_share_pct",
        "table_share_pct",
    ]

    x_all = np.array([[r[n] for n in seg_feat_names] for r in feature_rows], dtype=float) if feature_rows else np.zeros((0, len(seg_feat_names)), dtype=float)
    if len(x_all) > 0:
        mean_all = x_all.mean(axis=0)
        std_all = x_all.std(axis=0)
        std_all = np.where(std_all == 0, 1.0, std_all)
        x_all_z = (x_all - mean_all) / std_all
        labels, centroids_z = kmeans_fit(x_all_z, k=5, max_iter=80)
        centroids_orig = centroids_z * std_all + mean_all
    else:
        mean_all = np.zeros(len(seg_feat_names), dtype=float)
        std_all = np.ones(len(seg_feat_names), dtype=float)
        labels = np.array([], dtype=int)
        centroids_z = np.zeros((0, len(seg_feat_names)), dtype=float)
        centroids_orig = np.zeros((0, len(seg_feat_names)), dtype=float)

    idx = {name: i for i, name in enumerate(seg_feat_names)}

    # Map clusters to business-friendly archetype names.
    # We assign each archetype to one unique cluster based on centroid behavior.
    cluster_ids = list(range(len(centroids_orig)))
    remaining = set(cluster_ids)

    def pick_cluster(score_fn):
        if not remaining:
            return None
        best = max(remaining, key=score_fn)
        remaining.remove(best)
        return best

    seasonal_id = pick_cluster(lambda ci: float(centroids_orig[ci][idx["cv_pct"]]) + 0.3 * float(centroids_orig[ci][idx["peak_to_trough_ratio"]]))
    commuter_id = pick_cluster(lambda ci: float(centroids_orig[ci][idx["takeaway_share_pct"]]))
    premium_id = pick_cluster(lambda ci: float(centroids_orig[ci][idx["beverage_share_pct"]]) + 0.6 * float(centroids_orig[ci][idx["overall_margin_pct"]]))
    high_volume_low_margin_id = pick_cluster(lambda ci: float(centroids_orig[ci][idx["annual_sales_2025"]]) - 200000.0 * float(centroids_orig[ci][idx["overall_margin_pct"]]))
    social_dessert_id = pick_cluster(lambda ci: float(centroids_orig[ci][idx["dessert_share_pct"]]) + 0.4 * float(centroids_orig[ci][idx["dominant_group_share_pct"]]))

    cluster_label_map: Dict[int, str] = {}
    if seasonal_id is not None:
        cluster_label_map[seasonal_id] = "Seasonal/Tourist Branch"
    if commuter_id is not None:
        cluster_label_map[commuter_id] = "Commuter Grab-and-Go"
    if premium_id is not None:
        cluster_label_map[premium_id] = "Premium Beverage-Focused"
    if high_volume_low_margin_id is not None:
        cluster_label_map[high_volume_low_margin_id] = "High Volume, Low Margin"
    if social_dessert_id is not None:
        cluster_label_map[social_dessert_id] = "Social/Dessert Branch"

    for ci in cluster_ids:
        if ci not in cluster_label_map:
            cluster_label_map[ci] = f"Operational Archetype {ci + 1}"

    pred_map = {r["branch"]: r["predicted_jan_runrate_yoy_pct"] for r in ml_branch_performance_prediction}
    ml_branch_clusters: List[Dict] = []
    for i, r in enumerate(feature_rows):
        if i >= len(labels):
            continue
        cid = int(labels[i])
        ml_branch_clusters.append(
            {
                "branch": r["branch"],
                "cluster_id": cid + 1,
                "cluster_label": cluster_label_map.get(cid, f"Operational Archetype {cid + 1}"),
                "annual_sales_2025": round(r["annual_sales_2025"], 2),
                "annual_profit_2025": round(r["annual_profit_2025"], 2),
                "overall_margin_pct": round(r["overall_margin_pct"], 2),
                "beverage_share_pct": round(r["beverage_share_pct"], 2),
                "food_share_pct": round(r["food_share_pct"], 2),
                "dominant_group": r["dominant_group"],
                "dominant_group_share_pct": round(r["dominant_group_share_pct"], 2),
                "dessert_share_pct": round(r["dessert_share_pct"], 2),
                "takeaway_share_pct": round(r["takeaway_share_pct"], 2),
                "table_share_pct": round(r["table_share_pct"], 2),
                "cv_pct": round(r["cv_pct"], 2),
                "peak_to_trough_ratio": round(r["peak_to_trough_ratio"], 3),
                "anomaly_count_2025": int(r["anomaly_count_2025"]),
                "predicted_jan_runrate_yoy_pct": round(float(pred_map.get(r["branch"], 0.0)), 2),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_clusters.csv"),
        sorted(ml_branch_clusters, key=lambda r: (r["cluster_id"], r["branch"])),
        [
            "branch",
            "cluster_id",
            "cluster_label",
            "annual_sales_2025",
            "annual_profit_2025",
            "overall_margin_pct",
            "beverage_share_pct",
            "food_share_pct",
            "dominant_group",
            "dominant_group_share_pct",
            "dessert_share_pct",
            "takeaway_share_pct",
            "table_share_pct",
            "cv_pct",
            "peak_to_trough_ratio",
            "anomaly_count_2025",
            "predicted_jan_runrate_yoy_pct",
        ],
    )

    archetype_playbook = {
        "Commuter Grab-and-Go": {
            "menu_offer_strategy": "Fast combos (coffee + pastry), pre-batched morning sets",
            "pricing_strategy": "Time-window bundles and loyalty refill pricing",
            "operations_focus": "Queue speed, prep-before-rush, high-availability SKUs",
            "primary_kpi": "Morning combo attach rate",
        },
        "Social/Dessert Branch": {
            "menu_offer_strategy": "Dessert + drink pair bundles and shareable upsells",
            "pricing_strategy": "Bundle discount with margin floor and premium toppings upsell",
            "operations_focus": "Merchandising and upsell scripts",
            "primary_kpi": "Dessert-pair attach rate",
        },
        "Premium Beverage-Focused": {
            "menu_offer_strategy": "Signature beverage ladder and premium milk/syrup upgrades",
            "pricing_strategy": "Versioned pricing (good/better/best) and premium add-on pricing",
            "operations_focus": "Bar quality consistency and speed",
            "primary_kpi": "Premium beverage mix %",
        },
        "High Volume, Low Margin": {
            "menu_offer_strategy": "Re-bundle low-margin heroes with high-margin companions",
            "pricing_strategy": "Recipe-cost reset and minimum margin guardrails",
            "operations_focus": "Waste and COGS control",
            "primary_kpi": "Gross margin %",
        },
        "Seasonal/Tourist Branch": {
            "menu_offer_strategy": "Seasonal campaigns and peak-month curated bundles",
            "pricing_strategy": "Dynamic seasonal offers by demand window",
            "operations_focus": "Flexible staffing and inventory by seasonality",
            "primary_kpi": "Peak season conversion",
        },
    }

    ml_branch_archetypes: List[Dict] = []
    for row in ml_branch_clusters:
        play = archetype_playbook.get(row["cluster_label"], {
            "menu_offer_strategy": "Balanced cross-category bundle testing",
            "pricing_strategy": "Localized elasticity testing",
            "operations_focus": "Branch-level execution discipline",
            "primary_kpi": "Contribution margin",
        })
        row_out = dict(row)
        row_out["menu_offer_strategy"] = play["menu_offer_strategy"]
        row_out["pricing_strategy"] = play["pricing_strategy"]
        row_out["operations_focus"] = play["operations_focus"]
        row_out["primary_kpi"] = play["primary_kpi"]
        ml_branch_archetypes.append(row_out)

    write_csv(
        os.path.join(OUT_TABLES, "ml_branch_archetypes.csv"),
        sorted(ml_branch_archetypes, key=lambda r: (r["cluster_label"], r["branch"])),
        [
            "branch",
            "cluster_id",
            "cluster_label",
            "annual_sales_2025",
            "annual_profit_2025",
            "overall_margin_pct",
            "beverage_share_pct",
            "food_share_pct",
            "dominant_group",
            "dominant_group_share_pct",
            "dessert_share_pct",
            "takeaway_share_pct",
            "table_share_pct",
            "cv_pct",
            "peak_to_trough_ratio",
            "anomaly_count_2025",
            "predicted_jan_runrate_yoy_pct",
            "menu_offer_strategy",
            "pricing_strategy",
            "operations_focus",
            "primary_kpi",
        ],
    )

    archetype_rows = []
    for name, play in archetype_playbook.items():
        member_count = sum(1 for r in ml_branch_clusters if r["cluster_label"] == name)
        archetype_rows.append(
            {
                "archetype": name,
                "member_count": member_count,
                "menu_offer_strategy": play["menu_offer_strategy"],
                "pricing_strategy": play["pricing_strategy"],
                "operations_focus": play["operations_focus"],
                "primary_kpi": play["primary_kpi"],
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_archetype_playbook.csv"),
        sorted(archetype_rows, key=lambda r: r["member_count"], reverse=True),
        [
            "archetype",
            "member_count",
            "menu_offer_strategy",
            "pricing_strategy",
            "operations_focus",
            "primary_kpi",
        ],
    )

    centroid_rows = []
    for ci in range(len(centroids_orig)):
        members = [r for r in ml_branch_clusters if r["cluster_id"] == ci + 1]
        pred_avg = (
            sum(r["predicted_jan_runrate_yoy_pct"] for r in members) / len(members)
            if members
            else 0.0
        )
        c = centroids_orig[ci]
        centroid_rows.append(
            {
                "cluster_id": ci + 1,
                "cluster_label": cluster_label_map.get(ci, f"Operational Archetype {ci + 1}"),
                "member_count": len(members),
                "annual_sales_2025": round(float(c[idx["annual_sales_2025"]]), 2),
                "annual_profit_2025": round(float(c[idx["annual_profit_2025"]]), 2),
                "overall_margin_pct": round(float(c[idx["overall_margin_pct"]]), 2),
                "beverage_share_pct": round(float(c[idx["beverage_share_pct"]]), 2),
                "dominant_group_share_pct": round(float(c[idx["dominant_group_share_pct"]]), 2),
                "dessert_share_pct": round(float(c[idx["dessert_share_pct"]]), 2),
                "takeaway_share_pct": round(float(c[idx["takeaway_share_pct"]]), 2),
                "table_share_pct": round(float(c[idx["table_share_pct"]]), 2),
                "cv_pct": round(float(c[idx["cv_pct"]]), 2),
                "peak_to_trough_ratio": round(float(c[idx["peak_to_trough_ratio"]]), 3),
                "predicted_jan_runrate_yoy_pct": round(float(pred_avg), 2),
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "ml_cluster_centroids.csv"),
        sorted(centroid_rows, key=lambda r: r["cluster_id"]),
        [
            "cluster_id",
            "cluster_label",
            "member_count",
            "annual_sales_2025",
            "annual_profit_2025",
            "overall_margin_pct",
            "beverage_share_pct",
            "dominant_group_share_pct",
            "dessert_share_pct",
            "takeaway_share_pct",
            "table_share_pct",
            "cv_pct",
            "peak_to_trough_ratio",
            "predicted_jan_runrate_yoy_pct",
        ],
    )

    # Optimization ML: menu engineering + branch offer engine
    branch_sales_values = [float(v) for v in sales_2025_total_by_branch.values() if v > 0]
    low_sales_threshold = float(np.percentile(branch_sales_values, 30)) if branch_sales_values else 0.0
    pred_map = {r["branch"]: float(r["predicted_jan_runrate_yoy_pct"]) for r in ml_branch_performance_prediction}
    cluster_map = {r["branch"]: r["cluster_label"] for r in ml_branch_clusters}

    opt_target_branches: List[Dict] = []
    for branch, sales_val in sales_2025_total_by_branch.items():
        pred_yoy = float(pred_map.get(branch, 0.0))
        low_sales_flag = 1 if sales_val <= low_sales_threshold else 0
        decline_risk_flag = 1 if pred_yoy <= -20.0 else 0
        target_flag = 1 if (low_sales_flag == 1 or decline_risk_flag == 1) else 0
        priority_score = round((1.0 if low_sales_flag else 0.0) + max(0.0, min((-pred_yoy) / 100.0, 2.0)), 4)
        opt_target_branches.append(
            {
                "branch": branch,
                "sales_2025_total": round(float(sales_val), 2),
                "predicted_jan_runrate_yoy_pct": round(pred_yoy, 2),
                "cluster_label": cluster_map.get(branch, ""),
                "low_sales_flag": low_sales_flag,
                "decline_risk_flag": decline_risk_flag,
                "target_flag": target_flag,
                "priority_score": priority_score,
            }
        )

    write_csv(
        os.path.join(OUT_TABLES, "opt_target_branches.csv"),
        sorted(opt_target_branches, key=lambda r: (r["target_flag"], r["priority_score"], -r["sales_2025_total"]), reverse=True),
        [
            "branch",
            "sales_2025_total",
            "predicted_jan_runrate_yoy_pct",
            "cluster_label",
            "low_sales_flag",
            "decline_risk_flag",
            "target_flag",
            "priority_score",
        ],
    )

    product_global = defaultdict(lambda: {"revenue": 0.0, "profit": 0.0, "qty": 0.0, "cat": defaultdict(float)})
    branch_product_revenue = defaultdict(lambda: defaultdict(float))

    for r in item_profit:
        branch = r["branch"]
        product = r["product_desc"]
        rev = float(r["revenue"])
        prof = float(r["total_profit"])
        qty = float(r["qty"])
        category = r.get("category") or "UNKNOWN"

        product_global[product]["revenue"] += rev
        product_global[product]["profit"] += prof
        product_global[product]["qty"] += qty
        product_global[product]["cat"][category] += rev
        branch_product_revenue[branch][product] += rev

    total_network_sales_2025 = float(total_2025_sales) if total_2025_sales > 0 else 1.0

    product_meta: Dict[str, Dict] = {}
    for product, stats in product_global.items():
        rev = float(stats["revenue"])
        prof = float(stats["profit"])
        qty = float(stats["qty"])
        margin = (prof / rev * 100.0) if rev > 0 else 0.0
        cat_map_p = stats["cat"]
        dominant_category = max(cat_map_p.items(), key=lambda x: x[1])[0] if cat_map_p else "UNKNOWN"
        product_meta[product] = {
            "revenue": rev,
            "profit": prof,
            "qty": qty,
            "margin_pct": margin,
            "category": dominant_category,
            "network_share": (rev / total_network_sales_2025) if total_network_sales_2025 > 0 else 0.0,
        }

    eligible_products = [
        p
        for p, m in sorted(product_meta.items(), key=lambda kv: kv[1]["revenue"], reverse=True)
        if m["revenue"] > 0 and m["profit"] > 0
    ][:160]

    branch_list = sorted(sales_2025_total_by_branch.keys())
    bidx = {b: i for i, b in enumerate(branch_list)}
    pidx = {p: i for i, p in enumerate(eligible_products)}

    mat = np.zeros((len(eligible_products), len(branch_list)), dtype=float)
    for b in branch_list:
        b_total = float(sales_2025_total_by_branch.get(b, 0.0))
        if b_total <= 0:
            continue
        for p, rev in branch_product_revenue.get(b, {}).items():
            if p in pidx:
                mat[pidx[p], bidx[b]] = float(rev) / b_total

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_norm = mat / norms
    sim_mat = mat_norm @ mat_norm.T if len(eligible_products) > 0 else np.zeros((0, 0), dtype=float)

    opt_product_pair_affinity: List[Dict] = []
    max_products_for_pairs = min(120, len(eligible_products))
    for i in range(max_products_for_pairs):
        p1 = eligible_products[i]
        m1 = product_meta[p1]
        for j in range(i + 1, max_products_for_pairs):
            p2 = eligible_products[j]
            m2 = product_meta[p2]
            sim = float(sim_mat[i, j])
            if sim < 0.2:
                continue

            overlap = 0
            for b in branch_list:
                if branch_product_revenue.get(b, {}).get(p1, 0.0) > 0 and branch_product_revenue.get(b, {}).get(p2, 0.0) > 0:
                    overlap += 1
            if overlap < 4:
                continue

            pair_margin = (m1["margin_pct"] + m2["margin_pct"]) / 2.0
            pair_score = float(sim * np.log1p(min(m1["revenue"], m2["revenue"])) * (1.0 + max(pair_margin, 0.0) / 100.0))
            opt_product_pair_affinity.append(
                {
                    "product_a": p1,
                    "product_b": p2,
                    "category_a": m1["category"],
                    "category_b": m2["category"],
                    "similarity": round(sim, 4),
                    "branch_overlap": overlap,
                    "revenue_a": round(m1["revenue"], 2),
                    "revenue_b": round(m2["revenue"], 2),
                    "margin_a_pct": round(m1["margin_pct"], 2),
                    "margin_b_pct": round(m2["margin_pct"], 2),
                    "pair_margin_pct": round(pair_margin, 2),
                    "pair_score": round(pair_score, 4),
                }
            )

    opt_product_pair_affinity = sorted(opt_product_pair_affinity, key=lambda r: r["pair_score"], reverse=True)[:300]
    write_csv(
        os.path.join(OUT_TABLES, "opt_product_pair_affinity.csv"),
        opt_product_pair_affinity,
        [
            "product_a",
            "product_b",
            "category_a",
            "category_b",
            "similarity",
            "branch_overlap",
            "revenue_a",
            "revenue_b",
            "margin_a_pct",
            "margin_b_pct",
            "pair_margin_pct",
            "pair_score",
        ],
    )

    pair_options = defaultdict(list)
    for row in opt_product_pair_affinity:
        pair_options[row["product_a"]].append(
            {
                "pair_product": row["product_b"],
                "pair_similarity": row["similarity"],
                "pair_score": row["pair_score"],
                "pair_margin_pct": row["pair_margin_pct"],
                "anchor_category": row["category_a"],
                "pair_category": row["category_b"],
            }
        )
        pair_options[row["product_b"]].append(
            {
                "pair_product": row["product_a"],
                "pair_similarity": row["similarity"],
                "pair_score": row["pair_score"],
                "pair_margin_pct": row["pair_margin_pct"],
                "anchor_category": row["category_b"],
                "pair_category": row["category_a"],
            }
        )

    for p in pair_options:
        pair_options[p] = sorted(pair_options[p], key=lambda r: (r["pair_score"], r["pair_similarity"]), reverse=True)

    target_branch_names = [r["branch"] for r in opt_target_branches if r["target_flag"] == 1]
    opt_branch_bundle_recommendations: List[Dict] = []

    for branch in target_branch_names:
        b_total = float(sales_2025_total_by_branch.get(branch, 0.0))
        if b_total <= 0:
            continue

        branch_prod_map = branch_product_revenue.get(branch, {})
        anchors = sorted(
            [
                (p, float(rev))
                for p, rev in branch_prod_map.items()
                if rev > 0 and p in product_meta and product_meta[p]["margin_pct"] >= 45.0
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:25]

        used_pairs = set()
        branch_rows: List[Dict] = []

        for anchor, anchor_rev in anchors:
            options = pair_options.get(anchor, [])
            chosen = None
            for op in options:
                pair_product = op["pair_product"]
                key = tuple(sorted([anchor, pair_product]))
                if key in used_pairs:
                    continue

                pair_net_share = float(product_meta.get(pair_product, {}).get("network_share", 0.0))
                pair_rev_branch = float(branch_prod_map.get(pair_product, 0.0))
                pair_branch_share = (pair_rev_branch / b_total) if b_total > 0 else 0.0
                under_index_gap = pair_net_share - pair_branch_share
                if under_index_gap <= 0:
                    continue

                attach_uplift_pct = float(np.clip(under_index_gap * 120.0, 1.5, 9.0))
                base_exposed_revenue = max(anchor_rev * 0.25, 1.0)
                estimated_incremental_revenue = base_exposed_revenue * (attach_uplift_pct / 100.0) * max(op["pair_similarity"], 0.2)
                margin_factor = float(np.clip(op["pair_margin_pct"] / 100.0, 0.25, 0.9))
                estimated_incremental_profit = estimated_incremental_revenue * margin_factor
                offer_score = estimated_incremental_profit * (1.0 + op["pair_similarity"])

                anchor_cat = op["anchor_category"]
                pair_cat = op["pair_category"]
                if anchor_cat != pair_cat:
                    offer_type = "Cross-category combo"
                elif anchor_cat == "BEVERAGES":
                    offer_type = "Beverage pair offer"
                else:
                    offer_type = "Food bundle"

                chosen = {
                    "branch": branch,
                    "cluster_label": cluster_map.get(branch, ""),
                    "anchor_product": anchor,
                    "pair_product": pair_product,
                    "anchor_category": anchor_cat,
                    "pair_category": pair_cat,
                    "offer_type": offer_type,
                    "pair_similarity": round(float(op["pair_similarity"]), 4),
                    "pair_score": round(float(op["pair_score"]), 4),
                    "anchor_revenue_branch": round(anchor_rev, 2),
                    "pair_revenue_branch": round(pair_rev_branch, 2),
                    "branch_pair_share_pct": round(pair_branch_share * 100.0, 3),
                    "network_pair_share_pct": round(pair_net_share * 100.0, 3),
                    "attach_uplift_pct": round(attach_uplift_pct, 2),
                    "estimated_incremental_revenue": round(estimated_incremental_revenue, 2),
                    "estimated_incremental_profit": round(estimated_incremental_profit, 2),
                    "offer_score": round(offer_score, 4),
                    "rationale": f"Under-indexed vs network by {max(under_index_gap,0)*100:.2f} pts; high affinity pair score.",
                }
                break

            if chosen is not None:
                branch_rows.append(chosen)
                used_pairs.add(tuple(sorted([chosen["anchor_product"], chosen["pair_product"]])))

            if len(branch_rows) >= 6:
                break

        branch_rows = sorted(branch_rows, key=lambda r: r["offer_score"], reverse=True)[:5]
        opt_branch_bundle_recommendations.extend(branch_rows)

    write_csv(
        os.path.join(OUT_TABLES, "opt_branch_bundle_recommendations.csv"),
        sorted(opt_branch_bundle_recommendations, key=lambda r: (r["branch"], -r["offer_score"])),
        [
            "branch",
            "cluster_label",
            "anchor_product",
            "pair_product",
            "anchor_category",
            "pair_category",
            "offer_type",
            "pair_similarity",
            "pair_score",
            "anchor_revenue_branch",
            "pair_revenue_branch",
            "branch_pair_share_pct",
            "network_pair_share_pct",
            "attach_uplift_pct",
            "estimated_incremental_revenue",
            "estimated_incremental_profit",
            "offer_score",
            "rationale",
        ],
    )

    # Narrative report
    top5_branches = rank_rows(top_annual, "total_by_year", True, 5)
    jan_growth_ranked = sorted([r for r in jan_growth if isinstance(r["jan_runrate_yoy_pct"], float)], key=lambda r: r["jan_runrate_yoy_pct"], reverse=True)
    jan_decline_ranked = sorted([r for r in jan_growth if isinstance(r["jan_runrate_yoy_pct"], float)], key=lambda r: r["jan_runrate_yoy_pct"])
    top5_products = rank_rows(top_products, "profit", True, 5)
    low_margin5 = sorted(low_margin_high_vol, key=lambda r: r["margin_pct"])[:5]
    worst_mod = sorted(modifier_rows, key=lambda r: r["profit"])[:5]
    best_group = rank_rows(group_perf, "sales", True, 5)

    total_2025_sales = sum(r["total_by_year"] for r in totals_2025)
    top3_branch_share = (
        100.0 * sum(r["total_by_year"] for r in rank_rows(totals_2025, "total_by_year", True, 3)) / total_2025_sales
        if total_2025_sales
        else 0.0
    )

    total_group_sales = sum(r["sales"] for r in group_perf)
    top3_group_share = (
        100.0 * sum(r["sales"] for r in rank_rows(group_perf, "sales", True, 3)) / total_group_sales
        if total_group_sales
        else 0.0
    )

    modifier_loss_pool = sum(-r["profit"] for r in modifier_rows if r["profit"] < 0)
    august_sales = monthly_totals.get(2025, {}).get("August", 0.0)
    june_sales = monthly_totals.get(2025, {}).get("June", 0.0)
    aug_june_ratio = (august_sales / june_sales) if june_sales else 0.0


    top80_sku_count = next(
        (r["rank"] for r in product_profit_concentration if r["cumulative_profit_share_pct"] >= 80.0),
        len(product_profit_concentration),
    )
    branch_anomaly_flags = [r for r in branch_month_anomalies if r["anomaly_flag"] == 1]
    top_branch_anomalies = sorted(branch_anomaly_flags, key=lambda r: abs(r["robust_zscore"]), reverse=True)[:5]
    network_anomaly_flags = [r for r in network_month_anomalies if r["anomaly_flag"] == 1]

    seasonality_desc = sorted(seasonality, key=lambda r: r["sales_2025"], reverse=True)
    peak_month = seasonality_desc[0] if seasonality_desc else {"month": "", "seasonality_index": 0.0}
    trough_month = seasonality_desc[-1] if seasonality_desc else {"month": "", "seasonality_index": 0.0}
    volatile_branches = sorted(branch_seasonality_summary, key=lambda r: r["cv_pct"], reverse=True)[:3]


    network_forecast_2026_total = sum(r["forecast_sales_2026"] for r in ml_network_forecast)
    network_sales_2025_total = sum(r["sales_2025"] for r in ml_network_forecast)
    network_forecast_yoy = pct_change(network_forecast_2026_total, network_sales_2025_total) if network_sales_2025_total > 0 else None

    pred_rank = sorted(ml_branch_performance_prediction, key=lambda r: r["predicted_jan_runrate_yoy_pct"], reverse=True)
    pred_top = pred_rank[:3]
    pred_risk = pred_rank[-3:]

    cluster_count = len(centroid_rows)
    cluster_summary_str = ", ".join([f"{r['cluster_label']}: {r['member_count']} branches" for r in centroid_rows])
    archetype_top = sorted(centroid_rows, key=lambda r: r["member_count"], reverse=True)

    opt_target_count = sum(1 for r in opt_target_branches if r["target_flag"] == 1)
    opt_offer_count = len(opt_branch_bundle_recommendations)
    opt_total_incremental_profit = sum(r["estimated_incremental_profit"] for r in opt_branch_bundle_recommendations)
    opt_total_incremental_revenue = sum(r["estimated_incremental_revenue"] for r in opt_branch_bundle_recommendations)

    pair_counter = defaultdict(int)
    for r in opt_branch_bundle_recommendations:
        pair_key = f"{r['anchor_product']} + {r['pair_product']}"
        pair_counter[pair_key] += 1
    top_pairs = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)[:3]
    top_pairs_text = ", ".join([f"{k} ({v} branches)" for k, v in top_pairs]) if top_pairs else "n/a"

    report_path = os.path.join(OUT_REPORTS, "analysis_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stories Coffee - Analysis Summary\n\n")
        f.write("## Scope\n")
        f.write("- Data sources: four raw POS exports (2025 full year + January 2026)\n")
        f.write("- Revenue correction applied for `rep_s_00673_SMRY.csv` and subtotal rows where needed: `revenue = total_cost + total_profit`\n")
        f.write("- Branch names normalized to consistent canonical labels\n")
        f.write(f"- January 2026 treated as month-to-date snapshot (day {snapshot_day}) with run-rate normalization\n\n")

        f.write("## Key Findings\n")
        f.write("1. **Mixed January 2026 momentum, with growth concentrated in a small subset of branches.**\n")
        for r in jan_growth_ranked[:5]:
            f.write(f"   - {r['branch']}: run-rate Jan YoY {r['jan_runrate_yoy_pct']:.1f}% ({r['jan_2025']:.0f} vs run-rate {r['jan_2026_runrate']:.0f})\n")
        f.write("2. **Several legacy branches remain materially below prior-year pace and need local action plans.**\n")
        for r in jan_decline_ranked[:5]:
            f.write(f"   - {r['branch']}: run-rate Jan YoY {r['jan_runrate_yoy_pct']:.1f}%\n")
        f.write("3. **Profit concentration is heavy in beverage/frozen-yogurt SKUs.**\n")
        for r in top5_products:
            f.write(f"   - {r['product_desc']}: profit {r['profit']:.0f}, margin {r['margin_pct']:.1f}%\n")
        f.write("4. **High-volume low-margin outliers are clear and actionable.**\n")
        for r in low_margin5:
            f.write(f"   - {r['product_desc']}: qty {r['qty']:.0f}, margin {r['margin_pct']:.1f}%\n")
        f.write("5. **POS line-item view: replacement-milk modifiers can appear negative because revenue is recorded on the parent drink line, not on the modifier line.**\n")
        for r in worst_mod:
            f.write(f"   - {r['product_desc']}: profit {r['profit']:.0f}, margin {r['margin_pct']:.1f}%\n")

        f.write("\n## Top 2025 Branches by Annual Sales\n")
        for r in top5_branches:
            f.write(f"- {r['branch']}: {r['total_by_year']:.0f}\n")

        f.write("\n## Top Group Categories by Sales\n")
        for r in best_group:
            f.write(f"- {r['group_name']}: {r['sales']:.0f}\n")

        f.write("\n## Exploratory Analysis (Patterns, Seasonality, Anomalies)\n")
        f.write(f"- Hidden pattern: top {top80_sku_count} products account for ~80% of positive product profit (strong Pareto concentration).\n")
        f.write(f"- Hidden pattern: top 3 branches contribute {top3_branch_share:.1f}% of total 2025 sales.\n")
        f.write(
            f"- Seasonality: network peak month is {peak_month['month']} (index {peak_month['seasonality_index']:.2f}) "
            f"while trough month is {trough_month['month']} (index {trough_month['seasonality_index']:.2f}).\n"
        )
        if volatile_branches:
            f.write("- Branches with highest monthly volatility (CV%): " + ", ".join([f"{r['branch']} ({r['cv_pct']:.1f}%)" for r in volatile_branches]) + "\n")
        f.write(f"- Anomaly scan flagged {len(branch_anomaly_flags)} branch-month points for manual review (statistical outlier or sharp month-over-month shift).\n")
        for r in top_branch_anomalies[:3]:
            f.write(f"  - {r['branch']} / {r['month']}: z={r['robust_zscore']:.2f}, mom={r['mom_pct']}\n")
        if network_anomaly_flags:
            f.write("- Network-level anomaly months: " + ", ".join([r["month"] for r in network_anomaly_flags]) + "\n")

        f.write("\n## Predictive Modeling (ML Layer)\n")
        yoy_text = f"{network_forecast_yoy:.1f}%" if network_forecast_yoy is not None else "n/a"
        f.write(
            f"- Time-series forecast (trend x seasonality, January-anchored) projects 2026 network sales at {network_forecast_2026_total:.0f} "
            f"vs {network_sales_2025_total:.0f} in 2025 (YoY {yoy_text}).\n"
        )
        f.write(f"- Branch performance model (linear regression) uses margin/mix/volatility/anomaly features: R2={model_r2:.2f}, RMSE={model_rmse:.2f} YoY points (in-sample).\n")
        if pred_top:
            if pred_top[0]["predicted_jan_runrate_yoy_pct"] >= 0:
                f.write("- Predicted strongest 2026 branch momentum: " + ", ".join([f"{r['branch']} ({r['predicted_jan_runrate_yoy_pct']:.1f}%)" for r in pred_top]) + "\n")
            else:
                f.write("- Model indicates broad contraction risk; least-negative branches are: " + ", ".join([f"{r['branch']} ({r['predicted_jan_runrate_yoy_pct']:.1f}%)" for r in pred_top]) + "\n")
        if pred_risk:
            f.write("- Predicted highest-risk branches: " + ", ".join([f"{r['branch']} ({r['predicted_jan_runrate_yoy_pct']:.1f}%)" for r in pred_risk]) + "\n")
        if cluster_count > 0:
            f.write(f"- K-means segmentation identified {cluster_count} branch operating archetypes: {cluster_summary_str}.\n")

        f.write("\n## Location Analysis (Branch Segmentation)\n")
        if cluster_count > 0:
            f.write(f"- Branches were segmented into {cluster_count} operational archetypes: {cluster_summary_str}.\n")
        if archetype_top:
            top_arc = archetype_top[0]
            f.write(
                f"- Largest archetype: {top_arc['cluster_label']} ({top_arc['member_count']} branches), "
                f"with avg beverage share {top_arc['beverage_share_pct']:.1f}% and margin {top_arc['overall_margin_pct']:.1f}%.\n"
            )
        f.write("- Archetype playbooks were generated for offer design, pricing, and operations by location type.\n")

        f.write("\n## Optimization (Menu Engineering + Offer Engine)\n")
        f.write(
            f"- ML prioritization flagged {opt_target_count} branches as low-sales or decline-risk (bottom-sales threshold {low_sales_threshold:.0f}).\n"
        )
        f.write(
            f"- Generated {opt_offer_count} branch-level bundle offers with estimated total upside of +{opt_total_incremental_revenue:.0f} revenue and +{opt_total_incremental_profit:.0f} gross profit units.\n"
        )
        f.write(f"- Most reusable product pair themes across branches: {top_pairs_text}.\n")
        f.write("- Offer logic: keep anchor products already strong in each branch, then attach high-affinity under-indexed pair items through combo pricing.\n")

        f.write("\n## Executive Insight\n")
        f.write("Stories operates economically as a beverage and frozen-yogurt driven business, where a small group of high-margin beverage SKUs generates most profitability.\n\n")
        f.write("However, selected food items and replacement-milk pricing/cost gaps can create incremental margin leakage even when drinks are not discounted.\n\n")
        f.write("The strategic opportunity is to scale the beverage-led growth engine while systematically removing loss-making products and pricing inconsistencies.\n")

        f.write("\n## Implementation Roadmap\n")
        f.write("- Scale the **beverage-led growth engine**: prioritize inventory, staffing, and promotion support for beverage-dominant branches/groups with strongest contribution.\n")
        f.write("- Run a **menu margin sprint**: target high-volume products with <60% margin for recipe, pricing, or bundle redesign.\n")
        f.write("- Review **replacement-milk economics** using incremental costing: keep base drink pricing, and add/adjust surcharges only where replacement ingredients cost more.\n")
        f.write("- Use **seasonality planning** for inventory/labor calibration around peak-demand months.\n")

        f.write("\n## Expected Impact\n")
        f.write("- Improving VEGGIE SUB to a minimum positive margin could recover approximately +1.79M gross profit.\n")
        f.write(f"- Converting the current replacement-milk line-loss pool into cost-neutral pricing (after costing validation) represents up to +{modifier_loss_pool/1_000_000:.2f}M gross profit.\n")
        f.write(f"- Top 3 branches contribute about {top3_branch_share:.1f}% of 2025 sales, and top 3 product groups contribute about {top3_group_share:.1f}% of group sales; focused execution here maximizes ROI.\n")
        if aug_june_ratio > 0:
            f.write(f"- August demand is approximately {aug_june_ratio:.1f}x June, supporting stronger seasonal staffing/inventory planning.\n")

        f.write("\n## Confidence and Limitations\n")
        f.write("Results reflect POS export structure and should be validated against master product costing and branch metadata for production deployment. Modifier profitability in these exports is line-item based and should be interpreted as incremental, not full-drink profitability.\n")

    print(f"Wrote outputs to: {OUT_TABLES}")
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    make_outputs()
