from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
OUT_DIR = ROOT / "amap_selfdrive_model_outputs"
OUTPUT_DIR = ROOT / "outputs"

DAY_HOURS = 8.0
CAR_COST_YUAN_PER_KM_FOR_TWO = 0.55
HOTEL_ROOM_YUAN_PER_NIGHT = 260.0
EXTERNAL_FLIGHT_PROXY_YUAN_FOR_TWO = 6000.0


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def num(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spots = pd.read_csv(DATA_DIR / "spot_clean.csv", encoding="utf-8-sig").reset_index(drop=True)
    od = pd.read_csv(ENHANCED_DIR / "amap_driving_od_matrix_clean.csv", encoding="utf-8-sig")
    depot = pd.read_csv(ENHANCED_DIR / "amap_depot_access_matrix_clean.csv", encoding="utf-8-sig")
    return spots, od, depot


def build_matrices(spots: pd.DataFrame, od: pd.DataFrame, depot: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(spots)
    sid_to_idx = {sid: i + 1 for i, sid in enumerate(spots["spot_id"])}
    time = np.zeros((n + 1, n + 1), dtype=float)
    dist = np.zeros((n + 1, n + 1), dtype=float)
    cost = np.zeros((n + 1, n + 1), dtype=float)

    for _, row in od.iterrows():
        i = sid_to_idx[row["from_spot_id"]]
        j = sid_to_idx[row["to_spot_id"]]
        time[i, j] = num(row["driving_duration_hours"])
        dist[i, j] = num(row["driving_distance_km"])
        cost[i, j] = num(row.get("amap_selfdrive_cost_yuan_per_two"), dist[i, j] * CAR_COST_YUAN_PER_KM_FOR_TWO)

    for _, row in depot.iterrows():
        j = sid_to_idx[row["spot_id"]]
        if row["direction"] == "depot_to_spot":
            time[0, j] = num(row["driving_duration_hours"])
            dist[0, j] = num(row["driving_distance_km"])
            cost[0, j] = num(row["selfdrive_cost_yuan_per_two"])
        else:
            time[j, 0] = num(row["driving_duration_hours"])
            dist[j, 0] = num(row["driving_distance_km"])
            cost[j, 0] = num(row["selfdrive_cost_yuan_per_two"])
    return time, dist, cost


def service_hours(row: pd.Series, visit_multiplier: float = 1.0) -> float:
    # 高德OD已经到景点POI，避免重复加入原“城市到景区”的local_access时间。
    return num(row.get("visit_hours_mid"), 0.0) * visit_multiplier


def sequence_travel(seq: list[int], time: np.ndarray, dist: np.ndarray, cost: np.ndarray, return_to_depot: bool = True) -> tuple[float, float, float]:
    cur = 0
    travel = distance = money = 0.0
    for idx in seq:
        travel += time[cur, idx]
        distance += dist[cur, idx]
        money += cost[cur, idx]
        cur = idx
    if return_to_depot:
        travel += time[cur, 0]
        distance += dist[cur, 0]
        money += cost[cur, 0]
    return travel, distance, money


def route_metrics(
    seq: list[int],
    spots: pd.DataFrame,
    time: np.ndarray,
    dist: np.ndarray,
    cost: np.ndarray,
    visit_multiplier: float = 1.0,
    return_to_depot: bool = True,
    include_external_flight: bool = False,
) -> dict[str, float]:
    travel, distance, selfdrive_cost = sequence_travel(seq, time, dist, cost, return_to_depot=return_to_depot)
    service = sum(service_hours(spots.iloc[i - 1], visit_multiplier=visit_multiplier) for i in seq)
    total_hours = travel + service
    days = math.ceil(total_hours / DAY_HOURS) if total_hours > 0 else 0
    nights = max(days - 1, 0)
    tickets = sum(num(spots.iloc[i - 1]["ticket_high_total_yuan_per_person"]) for i in seq) * 2
    hotel = nights * HOTEL_ROOM_YUAN_PER_NIGHT
    external = EXTERNAL_FLIGHT_PROXY_YUAN_FOR_TWO if include_external_flight else 0.0
    return {
        "spots_count": float(len(seq)),
        "travel_hours": round(travel, 2),
        "service_hours": round(service, 2),
        "total_hours": round(total_hours, 2),
        "estimated_days": float(days),
        "nights": float(nights),
        "distance_km": round(distance, 1),
        "ticket_yuan_for_two": round(tickets, 1),
        "selfdrive_cost_yuan_for_two": round(selfdrive_cost, 1),
        "hotel_proxy_yuan_room": round(hotel, 1),
        "external_flight_proxy_yuan_for_two": round(external, 1),
        "total_proxy_yuan_excluding_meals": round(tickets + selfdrive_cost + hotel + external, 1),
    }


def two_opt(seq: list[int], time: np.ndarray, return_to_depot: bool = True, max_iter: int = 80) -> list[int]:
    if len(seq) < 4:
        return seq

    def travel(s: list[int]) -> float:
        cur = 0
        total = 0.0
        for idx in s:
            total += time[cur, idx]
            cur = idx
        if return_to_depot:
            total += time[cur, 0]
        return total

    best = seq[:]
    best_value = travel(best)
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best) + 1):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                val = travel(cand)
                if val < best_value - 1e-9:
                    best = cand
                    best_value = val
                    improved = True
    return best


def nearest_sequence(indices: list[int], spots: pd.DataFrame, time: np.ndarray, return_to_depot: bool = True) -> list[int]:
    remaining = set(indices)
    route: list[int] = []
    cur = 0
    while remaining:
        nxt = min(
            remaining,
            key=lambda idx: (
                time[cur, idx],
                -num(spots.iloc[idx - 1]["priority_score_for_op"]),
                spots.iloc[idx - 1]["spot_id"],
            ),
        )
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return two_opt(route, time, return_to_depot=return_to_depot)


def select_30day_route(spots: pd.DataFrame, time: np.ndarray, dist: np.ndarray, cost: np.ndarray) -> list[int]:
    candidates = [i + 1 for i, row in spots.iterrows() if not truthy(row["ordinary_tourist_restricted"])]
    full = nearest_sequence(candidates, spots, time)
    if route_metrics(full, spots, time, dist, cost, include_external_flight=True)["total_hours"] <= 30 * DAY_HOURS:
        return full

    selected: list[int] = []
    remaining = set(candidates)
    while remaining:
        base = route_metrics(nearest_sequence(selected, spots, time), spots, time, dist, cost)["total_hours"] if selected else 0.0
        best_idx = None
        best_route = None
        best_score = -math.inf
        for idx in remaining:
            trial = nearest_sequence(selected + [idx], spots, time)
            m = route_metrics(trial, spots, time, dist, cost, include_external_flight=True)
            if m["total_hours"] > 30 * DAY_HOURS:
                continue
            value = num(spots.iloc[idx - 1]["priority_score_for_op"], 1.0)
            score = value / max(m["total_hours"] - base, 0.5)
            if score > best_score:
                best_idx = idx
                best_route = trial
                best_score = score
        if best_idx is None or best_route is None:
            break
        selected = best_route
        remaining.remove(best_idx)
    return selected


def route_detail(
    route_id: str,
    seq: list[int],
    spots: pd.DataFrame,
    time: np.ndarray,
    dist: np.ndarray,
    return_to_depot: bool = True,
    visit_multiplier: float = 1.0,
) -> pd.DataFrame:
    rows = []
    cur = 0
    cumulative = 0.0
    for order, idx in enumerate(seq, 1):
        row = spots.iloc[idx - 1]
        travel = time[cur, idx]
        service = service_hours(row, visit_multiplier=visit_multiplier)
        cumulative += travel + service
        rows.append(
            {
                "route_id": route_id,
                "order": order,
                "spot_id": row["spot_id"],
                "spot_name": row["spot_name"],
                "region_cluster": row["region_cluster"],
                "travel_from_previous_hours": round(travel, 2),
                "distance_from_previous_km": round(dist[cur, idx], 1),
                "service_hours_used": round(service, 2),
                "cumulative_hours": round(cumulative, 2),
                "cumulative_day_equiv": round(cumulative / DAY_HOURS, 2),
                "requires_approval": row["requires_approval"],
                "requires_border_permit": row["requires_border_permit"],
            }
        )
        cur = idx
    if return_to_depot and seq:
        cumulative += time[cur, 0]
        rows.append(
            {
                "route_id": route_id,
                "order": len(seq) + 1,
                "spot_id": "END",
                "spot_name": "返回乌鲁木齐",
                "region_cluster": "",
                "travel_from_previous_hours": round(time[cur, 0], 2),
                "distance_from_previous_km": round(dist[cur, 0], 1),
                "service_hours_used": 0.0,
                "cumulative_hours": round(cumulative, 2),
                "cumulative_day_equiv": round(cumulative / DAY_HOURS, 2),
                "requires_approval": "",
                "requires_border_permit": "",
            }
        )
    return pd.DataFrame(rows)


def make_problem2_routes(spots: pd.DataFrame, time: np.ndarray) -> dict[str, tuple[list[int], bool]]:
    base = spots[~spots["ordinary_tourist_restricted"].map(truthy)]
    year1_clusters = {"乌鲁木齐周边", "北疆", "伊犁"}
    year1 = [i + 1 for i in base.index[base["region_cluster"].isin(year1_clusters)]]
    year2 = [i + 1 for i in base.index[~base["region_cluster"].isin(year1_clusters)]]
    optional = [i + 1 for i, row in spots.iterrows() if truthy(row["ordinary_tourist_restricted"])]
    return {
        "P2_AMAP_Y1_North_Ili_loop": (nearest_sequence(year1, spots, time, return_to_depot=True), True),
        "P2_AMAP_Y2_East_South_open": (nearest_sequence(year2, spots, time, return_to_depot=False), False),
        "P2_AMAP_Y2_Approval_Extension_open": (nearest_sequence(year2 + optional, spots, time, return_to_depot=False), False),
    }


def cultural_candidates(spots: pd.DataFrame) -> list[int]:
    return [i + 1 for i, row in spots.iterrows() if truthy(row["is_cultural"]) or truthy(row["ordinary_tourist_restricted"])]


def greedy_minmax_partition(indices: list[int], spots: pd.DataFrame, time: np.ndarray, dist: np.ndarray, cost: np.ndarray, k: int = 3) -> list[list[int]]:
    ordered = sorted(
        indices,
        key=lambda idx: (time[0, idx] + service_hours(spots.iloc[idx - 1], 4), num(spots.iloc[idx - 1]["priority_score_for_op"])),
        reverse=True,
    )
    groups: list[list[int]] = [[] for _ in range(k)]
    for idx in ordered:
        best_group = 0
        best_obj = (math.inf, math.inf)
        for g in range(k):
            trial_groups = [x[:] for x in groups]
            trial_groups[g].append(idx)
            route_times = []
            for group in trial_groups:
                route = nearest_sequence(group, spots, time) if group else []
                route_times.append(route_metrics(route, spots, time, dist, cost, visit_multiplier=4)["total_hours"])
            obj = (max(route_times), sum(route_times))
            if obj < best_obj:
                best_group = g
                best_obj = obj
        groups[best_group].append(idx)
    return [nearest_sequence(group, spots, time) for group in groups]


def route_from_names(names: Iterable[str], spots: pd.DataFrame, time: np.ndarray, return_to_depot: bool = True) -> list[int]:
    lookup = {row["spot_name"]: i + 1 for i, row in spots.iterrows()}
    return nearest_sequence([lookup[name] for name in names if name in lookup], spots, time, return_to_depot=return_to_depot)


def build_12day_library(spots: pd.DataFrame, time: np.ndarray, dist: np.ndarray, cost: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "R1_AMAP_Classic_North": ["天山天池", "世界魔鬼城", "喀纳斯湖", "禾木村", "五彩滩"],
        "R2_AMAP_Ili_Grassland": ["赛里木湖", "果子沟大桥", "霍城薰衣草花田", "那拉提草原", "喀拉峻大草原", "巴音布鲁克草原"],
        "R3_AMAP_East_Culture": ["达坂城古镇", "火焰山", "葡萄沟", "坎儿井民俗园", "交河故城", "高昌故城", "苏公塔", "吐峪沟麻扎村"],
        "R4_AMAP_South_Culture": ["喀什古城", "艾提尕尔清真寺", "香妃园", "帕米尔高原白沙湖", "石头城遗址", "天山神秘大峡谷", "库车王府", "克孜尔石窟"],
        "R5_AMAP_Bazhou_Desert": ["博斯腾湖", "罗布人村寨", "巴音布鲁克草原", "天山神秘大峡谷", "库车王府"],
        "R6_AMAP_Hotan_Extension": ["喀什古城", "和田博物馆", "千里葡萄长廊", "奥依塔克冰川"],
    }
    summaries = []
    details = []
    for route_id, names in specs.items():
        # 南疆延伸线按开放路径处理，表示可从喀什/和田离疆。
        return_to_depot = route_id not in {"R4_AMAP_South_Culture", "R6_AMAP_Hotan_Extension"}
        seq = route_from_names(names, spots, time, return_to_depot=return_to_depot)
        m = route_metrics(seq, spots, time, dist, cost, return_to_depot=return_to_depot)
        summaries.append(
            {
                "route_id": route_id,
                "spots_count": int(m["spots_count"]),
                "estimated_days": int(m["estimated_days"]),
                "within_12day_limit": m["estimated_days"] <= 12,
                "distance_km": m["distance_km"],
                "total_hours": m["total_hours"],
                "proxy_cost_yuan_excluding_meals": m["total_proxy_yuan_excluding_meals"],
                "return_to_urumqi": return_to_depot,
                "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in seq),
            }
        )
        details.append(route_detail(route_id, seq, spots, time, dist, return_to_depot=return_to_depot))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无数据）"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in df.columns) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, object], comparisons: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "新疆旅游高德驾车OD重实验报告.md"
    text = f"""# 新疆旅游线路安排：高德真实驾车OD重实验报告

## 1. 为什么要重跑

原有图论模型使用“枢纽公路边表 + Dijkstra闭包”估计景点间时间。现在已经取得高德 Web 服务返回的景点 POI 坐标和真实驾车 OD，因此应当新增一轮自驾/公路版实验，用真实景点到景点 OD 替代旧的枢纽闭包。

注意：本实验替换的是公路/自驾层，不直接替代铁路、航空、大巴构成的多层交通模型。

## 2. 数据替换口径

- 坐标：40 个景点全部使用高德 POI 坐标，匹配质量均为 high。
- OD：40 x 40 = 1600 条高德驾车 OD，自环已强制清洗为 0。
- 乌鲁木齐接入：新增乌鲁木齐市人民政府到各景点、各景点返回乌鲁木齐的 80 条驾车 OD。
- 服务时间：使用景点游玩时长 `visit_hours_mid`，不再叠加原始“城市到景区local_access”时间，避免与高德点到点驾车 OD 重复计算。

## 3. 四问重实验摘要

### 第一问：30天低成本多景点游

- 覆盖景点数：{summary['p1']['spots_count']}
- 高德驾车时间：{summary['p1']['travel_hours']} 小时
- 游玩服务时间：{summary['p1']['service_hours']} 小时
- 总时间：{summary['p1']['total_hours']} 小时，折算 {summary['p1']['estimated_days']} 天
- 高德驾车里程：{summary['p1']['distance_km']} km
- 两人不含餐费代理成本：{summary['p1']['total_proxy_yuan_excluding_meals']} 元

### 第二问：两年暑假路径覆盖

{summary['p2_table']}

### 第三问：三组文化考察 Min-Max

{summary['p3_table']}

### 第四问：五一12天候选线路

{summary['p4_table']}

## 4. 与旧枢纽闭包结果的差异

{markdown_table(comparisons)}

## 5. 建模结论

1. 高德 OD 更适合作为“公路/自驾真实边权”，能修正枢纽闭包低估或高估的景点间通行时间。
2. 对第一问这类 30 天普通旅游，使用高德 OD 后仍可覆盖较多景点，但路线顺序和里程成本更可信。
3. 对第二问和第四问，高德 OD 会让跨北疆-南疆的路线代价更直观，因此更容易解释为什么要分年、分区或开放式离疆。
4. 对多层交通模型，不应把高德驾车 OD 直接替代铁路/航空 OD；正确做法是用它校准公路层，然后与 12306、航班数据共同组成多层网络。
"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def export_workbook(tables: dict[str, pd.DataFrame]) -> Path:
    path = OUTPUT_DIR / "新疆旅游高德驾车OD重实验结果.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    spots, od, depot = read_inputs()
    time, dist, cost = build_matrices(spots, od, depot)

    p1_seq = select_30day_route(spots, time, dist, cost)
    p1_metrics = route_metrics(p1_seq, spots, time, dist, cost, include_external_flight=True)
    p1_summary = pd.DataFrame([{"route_id": "P1_AMAP_30day_selfdrive_OP", **p1_metrics, "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in p1_seq)}])
    p1_detail = route_detail("P1_AMAP_30day_selfdrive_OP", p1_seq, spots, time, dist)

    p2_rows = []
    p2_details = []
    for route_id, (seq, return_to_depot) in make_problem2_routes(spots, time).items():
        m = route_metrics(seq, spots, time, dist, cost, return_to_depot=return_to_depot)
        p2_rows.append({"route_id": route_id, **m, "return_to_urumqi": return_to_depot, "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in seq)})
        p2_details.append(route_detail(route_id, seq, spots, time, dist, return_to_depot=return_to_depot))
    p2_summary = pd.DataFrame(p2_rows)
    p2_detail = pd.concat(p2_details, ignore_index=True)

    groups = greedy_minmax_partition(cultural_candidates(spots), spots, time, dist, cost)
    p3_rows = []
    p3_details = []
    for i, seq in enumerate(groups, 1):
        route_id = f"P3_AMAP_Group_{i}"
        m = route_metrics(seq, spots, time, dist, cost, visit_multiplier=4)
        p3_rows.append({"route_id": route_id, **m, "route_sequence": " -> ".join(spots.iloc[j - 1]["spot_name"] for j in seq)})
        p3_details.append(route_detail(route_id, seq, spots, time, dist, visit_multiplier=4))
    p3_summary = pd.DataFrame(p3_rows)
    p3_summary["max_completion_hours"] = p3_summary["total_hours"].max()
    p3_summary["balance_gap_hours"] = p3_summary["total_hours"].max() - p3_summary["total_hours"].min()
    p3_detail = pd.concat(p3_details, ignore_index=True)

    p4_summary, p4_detail = build_12day_library(spots, time, dist, cost)

    old_p1_path = ROOT / "graph_model_outputs" / "problem1_summary.csv"
    comparisons = []
    if old_p1_path.exists():
        old = pd.read_csv(old_p1_path, encoding="utf-8-sig").iloc[0]
        comparisons.append({"metric": "P1覆盖景点数", "old_hub_closure": old["spots_count"], "amap_selfdrive": p1_metrics["spots_count"], "delta": p1_metrics["spots_count"] - old["spots_count"]})
        comparisons.append({"metric": "P1旅行时间小时", "old_hub_closure": old["travel_hours"], "amap_selfdrive": p1_metrics["travel_hours"], "delta": p1_metrics["travel_hours"] - old["travel_hours"]})
        comparisons.append({"metric": "P1总小时", "old_hub_closure": old["total_hours"], "amap_selfdrive": p1_metrics["total_hours"], "delta": p1_metrics["total_hours"] - old["total_hours"]})
        comparisons.append({"metric": "P1里程km", "old_hub_closure": old["distance_km"], "amap_selfdrive": p1_metrics["distance_km"], "delta": p1_metrics["distance_km"] - old["distance_km"]})
        comparisons.append({"metric": "P1代理成本元", "old_hub_closure": old["total_proxy_yuan_excluding_meals"], "amap_selfdrive": p1_metrics["total_proxy_yuan_excluding_meals"], "delta": p1_metrics["total_proxy_yuan_excluding_meals"] - old["total_proxy_yuan_excluding_meals"]})
    comparison_df = pd.DataFrame(comparisons)

    tables = {
        "problem1_amap_summary": p1_summary,
        "problem1_amap_detail": p1_detail,
        "problem2_amap_summary": p2_summary,
        "problem2_amap_detail": p2_detail,
        "problem3_amap_summary": p3_summary,
        "problem3_amap_detail": p3_detail,
        "problem4_amap_candidates": p4_summary,
        "problem4_amap_detail": p4_detail,
        "old_vs_amap_comparison": comparison_df,
    }
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    report_context = {
        "p1": p1_metrics,
        "p2_table": markdown_table(p2_summary[["route_id", "spots_count", "estimated_days", "distance_km", "total_proxy_yuan_excluding_meals", "return_to_urumqi"]]),
        "p3_table": markdown_table(p3_summary[["route_id", "spots_count", "estimated_days", "total_hours", "max_completion_hours", "balance_gap_hours"]]),
        "p4_table": markdown_table(p4_summary[["route_id", "spots_count", "estimated_days", "within_12day_limit", "distance_km", "proxy_cost_yuan_excluding_meals"]]),
    }
    report = write_report(report_context, comparison_df)
    workbook = export_workbook(tables)

    summary = {
        "problem1": p1_metrics,
        "problem2_routes": len(p2_summary),
        "problem3_groups": len(p3_summary),
        "problem4_candidate_routes": len(p4_summary),
        "report": str(report),
        "workbook": str(workbook),
    }
    (OUT_DIR / "amap_selfdrive_solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
