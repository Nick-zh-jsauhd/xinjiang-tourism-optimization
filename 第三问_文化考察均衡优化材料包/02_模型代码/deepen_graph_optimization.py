from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
OUT_DIR = ROOT / "graph_model_outputs"
REPORT_DIR = ROOT / "outputs"
DAY_HOURS = 8.0
START_HUB = "乌鲁木齐市"


EXACT_SCENIC_HUBS = {
    "天池风景区",
    "赛里木湖",
    "禾木村",
    "白哈巴",
    "可可托海",
    "巴音布鲁克",
}


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def num(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def build_graph(edges: pd.DataFrame) -> dict[str, list[tuple[str, float, float, str]]]:
    graph: dict[str, list[tuple[str, float, float, str]]] = defaultdict(list)
    for _, row in edges.iterrows():
        graph[row["from_hub_name"]].append(
            (
                row["to_hub_name"],
                num(row["time_hours"], math.inf),
                num(row["distance_km"], math.inf),
                row["edge_id"],
            )
        )
    return graph


def dijkstra(graph: dict[str, list[tuple[str, float, float, str]]], source: str) -> dict[str, dict[str, object]]:
    dist = {source: 0.0}
    distance_km = {source: 0.0}
    prev: dict[str, tuple[str, str]] = {}
    heap = [(0.0, 0.0, source)]
    while heap:
        cur_time, cur_km, node = heapq.heappop(heap)
        if cur_time > dist.get(node, math.inf):
            continue
        for nxt, edge_time, edge_km, edge_id in graph.get(node, []):
            cand = cur_time + edge_time
            cand_km = cur_km + edge_km
            if cand < dist.get(nxt, math.inf) - 1e-9:
                dist[nxt] = cand
                distance_km[nxt] = cand_km
                prev[nxt] = (node, edge_id)
                heapq.heappush(heap, (cand, cand_km, nxt))
    result = {}
    for target in dist:
        path = [target]
        edge_path = []
        cur = target
        while cur != source:
            p, eid = prev[cur]
            edge_path.append(eid)
            cur = p
            path.append(cur)
        result[target] = {
            "time_hours": dist[target],
            "distance_km": distance_km[target],
            "path_hubs": list(reversed(path)),
            "path_edges": list(reversed(edge_path)),
        }
    return result


def all_pairs_shortest(hubs: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    graph = build_graph(edges)
    lookup = {}
    rows = []
    for src in hubs["hub_name"]:
        result = dijkstra(graph, src)
        for dst in hubs["hub_name"]:
            item = result.get(dst)
            if item is None:
                time = math.inf
                km = math.inf
                path_hubs: list[str] = []
                path_edges: list[str] = []
            else:
                time = float(item["time_hours"])
                km = float(item["distance_km"])
                path_hubs = item["path_hubs"]
                path_edges = item["path_edges"]
            lookup[(src, dst)] = {
                "time_hours": time,
                "distance_km": km,
                "path_hubs": path_hubs,
                "path_edges": path_edges,
            }
            rows.append(
                {
                    "from_hub_name": src,
                    "to_hub_name": dst,
                    "shortest_time_hours": None if math.isinf(time) else round(time, 3),
                    "shortest_distance_km": None if math.isinf(km) else round(km, 3),
                    "path_hubs": " -> ".join(path_hubs),
                    "path_edges": " -> ".join(path_edges),
                }
            )
    return pd.DataFrame(rows), lookup


def service_hours(row: pd.Series, visit_multiplier: float = 1.0) -> float:
    visit = num(row.get("visit_hours_mid"), 0.0) * visit_multiplier
    hub = str(row.get("hub_name", ""))
    local = num(row.get("local_access_hours_from_text"), 0.0)
    if hub in EXACT_SCENIC_HUBS:
        local = 0.0
    return visit + local


def sequence_travel(seq: list[int], spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], start_hub: str, end_hub: str) -> tuple[float, float]:
    hubs = spots["hub_name"].tolist()
    travel_time = 0.0
    distance = 0.0
    cur = start_hub
    for idx in seq:
        nxt = hubs[idx]
        item = closure[(cur, nxt)]
        travel_time += float(item["time_hours"])
        distance += float(item["distance_km"])
        cur = nxt
    item = closure[(cur, end_hub)]
    travel_time += float(item["time_hours"])
    distance += float(item["distance_km"])
    return travel_time, distance


def two_opt(seq: list[int], spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], start_hub: str, end_hub: str, max_iter: int = 200) -> list[int]:
    if len(seq) < 4:
        return seq
    best = seq[:]
    best_time, _ = sequence_travel(best, spots, closure, start_hub, end_hub)
    improved = True
    count = 0
    while improved and count < max_iter:
        count += 1
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand_time, _ = sequence_travel(candidate, spots, closure, start_hub, end_hub)
                if cand_time < best_time - 1e-9:
                    best = candidate
                    best_time = cand_time
                    improved = True
        seq = best
    return best


def sequence_travel_hubs(seq: list[str], closure: dict[tuple[str, str], dict[str, object]], start_hub: str, end_hub: str) -> tuple[float, float]:
    travel_time = 0.0
    distance = 0.0
    cur = start_hub
    for nxt in seq:
        item = closure[(cur, nxt)]
        travel_time += float(item["time_hours"])
        distance += float(item["distance_km"])
        cur = nxt
    item = closure[(cur, end_hub)]
    travel_time += float(item["time_hours"])
    distance += float(item["distance_km"])
    return travel_time, distance


def two_opt_hubs(seq: list[str], closure: dict[tuple[str, str], dict[str, object]], start_hub: str, end_hub: str, max_iter: int = 200) -> list[str]:
    if len(seq) < 4:
        return seq
    best = seq[:]
    best_time, _ = sequence_travel_hubs(best, closure, start_hub, end_hub)
    improved = True
    count = 0
    while improved and count < max_iter:
        count += 1
        improved = False
        for i in range(0, len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand_time, _ = sequence_travel_hubs(candidate, closure, start_hub, end_hub)
                if cand_time < best_time - 1e-9:
                    best = candidate
                    best_time = cand_time
                    improved = True
    return best


def nearest_sequence(candidate_indices: list[int], spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], start_hub: str, end_hub: str) -> list[int]:
    if not candidate_indices:
        return []
    hub_to_spots: dict[str, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        hub_to_spots[str(spots.loc[idx, "hub_name"])].append(idx)
    remaining = set(hub_to_spots)
    hub_seq: list[str] = []
    cur = start_hub
    while remaining:
        nxt = min(
            remaining,
            key=lambda hub: (
                float(closure[(cur, hub)]["time_hours"]),
                -sum(num(spots.loc[idx, "priority_score_for_op"], 0.0) for idx in hub_to_spots[hub]),
            ),
        )
        hub_seq.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    hub_seq = two_opt_hubs(hub_seq, closure, start_hub, end_hub)
    seq: list[int] = []
    for hub in hub_seq:
        seq.extend(
            sorted(
                hub_to_spots[hub],
                key=lambda idx: (
                    -num(spots.loc[idx, "priority_score_for_op"], 0.0),
                    num(spots.loc[idx, "visit_hours_mid"], 0.0),
                    spots.loc[idx, "spot_id"],
                ),
            )
        )
    return seq


def compress_hub_jumps(route_detail_df: pd.DataFrame) -> int:
    hubs = [h for h in route_detail_df["hub_name"].tolist() if isinstance(h, str) and h]
    return sum(1 for a, b in zip(hubs, hubs[1:]) if a != b)


def route_metrics(
    seq: list[int],
    spots: pd.DataFrame,
    closure: dict[tuple[str, str], dict[str, object]],
    start_hub: str,
    end_hub: str,
    visit_multiplier: float = 1.0,
    include_external_flight: bool = False,
) -> dict[str, float]:
    travel, distance = sequence_travel(seq, spots, closure, start_hub, end_hub)
    service = sum(service_hours(spots.loc[idx], visit_multiplier=visit_multiplier) for idx in seq)
    total_hours = travel + service
    days = math.ceil(total_hours / DAY_HOURS)
    nights = max(days - 1, 0)
    tickets = sum(num(spots.loc[idx, "ticket_high_total_yuan_per_person"], 0.0) for idx in seq) * 2
    road_public_proxy = distance * 0.25 * 2
    hotel_avg = 210.0
    hotel = nights * hotel_avg
    external = 6000.0 if include_external_flight else 0.0
    return {
        "spots_count": float(len(seq)),
        "travel_hours": round(travel, 2),
        "service_hours": round(service, 2),
        "total_hours": round(total_hours, 2),
        "estimated_days": float(days),
        "nights": float(nights),
        "distance_km": round(distance, 1),
        "ticket_yuan_for_two": round(tickets, 1),
        "transport_proxy_yuan_for_two": round(road_public_proxy, 1),
        "hotel_proxy_yuan_room": round(hotel, 1),
        "external_flight_proxy_yuan_for_two": round(external, 1),
        "total_proxy_yuan_excluding_meals": round(tickets + road_public_proxy + hotel + external, 1),
    }


def route_detail(
    route_id: str,
    seq: list[int],
    spots: pd.DataFrame,
    closure: dict[tuple[str, str], dict[str, object]],
    start_hub: str,
    end_hub: str,
    visit_multiplier: float = 1.0,
) -> pd.DataFrame:
    rows = []
    cur = start_hub
    cumulative = 0.0
    for order, idx in enumerate(seq, 1):
        row = spots.loc[idx]
        item = closure[(cur, row["hub_name"])]
        travel = float(item["time_hours"])
        service = service_hours(row, visit_multiplier=visit_multiplier)
        cumulative += travel + service
        rows.append(
            {
                "route_id": route_id,
                "order": order,
                "spot_id": row["spot_id"],
                "spot_name": row["spot_name"],
                "region_cluster": row["region_cluster"],
                "hub_name": row["hub_name"],
                "travel_from_previous_hub_hours": round(travel, 2),
                "service_hours_used": round(service, 2),
                "cumulative_hours": round(cumulative, 2),
                "cumulative_day_equiv": round(cumulative / DAY_HOURS, 2),
                "ticket_total_yuan_per_person": row["ticket_high_total_yuan_per_person"],
                "requires_approval": row["requires_approval"],
                "requires_border_permit": row["requires_border_permit"],
                "map_confidence": row["map_confidence"],
            }
        )
        cur = row["hub_name"]
    if seq:
        end_item = closure[(cur, end_hub)]
        rows.append(
            {
                "route_id": route_id,
                "order": len(seq) + 1,
                "spot_id": "END",
                "spot_name": f"返回/离疆：{end_hub}",
                "region_cluster": "",
                "hub_name": end_hub,
                "travel_from_previous_hub_hours": round(float(end_item["time_hours"]), 2),
                "service_hours_used": 0.0,
                "cumulative_hours": round(cumulative + float(end_item["time_hours"]), 2),
                "cumulative_day_equiv": round((cumulative + float(end_item["time_hours"])) / DAY_HOURS, 2),
                "ticket_total_yuan_per_person": "",
                "requires_approval": "",
                "requires_border_permit": "",
                "map_confidence": "",
            }
        )
    return pd.DataFrame(rows)


def select_op_route(spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], limit_hours: float) -> list[int]:
    candidate = spots.index[~spots["ordinary_tourist_restricted"].map(truthy)].tolist()
    full_seq = nearest_sequence(candidate, spots, closure, START_HUB, START_HUB)
    metrics = route_metrics(full_seq, spots, closure, START_HUB, START_HUB)
    if metrics["total_hours"] <= limit_hours:
        return full_seq
    selected: list[int] = []
    remaining = set(candidate)
    while remaining:
        best_idx = None
        best_score = -math.inf
        best_seq = None
        for idx in remaining:
            trial = nearest_sequence(selected + [idx], spots, closure, START_HUB, START_HUB)
            trial_metrics = route_metrics(trial, spots, closure, START_HUB, START_HUB)
            if trial_metrics["total_hours"] > limit_hours:
                continue
            score = num(spots.loc[idx, "priority_score_for_op"], 1.0) / max(trial_metrics["total_hours"] - route_metrics(selected, spots, closure, START_HUB, START_HUB)["total_hours"], 1.0)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_seq = trial
        if best_idx is None or best_seq is None:
            break
        selected = best_seq
        remaining.remove(best_idx)
    return selected


def make_problem2_routes(spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]]) -> dict[str, list[int]]:
    base = spots[~spots["ordinary_tourist_restricted"].map(truthy)]
    year1_clusters = {"乌鲁木齐周边", "北疆", "伊犁"}
    year1 = base.index[base["region_cluster"].isin(year1_clusters)].tolist()
    year2 = base.index[~base["region_cluster"].isin(year1_clusters)].tolist()
    optional = spots.index[spots["ordinary_tourist_restricted"].map(truthy)].tolist()
    return {
        "P2_Y1_North_Ili": nearest_sequence(year1, spots, closure, START_HUB, START_HUB),
        "P2_Y2_East_South": nearest_sequence(year2, spots, closure, START_HUB, "喀什市"),
        "P2_Y2_Approval_Extension": nearest_sequence(year2 + optional, spots, closure, START_HUB, "喀什市"),
    }


def cultural_candidates(spots: pd.DataFrame) -> list[int]:
    mask = spots["is_cultural"].map(truthy) | spots["ordinary_tourist_restricted"].map(truthy)
    return spots.index[mask].tolist()


def greedy_minmax_partition(indices: list[int], spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], k: int = 3) -> list[list[int]]:
    start_times = {idx: float(closure[(START_HUB, spots.loc[idx, "hub_name"])]["time_hours"]) for idx in indices}
    ordered = sorted(indices, key=lambda idx: (start_times[idx], service_hours(spots.loc[idx], 4)), reverse=True)
    groups: list[list[int]] = [[] for _ in range(k)]
    for idx in ordered:
        best_group = 0
        best_max = math.inf
        best_sum = math.inf
        for g in range(k):
            trial_groups = [x[:] for x in groups]
            trial_groups[g].append(idx)
            route_times = []
            for members in trial_groups:
                if not members:
                    route_times.append(0.0)
                else:
                    seq = nearest_sequence(members, spots, closure, START_HUB, START_HUB)
                    route_times.append(route_metrics(seq, spots, closure, START_HUB, START_HUB, visit_multiplier=4)["total_hours"])
            max_time = max(route_times)
            sum_time = sum(route_times)
            if (max_time, sum_time) < (best_max, best_sum):
                best_group = g
                best_max = max_time
                best_sum = sum_time
        groups[best_group].append(idx)
    return [nearest_sequence(group, spots, closure, START_HUB, START_HUB) for group in groups]


def route_from_names(names: Iterable[str], spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]], start: str = START_HUB, end: str = START_HUB) -> list[int]:
    lookup = {row["spot_name"]: idx for idx, row in spots.iterrows()}
    indices = [lookup[name] for name in names if name in lookup]
    return nearest_sequence(indices, spots, closure, start, end)


def build_12day_library(spots: pd.DataFrame, closure: dict[tuple[str, str], dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_specs = {
        "R1_Classic_North": ["天山天池", "世界魔鬼城", "喀纳斯湖", "禾木村", "五彩滩"],
        "R2_Ili_Grassland": ["赛里木湖", "果子沟大桥", "霍城薰衣草花田", "那拉提草原", "喀拉峻大草原", "巴音布鲁克草原"],
        "R3_East_Culture": ["达坂城古镇", "火焰山", "葡萄沟", "坎儿井民俗园", "交河故城", "高昌故城", "苏公塔", "吐峪沟麻扎村"],
        "R4_South_Culture": ["喀什古城", "艾提尕尔清真寺", "香妃园", "帕米尔高原白沙湖", "石头城遗址", "天山神秘大峡谷", "库车王府", "克孜尔石窟"],
        "R5_Bazhou_Desert_Lake": ["博斯腾湖", "罗布人村寨", "巴音布鲁克草原", "天山神秘大峡谷", "库车王府"],
        "R6_Hotan_Extension": ["喀什古城", "和田博物馆", "千里葡萄长廊", "奥依塔克冰川"],
    }
    summaries = []
    details = []
    for route_id, names in route_specs.items():
        end = "喀什市" if route_id in {"R4_South_Culture", "R6_Hotan_Extension"} else START_HUB
        seq = route_from_names(names, spots, closure, START_HUB, end)
        metrics = route_metrics(seq, spots, closure, START_HUB, end)
        summaries.append(
            {
                "route_id": route_id,
                "route_theme": route_id.split("_", 1)[1],
                "spots_count": int(metrics["spots_count"]),
                "estimated_days": int(metrics["estimated_days"]),
                "within_12day_limit": metrics["estimated_days"] <= 12,
                "distance_km": metrics["distance_km"],
                "total_hours": metrics["total_hours"],
                "proxy_cost_yuan_excluding_meals": metrics["total_proxy_yuan_excluding_meals"],
                "capacity_status": "capacity_data_missing",
                "route_sequence": " -> ".join(spots.loc[idx, "spot_name"] for idx in seq),
            }
        )
        details.append(route_detail(route_id, seq, spots, closure, START_HUB, end))
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def write_report(context: dict[str, object]) -> Path:
    path = REPORT_DIR / "新疆旅游图论深化建模与求解报告.md"
    p1 = context["problem1"]
    p2 = context["problem2"]
    p3 = context["problem3"]
    p4 = context["problem4"]
    text = f"""# 新疆旅游线路安排：图论深化建模与求解报告

## 1. 建模深化的核心改变

原始题目不能直接用一个旅行商问题覆盖。基于清洗后的数据，本文把旅游系统拆成一个两层图：

- 枢纽交通图：`G_H=(H,E_H)`，节点为城市、县镇、交通枢纽，边为公路连接。本次数据有 25 个枢纽节点、56 条有向公路边。
- 景点挂接图：每个景点 `i∈P` 通过 `hub(i)` 挂接到最近枢纽。这样可以解决景点名与路网节点名不一致的问题。

在枢纽图上先做 Dijkstra 全源最短路，得到 `D_H(u,v)`；再把景点间距离定义为 `D_P(i,j)=D_H(hub(i),hub(j))`。这一步把原始稀疏图转成满足路径规划需要的度量闭包，是后续 OP、两阶段路径覆盖、mTSP 的共同底座。

## 2. 可计算数据结果

- 枢纽最短路闭包：625 对 hub-to-hub 最短路径。
- 景点度量闭包：1600 对 spot-to-spot 旅行时间/距离。
- 数据质量高优先级问题：{context['high_issue_count']} 条，主要是 OD 稀疏、容量缺失、审批景点和季节性道路。

## 3. 第一问：30 天低成本多景点游

推荐使用“字典序定向旅行问题”：先最大化可访问景点效用，再在同等覆盖下最小化费用。普通旅游默认排除 `ordinary_tourist_restricted=True` 的楼兰古城、尼雅遗址。

启发式求解采用：Dijkstra 闭包 + 最近邻构造 + 2-opt 改善。当前数据下，普通可达景点能够全部纳入 30 天时间上限。

- 覆盖景点数：{int(p1['spots_count'])}
- 估计总小时：{p1['total_hours']}
- 折算天数：{int(p1['estimated_days'])}
- 公路闭包距离：{p1['distance_km']} km
- 两人不含餐费代理估计：{p1['total_proxy_yuan_excluding_meals']} 元

注意：这里的费用是模型代理值，交通使用距离-费用代理，住宿按双人房估算；正式报价应再接入 12306、航班、包车实时报价。

## 4. 第二问：两年暑假路径覆盖

第二问不是把景点平均分两半，而是做两阶段路径覆盖。当前建议分法：

- 第一年：乌鲁木齐周边 + 北疆 + 伊犁，减少北疆/伊犁重复往返。
- 第二年：东疆 + 巴州南疆北线 + 南疆，以喀什作为开放式离疆端点。
- 楼兰古城、尼雅遗址作为审批扩展版本，不纳入普通旅游基准路线。

基准两年路线估计：

{p2}

## 5. 第三问：三组文化考察 Min-Max mTSP

第三问应最小化三组中最晚完成时间，而不是最小化总里程。本文筛选文化景点和审批遗址，使用贪心 Min-Max 分配 + 组内 TSP 改善。

三组结果摘要：

{p3}

研究场景中，楼兰古城和尼雅遗址可以进入候选集，但必须增加审批、向导、无人区安全保障等附加时间。当前表中只按 `4×游览时间 + 交通时间` 计算，是下界估计。

## 6. 第四问：五一 12 天路线库

第四问的数据仍缺景区容量、酒店房量、停车容量、摆渡容量、五一游客总量。因此当前只能先生成 12 天候选线路库，并保留容量参数。

候选线路摘要：

{p4}

比例分流应在获得 `Cap_g(r)` 后按题意计算：`f_r = F_total * Cap_g(r) / Σ Cap_g(q)`。若节点拥挤度超过阈值，再调整线路集合，而不是在比例约束已固定的情况下继续把 `f_r` 当自由变量优化。

## 7. 后续精确求解建议

1. 先用 `hub_shortest_paths.csv` 和 `spot_metric_closure.csv` 作为模型输入，不要直接从原始 Excel 建图。
2. 第一问可用 OP 的 0-1 整数规划精确化，小规模 40 点可先用 PuLP/OR-Tools。
3. 第二问用两阶段路径覆盖：`Σ_k y_i^k=1`，每年一条开放路径，并加入 MTZ 或单商品流消除子回路。
4. 第三问用 Min-Max mTSP：`min T`，约束每组 `T_k≤T`，并保留文化主题覆盖约束。
5. 第四问必须补容量数据，否则只能给路线库和比例分流公式，不能给具体游客人数。
"""
    REPORT_DIR.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无数据）"
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[c]) for c in df.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    hubs = read_table("hub_clean")
    edges = read_table("edge_road_directed")
    spots = read_table("spot_clean").reset_index(drop=True)
    issues = read_table("data_quality_issues")

    hub_paths, closure = all_pairs_shortest(hubs, edges)
    hub_paths.to_csv(OUT_DIR / "hub_shortest_paths.csv", index=False, encoding="utf-8-sig")

    spot_closure_rows = []
    for _, a in spots.iterrows():
        for _, b in spots.iterrows():
            item = closure[(a["hub_name"], b["hub_name"])]
            spot_closure_rows.append(
                {
                    "from_spot_id": a["spot_id"],
                    "from_spot_name": a["spot_name"],
                    "to_spot_id": b["spot_id"],
                    "to_spot_name": b["spot_name"],
                    "from_hub": a["hub_name"],
                    "to_hub": b["hub_name"],
                    "shortest_time_hours": round(float(item["time_hours"]), 3),
                    "shortest_distance_km": round(float(item["distance_km"]), 3),
                    "path_hubs": " -> ".join(item["path_hubs"]),
                }
            )
    pd.DataFrame(spot_closure_rows).to_csv(OUT_DIR / "spot_metric_closure.csv", index=False, encoding="utf-8-sig")

    p1_seq = select_op_route(spots, closure, limit_hours=30 * DAY_HOURS)
    p1_metrics = route_metrics(p1_seq, spots, closure, START_HUB, START_HUB, include_external_flight=True)
    p1_detail = route_detail("P1_30day_OP_proxy", p1_seq, spots, closure, START_HUB, START_HUB)
    pd.DataFrame([{"route_id": "P1_30day_OP_proxy", **p1_metrics}]).to_csv(OUT_DIR / "problem1_summary.csv", index=False, encoding="utf-8-sig")
    p1_detail.to_csv(OUT_DIR / "problem1_route_detail.csv", index=False, encoding="utf-8-sig")

    p2_routes = make_problem2_routes(spots, closure)
    p2_summaries = []
    p2_details = []
    for route_id, seq in p2_routes.items():
        end = "喀什市" if route_id != "P2_Y1_North_Ili" else START_HUB
        metrics = route_metrics(seq, spots, closure, START_HUB, end)
        p2_summaries.append({"route_id": route_id, **metrics, "route_sequence": " -> ".join(spots.loc[idx, "spot_name"] for idx in seq)})
        p2_details.append(route_detail(route_id, seq, spots, closure, START_HUB, end))
    p2_summary = pd.DataFrame(p2_summaries)
    p2_summary.to_csv(OUT_DIR / "problem2_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(p2_details, ignore_index=True).to_csv(OUT_DIR / "problem2_route_detail.csv", index=False, encoding="utf-8-sig")

    groups = greedy_minmax_partition(cultural_candidates(spots), spots, closure, k=3)
    p3_summaries = []
    p3_details = []
    for i, seq in enumerate(groups, 1):
        route_id = f"P3_Group_{i}"
        metrics = route_metrics(seq, spots, closure, START_HUB, START_HUB, visit_multiplier=4)
        p3_summaries.append({"route_id": route_id, **metrics, "route_sequence": " -> ".join(spots.loc[idx, "spot_name"] for idx in seq)})
        p3_details.append(route_detail(route_id, seq, spots, closure, START_HUB, START_HUB, visit_multiplier=4))
    p3_summary = pd.DataFrame(p3_summaries)
    p3_summary["max_completion_hours"] = p3_summary["total_hours"].max()
    p3_summary["balance_gap_hours"] = p3_summary["total_hours"].max() - p3_summary["total_hours"].min()
    p3_summary.to_csv(OUT_DIR / "problem3_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(p3_details, ignore_index=True).to_csv(OUT_DIR / "problem3_route_detail.csv", index=False, encoding="utf-8-sig")

    p4_summary, p4_detail = build_12day_library(spots, closure)
    p4_summary.to_csv(OUT_DIR / "problem4_candidate_routes.csv", index=False, encoding="utf-8-sig")
    p4_detail.to_csv(OUT_DIR / "problem4_candidate_route_detail.csv", index=False, encoding="utf-8-sig")

    context = {
        "high_issue_count": int((issues["severity"] == "High").sum()),
        "problem1": p1_metrics,
        "problem2": markdown_table(p2_summary[["route_id", "spots_count", "estimated_days", "distance_km", "total_proxy_yuan_excluding_meals"]]),
        "problem3": markdown_table(p3_summary[["route_id", "spots_count", "estimated_days", "total_hours", "max_completion_hours", "balance_gap_hours"]]),
        "problem4": markdown_table(p4_summary[["route_id", "spots_count", "estimated_days", "within_12day_limit", "proxy_cost_yuan_excluding_meals"]]),
    }
    report_path = write_report(context)

    summary = {
        "outputs": sorted(str(p) for p in OUT_DIR.glob("*.csv")),
        "report": str(report_path),
        "problem1": p1_metrics,
        "problem2_routes": len(p2_summary),
        "problem3_groups": len(p3_summary),
        "problem4_candidate_routes": len(p4_summary),
    }
    result_tables = {}
    for csv_path in sorted(OUT_DIR.glob("*.csv")):
        table = pd.read_csv(csv_path, encoding="utf-8-sig")
        result_tables[csv_path.stem] = json.loads(table.where(pd.notna(table), None).to_json(orient="records", force_ascii=False))
    (OUT_DIR / "tables.json").write_text(json.dumps(result_tables, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
