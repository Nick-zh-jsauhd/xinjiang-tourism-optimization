from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "adaptive_strategy_outputs"
OUTPUT_DIR = ROOT / "outputs"


def num(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


@dataclass(frozen=True)
class Persona:
    persona_id: str
    persona_name: str
    target_active_hours: float
    hard_active_hours: float
    long_transfer_hours: float
    max_days: int
    max_red_days: int
    min_mean_comfort: float
    max_drop: int


PERSONAS = [
    Persona("standard_active", "普通体力型30天主线", 7.5, 9.0, 4.5, 30, 3, 88.0, 0),
    Persona("family_comfort", "亲子舒适型30天改造", 6.5, 8.0, 3.5, 30, 3, 86.0, 8),
    Persona("senior_slow", "长者慢游型30天改造", 5.8, 7.2, 3.0, 30, 3, 84.0, 12),
    Persona("explorer_dense", "探索紧凑型30天改造", 8.7, 10.0, 5.5, 30, 2, 90.0, 0),
]


def load_spots() -> pd.DataFrame:
    spots = read_csv(ROOT / "model_data" / "spot_clean.csv")
    for col in [
        "visit_hours_mid",
        "ticket_high_total_yuan_per_person",
        "priority_score_for_op",
        "local_access_km_from_text",
    ]:
        spots[col] = spots[col].map(num)
    for col in [
        "is_cultural",
        "is_natural",
        "is_topic_preference",
        "requires_reservation",
        "requires_border_permit",
        "requires_approval",
        "ordinary_tourist_restricted",
    ]:
        spots[col] = spots[col].map(truthy)
    return spots


def build_edges() -> tuple[dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    od = read_csv(ROOT / "enhanced_data" / "amap_driving_od_matrix_clean.csv")
    depot = read_csv(ROOT / "enhanced_data" / "amap_depot_access_matrix_clean.csv")
    edge: dict[tuple[str, str], dict[str, float]] = {}
    depot_edge: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in od.iterrows():
        edge[(row["from_spot_id"], row["to_spot_id"])] = {
            "hours": num(row.get("driving_duration_hours")),
            "cost": num(row.get("amap_selfdrive_cost_yuan_per_two")),
            "km": num(row.get("driving_distance_km")),
        }
    for _, row in depot.iterrows():
        depot_edge[(row["direction"], row["spot_id"])] = {
            "hours": num(row.get("driving_duration_hours")),
            "cost": num(row.get("selfdrive_cost_yuan_per_two")),
            "km": num(row.get("driving_distance_km")),
        }
    return edge, depot_edge


def route_from_summary(spots: pd.DataFrame) -> list[str]:
    summary = read_csv(ROOT / "hybrid_30day_outputs" / "hard30_route_summary.csv").iloc[0]
    names = [part.strip() for part in str(summary["route_sequence"]).split("->")]
    name_to_id = dict(zip(spots["spot_name"], spots["spot_id"]))
    route = []
    for name in names:
        if name in name_to_id:
            route.append(name_to_id[name])
    return route


def spot_value(row: pd.Series) -> float:
    value = num(row.get("priority_score_for_op"), 3.0)
    if row.get("is_topic_preference"):
        value += 1.5
    if row.get("is_cultural"):
        value += 0.4
    if row.get("is_natural"):
        value += 0.3
    return value


def edge_lookup(
    prev: str | None,
    cur: str,
    edge: dict[tuple[str, str], dict[str, float]],
    depot_edge: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    if prev is None:
        return depot_edge.get(("depot_to_spot", cur), {"hours": 0.0, "cost": 0.0, "km": 0.0})
    return edge.get((prev, cur), {"hours": 99.0, "cost": 9999.0, "km": 9999.0})


def comfort_score(day_active: float, day_travel: float, visits: int, persona: Persona, risk: float, prev_long: bool) -> tuple[float, float]:
    active_over = max(0.0, day_active - persona.target_active_hours)
    hard_over = max(0.0, day_active - persona.hard_active_hours)
    transfer_over = max(0.0, day_travel - persona.long_transfer_hours)
    multi_spot = max(0.0, visits - 2.0)
    fatigue = (
        active_over ** 1.35 * 7.0
        + hard_over ** 1.55 * 9.0
        + transfer_over ** 1.35 * 5.5
        + risk * 15.0
        + multi_spot * 2.0
        + (1.5 if prev_long else 0.0)
    )
    return fatigue, clamp(100.0 - fatigue)


def schedule_route(
    route: list[str],
    spots: pd.DataFrame,
    edge: dict[tuple[str, str], dict[str, float]],
    depot_edge: dict[tuple[str, str], dict[str, float]],
    persona: Persona,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = spots.set_index("spot_id")
    days: list[dict[str, Any]] = []
    current = {
        "spot_ids": [],
        "spot_names": [],
        "travel_hours": 0.0,
        "service_hours": 0.0,
        "cost": 0.0,
        "km": 0.0,
        "risk": 0.0,
    }
    prev: str | None = None
    prev_day_long = False

    def append_transfer_day(spot_name: str, e: dict[str, float], risk: float) -> None:
        nonlocal prev_day_long
        fatigue, comfort = comfort_score(e["hours"], e["hours"], 0, persona, risk, prev_day_long)
        days.append(
            {
                "persona_id": persona.persona_id,
                "day": len(days) + 1,
                "visit_spots_count": 0,
                "visit_spots": f"跨区转场至{spot_name}",
                "day_travel_hours": round(e["hours"], 3),
                "day_service_hours": 0.0,
                "day_active_hours": round(e["hours"], 3),
                "transport_cost_yuan_for_two": round(e["cost"], 2),
                "distance_km": round(e["km"], 3),
                "risk_proxy": round(risk, 3),
                "fatigue_index": round(fatigue, 2),
                "comfort_score": round(comfort, 2),
                "stress_level": "red" if comfort < 72 else "yellow" if comfort < 84 else "green",
            }
        )
        prev_day_long = e["hours"] >= persona.long_transfer_hours

    def finish_day() -> None:
        nonlocal current, prev_day_long
        if not current["spot_ids"]:
            return
        active = current["travel_hours"] + current["service_hours"]
        fatigue, comfort = comfort_score(
            active,
            current["travel_hours"],
            len(current["spot_ids"]),
            persona,
            current["risk"],
            prev_day_long,
        )
        days.append(
            {
                "persona_id": persona.persona_id,
                "day": len(days) + 1,
                "visit_spots_count": len(current["spot_ids"]),
                "visit_spots": " -> ".join(current["spot_names"]),
                "day_travel_hours": round(current["travel_hours"], 3),
                "day_service_hours": round(current["service_hours"], 3),
                "day_active_hours": round(active, 3),
                "transport_cost_yuan_for_two": round(current["cost"], 2),
                "distance_km": round(current["km"], 3),
                "risk_proxy": round(current["risk"], 3),
                "fatigue_index": round(fatigue, 2),
                "comfort_score": round(comfort, 2),
                "stress_level": "red" if comfort < 72 else "yellow" if comfort < 84 else "green",
            }
        )
        prev_day_long = current["travel_hours"] >= persona.long_transfer_hours
        current = {
            "spot_ids": [],
            "spot_names": [],
            "travel_hours": 0.0,
            "service_hours": 0.0,
            "cost": 0.0,
            "km": 0.0,
            "risk": 0.0,
        }

    for sid in route:
        if sid not in meta.index:
            continue
        row = meta.loc[sid]
        e = edge_lookup(prev, sid, edge, depot_edge)
        service = num(row["visit_hours_mid"])
        risk = 0.035 + e["hours"] * 0.018
        if bool(row.get("requires_reservation")):
            risk += 0.025
        if bool(row.get("requires_border_permit")):
            risk += 0.035
        if bool(row.get("requires_approval")):
            risk += 0.05
        if e["hours"] >= persona.long_transfer_hours and e["hours"] + service > persona.hard_active_hours:
            finish_day()
            append_transfer_day(row["spot_name"], e, risk)
            prev = sid
            e = {"hours": 0.0, "cost": 0.0, "km": 0.0}
            risk = 0.02
        increment = e["hours"] + service
        if current["spot_ids"] and current["travel_hours"] + current["service_hours"] + increment > persona.hard_active_hours:
            finish_day()
        current["spot_ids"].append(sid)
        current["spot_names"].append(row["spot_name"])
        current["travel_hours"] += e["hours"]
        current["service_hours"] += service
        current["cost"] += e["cost"]
        current["km"] += e["km"]
        current["risk"] += risk
        prev = sid
    finish_day()

    day_df = pd.DataFrame(days)
    selected = spots[spots["spot_id"].isin(route)]
    summary = {
        "persona_id": persona.persona_id,
        "persona_name": persona.persona_name,
        "spots_count": len(route),
        "scheduled_days": int(len(day_df)),
        "total_value_score": round(float(selected.apply(spot_value, axis=1).sum()), 2),
        "topic_preference_spots": int(selected["is_topic_preference"].sum()),
        "cultural_spots": int(selected["is_cultural"].sum()),
        "natural_spots": int(selected["is_natural"].sum()),
        "total_travel_hours": round(float(day_df["day_travel_hours"].sum()), 3) if not day_df.empty else 0.0,
        "total_service_hours": round(float(day_df["day_service_hours"].sum()), 3) if not day_df.empty else 0.0,
        "total_active_hours": round(float(day_df["day_active_hours"].sum()), 3) if not day_df.empty else 0.0,
        "transport_cost_yuan_for_two": round(float(day_df["transport_cost_yuan_for_two"].sum()), 2) if not day_df.empty else 0.0,
        "distance_km": round(float(day_df["distance_km"].sum()), 3) if not day_df.empty else 0.0,
        "mean_comfort_score": round(float(day_df["comfort_score"].mean()), 2) if not day_df.empty else 0.0,
        "p10_comfort_score": round(float(day_df["comfort_score"].quantile(0.1)), 2) if not day_df.empty else 0.0,
        "red_days": int((day_df["stress_level"] == "red").sum()) if not day_df.empty else 0,
        "yellow_days": int((day_df["stress_level"] == "yellow").sum()) if not day_df.empty else 0,
        "route_sequence": " -> ".join(selected.set_index("spot_id").loc[route]["spot_name"].tolist()) if route else "",
    }
    return day_df, summary


def roi_drop_order(route: list[str], spots: pd.DataFrame, edge: dict[tuple[str, str], dict[str, float]], depot_edge: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    meta = spots.set_index("spot_id")
    rows = []
    prev: str | None = None
    for sid in route:
        row = meta.loc[sid]
        e = edge_lookup(prev, sid, edge, depot_edge)
        value = spot_value(row)
        burden = num(row["visit_hours_mid"]) + e["hours"] * 0.8 + e["cost"] / 170.0
        if bool(row.get("requires_reservation")):
            burden += 0.35
        rows.append(
            {
                "spot_id": sid,
                "spot_name": row["spot_name"],
                "protected": bool(row.get("is_topic_preference")),
                "priority": num(row.get("priority_score_for_op")),
                "roi": value / max(0.25, burden),
            }
        )
        prev = sid
    ordered = pd.DataFrame(rows).sort_values(["protected", "priority", "roi"], ascending=[True, True, True])
    return ordered["spot_id"].tolist()


def optimize_persona_route(
    persona: Persona,
    base_route: list[str],
    spots: pd.DataFrame,
    edge: dict[tuple[str, str], dict[str, float]],
    depot_edge: dict[tuple[str, str], dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    route = base_route.copy()
    drop_order = roi_drop_order(route, spots, edge, depot_edge)
    dropped: list[str] = []
    day_df, summary = schedule_route(route, spots, edge, depot_edge, persona)
    attempts = 0
    while (
        (summary["scheduled_days"] > persona.max_days or summary["red_days"] > persona.max_red_days or summary["mean_comfort_score"] < persona.min_mean_comfort)
        and len(dropped) < persona.max_drop
        and attempts < len(drop_order)
    ):
        candidate = drop_order[attempts]
        attempts += 1
        if candidate not in route:
            continue
        route.remove(candidate)
        dropped.append(candidate)
        day_df, summary = schedule_route(route, spots, edge, depot_edge, persona)

    meta = spots.set_index("spot_id")
    drop_rows = [
        {
            "persona_id": persona.persona_id,
            "drop_order": idx,
            "spot_id": sid,
            "spot_name": meta.loc[sid]["spot_name"] if sid in meta.index else sid,
            "region_cluster": meta.loc[sid]["region_cluster"] if sid in meta.index else "",
            "priority_score": num(meta.loc[sid]["priority_score_for_op"]) if sid in meta.index else 0.0,
            "is_topic_preference": bool(meta.loc[sid]["is_topic_preference"]) if sid in meta.index else False,
            "drop_reason": "释放天数/降低高疲劳日压力",
        }
        for idx, sid in enumerate(dropped, start=1)
    ]
    summary["dropped_spots_count"] = len(dropped)
    summary["dropped_spots"] = " -> ".join(row["spot_name"] for row in drop_rows)
    summary["constraint_status"] = (
        "satisfied"
        if summary["scheduled_days"] <= persona.max_days
        and summary["red_days"] <= persona.max_red_days
        and summary["mean_comfort_score"] >= persona.min_mean_comfort
        else "tradeoff_remaining"
    )
    summary["model_note"] = "基于真实高德OD重排程；按低体验收益比删除非核心景点"
    return day_df, pd.DataFrame(drop_rows), summary


def q1_persona_variants() -> dict[str, pd.DataFrame]:
    spots = load_spots()
    edge, depot_edge = build_edges()
    base_route = route_from_summary(spots)
    summary_rows = []
    day_frames = []
    drop_frames = []
    for persona in PERSONAS:
        day_df, drop_df, summary = optimize_persona_route(persona, base_route, spots, edge, depot_edge)
        summary_rows.append(summary)
        day_frames.append(day_df)
        if not drop_df.empty:
            drop_frames.append(drop_df)
    return {
        "q1_variant_summary": pd.DataFrame(summary_rows).sort_values(["constraint_status", "mean_comfort_score"], ascending=[True, False]),
        "q1_variant_days": pd.concat(day_frames, ignore_index=True),
        "q1_variant_dropped_spots": pd.concat(drop_frames, ignore_index=True) if drop_frames else pd.DataFrame(),
    }


def route_region_set(route_spot_ids: str, spot_region: dict[str, str]) -> set[str]:
    regions = set()
    for sid in str(route_spot_ids).split(";"):
        sid = sid.strip()
        if sid in spot_region:
            regions.add(spot_region[sid])
    return regions


def q4_adaptive_reallocation() -> dict[str, pd.DataFrame]:
    spots = load_spots()
    flow = read_csv(ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv")
    for col in ["route_capacity_persons_12day", "allocated_visitors", "attraction_score", "estimated_days"]:
        flow[col] = flow[col].map(num)
    spot_region = dict(zip(spots["spot_id"], spots["region_cluster"]))
    flow["region_set"] = flow["route_spot_ids"].map(lambda x: route_region_set(x, spot_region))

    scenarios = [1.05, 1.10, 1.20, 1.35]
    scenario_rows = []
    realloc_rows = []
    for multiplier in scenarios:
        requested = {row["column_id"]: num(row["allocated_visitors"]) * multiplier for _, row in flow.iterrows()}
        capacity = {row["column_id"]: num(row["route_capacity_persons_12day"]) for _, row in flow.iterrows()}
        served = {cid: min(requested[cid], capacity[cid]) for cid in requested}
        overflow = {cid: max(0.0, requested[cid] - capacity[cid]) for cid in requested}
        spare = {cid: max(0.0, capacity[cid] - served[cid]) for cid in requested}

        for source_id, extra in sorted(overflow.items(), key=lambda item: item[1], reverse=True):
            if extra <= 0:
                continue
            source_regions = flow.loc[flow["column_id"] == source_id, "region_set"].iloc[0]
            candidates = []
            for _, target in flow.iterrows():
                target_id = target["column_id"]
                if target_id == source_id or spare[target_id] <= 0:
                    continue
                target_regions = target["region_set"]
                overlap = len(source_regions & target_regions)
                union = len(source_regions | target_regions) or 1
                similarity = overlap / union
                attraction_density = num(target["attraction_score"]) / max(1.0, num(target["estimated_days"]))
                candidates.append((similarity, attraction_density, target_id))
            candidates.sort(reverse=True)
            remaining = extra
            for similarity, attraction_density, target_id in candidates:
                if remaining <= 0:
                    break
                if similarity <= 0 and attraction_density < 5.0:
                    continue
                moved = min(remaining, spare[target_id])
                if moved <= 0:
                    continue
                spare[target_id] -= moved
                served[target_id] += moved
                remaining -= moved
                realloc_rows.append(
                    {
                        "demand_multiplier": multiplier,
                        "source_column_id": source_id,
                        "target_column_id": target_id,
                        "reallocated_visitors": int(round(moved)),
                        "region_similarity": round(similarity, 3),
                        "target_attraction_per_day": round(attraction_density, 2),
                        "policy": "推荐作为同区或高吸引力替代线路分流",
                    }
                )
            overflow[source_id] = remaining

        total_requested = sum(requested.values())
        total_served = sum(min(requested[cid], capacity[cid]) for cid in requested)
        served_after = sum(served.values())
        unresolved = total_requested - served_after
        scenario_rows.append(
            {
                "demand_multiplier": multiplier,
                "requested_visitors": int(round(total_requested)),
                "served_without_reallocation": int(round(total_served)),
                "served_after_reallocation": int(round(served_after)),
                "additional_served_by_reallocation": int(round(served_after - total_served)),
                "unresolved_overflow_visitors": int(round(unresolved)),
                "routes_at_capacity_after_reallocation": int(sum(abs(served[cid] - capacity[cid]) < 1e-6 for cid in served)),
                "management_decision": "仅靠既有线路分流不足，需要新增运力/限流"
                if unresolved > 0
                else "可通过既有线路替代分流消化需求上浮",
            }
        )

    return {
        "q4_reallocation_summary": pd.DataFrame(scenario_rows),
        "q4_reallocation_plan": pd.DataFrame(realloc_rows),
    }


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    q1 = tables["q1_variant_summary"]
    q4 = tables["q4_reallocation_summary"]
    lines = [
        "# 新疆旅游自适应策略实验报告",
        "",
        "## 1. 研究目的",
        "",
        "上一轮人性化模型负责评价，本轮进一步做策略改造：对第一问按游客画像自动生成替代路线，对第四问在五一需求上浮时做替代线路分流。这样报告中既有最优路线，也有面向真实运营的备选策略。",
        "",
        "## 2. 第一问：游客画像路线改造",
        "",
        q1.to_markdown(index=False),
        "",
        "解释：`satisfied` 表示在30天内同时满足该画像的红色高压力日、平均舒适度和排程天数要求；`tradeoff_remaining` 表示即使删点后仍保留一定强度，应在论文中说明为高覆盖率方案而非舒适慢游方案。",
        "",
        "被删除景点明细：",
        "",
        tables["q1_variant_dropped_spots"].to_markdown(index=False) if not tables["q1_variant_dropped_spots"].empty else "无删点。",
        "",
        "## 3. 第四问：需求上浮下的替代线路分流",
        "",
        q4.to_markdown(index=False),
        "",
        "可执行分流动作：",
        "",
        tables["q4_reallocation_plan"].to_markdown(index=False) if not tables["q4_reallocation_plan"].empty else "没有可用分流空间。",
        "",
        "## 4. 可写入论文的模型升级",
        "",
        "1. 第一问从单一路线升级为画像适配路线族：同一底层OD图，不同人群对应不同疲劳阈值和删点策略。",
        "2. 第四问从静态容量分配升级为需求冲击后的自适应分流：先识别超载线路，再按区域相似度和吸引力密度寻找替代线路。",
        "3. 这两个实验都保留原问题主旋律：不是为了炫技而换问题，而是在低成本/覆盖/容量目标上加入真实人的体验和运营弹性。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    tables.update(q1_persona_variants())
    tables.update(q4_adaptive_reallocation())
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游自适应策略实验结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游自适应策略实验报告.md")
    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游自适应策略实验结果.xlsx",
        "report": "outputs/新疆旅游自适应策略实验报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
