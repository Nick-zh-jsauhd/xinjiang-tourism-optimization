from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "humanized_scenario_outputs"
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


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def load_spots() -> pd.DataFrame:
    spots = read_csv(ROOT / "model_data" / "spot_clean.csv")
    numeric_cols = [
        "visit_hours_mid",
        "ticket_high_total_yuan_per_person",
        "priority_score_for_op",
        "local_access_km_from_text",
        "local_access_hours_from_text",
    ]
    for col in numeric_cols:
        if col in spots.columns:
            spots[col] = spots[col].map(num)
    for col in [
        "is_cultural",
        "is_natural",
        "is_topic_preference",
        "requires_approval",
        "requires_border_permit",
        "requires_reservation",
        "high_altitude_or_remote",
        "ordinary_tourist_restricted",
    ]:
        if col in spots.columns:
            spots[col] = spots[col].map(truthy)
    return spots


def problem1_humanized(spots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = read_csv(ROOT / "hybrid_30day_outputs" / "hard30_route_days.csv")
    segments = read_csv(ROOT / "hybrid_30day_outputs" / "hard30_route_segments.csv")

    for col in [
        "day",
        "visit_spots_count",
        "day_travel_hours",
        "day_service_hours",
        "day_active_hours",
        "over_budget_hours",
        "transport_cost_yuan_per_two",
        "ticket_cost_yuan_per_two",
        "risk_score",
        "time_window_violations",
    ]:
        if col in days.columns:
            days[col] = days[col].map(num)

    personas = [
        {
            "persona_id": "standard_active",
            "persona_name": "普通体力型",
            "target_active_hours": 7.5,
            "hard_active_hours": 9.0,
            "long_transfer_hours": 4.5,
            "risk_weight": 14.0,
            "multi_spot_weight": 2.0,
        },
        {
            "persona_id": "family_comfort",
            "persona_name": "亲子舒适型",
            "target_active_hours": 6.5,
            "hard_active_hours": 8.0,
            "long_transfer_hours": 3.5,
            "risk_weight": 19.0,
            "multi_spot_weight": 3.2,
        },
        {
            "persona_id": "senior_slow",
            "persona_name": "长者慢游型",
            "target_active_hours": 5.8,
            "hard_active_hours": 7.2,
            "long_transfer_hours": 3.0,
            "risk_weight": 23.0,
            "multi_spot_weight": 4.0,
        },
        {
            "persona_id": "explorer_dense",
            "persona_name": "探索紧凑型",
            "target_active_hours": 8.7,
            "hard_active_hours": 10.0,
            "long_transfer_hours": 5.5,
            "risk_weight": 10.0,
            "multi_spot_weight": 1.2,
        },
    ]

    long_transfer_by_day = {
        int(num(row.get("day"))): 1.0 if num(row.get("day_travel_hours")) >= 5.5 else 0.0
        for _, row in days.iterrows()
    }

    stress_rows: list[dict[str, Any]] = []
    for _, day in days.iterrows():
        day_no = int(num(day.get("day")))
        prev_long = long_transfer_by_day.get(day_no - 1, 0.0)
        visit_spots = day.get("visit_spots", "")
        if pd.isna(visit_spots) or str(visit_spots).strip() == "":
            visit_spots = "无景点，跨区转场"
        for p in personas:
            active = num(day.get("day_active_hours"))
            travel = num(day.get("day_travel_hours"))
            risk = num(day.get("risk_score"))
            spots_count = num(day.get("visit_spots_count"))
            active_over = max(0.0, active - p["target_active_hours"])
            hard_over = max(0.0, active - p["hard_active_hours"])
            transfer_over = max(0.0, travel - p["long_transfer_hours"])
            multi_spot_load = max(0.0, spots_count - 2.0)
            fatigue_index = (
                active_over ** 1.35 * 7.0
                + hard_over ** 1.55 * 9.0
                + transfer_over ** 1.35 * 5.5
                + risk * p["risk_weight"]
                + multi_spot_load * p["multi_spot_weight"]
                + prev_long * 1.5
            )
            comfort_score = clamp(100.0 - fatigue_index)
            stress_rows.append(
                {
                    "problem": "Q1_30day_route",
                    "persona_id": p["persona_id"],
                    "persona_name": p["persona_name"],
                    "day": day_no,
                    "visit_spots": visit_spots,
                    "day_active_hours": round(active, 2),
                    "day_travel_hours": round(travel, 2),
                    "visit_spots_count": int(spots_count),
                    "risk_score": round(risk, 3),
                    "fatigue_index": round(fatigue_index, 2),
                    "comfort_score": round(comfort_score, 2),
                    "stress_level": "red" if comfort_score < 72 else "yellow" if comfort_score < 84 else "green",
                }
            )

    day_stress = pd.DataFrame(stress_rows)
    summary_rows = []
    for persona_id, grp in day_stress.groupby("persona_id"):
        persona_name = grp["persona_name"].iloc[0]
        red_days = int((grp["stress_level"] == "red").sum())
        yellow_days = int((grp["stress_level"] == "yellow").sum())
        mean_comfort = float(grp["comfort_score"].mean())
        p10_comfort = float(grp["comfort_score"].quantile(0.10))
        suggested_buffer_days = max(0, math.ceil(red_days * 0.7 + yellow_days * 0.25))
        summary_rows.append(
            {
                "problem": "Q1_30day_route",
                "persona_id": persona_id,
                "persona_name": persona_name,
                "mean_comfort_score": round(mean_comfort, 2),
                "p10_comfort_score": round(p10_comfort, 2),
                "red_days": red_days,
                "yellow_days": yellow_days,
                "suggested_buffer_days": suggested_buffer_days,
                "interpretation": (
                    "可作为正文主方案"
                    if persona_id == "standard_active" and red_days <= 2
                    else "需要增加缓冲或削减低收益景点"
                    if red_days >= 4
                    else "适合精力较好的游客，舒适型人群需预留弹性"
                ),
            }
        )
    persona_summary = pd.DataFrame(summary_rows).sort_values("mean_comfort_score", ascending=False)

    standard = day_stress[day_stress["persona_id"] == "standard_active"].copy()
    repairs = standard.sort_values(["fatigue_index", "day"], ascending=[False, True]).head(8)
    repair_rows = []
    for rank, (_, row) in enumerate(repairs.iterrows(), start=1):
        reason = []
        if row["day_travel_hours"] >= 5.0:
            reason.append("长距离转场")
        if row["day_active_hours"] >= 8.0:
            reason.append("活动时间偏长")
        if row["visit_spots_count"] >= 3:
            reason.append("单日景点密集")
        if row["risk_score"] >= 0.18:
            reason.append("交通/天气风险偏高")
        repair_rows.append(
            {
                "rank": rank,
                "day": int(row["day"]),
                "visit_spots": row["visit_spots"],
                "fatigue_index": row["fatigue_index"],
                "comfort_score": row["comfort_score"],
                "reason": "、".join(reason) if reason else "综合疲劳较高",
                "repair_action": "设置半日休整或把后续景点顺延一天",
            }
        )
    repair_days = pd.DataFrame(repair_rows)

    seg = segments[segments["activity_type"].astype(str) == "visit"].copy()
    for col in ["travel_hours", "service_hours", "active_increment_hours", "transport_cost_yuan_per_two", "risk_score"]:
        if col in seg.columns:
            seg[col] = seg[col].map(num)
    spot_meta = spots.set_index("spot_id")
    drop_rows = []
    for _, row in seg.iterrows():
        sid = row.get("spot_id")
        meta = spot_meta.loc[sid] if sid in spot_meta.index else pd.Series(dtype=object)
        priority = num(meta.get("priority_score_for_op", 3.0), 3.0)
        topic_bonus = 1.5 if bool(meta.get("is_topic_preference", False)) else 0.0
        value = priority + topic_bonus
        burden = (
            num(row.get("active_increment_hours")) * 1.0
            + num(row.get("travel_hours")) * 0.8
            + num(row.get("risk_score")) * 9.0
            + num(row.get("transport_cost_yuan_per_two")) / 160.0
        )
        roi = value / max(0.2, burden)
        drop_rows.append(
            {
                "spot_id": sid,
                "spot_name": row.get("to_name", ""),
                "day": int(num(row.get("day"))),
                "priority_score": round(priority, 2),
                "topic_preference": bool(meta.get("is_topic_preference", False)),
                "active_increment_hours": round(num(row.get("active_increment_hours")), 2),
                "travel_hours": round(num(row.get("travel_hours")), 2),
                "transport_cost_yuan_per_two": round(num(row.get("transport_cost_yuan_per_two")), 2),
                "risk_score": round(num(row.get("risk_score")), 3),
                "humanized_value_to_burden_ratio": round(roi, 3),
                "suggestion": "若坚持30天且面向舒适游客，可作为削减候选",
            }
        )
    drop_candidates = pd.DataFrame(drop_rows).sort_values(
        ["topic_preference", "humanized_value_to_burden_ratio"], ascending=[True, True]
    ).head(10)

    return {
        "q1_persona_summary": persona_summary,
        "q1_day_stress": day_stress,
        "q1_repair_days": repair_days,
        "q1_drop_candidates": drop_candidates,
    }


def problem2_gateway_frontier() -> dict[str, pd.DataFrame]:
    route_summary = read_csv(ROOT / "problem2_openpath_outputs" / "route_summary.csv")
    totals = read_csv(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    for col in [
        "spots_count",
        "estimated_days",
        "transport_cost_yuan_for_two",
        "travel_hours",
        "service_hours",
        "total_active_hours",
        "distance_km",
    ]:
        if col in route_summary.columns:
            route_summary[col] = route_summary[col].map(num)
    for col in [
        "covered_spots",
        "total_estimated_days",
        "max_year_days",
        "total_transport_cost_yuan_for_two",
        "total_distance_km",
    ]:
        if col in totals.columns:
            totals[col] = totals[col].map(num)

    rooted = totals[totals["scenario_id"] == "P2_ROOTED_URUMQI_MINCOST"].iloc[0]
    open_row = totals[totals["scenario_id"] == "P2_OPEN_GATEWAY_LOWER_BOUND"].iloc[0]
    rooted_cost = num(rooted["total_transport_cost_yuan_for_two"])
    open_cost = num(open_row["total_transport_cost_yuan_for_two"])
    savings = rooted_cost - open_cost

    premiums = list(range(0, 2250, 250))
    sensitivity_rows = []
    for premium in premiums:
        effective_open = open_cost + premium
        sensitivity_rows.append(
            {
                "external_multicity_premium_yuan_for_two": premium,
                "rooted_effective_cost_yuan_for_two": round(rooted_cost, 2),
                "open_gateway_effective_cost_yuan_for_two": round(effective_open, 2),
                "recommended_policy": "开放式多口岸" if effective_open <= rooted_cost else "乌鲁木齐起讫",
                "cost_delta_open_minus_rooted": round(effective_open - rooted_cost, 2),
            }
        )

    year_rows = []
    for scenario_id, grp in route_summary.groupby("scenario_id"):
        days = list(grp["estimated_days"])
        costs = list(grp["transport_cost_yuan_for_two"])
        travel = list(grp["travel_hours"])
        year_rows.append(
            {
                "scenario_id": scenario_id,
                "year_count": len(grp),
                "spots_total": int(grp["spots_count"].sum()),
                "days_total": int(sum(days)),
                "max_year_days": int(max(days)),
                "day_balance_gap": int(max(days) - min(days)),
                "cost_balance_gap_yuan_for_two": round(max(costs) - min(costs), 2),
                "travel_balance_gap_hours": round(max(travel) - min(travel), 2),
                "total_transport_cost_yuan_for_two": round(sum(costs), 2),
                "humanized_score": round(
                    clamp(100 - (max(days) - min(days)) * 3.0 - (max(travel) - min(travel)) * 0.35 - sum(costs) / 600.0),
                    2,
                ),
                "interpretation": "两年负担均衡且费用最低"
                if scenario_id == "P2_OPEN_GATEWAY_LOWER_BOUND"
                else "口径保守、可落地，但第二年负担明显更重",
            }
        )

    threshold = pd.DataFrame(
        [
            {
                "comparison": "开放式多口岸 vs 乌鲁木齐起讫",
                "rooted_cost_yuan_for_two": round(rooted_cost, 2),
                "open_gateway_cost_yuan_for_two": round(open_cost, 2),
                "max_allowable_external_premium_yuan_for_two": round(savings, 2),
                "decision_rule": "若两人多口岸大交通额外差价小于该阈值，则开放式方案总费用更优",
            }
        ]
    )

    return {
        "q2_gateway_threshold": threshold,
        "q2_gateway_sensitivity": pd.DataFrame(sensitivity_rows),
        "q2_year_balance": pd.DataFrame(year_rows).sort_values("humanized_score", ascending=False),
    }


def problem3_mission_fairness() -> dict[str, pd.DataFrame]:
    summary = read_csv(ROOT / "enhanced_model_outputs" / "problem3_minmax_summary.csv")
    for col in [
        "spots_count",
        "travel_hours",
        "service_hours",
        "total_hours",
        "days",
        "transport_cost_yuan",
        "ticket_yuan",
        "risk_score",
        "objective_cost_yuan",
        "culture_value_total",
    ]:
        if col in summary.columns:
            summary[col] = summary[col].map(num)

    max_hours = float(summary["total_hours"].max())
    min_hours = float(summary["total_hours"].min())
    max_culture = float(summary["culture_value_total"].max())
    min_culture = float(summary["culture_value_total"].min())
    risk_mean = float(summary["risk_score"].mean())

    rows = []
    for _, row in summary.iterrows():
        travel_share = num(row["travel_hours"]) / max(1.0, num(row["total_hours"]))
        field_difficulty = (
            num(row["risk_score"]) * 28.0
            + travel_share * 35.0
            + max(0.0, num(row["days"]) - 10) * 2.0
            + (1.0 if num(row["spots_count"]) <= 2 else 0.0) * 7.0
        )
        rows.append(
            {
                "route_id": row["route_id"],
                "spots_count": int(num(row["spots_count"])),
                "days": int(num(row["days"])),
                "total_hours": round(num(row["total_hours"]), 2),
                "culture_value_total": round(num(row["culture_value_total"]), 2),
                "risk_score": round(num(row["risk_score"]), 3),
                "travel_share": round(travel_share, 3),
                "field_difficulty_score": round(field_difficulty, 2),
                "buffer_days_recommended": max(1, math.ceil(num(row["risk_score"]) * 1.2 + max(0, num(row["days"]) - 10) * 0.35)),
                "mission_note": "专项审批与高难远程考察组"
                if num(row["spots_count"]) <= 2
                else "文化点密集串联组"
                if num(row["culture_value_total"]) >= risk_mean * 30
                else "跨区补充考察组",
            }
        )
    group_eval = pd.DataFrame(rows).sort_values("field_difficulty_score", ascending=False)

    fairness = pd.DataFrame(
        [
            {
                "metric": "time_balance_gap_hours",
                "value": round(max_hours - min_hours, 2),
                "interpretation": "三组完成时间差控制在约1.5天内，任务均衡可解释",
            },
            {
                "metric": "culture_value_gap",
                "value": round(max_culture - min_culture, 2),
                "interpretation": "文化价值不完全均衡，因为楼兰/尼雅的访问难度主导了第一组任务量",
            },
            {
                "metric": "mean_risk_score",
                "value": round(risk_mean, 3),
                "interpretation": "第三问应按研究队/专项考察配置保障，而不是普通游客口径",
            },
        ]
    )

    return {"q3_group_eval": group_eval, "q3_fairness": fairness}


def problem4_capacity_stress() -> dict[str, pd.DataFrame]:
    flow = read_csv(ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv")
    for col in [
        "spots_count",
        "estimated_days",
        "route_capacity_persons_12day",
        "attraction_score",
        "allocated_visitors",
        "allocation_share",
        "binding_capacity_upper",
    ]:
        if col in flow.columns:
            flow[col] = flow[col].map(num)

    flow["utilization"] = flow["allocated_visitors"] / flow["route_capacity_persons_12day"].replace(0, np.nan)
    flow["unused_capacity"] = flow["route_capacity_persons_12day"] - flow["allocated_visitors"]
    flow["attraction_per_day"] = flow["attraction_score"] / flow["estimated_days"].replace(0, np.nan)
    flow["bottleneck_level"] = np.where(
        flow["utilization"] >= 0.98,
        "red",
        np.where(flow["utilization"] >= 0.85, "yellow", "green"),
    )
    bottlenecks = flow[
        [
            "column_id",
            "route_theme",
            "estimated_days",
            "route_capacity_persons_12day",
            "allocated_visitors",
            "utilization",
            "unused_capacity",
            "attraction_score",
            "attraction_per_day",
            "bottleneck_level",
            "route_sequence",
        ]
    ].copy()
    bottlenecks["utilization"] = bottlenecks["utilization"].round(3)
    bottlenecks["attraction_per_day"] = bottlenecks["attraction_per_day"].round(2)
    level_rank = {"red": 3, "yellow": 2, "green": 1}
    bottlenecks["_level_rank"] = bottlenecks["bottleneck_level"].map(level_rank).fillna(0)
    bottlenecks = bottlenecks.sort_values(["_level_rank", "utilization", "attraction_per_day"], ascending=[False, False, False])
    bottlenecks = bottlenecks.drop(columns=["_level_rank"])

    multipliers = [0.8, 1.0, 1.1, 1.2, 1.35]
    rows = []
    base_alloc = float(flow["allocated_visitors"].sum())
    total_capacity = float(flow["route_capacity_persons_12day"].sum())
    for m in multipliers:
        requested = flow["allocated_visitors"] * m
        served = np.minimum(requested, flow["route_capacity_persons_12day"])
        rejected = requested - served
        rows.append(
            {
                "demand_multiplier": m,
                "requested_visitors": int(round(base_alloc * m)),
                "served_visitors_if_no_reallocation": int(round(float(served.sum()))),
                "rejected_or_overflow_visitors": int(round(float(rejected.sum()))),
                "system_capacity": int(round(total_capacity)),
                "system_utilization": round(float(served.sum()) / max(1.0, total_capacity), 3),
                "routes_over_95pct_capacity": int(((requested / flow["route_capacity_persons_12day"]) >= 0.95).sum()),
                "interpretation": "需求超过总容量，需要预约限流和分流"
                if base_alloc * m > total_capacity
                else "总容量尚可，但热门线路可能局部拥堵",
            }
        )
    stress = pd.DataFrame(rows)

    return {"q4_bottlenecks": bottlenecks, "q4_demand_stress": stress}


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    q1 = tables["q1_persona_summary"]
    q2 = tables["q2_gateway_threshold"].iloc[0]
    q2_balance = tables["q2_year_balance"]
    q3 = tables["q3_group_eval"]
    q4 = tables["q4_demand_stress"]
    q4_b = tables["q4_bottlenecks"]

    lines: list[str] = [
        "# 新疆旅游人性化场景强化模型与实验报告",
        "",
        "## 1. 本轮研究定位",
        "",
        "本轮不重复证明已有路线能否求出，而是在现有图论/运筹模型上增加一层“游客体验与现实扰动”的决策评价。核心思想是：原模型给出可行与低成本，本轮模型回答这条路线是否舒服、是否抗扰动、是否适合不同人群、是否便于答辩解释。",
        "",
        "新增统一评价函数：",
        "",
        "```text",
        "HumanizedScore = 100",
        "  - f(单日活动超载)",
        "  - f(长距离转场)",
        "  - f(连续高负荷)",
        "  - f(预约/天气/交通风险)",
        "  - f(人群画像不匹配)",
        "```",
        "",
        "其中单日疲劳采用凸惩罚而不是线性惩罚，因为真实旅行中 9 小时活动比 7 小时活动带来的疲劳不是简单多 2 小时，而是明显加速上升。",
        "",
        "## 2. 第一问：30天路线的人性化复核",
        "",
        q1.to_markdown(index=False),
        "",
        "结论：当前 30 天硬约束路线适合作为“普通体力型”主方案，但对亲子舒适型和长者慢游型偏紧。它是一个高覆盖率、低成本的竞赛型解，不应包装成所有游客都轻松的慢游方案。",
        "",
        "最建议插入缓冲或拆分的高压力日期：",
        "",
        tables["q1_repair_days"].to_markdown(index=False),
        "",
        "若题目必须坚持 30 天内完成，则可把以下低体验收益比景点作为舒适版削减候选，削减后应重新运行局部插入/2-opt 修复：",
        "",
        tables["q1_drop_candidates"].to_markdown(index=False),
        "",
        "## 3. 第二问：两年暑假的进出疆口岸决策",
        "",
        f"乌鲁木齐起讫主模型费用为 {q2['rooted_cost_yuan_for_two']:.2f} 元，开放式多口岸下界为 {q2['open_gateway_cost_yuan_for_two']:.2f} 元。两人多口岸大交通额外差价只要低于 {q2['max_allowable_external_premium_yuan_for_two']:.2f} 元，开放式方案在总费用上就更优。",
        "",
        q2_balance.to_markdown(index=False),
        "",
        "这给第二问一个更现实的答辩口径：正文用乌鲁木齐起讫保证保守落地；强化版用开放式口岸说明如果补齐喀什、伊宁、阿勒泰等多机场/铁路票价，模型可自动选择更经济的跨年入离疆策略。",
        "",
        "## 4. 第三问：文化专项考察的任务均衡",
        "",
        q3.to_markdown(index=False),
        "",
        tables["q3_fairness"].to_markdown(index=False),
        "",
        "结论：第三问不应按景点数量平均分配。楼兰古城和尼雅遗址只有两个点，但审批、远程交通和现场考察强度高，因此作为一个专项组是合理的。更贴近现实的模型应给每组配置风险缓冲天，而不是只比较景点数。",
        "",
        "## 5. 第四问：五一接待容量与拥堵压力",
        "",
        q4.to_markdown(index=False),
        "",
        "路线瓶颈排序：",
        "",
        q4_b.head(9).to_markdown(index=False),
        "",
        "结论：第四问的关键不是找一条最优路线，而是路线产品组合与容量分流。当前系统在基准需求下接近满负荷，多数线路已经达到容量上限；若需求增加 20% 以上，必须启用预约限流、价格引导或新增替代线路。",
        "",
        "## 6. 建议纳入最终论文的 fancy 模型层",
        "",
        "1. 在第一问主模型后增加游客画像鲁棒性分析：同一路线对普通体力型、亲子型、长者型、探索型分别评价。",
        "2. 在第二问增加口岸选择阈值模型：多口岸大交通额外差价低于阈值时，开放式方案更优。",
        "3. 在第三问增加任务难度公平性：用完成时间、风险、远程交通占比共同刻画小组负担。",
        "4. 在第四问增加需求压力测试：把单点容量约束升级为线路产品容量、预约限流和需求溢出的组合模型。",
        "5. 在答辩中明确分层：确定性最优解负责可行性，Monte Carlo 负责现实扰动，人性化评分负责游客体验。",
        "",
        "## 7. 生成文件",
        "",
        "- `outputs/新疆旅游人性化场景强化结果.xlsx`",
        "- `humanized_scenario_outputs/*.csv`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    spots = load_spots()
    tables: dict[str, pd.DataFrame] = {}
    tables.update(problem1_humanized(spots))
    tables.update(problem2_gateway_frontier())
    tables.update(problem3_mission_fairness())
    tables.update(problem4_capacity_stress())

    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    to_excel(tables, OUTPUT_DIR / "新疆旅游人性化场景强化结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游人性化场景强化模型与实验报告.md")

    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游人性化场景强化结果.xlsx",
        "report": "outputs/新疆旅游人性化场景强化模型与实验报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
