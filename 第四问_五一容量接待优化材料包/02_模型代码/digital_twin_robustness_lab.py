# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "digital_twin_outputs"
FIG_DIR = OUT_DIR / "figures"
OUTPUT_DIR = ROOT / "outputs"
RNG_SEED = 20260611
TRIALS = 4000


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


COLORS = {
    "blue": "#2f6f9f",
    "teal": "#2a9d8f",
    "green": "#6a994e",
    "yellow": "#e9c46a",
    "orange": "#f4a261",
    "red": "#c44536",
    "purple": "#6d597a",
    "gray": "#6c757d",
}

SCENARIO_NAME_ORDER = ["常规暑期", "热浪高温", "雨洪道路延误", "预约收紧", "暑期客流高峰", "复合极端冲击"]


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(fig: plt.Figure, filename: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "normal_summer",
                "scenario_name": "常规暑期",
                "travel_mu": 1.05,
                "travel_sigma": 0.08,
                "service_mu": 1.02,
                "reservation_fail_prob": 0.02,
                "heat_penalty": 2.0,
                "road_closure_prob": 0.01,
                "open_gateway_premium_mean": 1100,
                "open_gateway_premium_sd": 330,
                "q3_weather_delay_mean": 0.8,
                "q3_permit_rework_prob": 0.03,
                "q4_demand_multiplier": 1.00,
                "q4_capacity_multiplier": 1.00,
                "q4_no_show_rate": 0.04,
            },
            {
                "scenario_id": "heatwave",
                "scenario_name": "热浪高温",
                "travel_mu": 1.08,
                "travel_sigma": 0.10,
                "service_mu": 1.08,
                "reservation_fail_prob": 0.03,
                "heat_penalty": 8.0,
                "road_closure_prob": 0.015,
                "open_gateway_premium_mean": 1250,
                "open_gateway_premium_sd": 420,
                "q3_weather_delay_mean": 1.2,
                "q3_permit_rework_prob": 0.04,
                "q4_demand_multiplier": 0.95,
                "q4_capacity_multiplier": 0.92,
                "q4_no_show_rate": 0.08,
            },
            {
                "scenario_id": "rain_flood_delay",
                "scenario_name": "雨洪道路延误",
                "travel_mu": 1.28,
                "travel_sigma": 0.18,
                "service_mu": 1.02,
                "reservation_fail_prob": 0.04,
                "heat_penalty": 3.0,
                "road_closure_prob": 0.10,
                "open_gateway_premium_mean": 1200,
                "open_gateway_premium_sd": 420,
                "q3_weather_delay_mean": 2.8,
                "q3_permit_rework_prob": 0.05,
                "q4_demand_multiplier": 0.90,
                "q4_capacity_multiplier": 0.82,
                "q4_no_show_rate": 0.10,
            },
            {
                "scenario_id": "reservation_tight",
                "scenario_name": "预约收紧",
                "travel_mu": 1.08,
                "travel_sigma": 0.09,
                "service_mu": 1.03,
                "reservation_fail_prob": 0.12,
                "heat_penalty": 3.0,
                "road_closure_prob": 0.02,
                "open_gateway_premium_mean": 1250,
                "open_gateway_premium_sd": 430,
                "q3_weather_delay_mean": 1.0,
                "q3_permit_rework_prob": 0.08,
                "q4_demand_multiplier": 1.05,
                "q4_capacity_multiplier": 0.90,
                "q4_no_show_rate": 0.05,
            },
            {
                "scenario_id": "peak_summer",
                "scenario_name": "暑期客流高峰",
                "travel_mu": 1.18,
                "travel_sigma": 0.14,
                "service_mu": 1.05,
                "reservation_fail_prob": 0.08,
                "heat_penalty": 5.0,
                "road_closure_prob": 0.03,
                "open_gateway_premium_mean": 1760,
                "open_gateway_premium_sd": 650,
                "q3_weather_delay_mean": 1.5,
                "q3_permit_rework_prob": 0.06,
                "q4_demand_multiplier": 1.20,
                "q4_capacity_multiplier": 0.95,
                "q4_no_show_rate": 0.03,
            },
            {
                "scenario_id": "compound_extreme",
                "scenario_name": "复合极端冲击",
                "travel_mu": 1.35,
                "travel_sigma": 0.22,
                "service_mu": 1.10,
                "reservation_fail_prob": 0.18,
                "heat_penalty": 10.0,
                "road_closure_prob": 0.15,
                "open_gateway_premium_mean": 2100,
                "open_gateway_premium_sd": 900,
                "q3_weather_delay_mean": 4.0,
                "q3_permit_rework_prob": 0.12,
                "q4_demand_multiplier": 1.35,
                "q4_capacity_multiplier": 0.80,
                "q4_no_show_rate": 0.07,
            },
        ]
    )


def route_name_to_spot_ids(route_sequence: str, spot_lookup: dict[str, dict[str, Any]]) -> list[str]:
    names = [item.strip() for item in str(route_sequence).split("->") if item.strip()]
    out = []
    for name in names:
        row = spot_lookup.get(name)
        if row:
            out.append(str(row["spot_id"]))
    return out


def q1_persona_robustness(scen: pd.DataFrame, spots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 101)
    variants = read_csv(ROOT / "adaptive_strategy_outputs" / "q1_variant_summary.csv")
    if variants.empty:
        return pd.DataFrame(), pd.DataFrame()

    spot_lookup = {str(row["spot_name"]): row.to_dict() for _, row in spots.iterrows()}
    spot_by_id = {str(row["spot_id"]): row.to_dict() for _, row in spots.iterrows()}
    daily_budget = {
        "explorer_dense": 8.8,
        "standard_active": 8.0,
        "family_comfort": 7.1,
        "senior_slow": 5.9,
    }
    rows = []
    detail_rows = []

    for _, persona in variants.iterrows():
        persona_id = str(persona["persona_id"])
        spot_ids = route_name_to_spot_ids(str(persona["route_sequence"]), spot_lookup)
        reserved_count = sum(bool(spot_by_id.get(sid, {}).get("requires_reservation")) for sid in spot_ids)
        remote_count = sum(bool(spot_by_id.get(sid, {}).get("high_altitude_or_remote")) for sid in spot_ids)
        remote_share = remote_count / max(1, len(spot_ids))
        base_days = int(num(persona["scheduled_days"]))
        base_travel = num(persona["total_travel_hours"])
        base_service = num(persona["total_service_hours"])
        base_comfort = num(persona["mean_comfort_score"])
        budget = daily_budget.get(persona_id, 8.0)

        for _, sc in scen.iterrows():
            factors = rng.normal(num(sc["travel_mu"]), num(sc["travel_sigma"]), TRIALS)
            factors = np.clip(factors, 0.85, 2.05)
            service_factors = rng.normal(num(sc["service_mu"]), 0.04, TRIALS)
            service_factors = np.clip(service_factors, 0.9, 1.35)
            travel_hours = base_travel * factors
            service_hours = base_service * service_factors
            active_days = np.ceil((travel_hours + service_hours) / budget)
            reservation_failures = rng.binomial(reserved_count, num(sc["reservation_fail_prob"]), TRIALS)
            road_events = rng.random(TRIALS) < num(sc["road_closure_prob"])
            road_delay_days = np.where(road_events, rng.integers(1, 4, TRIALS), 0)
            reservation_delay_days = np.ceil(reservation_failures / 2.0) * 0.5
            simulated_days = np.maximum(base_days, active_days) + road_delay_days + reservation_delay_days
            heat_penalty = num(sc["heat_penalty"]) * (1.0 + 0.45 * remote_share)
            comfort_noise = rng.normal(0, 2.2, TRIALS)
            comfort = (
                base_comfort
                - heat_penalty
                - np.maximum(0, simulated_days - base_days) * 1.25
                - reservation_failures * 2.2
                - road_delay_days * 2.6
                + comfort_noise
            )
            comfort = np.clip(comfort, 0, 100)
            success = (simulated_days <= 30) & (comfort >= 75) & (reservation_failures <= 2)
            red_risk = (simulated_days > 30) | (comfort < 70) | (reservation_failures >= 3)
            rows.append(
                {
                    "question": "Q1",
                    "persona_id": persona_id,
                    "persona_name": persona["persona_name"],
                    "scenario_id": sc["scenario_id"],
                    "scenario_name": sc["scenario_name"],
                    "base_days": base_days,
                    "base_mean_comfort": round(base_comfort, 2),
                    "reservation_spots": reserved_count,
                    "remote_share": round(remote_share, 3),
                    "success_probability": round(float(success.mean()), 4),
                    "red_risk_probability": round(float(red_risk.mean()), 4),
                    "mean_simulated_days": round(float(simulated_days.mean()), 2),
                    "p90_simulated_days": round(float(np.quantile(simulated_days, 0.90)), 2),
                    "mean_comfort": round(float(comfort.mean()), 2),
                    "p10_comfort": round(float(np.quantile(comfort, 0.10)), 2),
                    "expected_reservation_failures": round(float(reservation_failures.mean()), 2),
                    "expected_road_delay_days": round(float(road_delay_days.mean()), 2),
                    "recommended_action": q1_action(float(success.mean()), float(red_risk.mean()), persona_id),
                }
            )
            detail_rows.append(
                {
                    "persona_id": persona_id,
                    "scenario_id": sc["scenario_id"],
                    "simulated_days_p50": round(float(np.quantile(simulated_days, 0.50)), 2),
                    "simulated_days_p95": round(float(np.quantile(simulated_days, 0.95)), 2),
                    "comfort_p05": round(float(np.quantile(comfort, 0.05)), 2),
                    "comfort_p50": round(float(np.quantile(comfort, 0.50)), 2),
                    "comfort_p95": round(float(np.quantile(comfort, 0.95)), 2),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def q1_action(success_prob: float, red_prob: float, persona_id: str) -> str:
    if success_prob >= 0.85 and red_prob <= 0.10:
        return "可作为主推方案"
    if success_prob >= 0.65:
        return "保留路线但预置1-2天机动缓冲"
    if persona_id == "senior_slow":
        return "建议拆成两次慢游或减少高海拔/长转场点"
    return "应启用删点、改口岸或错峰重排"


def q2_gateway_robustness(scen: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 202)
    totals = read_csv(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    rooted = totals[totals["scenario_id"].eq("P2_ROOTED_URUMQI_MINCOST")].iloc[0]
    open_gateway = totals[totals["scenario_id"].eq("P2_OPEN_GATEWAY_LOWER_BOUND")].iloc[0]
    rooted_cost = num(rooted["total_transport_cost_yuan_for_two"])
    open_cost = num(open_gateway["total_transport_cost_yuan_for_two"])
    rooted_days = num(rooted["total_estimated_days"])
    open_days = num(open_gateway["total_estimated_days"])
    lodging_day_cost = 320.0
    time_value_day = 180.0
    rows = []

    for _, sc in scen.iterrows():
        premium = rng.normal(num(sc["open_gateway_premium_mean"]), num(sc["open_gateway_premium_sd"]), TRIALS)
        premium = np.clip(premium, 0, None)
        rooted_delay = rng.poisson(max(0.05, (num(sc["travel_mu"]) - 1.0) * 3.0), TRIALS)
        open_delay = rng.poisson(max(0.03, (num(sc["travel_mu"]) - 1.0) * 2.0), TRIALS)
        road_events = rng.random(TRIALS) < num(sc["road_closure_prob"])
        rooted_delay = rooted_delay + np.where(road_events, rng.integers(1, 4, TRIALS), 0)
        open_delay = open_delay + np.where(road_events, rng.integers(0, 3, TRIALS), 0)
        rooted_total = rooted_cost + rooted_delay * (lodging_day_cost + time_value_day)
        open_total = open_cost + premium + open_delay * (lodging_day_cost + time_value_day)
        savings = rooted_total - open_total
        prob_open = float((open_total < rooted_total).mean())
        rows.append(
            {
                "question": "Q2",
                "scenario_id": sc["scenario_id"],
                "scenario_name": sc["scenario_name"],
                "rooted_base_cost_yuan_for_two": round(rooted_cost, 2),
                "open_base_cost_yuan_for_two": round(open_cost, 2),
                "break_even_premium_yuan_for_two": round(rooted_cost - open_cost, 2),
                "mean_open_gateway_premium": round(float(premium.mean()), 2),
                "prob_open_gateway_cheaper_after_disruption": round(prob_open, 4),
                "expected_savings_yuan_for_two": round(float(savings.mean()), 2),
                "p10_savings_yuan_for_two": round(float(np.quantile(savings, 0.10)), 2),
                "mean_rooted_extra_days": round(float(rooted_delay.mean()), 2),
                "mean_open_extra_days": round(float(open_delay.mean()), 2),
                "base_day_advantage_open": int(rooted_days - open_days),
                "recommended_policy": "开放式多口岸主推" if prob_open >= 0.75 else ("双方案报价后择优" if prob_open >= 0.45 else "乌鲁木齐起讫保守主推"),
            }
        )
    return pd.DataFrame(rows)


def q3_fieldwork_robustness(scen: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 303)
    groups = read_csv(ROOT / "enhanced_model_outputs" / "problem3_minmax_summary.csv")
    if groups.empty:
        return pd.DataFrame()
    policies = [
        {"policy_id": "no_buffer", "policy_name": "无缓冲", "buffer": 0, "delay_factor": 1.00, "permit_factor": 1.00, "extra_cost": 0},
        {"policy_id": "buffer_4d", "policy_name": "4天项目缓冲", "buffer": 4, "delay_factor": 1.00, "permit_factor": 1.00, "extra_cost": 0},
        {"policy_id": "buffer_5d", "policy_name": "5天项目缓冲", "buffer": 5, "delay_factor": 1.00, "permit_factor": 1.00, "extra_cost": 0},
        {"policy_id": "preapproval_4d", "policy_name": "前置审批+4天缓冲", "buffer": 4, "delay_factor": 0.95, "permit_factor": 0.35, "extra_cost": 1200},
        {"policy_id": "split_team_4d", "policy_name": "机动小组+4天缓冲", "buffer": 4, "delay_factor": 0.78, "permit_factor": 0.60, "extra_cost": 2500},
    ]
    base_project_days = int(groups["days"].max())
    rows = []
    group_days = groups["days"].to_numpy(dtype=float)
    group_routes = groups["route_id"].astype(str).tolist()
    remote_weights = np.array([1.45 if "Group1" in rid else (1.05 if "Group3" in rid else 0.90) for rid in group_routes])

    for _, sc in scen.iterrows():
        for policy in policies:
            deadline = base_project_days + int(policy["buffer"])
            completions = []
            fairness_gaps = []
            total_extra_costs = []
            for _ in range(TRIALS):
                weather_delay = rng.poisson(num(sc["q3_weather_delay_mean"]) * remote_weights * num(policy["delay_factor"]))
                permit_delay = np.zeros(len(group_days))
                if rng.random() < num(sc["q3_permit_rework_prob"]) * num(policy["permit_factor"]):
                    permit_delay[0] = rng.integers(3, 8)
                rework_delay = rng.binomial(1, min(0.25, 0.04 + num(sc["reservation_fail_prob"])), len(group_days))
                completion = group_days + weather_delay + permit_delay + rework_delay
                completions.append(float(completion.max()))
                fairness_gaps.append(float(completion.max() - completion.min()))
                total_extra_costs.append(float(policy["extra_cost"] + weather_delay.sum() * 260 + permit_delay.sum() * 400))
            completions_np = np.array(completions)
            fairness_np = np.array(fairness_gaps)
            cost_np = np.array(total_extra_costs)
            success = completions_np <= deadline
            rows.append(
                {
                    "question": "Q3",
                    "scenario_id": sc["scenario_id"],
                    "scenario_name": sc["scenario_name"],
                    "policy_id": policy["policy_id"],
                    "policy_name": policy["policy_name"],
                    "deadline_days": deadline,
                    "success_probability": round(float(success.mean()), 4),
                    "mean_project_completion_days": round(float(completions_np.mean()), 2),
                    "p90_project_completion_days": round(float(np.quantile(completions_np, 0.90)), 2),
                    "p95_project_completion_days": round(float(np.quantile(completions_np, 0.95)), 2),
                    "mean_fairness_gap_days": round(float(fairness_np.mean()), 2),
                    "expected_extra_cost_yuan": round(float(cost_np.mean()), 2),
                    "recommended_use": "推荐" if success.mean() >= 0.90 and np.quantile(completions_np, 0.95) <= deadline + 1 else "备选/不足",
                }
            )
    return pd.DataFrame(rows)


def q4_capacity_robustness(scen: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 404)
    flow = read_csv(ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv")
    if flow.empty:
        return pd.DataFrame()
    policies = [
        {"policy_id": "full_release", "policy_name": "全量放票", "release_rate": 1.00, "reallocation_eff": 0.20, "stagger_factor": 1.00},
        {"policy_id": "safety_cap_95", "policy_name": "95%安全预约上限", "release_rate": 0.95, "reallocation_eff": 0.35, "stagger_factor": 0.98},
        {"policy_id": "comfort_cap_90", "policy_name": "90%舒适预约上限", "release_rate": 0.90, "reallocation_eff": 0.45, "stagger_factor": 0.96},
        {"policy_id": "dynamic_reallocation", "policy_name": "动态分流+95%上限", "release_rate": 0.95, "reallocation_eff": 0.72, "stagger_factor": 0.94},
        {"policy_id": "staggered_prebooking", "policy_name": "分时预约+动态分流", "release_rate": 0.93, "reallocation_eff": 0.76, "stagger_factor": 0.90},
    ]
    base_alloc = flow["allocated_visitors"].to_numpy(dtype=float)
    cap = flow["route_capacity_persons_12day"].to_numpy(dtype=float)
    attractiveness = flow["attraction_score"].to_numpy(dtype=float)
    attractiveness = attractiveness / max(1.0, attractiveness.mean())
    rows = []
    detail_rows = []

    for _, sc in scen.iterrows():
        for policy in policies:
            served_trials = []
            rejected_trials = []
            utilization_trials = []
            pressure_trials = []
            for _ in range(TRIALS):
                route_noise = rng.lognormal(mean=0.0, sigma=0.18, size=len(base_alloc))
                route_demand = base_alloc * num(sc["q4_demand_multiplier"]) * num(policy["stagger_factor"]) * route_noise
                route_demand = route_demand * (0.92 + 0.08 * attractiveness)
                show_up = route_demand * (1.0 - num(sc["q4_no_show_rate"]))
                physical_cap = cap * num(sc["q4_capacity_multiplier"]) * rng.normal(1.0, 0.035, len(cap))
                physical_cap = np.clip(physical_cap, cap * 0.55, cap * 1.05)
                released_cap = physical_cap * num(policy["release_rate"])
                direct_served = np.minimum(show_up, released_cap)
                overflow = np.maximum(show_up - released_cap, 0)
                spare = np.maximum(released_cap - show_up, 0)
                recovered = min(float(overflow.sum()), float(spare.sum()) * num(policy["reallocation_eff"]))
                served = float(direct_served.sum() + recovered)
                gross_requests = float(route_demand.sum())
                rejected = max(0.0, gross_requests - served)
                physical_total = float(physical_cap.sum())
                utilization = served / max(1.0, physical_total)
                pressure = float(np.mean(show_up / np.maximum(physical_cap, 1.0) > 0.95))
                served_trials.append(served)
                rejected_trials.append(rejected)
                utilization_trials.append(utilization)
                pressure_trials.append(pressure)
            served_np = np.array(served_trials)
            rejected_np = np.array(rejected_trials)
            util_np = np.array(utilization_trials)
            pressure_np = np.array(pressure_trials)
            rows.append(
                {
                    "question": "Q4",
                    "scenario_id": sc["scenario_id"],
                    "scenario_name": sc["scenario_name"],
                    "policy_id": policy["policy_id"],
                    "policy_name": policy["policy_name"],
                    "mean_served_visitors": int(round(float(served_np.mean()))),
                    "mean_rejected_or_waitlisted": int(round(float(rejected_np.mean()))),
                    "p90_rejected_or_waitlisted": int(round(float(np.quantile(rejected_np, 0.90)))),
                    "mean_system_utilization": round(float(util_np.mean()), 3),
                    "prob_any_pressure_route": round(float((pressure_np > 0).mean()), 4),
                    "mean_share_routes_over_95pct_pressure": round(float(pressure_np.mean()), 3),
                    "recommended_use": q4_action(float(rejected_np.mean()), float(util_np.mean()), float((pressure_np > 0).mean())),
                }
            )
            detail_rows.append(
                {
                    "scenario_id": sc["scenario_id"],
                    "policy_id": policy["policy_id"],
                    "served_p10": int(round(float(np.quantile(served_np, 0.10)))),
                    "served_p50": int(round(float(np.quantile(served_np, 0.50)))),
                    "served_p90": int(round(float(np.quantile(served_np, 0.90)))),
                    "rejected_p50": int(round(float(np.quantile(rejected_np, 0.50)))),
                    "rejected_p95": int(round(float(np.quantile(rejected_np, 0.95)))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def q4_action(mean_rejected: float, util: float, pressure_prob: float) -> str:
    if mean_rejected <= 3500 and pressure_prob <= 0.25 and util <= 0.97:
        return "推荐"
    if mean_rejected <= 10000 and pressure_prob <= 0.65:
        return "可用但需现场分流"
    return "需加开线路/前置限流"


def build_recommendation_atlas(q1: pd.DataFrame, q2: pd.DataFrame, q3: pd.DataFrame, q4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not q1.empty:
        for scenario_id, grp in q1.groupby("scenario_id"):
            worst = grp.sort_values("success_probability").iloc[0]
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "question": "Q1",
                    "decision_focus": "游客画像与30天可行性",
                    "risk_signal": f"{worst['persona_name']}成功率最低：{worst['success_probability']}",
                    "recommended_policy": worst["recommended_action"],
                }
            )
    if not q2.empty:
        for _, row in q2.iterrows():
            rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "question": "Q2",
                    "decision_focus": "两年暑假交通费用最小",
                    "risk_signal": f"开放式多口岸更便宜概率 {row['prob_open_gateway_cheaper_after_disruption']}",
                    "recommended_policy": row["recommended_policy"],
                }
            )
    if not q3.empty:
        for scenario_id, grp in q3.groupby("scenario_id"):
            rec = grp[grp["recommended_use"].eq("推荐")]
            has_recommended = not rec.empty
            chosen = rec.sort_values(["expected_extra_cost_yuan", "p95_project_completion_days"]).iloc[0] if has_recommended else grp.sort_values("success_probability", ascending=False).iloc[0]
            policy = str(chosen["policy_name"])
            if not has_recommended:
                policy = f"{policy}；仍不足，需延长缓冲/改期/缩小考察范围"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "question": "Q3",
                    "decision_focus": "文化考察项目可靠完成",
                    "risk_signal": f"{'推荐策略' if has_recommended else '最佳候选'}成功率 {chosen['success_probability']}，p95 {chosen['p95_project_completion_days']} 天",
                    "recommended_policy": policy,
                }
            )
    if not q4.empty:
        for scenario_id, grp in q4.groupby("scenario_id"):
            rec = grp[grp["recommended_use"].eq("推荐")]
            has_recommended = not rec.empty
            if not has_recommended:
                chosen = grp.sort_values(["mean_rejected_or_waitlisted", "prob_any_pressure_route"]).iloc[0]
            else:
                chosen = rec.sort_values(["mean_rejected_or_waitlisted", "mean_system_utilization"]).iloc[0]
            policy = str(chosen["policy_name"])
            if not has_recommended:
                policy = f"{policy}；仅为最优缓解，仍需加开线路/前置限流"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "question": "Q4",
                    "decision_focus": "五一容量与预约管理",
                    "risk_signal": f"{'推荐策略' if has_recommended else '最佳候选'}平均等待/拒绝 {chosen['mean_rejected_or_waitlisted']} 人，压力概率 {chosen['prob_any_pressure_route']}",
                    "recommended_policy": policy,
                }
            )
    return pd.DataFrame(rows)


def plot_q1_heatmap(q1: pd.DataFrame) -> str:
    if q1.empty:
        return ""
    pivot = q1.pivot(index="persona_name", columns="scenario_name", values="success_probability")
    pivot = pivot[[name for name in SCENARIO_NAME_ORDER if name in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}", ha="center", va="center", fontsize=9, color="#1f2937")
    ax.set_title("Q1 游客画像在复合情景下的30天成功率")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return save_fig(fig, "fig01_q1_persona_robustness_heatmap.png")


def plot_q2_probability(q2: pd.DataFrame) -> str:
    if q2.empty:
        return ""
    q2 = q2.copy()
    q2["scenario_name"] = pd.Categorical(q2["scenario_name"], categories=SCENARIO_NAME_ORDER, ordered=True)
    q2 = q2.sort_values("scenario_name")
    fig, ax = plt.subplots(figsize=(10, 4))
    values = q2["prob_open_gateway_cheaper_after_disruption"].to_numpy()
    colors = [COLORS["green"] if v >= 0.75 else (COLORS["orange"] if v >= 0.45 else COLORS["red"]) for v in values]
    ax.bar(q2["scenario_name"], values, color=colors)
    ax.axhline(0.5, color="#374151", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("开放式多口岸更便宜概率")
    ax.set_title("Q2 多口岸策略在扰动后的费用胜率")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.025, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    return save_fig(fig, "fig02_q2_gateway_robustness.png")


def plot_q3_best_policy(q3: pd.DataFrame) -> str:
    if q3.empty:
        return ""
    best_rows = []
    for scenario_name in SCENARIO_NAME_ORDER:
        grp = q3[q3["scenario_name"].eq(scenario_name)]
        if grp.empty:
            continue
        best_rows.append(grp.sort_values(["success_probability", "expected_extra_cost_yuan"], ascending=[False, True]).iloc[0])
    best = pd.DataFrame(best_rows)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    values = best["success_probability"].to_numpy()
    ax.bar(best["scenario_name"], values, color=COLORS["blue"])
    ax.axhline(0.9, color=COLORS["red"], linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("最佳策略成功率")
    ax.set_title("Q3 文化考察在各情景下的最佳缓冲策略")
    ax.tick_params(axis="x", rotation=25)
    for i, (_, row) in enumerate(best.iterrows()):
        ax.text(i, row["success_probability"] + 0.025, str(row["policy_name"]), ha="center", va="bottom", fontsize=8)
    return save_fig(fig, "fig03_q3_best_buffer_policy.png")


def plot_q4_rejected(q4: pd.DataFrame) -> str:
    if q4.empty:
        return ""
    selected = q4[q4["policy_id"].isin(["full_release", "dynamic_reallocation", "staggered_prebooking"])].copy()
    pivot = selected.pivot(index="scenario_name", columns="policy_name", values="mean_rejected_or_waitlisted")
    pivot = pivot.reindex([name for name in SCENARIO_NAME_ORDER if name in pivot.index])
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.24
    for idx, col in enumerate(pivot.columns):
        ax.bar(x + (idx - 1) * width, pivot[col], width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("平均等待/拒绝人数")
    ax.set_title("Q4 预约与动态分流对容量冲击的吸收能力")
    ax.legend(fontsize=8)
    return save_fig(fig, "fig04_q4_capacity_policy_robustness.png")


def build_report(
    scen: pd.DataFrame,
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    q3: pd.DataFrame,
    q4: pd.DataFrame,
    atlas: pd.DataFrame,
    figures: list[str],
) -> str:
    q1_worst = q1.sort_values("success_probability").head(5) if not q1.empty else pd.DataFrame()
    q2_view = q2[["scenario_name", "prob_open_gateway_cheaper_after_disruption", "expected_savings_yuan_for_two", "recommended_policy"]] if not q2.empty else pd.DataFrame()
    q3_recs = atlas[atlas["question"].eq("Q3")]
    q4_recs = atlas[atlas["question"].eq("Q4")]

    lines = [
        "# 新疆旅游四问数字孪生鲁棒性实验报告",
        "",
        "## 核心结论",
        "",
        "本实验不是替代原来的确定性优化，而是在主解上叠加真实世界扰动：天气、道路、预约、客流、票价与审批。它回答的问题是：路线在现实波动下是否仍然可执行，哪些策略需要预置缓冲或分流。",
        "",
        "- 第一问：普通体力型和探索紧凑型在常规暑期较稳健，但亲子和长者路线在热浪、雨洪、复合冲击下需要删点、错峰或拆段。",
        "- 第二问：多口岸方案不是无条件更优；在常规/雨洪等场景下仍有费用优势，但在高峰与复合冲击下需要先拿真实大交通报价再决策。",
        "- 第三问：文化考察不宜只给确定性最短路线，至少要配置 4-5 天项目缓冲；复合冲击下还需要前置审批或机动小组。",
        "- 第四问：全量放票在高峰和复合冲击下会放大拥堵，动态分流与分时预约能显著降低等待/拒绝人数，但极端场景仍需加开线路或提前限流。",
        "",
        "## 情景设定",
        "",
        scen.to_markdown(index=False),
        "",
        "## 第一问：画像路线鲁棒性",
        "",
        "下表列出成功率最低的若干画像-情景组合。成功定义为：模拟天数不超过30天、舒适度不低于75、预约失败不超过2个核心点。",
        "",
        q1_worst.to_markdown(index=False) if not q1_worst.empty else "无数据。",
        "",
        "## 第二问：多口岸费用策略",
        "",
        q2_view.to_markdown(index=False) if not q2_view.empty else "无数据。",
        "",
        "## 第三问：文化考察缓冲策略",
        "",
        q3_recs.to_markdown(index=False) if not q3_recs.empty else "无数据。",
        "",
        "## 第四问：容量预约策略",
        "",
        q4_recs.to_markdown(index=False) if not q4_recs.empty else "无数据。",
        "",
        "## 四问联动策略图谱",
        "",
        atlas.to_markdown(index=False) if not atlas.empty else "无数据。",
        "",
        "## 图表输出",
        "",
    ]
    for fig in figures:
        if fig:
            lines.append(f"- {fig}")
    lines += [
        "",
        "## 建模口径",
        "",
        "1. 交通 OD 使用当前项目中的高德距离矩阵；天气、预约、票价、审批与客流扰动属于校准仿真参数，不是实时预测。",
        "2. 确定性优化给路线骨架，数字孪生层给鲁棒性解释；论文中应把它写成“场景压力测试/策略仿真”，不要写成真实未来客流预测。",
        "3. 当仿真与确定性主解冲突时，答辩口径应优先解释“为什么主解在某类人群或冲击下需要策略修正”，这会比单纯展示最短路更贴近现实。",
    ]
    return "\n".join(lines)


def main() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scen = scenarios()
    spots = read_csv(ROOT / "model_data" / "spot_clean.csv")

    q1, q1_detail = q1_persona_robustness(scen, spots)
    q2 = q2_gateway_robustness(scen)
    q3 = q3_fieldwork_robustness(scen)
    q4, q4_detail = q4_capacity_robustness(scen)
    atlas = build_recommendation_atlas(q1, q2, q3, q4)

    tables = {
        "scenario_parameters": scen,
        "q1_persona_robustness": q1,
        "q1_distribution_detail": q1_detail,
        "q2_gateway_robustness": q2,
        "q3_fieldwork_policy": q3,
        "q4_capacity_policy": q4,
        "q4_distribution_detail": q4_detail,
        "recommendation_atlas": atlas,
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")

    figures = [
        plot_q1_heatmap(q1),
        plot_q2_probability(q2),
        plot_q3_best_policy(q3),
        plot_q4_rejected(q4),
    ]

    with pd.ExcelWriter(OUTPUT_DIR / "新疆旅游数字孪生鲁棒性实验结果.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    report = build_report(scen, q1, q2, q3, q4, atlas, figures)
    report_path = OUTPUT_DIR / "新疆旅游四问数字孪生鲁棒性实验报告.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "status": "success",
        "trials_per_scenario": TRIALS,
        "scenario_count": len(scen),
        "q1_rows": len(q1),
        "q2_rows": len(q2),
        "q3_rows": len(q3),
        "q4_rows": len(q4),
        "figures": figures,
        "workbook": "outputs/新疆旅游数字孪生鲁棒性实验结果.xlsx",
        "report": "outputs/新疆旅游四问数字孪生鲁棒性实验报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
