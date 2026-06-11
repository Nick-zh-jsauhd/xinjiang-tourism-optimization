from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "policy_simulation_outputs"
OUTPUT_DIR = ROOT / "outputs"
RNG = np.random.default_rng(20260611)


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
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])


def q2_gateway_price_uncertainty(trials: int = 8000) -> dict[str, pd.DataFrame]:
    totals = read_csv(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    for col in ["total_transport_cost_yuan_for_two", "total_estimated_days", "max_year_days"]:
        totals[col] = totals[col].map(num)
    rooted = totals[totals["scenario_id"] == "P2_ROOTED_URUMQI_MINCOST"].iloc[0]
    open_row = totals[totals["scenario_id"] == "P2_OPEN_GATEWAY_LOWER_BOUND"].iloc[0]
    rooted_cost = num(rooted["total_transport_cost_yuan_for_two"])
    open_cost = num(open_row["total_transport_cost_yuan_for_two"])
    savings_threshold = rooted_cost - open_cost

    scenarios = [
        ("optimistic_multicity", 0.0, 450.0, 1500.0),
        ("balanced_multicity", 100.0, 800.0, 2400.0),
        ("peak_summer_multicity", 300.0, 1400.0, 3600.0),
    ]
    trial_rows = []
    summary_rows = []
    for scenario_id, low, mode, high in scenarios:
        premiums = RNG.triangular(low, mode, high, size=trials)
        open_effective = open_cost + premiums
        rooted_effective = np.full(trials, rooted_cost)
        open_wins = open_effective < rooted_effective
        savings = rooted_effective - open_effective
        for i in range(min(trials, 1200)):
            trial_rows.append(
                {
                    "scenario_id": scenario_id,
                    "trial": i + 1,
                    "external_multicity_premium_yuan_for_two": round(float(premiums[i]), 2),
                    "open_gateway_effective_cost_yuan_for_two": round(float(open_effective[i]), 2),
                    "rooted_effective_cost_yuan_for_two": round(rooted_cost, 2),
                    "open_gateway_savings_yuan_for_two": round(float(savings[i]), 2),
                    "chosen_policy": "open_gateway" if open_wins[i] else "urumqi_rooted",
                }
            )
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "rooted_cost_yuan_for_two": round(rooted_cost, 2),
                "open_gateway_base_cost_yuan_for_two": round(open_cost, 2),
                "break_even_premium_yuan_for_two": round(savings_threshold, 2),
                "premium_mean": round(float(np.mean(premiums)), 2),
                "premium_p75": round(float(np.quantile(premiums, 0.75)), 2),
                "premium_p95": round(float(np.quantile(premiums, 0.95)), 2),
                "prob_open_gateway_cheaper": round(float(np.mean(open_wins)), 4),
                "expected_savings_if_choose_open_yuan_for_two": round(float(np.mean(savings)), 2),
                "p10_savings_yuan_for_two": round(float(np.quantile(savings, 0.10)), 2),
                "recommended_policy": "开放式多口岸作为主方案"
                if float(np.mean(open_wins)) >= 0.75
                else "正文保留乌鲁木齐起讫，开放式作为条件方案"
                if float(np.mean(open_wins)) >= 0.4
                else "暑期高峰下优先保守采用乌鲁木齐起讫",
            }
        )
    return {
        "q2_gateway_price_summary": pd.DataFrame(summary_rows),
        "q2_gateway_price_trials": pd.DataFrame(trial_rows),
    }


def q3_fieldwork_buffer_simulation(trials: int = 8000) -> dict[str, pd.DataFrame]:
    groups = read_csv(ROOT / "enhanced_model_outputs" / "problem3_minmax_summary.csv")
    for col in ["days", "travel_hours", "service_hours", "total_hours", "risk_score", "spots_count"]:
        groups[col] = groups[col].map(num)
    base_project_days = int(groups["days"].max())

    group_delay_rows = []
    project_rows = []
    completion_samples: dict[str, np.ndarray] = {}
    for _, row in groups.iterrows():
        route_id = row["route_id"]
        base_days = num(row["days"])
        risk = num(row["risk_score"])
        travel_intensity = num(row["travel_hours"]) / max(1.0, num(row["total_hours"]))
        remote_bonus = 0.45 if num(row["spots_count"]) <= 2 else 0.0
        lam = 0.35 + 0.55 * risk + 0.7 * travel_intensity + remote_bonus
        minor_delay = RNG.poisson(lam=lam, size=trials)
        severe_event = RNG.binomial(1, min(0.55, 0.10 + 0.12 * risk + remote_bonus * 0.18), size=trials)
        severe_delay = severe_event * RNG.integers(1, 4, size=trials)
        completion = base_days + minor_delay + severe_delay
        completion_samples[route_id] = completion
        for buffer_days in range(0, 8):
            on_time = completion <= base_days + buffer_days
            group_delay_rows.append(
                {
                    "route_id": route_id,
                    "base_days": int(base_days),
                    "buffer_days": buffer_days,
                    "deadline_days": int(base_days + buffer_days),
                    "on_time_probability": round(float(np.mean(on_time)), 4),
                    "mean_completion_days": round(float(np.mean(completion)), 2),
                    "p90_completion_days": round(float(np.quantile(completion, 0.90)), 2),
                    "p95_completion_days": round(float(np.quantile(completion, 0.95)), 2),
                    "recommended": bool(float(np.mean(on_time)) >= 0.9 and buffer_days <= 5),
                }
            )

    stacked = np.vstack([completion_samples[row["route_id"]] for _, row in groups.iterrows()])
    project_completion = stacked.max(axis=0)
    for buffer_days in range(0, 10):
        deadline = base_project_days + buffer_days
        project_rows.append(
            {
                "base_project_days": base_project_days,
                "buffer_days": buffer_days,
                "project_deadline_days": deadline,
                "all_groups_on_time_probability": round(float(np.mean(project_completion <= deadline)), 4),
                "mean_project_completion_days": round(float(np.mean(project_completion)), 2),
                "p90_project_completion_days": round(float(np.quantile(project_completion, 0.90)), 2),
                "p95_project_completion_days": round(float(np.quantile(project_completion, 0.95)), 2),
                "policy": "可作为90%可靠性缓冲"
                if float(np.mean(project_completion <= deadline)) >= 0.9
                else "缓冲不足",
            }
        )

    group_df = pd.DataFrame(group_delay_rows)
    rec_rows = []
    for route_id, grp in group_df.groupby("route_id"):
        ok = grp[grp["on_time_probability"] >= 0.9]
        best = ok.iloc[0] if not ok.empty else grp.iloc[-1]
        rec_rows.append(
            {
                "route_id": route_id,
                "recommended_buffer_days": int(best["buffer_days"]),
                "on_time_probability": best["on_time_probability"],
                "deadline_days": int(best["deadline_days"]),
                "note": "按90%单组可靠性配置缓冲",
            }
        )
    return {
        "q3_group_buffer_curve": group_df,
        "q3_group_buffer_recommend": pd.DataFrame(rec_rows),
        "q3_project_buffer_curve": pd.DataFrame(project_rows),
    }


def q4_reservation_control() -> dict[str, pd.DataFrame]:
    flow = read_csv(ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv")
    for col in ["route_capacity_persons_12day", "allocated_visitors", "attraction_score"]:
        flow[col] = flow[col].map(num)

    multipliers = [1.0, 1.05, 1.10, 1.20, 1.35]
    cap_policies = [
        ("full_capacity", 1.00),
        ("safety_cap_95pct", 0.95),
        ("comfort_cap_90pct", 0.90),
    ]
    rows = []
    route_rows = []
    for demand_multiplier in multipliers:
        requested = flow["allocated_visitors"].to_numpy(dtype=float) * demand_multiplier
        physical_capacity = flow["route_capacity_persons_12day"].to_numpy(dtype=float)
        for policy_id, cap_ratio in cap_policies:
            booking_cap = physical_capacity * cap_ratio
            initially_accepted = np.minimum(requested, booking_cap)
            waitlist = np.maximum(0.0, requested - booking_cap)
            spare = np.maximum(0.0, booking_cap - initially_accepted)
            # Pool spare capacity to simulate a simple centralized waitlist.
            pooled_spare = float(spare.sum())
            waitlist_remaining = float(waitlist.sum())
            waitlist_served = min(pooled_spare, waitlist_remaining)
            served_total = float(initially_accepted.sum() + waitlist_served)
            rejected = float(requested.sum() - served_total)
            physical_overload_risk = float(np.mean(requested > physical_capacity))
            comfort_congestion_index = float(np.mean(initially_accepted / np.maximum(1.0, physical_capacity)))
            rows.append(
                {
                    "demand_multiplier": demand_multiplier,
                    "policy_id": policy_id,
                    "booking_cap_ratio": cap_ratio,
                    "requested_visitors": int(round(float(requested.sum()))),
                    "served_visitors": int(round(served_total)),
                    "waitlist_or_rejected_visitors": int(round(rejected)),
                    "capacity_utilization": round(served_total / max(1.0, float(physical_capacity.sum())), 3),
                    "mean_route_load_before_pooling": round(comfort_congestion_index, 3),
                    "routes_with_physical_overload_risk": int((requested > physical_capacity).sum()),
                    "policy_interpretation": "最大接待优先"
                    if policy_id == "full_capacity"
                    else "保留少量安全余量，适合不确定天气/交通"
                    if policy_id == "safety_cap_95pct"
                    else "体验优先，适合高端/亲子客群",
                }
            )
            for idx, route in flow.iterrows():
                route_rows.append(
                    {
                        "demand_multiplier": demand_multiplier,
                        "policy_id": policy_id,
                        "column_id": route["column_id"],
                        "route_theme": route["route_theme"],
                        "requested_visitors": int(round(requested[idx])),
                        "booking_cap": int(round(booking_cap[idx])),
                        "initially_accepted": int(round(initially_accepted[idx])),
                        "waitlist": int(round(waitlist[idx])),
                    }
                )
    return {
        "q4_reservation_policy_summary": pd.DataFrame(rows),
        "q4_reservation_route_detail": pd.DataFrame(route_rows),
    }


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    q2 = tables["q2_gateway_price_summary"]
    q3_group = tables["q3_group_buffer_recommend"]
    q3_project = tables["q3_project_buffer_curve"]
    q4 = tables["q4_reservation_policy_summary"]
    project_ok = q3_project[q3_project["all_groups_on_time_probability"] >= 0.9]
    project_buffer = int(project_ok.iloc[0]["buffer_days"]) if not project_ok.empty else int(q3_project.iloc[-1]["buffer_days"])

    lines = [
        "# 新疆旅游政策仿真实验报告",
        "",
        "## 1. 目的",
        "",
        "本轮把不确定性显式放进模型：第二问考虑多口岸外部大交通差价波动，第三问考虑文化考察延期风险，第四问比较预约上限策略。它服务于论文中的“仿真与策略鲁棒性”章节。",
        "",
        "## 2. 第二问：多口岸价格不确定性",
        "",
        q2.to_markdown(index=False),
        "",
        "解释：开放式多口岸路线在新疆境内交通费用上更低，但是否作为主方案取决于外部机票/铁路差价。仿真给出的是概率口径：当暑期多口岸溢价较低或中等时，开放式方案更有优势；高峰溢价很高时，乌鲁木齐起讫更保守。",
        "",
        "## 3. 第三问：文化考察缓冲天数",
        "",
        q3_group.to_markdown(index=False),
        "",
        f"以三组同时完成为目标，项目层面达到约90%可靠性需要预留 {project_buffer} 天缓冲。对应曲线如下：",
        "",
        q3_project.to_markdown(index=False),
        "",
        "解释：第三问的高级口径不是“路线算完即可”，而是把文化专项看成野外/远程调研项目；楼兰、尼雅等点存在审批、道路、天气和现场工作不确定性，应给出缓冲天数。",
        "",
        "## 4. 第四问：预约上限策略",
        "",
        q4.to_markdown(index=False),
        "",
        "解释：满容量策略最大化接待人数，但体验和扰动风险较高；95%安全上限适合普通节假日运营；90%舒适上限适合高端团或亲子团。论文中可以把它写成“容量约束 + 服务水平约束”的双目标管理模型。",
        "",
        "## 5. 可并入最终模型体系的表述",
        "",
        "1. 确定性优化给出路线主解。",
        "2. Monte Carlo 给出外部价格、延期、需求波动下的策略可靠性。",
        "3. 人性化评分和预约上限把游客体验转化为可计算约束。",
        "4. 最终答案不是单个最短路，而是主方案、条件方案和应急策略组成的路线决策系统。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    tables.update(q2_gateway_price_uncertainty())
    tables.update(q3_fieldwork_buffer_simulation())
    tables.update(q4_reservation_control())
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游政策仿真实验结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游政策仿真实验报告.md")
    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游政策仿真实验结果.xlsx",
        "report": "outputs/新疆旅游政策仿真实验报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
