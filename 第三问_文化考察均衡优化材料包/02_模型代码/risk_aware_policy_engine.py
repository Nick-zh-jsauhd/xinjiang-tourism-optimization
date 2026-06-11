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
OUT_DIR = ROOT / "risk_policy_outputs"
FIG_DIR = OUT_DIR / "figures"
OUTPUT_DIR = ROOT / "outputs"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


SCENARIO_ORDER = ["normal_summer", "heatwave", "rain_flood_delay", "reservation_tight", "peak_summer", "compound_extreme"]
SCENARIO_NAME = {
    "normal_summer": "常规暑期",
    "heatwave": "热浪高温",
    "rain_flood_delay": "雨洪道路延误",
    "reservation_tight": "预约收紧",
    "peak_summer": "暑期客流高峰",
    "compound_extreme": "复合极端冲击",
}
RISK_PROFILES = {
    "optimistic": {
        "profile_name": "乐观效率型",
        "tail_lambda": 0.15,
        "weights": [0.45, 0.15, 0.10, 0.10, 0.15, 0.05],
        "failure_penalty": 2500,
        "service_penalty": 0.35,
        "min_scenario_success_floor": 0.00,
    },
    "balanced": {
        "profile_name": "均衡稳健型",
        "tail_lambda": 0.45,
        "weights": [0.30, 0.15, 0.15, 0.15, 0.20, 0.05],
        "failure_penalty": 4500,
        "service_penalty": 0.60,
        "min_scenario_success_floor": 0.15,
    },
    "conservative": {
        "profile_name": "保守可靠型",
        "tail_lambda": 0.85,
        "weights": [0.20, 0.15, 0.20, 0.15, 0.20, 0.10],
        "failure_penalty": 7000,
        "service_penalty": 0.85,
        "min_scenario_success_floor": 0.25,
    },
    "extreme_safe": {
        "profile_name": "极端安全型",
        "tail_lambda": 1.35,
        "weights": [0.10, 0.10, 0.15, 0.15, 0.25, 0.25],
        "failure_penalty": 10000,
        "service_penalty": 1.15,
        "min_scenario_success_floor": 0.38,
    },
}


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


def weighted_mean(values: list[float], weights: list[float]) -> float:
    arr = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    return float(np.sum(arr * w) / max(1e-12, np.sum(w)))


def weighted_cvar(values: list[float], weights: list[float], alpha: float = 0.75) -> float:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total_w = sum(weights)
    target_tail = max(1e-12, total_w * (1.0 - alpha))
    acc = 0.0
    weighted_sum = 0.0
    for value, weight in reversed(pairs):
        take = min(weight, target_tail - acc)
        if take <= 0:
            break
        weighted_sum += value * take
        acc += take
    return weighted_sum / max(1e-12, acc)


def score_policy(losses: list[float], weights: list[float], tail_lambda: float) -> tuple[float, float, float]:
    exp_loss = weighted_mean(losses, weights)
    tail_loss = weighted_cvar(losses, weights, alpha=0.75)
    objective = exp_loss + tail_lambda * tail_loss
    return exp_loss, tail_loss, objective


def q1_candidate_rows(q1: pd.DataFrame) -> pd.DataFrame:
    if q1.empty:
        return pd.DataFrame()
    policies = [
        {
            "policy_id": "as_planned",
            "policy_name": "按原画像路线执行",
            "cost_penalty_yuan": 0,
            "coverage_loss_spots": 0,
            "success_lift": 0.00,
            "red_risk_cut": 0.00,
            "comfort_lift": 0.0,
        },
        {
            "policy_id": "buffer_light",
            "policy_name": "预留1-2天机动缓冲",
            "cost_penalty_yuan": 650,
            "coverage_loss_spots": 0,
            "success_lift": 0.11,
            "red_risk_cut": 0.16,
            "comfort_lift": 3.0,
        },
        {
            "policy_id": "drop_low_value",
            "policy_name": "删减低收益点并错峰预约",
            "cost_penalty_yuan": 300,
            "coverage_loss_spots": 2,
            "success_lift": 0.23,
            "red_risk_cut": 0.32,
            "comfort_lift": 5.5,
        },
        {
            "policy_id": "split_slow_trip",
            "policy_name": "拆成两段慢游",
            "cost_penalty_yuan": 1800,
            "coverage_loss_spots": 0,
            "success_lift": 0.45,
            "red_risk_cut": 0.55,
            "comfort_lift": 8.0,
        },
    ]
    rows = []
    for _, base in q1.iterrows():
        base_success = num(base["success_probability"])
        base_red = num(base["red_risk_probability"])
        base_comfort = num(base["mean_comfort"])
        for policy in policies:
            if num(base["base_days"]) >= 30 and policy["policy_id"] == "buffer_light":
                success = min(0.995, base_success + 0.05)
                red = max(0.0, base_red - 0.08)
                feasible_note = "满30天路线无法直接加缓冲，需要同步删点"
            else:
                success = min(0.995, base_success + policy["success_lift"] * (1.0 - base_success))
                red = max(0.0, base_red * (1.0 - policy["red_risk_cut"]))
                feasible_note = "可实施"
            comfort = min(100.0, base_comfort + policy["comfort_lift"])
            rows.append(
                {
                    "question": "Q1",
                    "persona_id": base["persona_id"],
                    "persona_name": base["persona_name"],
                    "scenario_id": base["scenario_id"],
                    "scenario_name": base["scenario_name"],
                    "policy_id": policy["policy_id"],
                    "policy_name": policy["policy_name"],
                    "adjusted_success_probability": round(success, 4),
                    "adjusted_red_risk_probability": round(red, 4),
                    "adjusted_mean_comfort": round(comfort, 2),
                    "cost_penalty_yuan": policy["cost_penalty_yuan"],
                    "coverage_loss_spots": policy["coverage_loss_spots"],
                    "feasible_note": feasible_note,
                }
            )
    return pd.DataFrame(rows)


def q1_profile_selection(q1_candidates: pd.DataFrame) -> pd.DataFrame:
    if q1_candidates.empty:
        return pd.DataFrame()
    rows = []
    for profile_id, profile in RISK_PROFILES.items():
        weights = list(profile["weights"])
        for persona_id, persona_grp in q1_candidates.groupby("persona_id"):
            for policy_id, grp in persona_grp.groupby("policy_id"):
                grp = grp.set_index("scenario_id").reindex(SCENARIO_ORDER).reset_index()
                losses = []
                for _, row in grp.iterrows():
                    failure = 1.0 - num(row["adjusted_success_probability"])
                    red = num(row["adjusted_red_risk_probability"])
                    comfort_shortfall = max(0.0, 82.0 - num(row["adjusted_mean_comfort"]))
                    loss = (
                        profile["failure_penalty"] * failure
                        + 2200 * red
                        + 95 * comfort_shortfall
                        + num(row["cost_penalty_yuan"])
                        + 520 * num(row["coverage_loss_spots"])
                    )
                    losses.append(float(loss))
                exp_loss, tail_loss, objective = score_policy(losses, weights, num(profile["tail_lambda"]))
                sample = grp.iloc[0]
                rows.append(
                    {
                        "question": "Q1",
                        "profile_id": profile_id,
                        "profile_name": profile["profile_name"],
                        "persona_id": persona_id,
                        "persona_name": sample["persona_name"],
                        "policy_id": policy_id,
                        "policy_name": sample["policy_name"],
                        "expected_loss": round(exp_loss, 2),
                        "tail_loss_cvar75": round(tail_loss, 2),
                        "risk_adjusted_objective": round(objective, 2),
                        "min_success_probability": round(float(grp["adjusted_success_probability"].min()), 4),
                        "weighted_success_probability": round(weighted_mean(grp["adjusted_success_probability"].tolist(), weights), 4),
                        "weighted_red_risk": round(weighted_mean(grp["adjusted_red_risk_probability"].tolist(), weights), 4),
                        "coverage_loss_spots": int(num(sample["coverage_loss_spots"])),
                        "cost_penalty_yuan": int(num(sample["cost_penalty_yuan"])),
                    }
                )
    table = pd.DataFrame(rows)
    table["is_selected"] = False
    for keys, grp in table.groupby(["profile_id", "persona_id"]):
        idx = grp["risk_adjusted_objective"].idxmin()
        table.loc[idx, "is_selected"] = True
    return table.sort_values(["profile_id", "persona_id", "risk_adjusted_objective"])


def q2_profile_selection(q2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if q2.empty:
        return pd.DataFrame()
    q2 = q2.set_index("scenario_id").reindex(SCENARIO_ORDER).reset_index()
    policies = [
        {"policy_id": "rooted_urumqi", "policy_name": "乌鲁木齐起讫保守方案", "quote_cost": 0, "option_bonus": 0},
        {"policy_id": "open_gateway", "policy_name": "开放式多口岸方案", "quote_cost": 0, "option_bonus": 0},
        {"policy_id": "quote_then_choose", "policy_name": "双方案实时比价后择优", "quote_cost": 120, "option_bonus": 0.72},
    ]
    for profile_id, profile in RISK_PROFILES.items():
        weights = list(profile["weights"])
        for policy in policies:
            losses = []
            chosen_prob = []
            for _, row in q2.iterrows():
                rooted = num(row["rooted_base_cost_yuan_for_two"]) + num(row["mean_rooted_extra_days"]) * 500
                open_cost = rooted - num(row["expected_savings_yuan_for_two"])
                if policy["policy_id"] == "rooted_urumqi":
                    cost = rooted
                    regret = max(0.0, num(row["expected_savings_yuan_for_two"]))
                    open_prob = 0.0
                elif policy["policy_id"] == "open_gateway":
                    cost = open_cost
                    regret = max(0.0, -num(row["expected_savings_yuan_for_two"]))
                    open_prob = 1.0
                else:
                    option_bonus = num(policy["option_bonus"])
                    saving = num(row["expected_savings_yuan_for_two"])
                    cost = min(rooted, open_cost) + num(policy["quote_cost"]) - max(0.0, saving) * 0.10
                    regret = abs(saving) * (1.0 - option_bonus) * 0.25
                    open_prob = num(row["prob_open_gateway_cheaper_after_disruption"])
                risk_loss = cost + profile["service_penalty"] * regret
                losses.append(float(risk_loss))
                chosen_prob.append(float(open_prob))
            exp_loss, tail_loss, objective = score_policy(losses, weights, num(profile["tail_lambda"]))
            rows.append(
                {
                    "question": "Q2",
                    "profile_id": profile_id,
                    "profile_name": profile["profile_name"],
                    "policy_id": policy["policy_id"],
                    "policy_name": policy["policy_name"],
                    "expected_loss_or_cost": round(exp_loss, 2),
                    "tail_loss_cvar75": round(tail_loss, 2),
                    "risk_adjusted_objective": round(objective, 2),
                    "weighted_open_gateway_probability": round(weighted_mean(chosen_prob, weights), 4),
                }
            )
    table = pd.DataFrame(rows)
    table["is_selected"] = False
    for profile_id, grp in table.groupby("profile_id"):
        idx = grp["risk_adjusted_objective"].idxmin()
        table.loc[idx, "is_selected"] = True
    return table.sort_values(["profile_id", "risk_adjusted_objective"])


def q3_profile_selection(q3: pd.DataFrame) -> pd.DataFrame:
    if q3.empty:
        return pd.DataFrame()
    rows = []
    for profile_id, profile in RISK_PROFILES.items():
        weights = list(profile["weights"])
        for policy_id, grp in q3.groupby("policy_id"):
            grp = grp.set_index("scenario_id").reindex(SCENARIO_ORDER).reset_index()
            losses = []
            for _, row in grp.iterrows():
                failure = 1.0 - num(row["success_probability"])
                p95_over = max(0.0, num(row["p95_project_completion_days"]) - num(row["deadline_days"]))
                fairness = num(row["mean_fairness_gap_days"])
                chance_floor_gap = max(0.0, num(profile["min_scenario_success_floor"]) - num(row["success_probability"]))
                loss = (
                    num(row["expected_extra_cost_yuan"])
                    + profile["failure_penalty"] * failure
                    + 650 * p95_over
                    + 85 * fairness
                    + 2.4 * profile["failure_penalty"] * chance_floor_gap
                )
                losses.append(float(loss))
            exp_loss, tail_loss, objective = score_policy(losses, weights, num(profile["tail_lambda"]))
            sample = grp.iloc[0]
            rows.append(
                {
                    "question": "Q3",
                    "profile_id": profile_id,
                    "profile_name": profile["profile_name"],
                    "policy_id": policy_id,
                    "policy_name": sample["policy_name"],
                    "expected_loss": round(exp_loss, 2),
                    "tail_loss_cvar75": round(tail_loss, 2),
                    "risk_adjusted_objective": round(objective, 2),
                    "weighted_success_probability": round(weighted_mean(grp["success_probability"].tolist(), weights), 4),
                    "worst_success_probability": round(float(grp["success_probability"].min()), 4),
                    "max_p95_completion_days": round(float(grp["p95_project_completion_days"].max()), 2),
                }
            )
    table = pd.DataFrame(rows)
    table["is_selected"] = False
    for profile_id, grp in table.groupby("profile_id"):
        idx = grp["risk_adjusted_objective"].idxmin()
        table.loc[idx, "is_selected"] = True
    return table.sort_values(["profile_id", "risk_adjusted_objective"])


def q4_profile_selection(q4: pd.DataFrame) -> pd.DataFrame:
    if q4.empty:
        return pd.DataFrame()
    rows = []
    for profile_id, profile in RISK_PROFILES.items():
        weights = list(profile["weights"])
        for policy_id, grp in q4.groupby("policy_id"):
            grp = grp.set_index("scenario_id").reindex(SCENARIO_ORDER).reset_index()
            losses = []
            for _, row in grp.iterrows():
                rejected = num(row["mean_rejected_or_waitlisted"])
                p90_rejected = num(row["p90_rejected_or_waitlisted"])
                pressure = num(row["prob_any_pressure_route"])
                utilization_gap = max(0.0, num(row["mean_system_utilization"]) - 0.93)
                loss = (
                    rejected * profile["service_penalty"]
                    + 0.35 * p90_rejected
                    + 8500 * pressure
                    + 30000 * utilization_gap
                )
                losses.append(float(loss))
            exp_loss, tail_loss, objective = score_policy(losses, weights, num(profile["tail_lambda"]))
            sample = grp.iloc[0]
            rows.append(
                {
                    "question": "Q4",
                    "profile_id": profile_id,
                    "profile_name": profile["profile_name"],
                    "policy_id": policy_id,
                    "policy_name": sample["policy_name"],
                    "expected_loss": round(exp_loss, 2),
                    "tail_loss_cvar75": round(tail_loss, 2),
                    "risk_adjusted_objective": round(objective, 2),
                    "weighted_rejected_or_waitlisted": int(round(weighted_mean(grp["mean_rejected_or_waitlisted"].tolist(), weights))),
                    "worst_rejected_or_waitlisted": int(round(float(grp["mean_rejected_or_waitlisted"].max()))),
                    "weighted_pressure_probability": round(weighted_mean(grp["prob_any_pressure_route"].tolist(), weights), 4),
                }
            )
    table = pd.DataFrame(rows)
    table["is_selected"] = False
    for profile_id, grp in table.groupby("profile_id"):
        idx = grp["risk_adjusted_objective"].idxmin()
        table.loc[idx, "is_selected"] = True
    return table.sort_values(["profile_id", "risk_adjusted_objective"])


def build_integrated_portfolio(q1_sel: pd.DataFrame, q2_sel: pd.DataFrame, q3_sel: pd.DataFrame, q4_sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile_id, profile in RISK_PROFILES.items():
        q2 = q2_sel[(q2_sel["profile_id"].eq(profile_id)) & (q2_sel["is_selected"])].iloc[0]
        q3 = q3_sel[(q3_sel["profile_id"].eq(profile_id)) & (q3_sel["is_selected"])].iloc[0]
        q4 = q4_sel[(q4_sel["profile_id"].eq(profile_id)) & (q4_sel["is_selected"])].iloc[0]
        q1 = q1_sel[(q1_sel["profile_id"].eq(profile_id)) & (q1_sel["is_selected"])]
        q1_policy_summary = "；".join(
            f"{row['persona_name']}->{row['policy_name']}" for _, row in q1.sort_values("persona_id").iterrows()
        )
        rows.append(
            {
                "profile_id": profile_id,
                "profile_name": profile["profile_name"],
                "q1_persona_policy": q1_policy_summary,
                "q2_transport_policy": q2["policy_name"],
                "q3_fieldwork_policy": q3["policy_name"],
                "q4_capacity_policy": q4["policy_name"],
                "portfolio_interpretation": portfolio_interpretation(profile_id, q2["policy_name"], q3["policy_name"], q4["policy_name"]),
            }
        )
    return pd.DataFrame(rows)


def portfolio_interpretation(profile_id: str, q2: str, q3: str, q4: str) -> str:
    if profile_id == "optimistic":
        return f"强调效率与覆盖，交通采用{q2}，文化考察和容量策略保留基本缓冲。"
    if profile_id == "balanced":
        return f"适合论文主口径：{q2}，{q3}，五一采用{q4}，兼顾成本与可靠性。"
    if profile_id == "conservative":
        return f"适合答辩防守口径：优先降低失败概率，采用{q2}、{q3}、{q4}。"
    return f"面向极端情景压力测试，所有问都按尾部风险控制，采用{q2}、{q3}、{q4}。"


def plot_policy_matrix(portfolio: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    display = portfolio[["profile_name", "q2_transport_policy", "q3_fieldwork_policy", "q4_capacity_policy"]].copy()
    table = ax.table(cellText=display.values, colLabels=["风险偏好", "Q2交通", "Q3考察", "Q4容量"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#2f6f9f")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#e9f2f7")
    ax.set_title("风险偏好下的跨问题策略组合", pad=12, fontsize=15)
    return save_fig(fig, "fig01_integrated_policy_matrix.png")


def plot_q2_frontier(q2_sel: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    for policy, grp in q2_sel.groupby("policy_name"):
        ax.plot(grp["profile_name"], grp["risk_adjusted_objective"], marker="o", label=policy)
    ax.set_title("Q2 风险偏好下的交通策略目标值")
    ax.set_ylabel("风险调整目标值")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    return save_fig(fig, "fig02_q2_risk_frontier.png")


def plot_q3_q4_selected(q3_sel: pd.DataFrame, q4_sel: pd.DataFrame) -> str:
    q3_best = q3_sel[q3_sel["is_selected"]].copy()
    q4_best = q4_sel[q4_sel["is_selected"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(q3_best["profile_name"], q3_best["weighted_success_probability"], color=COLORS["green"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Q3 被选策略的加权成功率")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(q4_best["profile_name"], q4_best["weighted_rejected_or_waitlisted"], color=COLORS["orange"])
    axes[1].set_title("Q4 被选策略的加权等待/拒绝人数")
    axes[1].tick_params(axis="x", rotation=20)
    return save_fig(fig, "fig03_q3_q4_selected_risk_metrics.png")


def build_report(
    q1_sel: pd.DataFrame,
    q2_sel: pd.DataFrame,
    q3_sel: pd.DataFrame,
    q4_sel: pd.DataFrame,
    portfolio: pd.DataFrame,
    figures: list[str],
) -> str:
    q1_selected = q1_sel[q1_sel["is_selected"]].copy()
    q2_selected = q2_sel[q2_sel["is_selected"]].copy()
    q3_selected = q3_sel[q3_sel["is_selected"]].copy()
    q4_selected = q4_sel[q4_sel["is_selected"]].copy()
    lines = [
        "# 新疆旅游四问风险感知策略选择模型报告",
        "",
        "## 建模结论",
        "",
        "本层把数字孪生仿真的多情景结果转化为决策模型。核心目标不是重新求一条路，而是在不同风险偏好下选择策略组合：",
        "",
        "\\[ \\min_p \\; E_w[L(p,\\omega)] + \\lambda \\operatorname{CVaR}_{75\\%}(L(p,\\omega)) \\]",
        "",
        "其中 \\(p\\) 是候选策略，\\(\\omega\\) 是常规暑期、热浪、雨洪、预约收紧、客流高峰、复合极端等情景，\\(\\lambda\\) 表示风险厌恶程度。这个表达把“期望表现”和“尾部风险”放在同一个框架下。",
        "",
        "## 跨问题策略组合",
        "",
        portfolio.to_markdown(index=False),
        "",
        "## Q1 画像策略选择",
        "",
        q1_selected[[
            "profile_name",
            "persona_name",
            "policy_name",
            "weighted_success_probability",
            "weighted_red_risk",
            "min_success_probability",
            "risk_adjusted_objective",
        ]].to_markdown(index=False),
        "",
        "## Q2 交通策略选择",
        "",
        q2_selected[[
            "profile_name",
            "policy_name",
            "expected_loss_or_cost",
            "tail_loss_cvar75",
            "risk_adjusted_objective",
            "weighted_open_gateway_probability",
        ]].to_markdown(index=False),
        "",
        "## Q3 文化考察策略选择",
        "",
        q3_selected[[
            "profile_name",
            "policy_name",
            "weighted_success_probability",
            "worst_success_probability",
            "max_p95_completion_days",
            "risk_adjusted_objective",
        ]].to_markdown(index=False),
        "",
        "## Q4 容量预约策略选择",
        "",
        q4_selected[[
            "profile_name",
            "policy_name",
            "weighted_rejected_or_waitlisted",
            "worst_rejected_or_waitlisted",
            "weighted_pressure_probability",
            "risk_adjusted_objective",
        ]].to_markdown(index=False),
        "",
        "## 图表输出",
        "",
    ]
    lines.extend(f"- {fig}" for fig in figures if fig)
    lines += [
        "",
        "## 解释口径",
        "",
        "1. 这不是单纯的启发式排序，而是基于情景权重、失败惩罚和尾部损失的策略选择模型。",
        "2. 风险偏好不同，策略不同是合理结果；论文可把“均衡稳健型”作为正文主方案，把“保守可靠型/极端安全型”作为敏感性与应急预案。",
        "3. Q1 的删点/缓冲/拆段策略是基于已得到画像路线的后验策略修正，属于决策层，不改变原问题的主路线求解逻辑。",
        "4. Q2 的“双方案实时比价后择优”适合真实旅行决策：模型先给口岸阈值，出行前用 12306/航班报价决定执行哪条方案。",
    ]
    return "\n".join(lines)


def main() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q1 = read_csv(ROOT / "digital_twin_outputs" / "q1_persona_robustness.csv")
    q2 = read_csv(ROOT / "digital_twin_outputs" / "q2_gateway_robustness.csv")
    q3 = read_csv(ROOT / "digital_twin_outputs" / "q3_fieldwork_policy.csv")
    q4 = read_csv(ROOT / "digital_twin_outputs" / "q4_capacity_policy.csv")

    q1_candidates = q1_candidate_rows(q1)
    q1_selection = q1_profile_selection(q1_candidates)
    q2_selection = q2_profile_selection(q2)
    q3_selection = q3_profile_selection(q3)
    q4_selection = q4_profile_selection(q4)
    portfolio = build_integrated_portfolio(q1_selection, q2_selection, q3_selection, q4_selection)

    tables = {
        "q1_policy_candidates": q1_candidates,
        "q1_policy_selection": q1_selection,
        "q2_policy_selection": q2_selection,
        "q3_policy_selection": q3_selection,
        "q4_policy_selection": q4_selection,
        "integrated_policy_portfolio": portfolio,
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")

    figures = [
        plot_policy_matrix(portfolio),
        plot_q2_frontier(q2_selection),
        plot_q3_q4_selected(q3_selection, q4_selection),
    ]

    with pd.ExcelWriter(OUTPUT_DIR / "新疆旅游风险感知策略选择模型结果.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    report = build_report(q1_selection, q2_selection, q3_selection, q4_selection, portfolio, figures)
    report_path = OUTPUT_DIR / "新疆旅游四问风险感知策略选择模型报告.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "status": "success",
        "risk_profiles": len(RISK_PROFILES),
        "q1_selection_rows": len(q1_selection),
        "q2_selection_rows": len(q2_selection),
        "q3_selection_rows": len(q3_selection),
        "q4_selection_rows": len(q4_selection),
        "figures": figures,
        "workbook": "outputs/新疆旅游风险感知策略选择模型结果.xlsx",
        "report": "outputs/新疆旅游四问风险感知策略选择模型报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
