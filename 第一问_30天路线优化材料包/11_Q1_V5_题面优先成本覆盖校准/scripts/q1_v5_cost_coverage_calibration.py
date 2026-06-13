from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
Q1_V3 = ROOT / "第一问_30天路线优化材料包" / "09_Q1_V3_鲁棒联合优化"
OUT = ROOT / "第一问_30天路线优化材料包" / "11_Q1_V5_题面优先成本覆盖校准"

OPERATIONAL_SUCCESS_THRESHOLD = 0.80
STRICT_COMFORT_REFERENCE_THRESHOLD = 0.40
FORBIDDEN_APPROVAL_SPOTS = {"P007", "P038"}  # 楼兰古城、尼雅遗址，普通游客不作为基准必选点。


def ensure_dirs() -> None:
    for child in ("outputs", "reports", "scripts"):
        (OUT / child).mkdir(parents=True, exist_ok=True)


def read_candidates() -> pd.DataFrame:
    df = pd.read_csv(Q1_V3 / "outputs" / "q1_v3_candidate_routes_enriched.csv", encoding="utf-8-sig")
    numeric_cols = [
        "spots_count",
        "planned_trip_days",
        "buffer_days",
        "red_days",
        "yellow_days",
        "time_window_violations",
        "late_after_hard_end_violations",
        "transport_cost_yuan_for_two",
        "ticket_cost_yuan_for_two",
        "rough_hotel_cost_yuan",
        "total_cost_yuan_excluding_meals",
        "total_travel_hours",
        "mean_comfort_score",
        "p10_comfort_score",
        "operational_success_probability",
        "strict_comfort_success_probability",
        "expected_cost_yuan",
        "p95_cost_yuan",
        "cvar75_loss",
        "robust_utility",
        "family_utility",
        "senior_utility",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "hard_constraints_pass",
        "schedule_strict_feasible",
        "schedule_soft_feasible",
        "schedule_feasible",
        "operational_chance_constraint_pass",
        "strict_comfort_chance_constraint_pass",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true")

    def ordinary_route(seq: str) -> bool:
        parts = {part.strip() for part in str(seq).split("->") if part.strip()}
        return not bool(parts & FORBIDDEN_APPROVAL_SPOTS)

    df["ordinary_tourist_route"] = df["spot_id_sequence"].map(ordinary_route)
    df["topic_executable_feasible"] = (
        df["ordinary_tourist_route"]
        & df["hard_constraints_pass"]
        & (df["planned_trip_days"] <= 30)
        & df["schedule_strict_feasible"]
        & (df["time_window_violations"] == 0)
    )
    df["operational_robust_feasible"] = (
        df["topic_executable_feasible"]
        & (df["operational_success_probability"] >= OPERATIONAL_SUCCESS_THRESHOLD)
    )
    df["strict_comfort_reference_feasible"] = (
        df["topic_executable_feasible"]
        & (df["strict_comfort_success_probability"] >= STRICT_COMFORT_REFERENCE_THRESHOLD)
    )
    df["cost_per_spot_yuan"] = df["total_cost_yuan_excluding_meals"] / df["spots_count"]
    return df


def route_record(row: pd.Series, role: str, note: str, baseline: Optional[pd.Series] = None) -> Dict[str, object]:
    record: Dict[str, object] = {
        "solution_role": role,
        "route_id": row["route_id"],
        "spots_count": int(row["spots_count"]),
        "total_cost_yuan_excluding_meals": round(float(row["total_cost_yuan_excluding_meals"]), 2),
        "transport_cost_yuan_for_two": round(float(row["transport_cost_yuan_for_two"]), 2),
        "ticket_cost_yuan_for_two": round(float(row["ticket_cost_yuan_for_two"]), 2),
        "rough_hotel_cost_yuan": round(float(row["rough_hotel_cost_yuan"]), 2),
        "cost_per_spot_yuan": round(float(row["cost_per_spot_yuan"]), 2),
        "total_travel_hours": round(float(row["total_travel_hours"]), 3),
        "planned_trip_days": int(row["planned_trip_days"]),
        "buffer_days": int(row["buffer_days"]),
        "red_days": int(row["red_days"]),
        "yellow_days": int(row["yellow_days"]),
        "time_window_violations": int(row["time_window_violations"]),
        "mean_comfort_score": round(float(row["mean_comfort_score"]), 3),
        "p10_comfort_score": round(float(row["p10_comfort_score"]), 3),
        "operational_success_probability": round(float(row["operational_success_probability"]), 3),
        "strict_comfort_success_probability": round(float(row["strict_comfort_success_probability"]), 3),
        "p95_cost_yuan": round(float(row["p95_cost_yuan"]), 2),
        "cvar75_loss": round(float(row["cvar75_loss"]), 3),
        "topic_executable_feasible": bool(row["topic_executable_feasible"]),
        "operational_robust_feasible": bool(row["operational_robust_feasible"]),
        "strict_comfort_reference_feasible": bool(row["strict_comfort_reference_feasible"]),
        "route_sequence": row["route_sequence"],
        "selection_note": note,
    }
    if baseline is not None:
        base_cost = float(baseline["total_cost_yuan_excluding_meals"])
        record["spots_delta_vs_topic_primary"] = int(row["spots_count"] - baseline["spots_count"])
        record["cost_saving_vs_topic_primary_yuan"] = round(base_cost - float(row["total_cost_yuan_excluding_meals"]), 2)
        record["cost_saving_vs_topic_primary_pct"] = round((base_cost - float(row["total_cost_yuan_excluding_meals"])) / base_cost, 6)
        record["operational_success_delta_vs_topic_primary"] = round(
            float(row["operational_success_probability"]) - float(baseline["operational_success_probability"]), 6
        )
        record["buffer_days_delta_vs_topic_primary"] = int(row["buffer_days"] - baseline["buffer_days"])
        record["red_days_delta_vs_topic_primary"] = int(row["red_days"] - baseline["red_days"])
    return record


def best_by_lexicographic(df: pd.DataFrame, mask_col: str) -> Optional[pd.Series]:
    feasible = df[df[mask_col]].copy()
    if feasible.empty:
        return None
    return feasible.sort_values(
        ["spots_count", "total_cost_yuan_excluding_meals", "operational_success_probability"],
        ascending=[False, True, False],
    ).iloc[0]


def min_cost_for_q(df: pd.DataFrame, mask_col: str, q: int, exact: bool = True) -> Optional[pd.Series]:
    feasible = df[df[mask_col]].copy()
    feasible = feasible[feasible["spots_count"].eq(q) if exact else feasible["spots_count"].ge(q)]
    if feasible.empty:
        return None
    return feasible.sort_values(
        ["total_cost_yuan_excluding_meals", "operational_success_probability", "buffer_days"],
        ascending=[True, False, False],
    ).iloc[0]


def frontier_table(df: pd.DataFrame) -> pd.DataFrame:
    policies = [
        (
            "topic_executable_min_cost",
            "题面可执行：30天、严格排程、0时间窗违规、普通游客路线；每个覆盖数下费用最低",
            "topic_executable_feasible",
        ),
        (
            "operational_robust_min_cost",
            f"题面+运营鲁棒：在题面可执行基础上要求运营成功率>={OPERATIONAL_SUCCESS_THRESHOLD:.2f}",
            "operational_robust_feasible",
        ),
        (
            "strict_comfort_reference_min_cost",
            f"舒适参考：在题面可执行基础上要求严格舒适成功率>={STRICT_COMFORT_REFERENCE_THRESHOLD:.2f}",
            "strict_comfort_reference_feasible",
        ),
    ]
    rows: List[Dict[str, object]] = []
    q_values = sorted(int(v) for v in df["spots_count"].dropna().unique())
    for policy_id, policy_note, mask_col in policies:
        for q in q_values:
            exact_row = min_cost_for_q(df, mask_col, q, exact=True)
            at_least_row = min_cost_for_q(df, mask_col, q, exact=False)
            for coverage_mode, row in [("exact_q", exact_row), ("at_least_q", at_least_row)]:
                if row is None:
                    rows.append(
                        {
                            "policy_id": policy_id,
                            "policy_note": policy_note,
                            "coverage_mode": coverage_mode,
                            "coverage_threshold_q": q,
                            "status": "infeasible_in_candidate_set",
                        }
                    )
                    continue
                rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_note": policy_note,
                        "coverage_mode": coverage_mode,
                        "coverage_threshold_q": q,
                        "status": "selected",
                        "route_id": row["route_id"],
                        "spots_count": int(row["spots_count"]),
                        "total_cost_yuan_excluding_meals": round(float(row["total_cost_yuan_excluding_meals"]), 2),
                        "transport_cost_yuan_for_two": round(float(row["transport_cost_yuan_for_two"]), 2),
                        "ticket_cost_yuan_for_two": round(float(row["ticket_cost_yuan_for_two"]), 2),
                        "rough_hotel_cost_yuan": round(float(row["rough_hotel_cost_yuan"]), 2),
                        "planned_trip_days": int(row["planned_trip_days"]),
                        "buffer_days": int(row["buffer_days"]),
                        "red_days": int(row["red_days"]),
                        "operational_success_probability": round(float(row["operational_success_probability"]), 3),
                        "strict_comfort_success_probability": round(float(row["strict_comfort_success_probability"]), 3),
                        "mean_comfort_score": round(float(row["mean_comfort_score"]), 3),
                        "p10_comfort_score": round(float(row["p10_comfort_score"]), 3),
                        "route_sequence": row["route_sequence"],
                    }
                )
    return pd.DataFrame(rows)


def marginal_table(frontier: pd.DataFrame) -> pd.DataFrame:
    base = frontier[
        (frontier["policy_id"] == "topic_executable_min_cost")
        & (frontier["coverage_mode"] == "exact_q")
        & (frontier["status"] == "selected")
    ].copy()
    base = base.sort_values("coverage_threshold_q")
    rows = []
    prev = None
    for _, row in base.iterrows():
        item = {
            "coverage_q": int(row["coverage_threshold_q"]),
            "route_id": row["route_id"],
            "min_cost_yuan": float(row["total_cost_yuan_excluding_meals"]),
            "buffer_days": int(row["buffer_days"]),
            "red_days": int(row["red_days"]),
            "operational_success_probability": float(row["operational_success_probability"]),
        }
        if prev is None:
            item["delta_cost_from_previous_q"] = ""
            item["delta_spots_from_previous_q"] = ""
            item["marginal_cost_per_added_spot"] = ""
        else:
            dc = float(row["total_cost_yuan_excluding_meals"]) - float(prev["total_cost_yuan_excluding_meals"])
            dq = int(row["coverage_threshold_q"]) - int(prev["coverage_threshold_q"])
            item["delta_cost_from_previous_q"] = round(dc, 2)
            item["delta_spots_from_previous_q"] = dq
            item["marginal_cost_per_added_spot"] = round(dc / dq, 2) if dq else ""
        rows.append(item)
        prev = row
    return pd.DataFrame(rows)


def selected_comparison(df: pd.DataFrame) -> pd.DataFrame:
    topic_primary = best_by_lexicographic(df, "topic_executable_feasible")
    if topic_primary is None:
        raise RuntimeError("No topic-executable Q1 candidate found.")

    rows: List[Dict[str, object]] = []
    choices = [
        (
            topic_primary,
            "题面优先最高覆盖方案",
            "先最大化30天严格可执行景点数，再在该覆盖数下最小化除吃饭外总费用。",
        ),
        (
            best_by_lexicographic(df, "operational_robust_feasible"),
            "题面+运营鲁棒方案",
            f"在题面可执行基础上加入运营成功率>={OPERATIONAL_SUCCESS_THRESHOLD:.2f}，再按覆盖数优先、费用次优选择。",
        ),
        (
            min_cost_for_q(df, "topic_executable_feasible", 24, exact=True),
            "24景点最低成本对照",
            "固定24景点覆盖数，按题面费用目标选择最低费用路线，用于对照原Q1-V3主推方案。",
        ),
        (
            df.loc[df["route_id"].eq("Q1V3_Q24_120")].iloc[0],
            "原Q1-V3运营鲁棒主推",
            "保留既有Q1-V3主推方案，作为现实鲁棒性更强的解释层方案。",
        ),
        (
            best_by_lexicographic(df, "strict_comfort_reference_feasible"),
            "舒适保守参考方案",
            f"在题面可执行基础上要求严格舒适成功率>={STRICT_COMFORT_REFERENCE_THRESHOLD:.2f}，用于低强度偏好对照。",
        ),
    ]
    for row, role, note in choices:
        if row is not None:
            rows.append(route_record(row, role, note, baseline=topic_primary))
    return pd.DataFrame(rows)


def objective_hierarchy() -> pd.DataFrame:
    rows = [
        {
            "level": "L0",
            "name": "题面可执行性约束",
            "mathematical_role": "feasibility",
            "definition": "普通游客路线、30天内、严格日程可执行、0时间窗违规、硬约束通过。",
            "reason": "对应题目中的“合适旅游路线”，避免把不可执行的高覆盖路线作为主答案。",
        },
        {
            "level": "L1",
            "name": "覆盖数最大化",
            "mathematical_role": "primary_objective",
            "definition": "maximize N(R)",
            "reason": "对应“游尽可能多的地方”。",
        },
        {
            "level": "L2",
            "name": "除吃饭外总费用最小化",
            "mathematical_role": "secondary_objective",
            "definition": "minimize C(R)=交通费+门票/预约费+住宿费+必要本地接驳费用",
            "reason": "对应“花最少的钱”及“估算除吃饭之外的费用”。",
        },
        {
            "level": "L3",
            "name": "现实质量与鲁棒性评价",
            "mathematical_role": "tie_breaker_and_audit",
            "definition": "缓冲日、舒适度、红色压力日、运营成功率、严格舒适成功率、CVaR等。",
            "reason": "作为模型亮点与推荐方案解释，不抢占题面主目标。",
        },
    ]
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table renderer."""

    if df.empty:
        return "_无记录_"
    rendered = df.copy()
    for col in rendered.columns:
        rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, frontier: pd.DataFrame, selected: pd.DataFrame, marginal: pd.DataFrame) -> None:
    outputs = OUT / "outputs"
    reports = OUT / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    objective_hierarchy().to_csv(outputs / "q1_v5_objective_hierarchy.csv", index=False, encoding="utf-8-sig")
    frontier.to_csv(outputs / "q1_v5_cost_coverage_frontier.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(outputs / "q1_v5_selected_comparison.csv", index=False, encoding="utf-8-sig")
    marginal.to_csv(outputs / "q1_v5_topic_marginal_cost.csv", index=False, encoding="utf-8-sig")

    candidate_cols = [
        "route_id",
        "spots_count",
        "total_cost_yuan_excluding_meals",
        "transport_cost_yuan_for_two",
        "ticket_cost_yuan_for_two",
        "rough_hotel_cost_yuan",
        "planned_trip_days",
        "buffer_days",
        "red_days",
        "time_window_violations",
        "operational_success_probability",
        "strict_comfort_success_probability",
        "topic_executable_feasible",
        "operational_robust_feasible",
        "strict_comfort_reference_feasible",
        "route_sequence",
    ]
    df[candidate_cols].sort_values(
        ["topic_executable_feasible", "spots_count", "total_cost_yuan_excluding_meals"],
        ascending=[False, False, True],
    ).to_csv(outputs / "q1_v5_candidate_reclassified.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(reports / "新疆旅游第一问Q1_V5题面优先成本覆盖校准结果.xlsx") as writer:
            objective_hierarchy().to_excel(writer, sheet_name="objective_hierarchy", index=False)
            selected.to_excel(writer, sheet_name="selected_comparison", index=False)
            frontier.to_excel(writer, sheet_name="cost_coverage_frontier", index=False)
            marginal.to_excel(writer, sheet_name="marginal_cost", index=False)
    except Exception:
        # Excel is a convenience artifact; CSV/Markdown are the authoritative outputs.
        pass


def make_report(frontier: pd.DataFrame, selected: pd.DataFrame, marginal: pd.DataFrame) -> str:
    def row_by_role(role: str) -> pd.Series:
        return selected[selected["solution_role"].eq(role)].iloc[0]

    topic = row_by_role("题面优先最高覆盖方案")
    operational = row_by_role("题面+运营鲁棒方案")
    robust = row_by_role("原Q1-V3运营鲁棒主推")
    comfort = row_by_role("舒适保守参考方案")
    q24_cost = row_by_role("24景点最低成本对照")

    selected_table = dataframe_to_markdown(selected[
        [
            "solution_role",
            "route_id",
            "spots_count",
            "total_cost_yuan_excluding_meals",
            "buffer_days",
            "red_days",
            "operational_success_probability",
            "strict_comfort_success_probability",
            "cost_saving_vs_topic_primary_yuan",
        ]
    ])

    marginal_table_md = dataframe_to_markdown(marginal)

    return f"""# 新疆旅游第一问 Q1-V5 题面优先成本-覆盖校准报告

## 1. 实验定位

Q1-V3 的鲁棒多目标模型仍然有效，但第一问题面首先要求“一个月内花最少的钱游尽可能多的地方，并估算除吃饭之外的费用”。因此，本轮不推翻原方案，而是在 Q1-V3 候选路线空间上新增一个题面优先校准层：

1. 先判定路线是否为30天内严格可执行的普通游客路线；
2. 再优先最大化游览景点数；
3. 在同等覆盖数下最小化除吃饭外总费用；
4. 最后用缓冲日、舒适度、红色压力日和运营成功率解释推荐差异。

## 2. 目标层级

数学上采用字典序目标：

```text
L0: R 属于题面可执行集合
L1: maximize N(R)
L2: minimize C(R), where C=交通费+门票/预约费+住宿费+必要本地接驳费用
L3: evaluate robustness and comfort
```

其中 L3 是“合适路线”的增强解释，不抢占题面主目标。

## 3. 关键方案对比

{selected_table}

## 4. 主要发现

- **题面优先最高覆盖方案**为 `{topic['route_id']}`：30天严格可执行，覆盖 `{int(topic['spots_count'])}` 个景点，除吃饭外费用 `{float(topic['total_cost_yuan_excluding_meals']):.2f}` 元。但它只有 `{int(topic['buffer_days'])}` 天缓冲、`{int(topic['red_days'])}` 个红色压力日，运营成功率为 `{float(topic['operational_success_probability']):.3f}`，说明高覆盖方案更接近“极限执行”。
- **题面+运营鲁棒方案**为 `{operational['route_id']}`：覆盖 `{int(operational['spots_count'])}` 个景点，费用 `{float(operational['total_cost_yuan_excluding_meals']):.2f}` 元，比题面最高覆盖方案少 `{abs(int(operational['spots_delta_vs_topic_primary']))}` 个景点，但节省 `{float(operational['cost_saving_vs_topic_primary_yuan']):.2f}` 元，运营成功率提升到 `{float(operational['operational_success_probability']):.3f}`。
- **原 Q1-V3 运营鲁棒主推** `{robust['route_id']}` 仍然有效：覆盖 `{int(robust['spots_count'])}` 个景点，费用 `{float(robust['total_cost_yuan_excluding_meals']):.2f}` 元，比题面最高覆盖方案节省 `{float(robust['cost_saving_vs_topic_primary_yuan']):.2f}` 元，并提供 `{int(robust['buffer_days'])}` 天缓冲和 `{float(robust['operational_success_probability']):.3f}` 的运营成功率。
- 固定24景点时，最低成本对照 `{q24_cost['route_id']}` 的费用为 `{float(q24_cost['total_cost_yuan_excluding_meals']):.2f}` 元，低于原 Q1-V3 主推 `{robust['route_id']}`。因此论文中可以把原主推解释为“鲁棒综合推荐”，而把 `{q24_cost['route_id']}` 作为“同覆盖数最低费用对照”。
- **舒适保守参考方案** `{comfort['route_id']}` 覆盖 `{int(comfort['spots_count'])}` 个景点，费用 `{float(comfort['total_cost_yuan_excluding_meals']):.2f}` 元，严格舒适成功率为 `{float(comfort['strict_comfort_success_probability']):.3f}`，适合作为低强度偏好备选。

## 5. 题面前沿的边际费用

{marginal_table_md}

边际费用显示，从26景点提升到28景点需要额外增加较高费用，同时缓冲日下降、运营成功率下降。因此“28景点”适合作为题面优先最高覆盖答案；“26景点”和“24景点”则适合作为更现实的鲁棒执行答案。

## 6. 论文表述建议

第一问最终可以形成双层答案：

- **题面主答案**：先给出成本-覆盖前沿，并指出30天严格可执行的最高覆盖候选为 `{topic['route_id']}`，费用 `{float(topic['total_cost_yuan_excluding_meals']):.2f}` 元。
- **现实推荐答案**：在考虑缓冲日、红色压力日和运营成功率后，推荐 `{operational['route_id']}` 或保留原 `{robust['route_id']}` 作为更稳健方案。

这样既忠实于题面“多游、省钱”的主目标，又保留 Q1-V3 在真实执行风险上的优势。
"""


def write_readme() -> None:
    text = """# Q1-V5 题面优先成本-覆盖校准

本目录在保留 Q1-V3 鲁棒多目标方案的基础上，按题面“花最少的钱游尽可能多的地方”重新校准第一问的求解叙事。

核心输出：

- `outputs/q1_v5_objective_hierarchy.csv`：题面优先目标层级；
- `outputs/q1_v5_cost_coverage_frontier.csv`：成本-覆盖前沿；
- `outputs/q1_v5_selected_comparison.csv`：题面最高覆盖、运营鲁棒、原Q1-V3主推等方案对比；
- `outputs/q1_v5_topic_marginal_cost.csv`：题面前沿边际费用；
- `reports/新疆旅游第一问Q1_V5题面优先成本覆盖校准报告.md`：论文口径报告。

复现：

```powershell
E:\\Anaconda\\envs\\xj-opt\\python.exe "E:\\Desktop\\运筹学项目\\第一问_30天路线优化材料包\\11_Q1_V5_题面优先成本覆盖校准\\scripts\\q1_v5_cost_coverage_calibration.py"
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = read_candidates()
    frontier = frontier_table(df)
    selected = selected_comparison(df)
    marginal = marginal_table(frontier)
    write_outputs(df, frontier, selected, marginal)
    report = make_report(frontier, selected, marginal)
    (OUT / "reports" / "新疆旅游第一问Q1_V5题面优先成本覆盖校准报告.md").write_text(report, encoding="utf-8")
    write_readme()
    print("Q1-V5 cost-coverage calibration complete.")
    print(selected[["solution_role", "route_id", "spots_count", "total_cost_yuan_excluding_meals", "operational_success_probability"]].to_string(index=False))


if __name__ == "__main__":
    main()
