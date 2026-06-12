from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
OUTPUT_DIR = ROOT / "outputs"
OUT_DIR = ROOT / "research_synthesis_outputs"


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


def maybe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_csv(path)
    return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])


def build_artifact_index() -> pd.DataFrame:
    artifacts = [
        ("基础数据底座", "outputs/新疆旅游优化建模数据底座.xlsx", "清洗后的景点、交通、住宿、费用数据"),
        ("第一问主结果", "outputs/新疆旅游30天硬约束混合元启发式求解结果.xlsx", "30天硬约束混合元启发式路线"),
        ("第二问重建结果", "outputs/新疆旅游第二问交通费用最小化重建结果.xlsx", "两年路径覆盖与开放式下界"),
        ("四问收束方案", "outputs/新疆旅游线路安排四问最终收束方案.md", "四问正文主口径"),
        ("人性化评价", "outputs/新疆旅游人性化场景强化结果.xlsx", "疲劳、舒适度、画像适配"),
        ("自适应策略", "outputs/新疆旅游自适应策略实验结果.xlsx", "画像改造路线与五一分流"),
        ("政策仿真", "outputs/新疆旅游政策仿真实验结果.xlsx", "多口岸价格、延期、预约上限仿真"),
        ("随机联合优化", "outputs/新疆旅游随机仿真与联合优化结果.xlsx", "Monte Carlo与联合调度拓展"),
    ]
    rows = []
    for name, rel, note in artifacts:
        path = ROOT / rel
        last_write_time = ""
        if path.exists():
            last_write_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "artifact_name": name,
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "last_write_time": last_write_time,
                "role": note,
            }
        )
    return pd.DataFrame(rows)


def build_master_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    q1 = maybe_read(ROOT / "hybrid_30day_outputs" / "hard30_route_summary.csv")
    if not q1.empty:
        row = q1.iloc[0]
        rows.append(
            {
                "question": "Q1",
                "model_layer": "确定性主模型",
                "model_name": "HYBRID_30D_ACO_ALNS_SA",
                "primary_metric": "30天内高覆盖低成本路线",
                "key_result": f"{int(num(row['optimized_spots_count']))}景点，{int(num(row['scheduled_days']))}天，总成本{num(row['itinerary_proxy_cost_yuan_excluding_meals']):.2f}元",
                "evidence_table": "hybrid_30day_outputs/hard30_route_summary.csv",
            }
        )

    q1_variant = maybe_read(ROOT / "adaptive_strategy_outputs" / "q1_variant_summary.csv")
    if not q1_variant.empty:
        for persona_id in ["standard_active", "family_comfort", "senior_slow"]:
            item = q1_variant[q1_variant["persona_id"] == persona_id]
            if item.empty:
                continue
            row = item.iloc[0]
            rows.append(
                {
                    "question": "Q1",
                    "model_layer": "人性化/策略层",
                    "model_name": row["persona_name"],
                    "primary_metric": "画像适配路线",
                    "key_result": f"{int(num(row['spots_count']))}景点，{int(num(row['scheduled_days']))}天，均值舒适度{num(row['mean_comfort_score']):.2f}，状态{row['constraint_status']}",
                    "evidence_table": "adaptive_strategy_outputs/q1_variant_summary.csv",
                }
            )

    q2 = maybe_read(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    if not q2.empty:
        for _, row in q2.iterrows():
            rows.append(
                {
                    "question": "Q2",
                    "model_layer": "路径覆盖主模型/下界",
                    "model_name": row["scenario_id"],
                    "primary_metric": "两年覆盖38个普通景点",
                    "key_result": f"{int(num(row['total_estimated_days']))}天，交通费{num(row['total_transport_cost_yuan_for_two']):.2f}元，最大单年{int(num(row['max_year_days']))}天",
                    "evidence_table": "problem2_openpath_outputs/scenario_totals.csv",
                }
            )

    q2_policy = maybe_read(ROOT / "policy_simulation_outputs" / "q2_gateway_price_summary.csv")
    if not q2_policy.empty:
        for _, row in q2_policy.iterrows():
            rows.append(
                {
                    "question": "Q2",
                    "model_layer": "政策仿真层",
                    "model_name": row["scenario_id"],
                    "primary_metric": "开放式多口岸更便宜概率",
                    "key_result": f"胜率{num(row['prob_open_gateway_cheaper']) * 100:.2f}%，期望节省{num(row['expected_savings_if_choose_open_yuan_for_two']):.2f}元",
                    "evidence_table": "policy_simulation_outputs/q2_gateway_price_summary.csv",
                }
            )

    q3 = maybe_read(ROOT / "enhanced_model_outputs" / "problem3_minmax_summary.csv")
    if not q3.empty:
        rows.append(
            {
                "question": "Q3",
                "model_layer": "确定性主模型",
                "model_name": "S3_MinMax_MultiTeam",
                "primary_metric": "三组文化考察最小最大完成时间",
                "key_result": f"最大完成{num(q3['total_hours'].map(num).max()):.2f}小时，时间差{num(q3['balance_gap_hours'].map(num).max()):.2f}小时",
                "evidence_table": "enhanced_model_outputs/problem3_minmax_summary.csv",
            }
        )

    q3_policy = maybe_read(ROOT / "policy_simulation_outputs" / "q3_project_buffer_curve.csv")
    if not q3_policy.empty:
        ok = q3_policy[q3_policy["all_groups_on_time_probability"].map(num) >= 0.9]
        row = ok.iloc[0] if not ok.empty else q3_policy.iloc[-1]
        rows.append(
            {
                "question": "Q3",
                "model_layer": "风险仿真层",
                "model_name": "Fieldwork_Buffer_MC",
                "primary_metric": "项目级90%可靠性缓冲",
                "key_result": f"建议缓冲{int(num(row['buffer_days']))}天，完成概率{num(row['all_groups_on_time_probability']) * 100:.2f}%",
                "evidence_table": "policy_simulation_outputs/q3_project_buffer_curve.csv",
            }
        )

    q4 = maybe_read(ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv")
    if not q4.empty:
        rows.append(
            {
                "question": "Q4",
                "model_layer": "容量分配主模型",
                "model_name": "RouteColumn_CapacityFlow",
                "primary_metric": "五一12天接待路线组合",
                "key_result": f"{len(q4)}条线路，分配{int(q4['allocated_visitors'].map(num).sum())}人，总容量{int(q4['route_capacity_persons_12day'].map(num).sum())}人",
                "evidence_table": "enhanced_model_outputs/problem4_capacity_flow.csv",
            }
        )

    q4_strategy = maybe_read(ROOT / "adaptive_strategy_outputs" / "q4_reallocation_summary.csv")
    if not q4_strategy.empty:
        row = q4_strategy[q4_strategy["demand_multiplier"].map(num) == 1.10]
        if not row.empty:
            item = row.iloc[0]
            rows.append(
                {
                    "question": "Q4",
                    "model_layer": "自适应分流层",
                    "model_name": "DemandShock_Reallocation",
                    "primary_metric": "需求上浮10%的分流效果",
                    "key_result": f"分流新增接待{int(num(item['additional_served_by_reallocation']))}人，仍溢出{int(num(item['unresolved_overflow_visitors']))}人",
                    "evidence_table": "adaptive_strategy_outputs/q4_reallocation_summary.csv",
                }
            )

    q4_policy = maybe_read(ROOT / "policy_simulation_outputs" / "q4_reservation_policy_summary.csv")
    if not q4_policy.empty:
        row = q4_policy[(q4_policy["demand_multiplier"].map(num) == 1.10) & (q4_policy["policy_id"] == "safety_cap_95pct")]
        if not row.empty:
            item = row.iloc[0]
            rows.append(
                {
                    "question": "Q4",
                    "model_layer": "预约管理层",
                    "model_name": "SafetyCap95",
                    "primary_metric": "95%预约上限",
                    "key_result": f"接待{int(num(item['served_visitors']))}人，等待/拒绝{int(num(item['waitlist_or_rejected_visitors']))}人，利用率{num(item['capacity_utilization']) * 100:.1f}%",
                    "evidence_table": "policy_simulation_outputs/q4_reservation_policy_summary.csv",
                }
            )

    return pd.DataFrame(rows)


def build_model_stack() -> pd.DataFrame:
    rows = [
        ("数据层", "景点属性、高德OD、住宿锚点、容量与费用参数", "真实交通OD + 题面/Excel清洗数据 + 场景校准参数"),
        ("确定性优化层", "PCOP、两年路径覆盖、MinMax Multi-TSP、容量流", "回答四问的主结果"),
        ("元启发式层", "ACO + ALNS + SA + 2-opt/relocate", "在复杂约束下生成高质量路线"),
        ("人性化层", "疲劳凸惩罚、游客画像、长途转场拆日", "把人的体验转化为可计算约束"),
        ("仿真层", "Monte Carlo价格、延期、需求扰动", "评估路线和政策在真实波动下是否稳健"),
        ("策略层", "多口岸阈值、缓冲天、预约上限、替代线路分流", "形成可答辩、可运营的条件方案"),
    ]
    return pd.DataFrame(rows, columns=["layer", "content", "purpose"])


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = tables["master_metrics"]
    stack = tables["model_stack"]
    artifacts = tables["artifact_index"]
    q1 = metrics[metrics["question"] == "Q1"]
    q2 = metrics[metrics["question"] == "Q2"]
    q3 = metrics[metrics["question"] == "Q3"]
    q4 = metrics[metrics["question"] == "Q4"]

    lines = [
        "# 新疆旅游四问 Fancy 建模科研总报告",
        "",
        "## 1. 总体定位",
        "",
        "本项目已经不只是求一条路线，而是形成了一个分层旅游决策系统：先用真实与校准数据构造加权图，再用运筹优化给出四问主结果，随后加入游客画像、疲劳、容量、价格、延期和需求扰动，使模型更贴近真实旅游场景。",
        "",
        "核心表达可以概括为：",
        "",
        "```text",
        "真实数据底座 -> 确定性优化主解 -> 元启发式增强 -> 人性化评价 -> Monte Carlo仿真 -> 条件策略/应急方案",
        "```",
        "",
        "## 2. 模型分层",
        "",
        stack.to_markdown(index=False),
        "",
        "## 3. 四问主结论总览",
        "",
        metrics.to_markdown(index=False),
        "",
        "## 4. 第一问：30天路线",
        "",
        q1.to_markdown(index=False),
        "",
        "第一问正文建议采用 30 天硬约束混合元启发式作为主答案。强化口径是：普通体力型和探索型可以不删点完成；亲子舒适型建议删去少量低优先级非核心点；长者慢游型不建议硬塞30天，应作为慢游拆期方案。",
        "",
        "## 5. 第二问：两年暑假路线",
        "",
        q2.to_markdown(index=False),
        "",
        "第二问正文建议保留乌鲁木齐起讫模型作为保守主方案，同时把开放式多口岸作为条件方案。阈值逻辑是：若两人多口岸外部大交通额外差价低于 1347.99 元，则开放式方案在总费用上更优。第二问 DFJ 精确求证仍未生成最终 summary，不能替换当前主答案。",
        "",
        "## 6. 第三问：文化考察",
        "",
        q3.to_markdown(index=False),
        "",
        "第三问不应按景点数量平均分配，而应按完成时间、远程交通、审批/风险和文化任务量均衡。仿真表明项目级90%可靠性需要约4天缓冲，因此论文中应把第三问称为文化专项调研调度，而不是普通旅游路线。",
        "",
        "## 7. 第四问：五一接待",
        "",
        q4.to_markdown(index=False),
        "",
        "第四问应强调路线产品组合和容量管理。基准需求已经接近满负荷；需求上浮10%时，单靠既有替代线路仍有溢出，因此需要预约上限、价格引导、新增运力或分流宣传。",
        "",
        "## 8. 论文中建议突出的创新点",
        "",
        "1. 统一加权图：把景点、交通、住宿、容量、风险放入同一数据底座。",
        "2. 四问分层建模：PCOP、两年路径覆盖、MinMax Multi-TSP、容量流分别服务不同问题。",
        "3. 混合元启发式：用 ACO 初始化、ALNS 破坏修复、SA 接受准则处理大规模约束。",
        "4. 游客画像：把亲子、长者、探索型游客的疲劳阈值写入模型。",
        "5. 运营策略：多口岸阈值、文化缓冲天、预约上限和需求分流让结果更贴近实际。",
        "",
        "## 9. 交付文件索引",
        "",
        artifacts.to_markdown(index=False),
        "",
        "## 10. 后续还能继续强化的方向",
        "",
        "1. 若安装 Gurobi/CPLEX，可把第二问 DFJ lazy constraints 改成真正 branch-and-cut，提升证明效率。",
        "2. 补真实酒店房态、景区预约余量后，第四问可以升级为动态容量控制。",
        "3. 补真实多机场机票/铁路票价后，第二问开放式口岸方案可从下界变成正式主模型候选。",
        "4. 进一步把天气、道路封闭和客流热力接入为时变边权，形成在线重规划模型。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables = {
        "master_metrics": build_master_metrics(),
        "model_stack": build_model_stack(),
        "artifact_index": build_artifact_index(),
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游四问Fancy建模科研总览.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游四问Fancy建模科研总报告.md")
    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游四问Fancy建模科研总览.xlsx",
        "report": "outputs/新疆旅游四问Fancy建模科研总报告.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
