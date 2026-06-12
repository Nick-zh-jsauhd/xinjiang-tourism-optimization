# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "final_research_package_v2_outputs"
FIG_DIR = OUT_DIR / "figures"
OUTPUT_DIR = ROOT / "outputs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


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


def file_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(read_csv(path))
    except Exception:
        return None


def build_final_answers() -> pd.DataFrame:
    master = read_csv(ROOT / "research_synthesis_outputs" / "master_metrics.csv")
    portfolio = read_csv(ROOT / "risk_policy_outputs" / "integrated_policy_portfolio.csv")
    balanced = portfolio[portfolio["profile_id"].eq("balanced")].iloc[0] if not portfolio.empty else {}
    q1_standard = master[(master["question"].eq("Q1")) & (master["model_name"].eq("普通体力型30天主线"))]
    q1_main = master[(master["question"].eq("Q1")) & (master["model_name"].eq("HYBRID_30D_ACO_ALNS_SA"))]
    q2_root = master[(master["question"].eq("Q2")) & (master["model_name"].eq("P2_ROOTED_URUMQI_MINCOST"))]
    q2_open = master[(master["question"].eq("Q2")) & (master["model_name"].eq("P2_OPEN_GATEWAY_LOWER_BOUND"))]
    q3_main = master[(master["question"].eq("Q3")) & (master["model_name"].eq("S3_MinMax_MultiTeam"))]
    q4_main = master[(master["question"].eq("Q4")) & (master["model_name"].eq("RouteColumn_CapacityFlow"))]

    def result_text(df: pd.DataFrame) -> str:
        return str(df.iloc[0]["key_result"]) if not df.empty else "NA"

    return pd.DataFrame(
        [
            {
                "question": "Q1",
                "final_answer": f"主线采用30天硬约束混合元启发式：{result_text(q1_main)}；普通体力型可用画像路线：{result_text(q1_standard)}。",
                "humanized_policy": str(balanced.get("q1_persona_policy", "按画像分层给路线")),
                "evidence": "hybrid_30day_outputs/hard30_route_summary.csv; adaptive_strategy_outputs/q1_variant_summary.csv; risk_policy_outputs/q1_policy_selection.csv",
                "claim_strength": "强可行解 + 人群画像鲁棒性；不声称全局最优",
            },
            {
                "question": "Q2",
                "final_answer": f"保守主方案：{result_text(q2_root)}；开放式多口岸下界：{result_text(q2_open)}；当两人外部大交通额外差价低于1347.99元时开放式更优。",
                "humanized_policy": str(balanced.get("q2_transport_policy", "双方案实时比价后择优")),
                "evidence": "problem2_openpath_outputs/scenario_totals.csv; policy_simulation_outputs/q2_gateway_price_summary.csv; risk_policy_outputs/q2_policy_selection.csv",
                "claim_strength": "交通费用主模型 + 条件策略；DFJ精确证明未完成",
            },
            {
                "question": "Q3",
                "final_answer": f"三组文化考察采用MinMax均衡：{result_text(q3_main)}。",
                "humanized_policy": str(balanced.get("q3_fieldwork_policy", "5天项目缓冲")),
                "evidence": "enhanced_model_outputs/problem3_minmax_summary.csv; policy_simulation_outputs/q3_project_buffer_curve.csv; risk_policy_outputs/q3_policy_selection.csv",
                "claim_strength": "确定性均衡 + 风险缓冲策略",
            },
            {
                "question": "Q4",
                "final_answer": f"五一容量流主方案：{result_text(q4_main)}。",
                "humanized_policy": str(balanced.get("q4_capacity_policy", "分时预约+动态分流")),
                "evidence": "enhanced_model_outputs/problem4_capacity_flow.csv; adaptive_strategy_outputs/q4_reallocation_summary.csv; digital_twin_outputs/q4_capacity_policy.csv",
                "claim_strength": "容量分配 + 预约/分流压力测试",
            },
        ]
    )


def build_model_stack() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "数据底座",
                "role": "把题面Excel、景点、门票、时长、OD、容量、酒店等统一成可建模表",
                "main_artifacts": "outputs/新疆旅游优化建模数据底座.xlsx; enhanced_data/*; model_data/*",
                "value": "解决原始数据散、口径不统一的问题",
            },
            {
                "layer": "确定性优化",
                "role": "分别回答四问的核心目标：30天路线、两年交通费、文化MinMax、五一容量流",
                "main_artifacts": "hybrid_30day_outputs; problem2_openpath_outputs; enhanced_model_outputs",
                "value": "给出可计算、可复现的主答案",
            },
            {
                "layer": "元启发式/局部搜索",
                "role": "ACO+ALNS+SA+2-opt 构造高质量可行路线，并修复时间/舒适度约束",
                "main_artifacts": "hybrid_metaheuristic_outputs; hybrid_30day_outputs",
                "value": "提升路径质量和工程可运行性",
            },
            {
                "layer": "人性化画像",
                "role": "将普通体力、探索紧凑、亲子舒适、长者慢游映射为不同天数/舒适度/删点策略",
                "main_artifacts": "adaptive_strategy_outputs; humanized_scenario_outputs",
                "value": "让路线从数学最短路变成可旅行方案",
            },
            {
                "layer": "数字孪生仿真",
                "role": "同时扰动天气、道路、预约、票价、审批、客流，验证路线和策略鲁棒性",
                "main_artifacts": "digital_twin_outputs; outputs/新疆旅游四问数字孪生鲁棒性实验报告.md",
                "value": "解释真实世界不确定性下模型是否扛得住",
            },
            {
                "layer": "风险感知决策",
                "role": "用期望损失+CVaR尾部风险为不同风险偏好选择策略组合",
                "main_artifacts": "risk_policy_outputs; outputs/新疆旅游四问风险感知策略选择模型报告.md",
                "value": "将仿真结果转化为论文主方案、保守方案和应急方案",
            },
            {
                "layer": "求解效率研究",
                "role": "说明HiGHS/DFJ瓶颈，并给出OR-Tools warm start + Gurobi/CPLEX lazy cuts升级路径",
                "main_artifacts": "solver_efficiency_outputs; scripts/solve_problem2_ortools_routing.py",
                "value": "避免把未证明最优的解误写成全局最优，同时给出工程强化方向",
            },
        ]
    )


def build_evidence_registry() -> pd.DataFrame:
    items = [
        ("Q1主路线", ROOT / "hybrid_30day_outputs" / "hard30_route_summary.csv", "32景点30天硬约束路线"),
        ("Q1画像路线", ROOT / "adaptive_strategy_outputs" / "q1_variant_summary.csv", "四类游客画像路线与舒适度"),
        ("Q2交通主模型", ROOT / "problem2_openpath_outputs" / "scenario_totals.csv", "乌鲁木齐起讫与开放式多口岸对比"),
        ("Q2价格仿真", ROOT / "policy_simulation_outputs" / "q2_gateway_price_summary.csv", "多口岸票价溢价敏感性"),
        ("Q2效率审计", ROOT / "solver_efficiency_outputs" / "highs_log_status.csv", "DFJ/flow/long MILP日志状态"),
        ("Q3文化MinMax", ROOT / "enhanced_model_outputs" / "problem3_minmax_summary.csv", "三组文化考察均衡结果"),
        ("Q3缓冲仿真", ROOT / "policy_simulation_outputs" / "q3_project_buffer_curve.csv", "项目缓冲天数成功概率"),
        ("Q4容量流", ROOT / "enhanced_model_outputs" / "problem4_capacity_flow.csv", "五一线路容量与分配"),
        ("Q4分流策略", ROOT / "adaptive_strategy_outputs" / "q4_reallocation_summary.csv", "需求冲击下动态分流"),
        ("数字孪生", ROOT / "digital_twin_outputs" / "recommendation_atlas.csv", "四问复合情景风险图谱"),
        ("风险策略", ROOT / "risk_policy_outputs" / "integrated_policy_portfolio.csv", "不同风险偏好策略组合"),
        ("视觉图表", ROOT / "visual_assets_outputs" / "visual_index.csv", "PPT图表索引"),
    ]
    rows = []
    for name, path, purpose in items:
        rows.append(
            {
                "evidence_name": name,
                "path": str(path),
                "exists": path.exists(),
                "rows_if_csv": file_rows(path),
                "purpose": purpose,
            }
        )
    return pd.DataFrame(rows)


def build_data_truth_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_domain": "景点名称/门票/游览时长/偏好属性",
                "source_type": "题面Excel清洗",
                "current_status": "可作为主数据",
                "paper_wording": "来源于题目附件并进行结构化清洗",
                "risk": "个别开闭园和价格可能随季节变化",
            },
            {
                "data_domain": "新疆境内自驾OD距离/时间",
                "source_type": "高德API",
                "current_status": "真实API距离矩阵",
                "paper_wording": "使用高德地图驾车距离与时长作为交通网络权重",
                "risk": "不是实时逐小时拥堵预测",
            },
            {
                "data_domain": "12306/航班外部大交通",
                "source_type": "未接入实时数据；用阈值与仿真替代",
                "current_status": "条件策略",
                "paper_wording": "建立多口岸费用阈值，出行前用实时票价决策",
                "risk": "不能写成已抓取真实票价",
            },
            {
                "data_domain": "酒店房量/价格",
                "source_type": "校准模拟",
                "current_status": "用于约束与风险解释",
                "paper_wording": "作为住宿可得性情景参数",
                "risk": "需要携程/美团/飞猪等接口才能实时化",
            },
            {
                "data_domain": "景区容量/预约余量",
                "source_type": "规则校准+仿真",
                "current_status": "可做压力测试",
                "paper_wording": "容量参数用于模拟预约与分流策略",
                "risk": "不能宣称为实时预约余量",
            },
            {
                "data_domain": "天气/道路封闭/审批延误",
                "source_type": "情景模拟",
                "current_status": "用于数字孪生压力测试",
                "paper_wording": "构造复合扰动情景进行鲁棒性检验",
                "risk": "不是气象预测模型",
            },
        ]
    )


def build_paper_insert_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_section": "摘要/创新点",
                "insert_content": "提出数据底座-确定性优化-元启发式-数字孪生-风险感知决策的分层建模框架",
                "evidence": "final_research_package_v2_outputs/model_stack_v2.csv",
            },
            {
                "paper_section": "数据预处理",
                "insert_content": "说明高德OD为真实API，酒店/容量/预约/天气为校准仿真或情景参数",
                "evidence": "final_research_package_v2_outputs/data_truth_table.csv",
            },
            {
                "paper_section": "问题一模型",
                "insert_content": "从单一30天路线扩展到游客画像和CVaR风险策略，解释亲子/长者为什么需要拆段",
                "evidence": "adaptive_strategy_outputs/q1_variant_summary.csv; risk_policy_outputs/q1_policy_selection.csv",
            },
            {
                "paper_section": "问题二模型",
                "insert_content": "保留乌鲁木齐起讫主方案，同时加入开放式多口岸阈值和实时比价策略",
                "evidence": "problem2_openpath_outputs/scenario_totals.csv; risk_policy_outputs/q2_policy_selection.csv",
            },
            {
                "paper_section": "问题三模型",
                "insert_content": "MinMax主模型后加入项目缓冲和机动小组策略，区分普通与极端情景",
                "evidence": "enhanced_model_outputs/problem3_minmax_summary.csv; risk_policy_outputs/q3_policy_selection.csv",
            },
            {
                "paper_section": "问题四模型",
                "insert_content": "容量流后加入预约上限、动态分流、分时预约，说明极端场景仍需前置限流",
                "evidence": "enhanced_model_outputs/problem4_capacity_flow.csv; digital_twin_outputs/q4_capacity_policy.csv",
            },
            {
                "paper_section": "算法与复杂度",
                "insert_content": "说明SciPy/HiGHS可得高质量可行解，DFJ全局证明未完成；提出OR-Tools+Gurobi升级方向",
                "evidence": "solver_efficiency_outputs/highs_log_status.csv",
            },
        ]
    )


def build_slide_storyline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "问题与挑战", "新疆景点分散、交通跨度大、暑期风险强，不能只做最短路。"),
            (2, "数据底座", "题面数据 + 高德OD + 容量/住宿/预约情景参数。"),
            (3, "总体模型架构", "六层系统：数据、确定性、启发式、人性化、数字孪生、风险决策。"),
            (4, "Q1路线主解", "32景点30天硬约束路线，并给游客画像分层。"),
            (5, "Q1人性化修正", "亲子/长者在压力情景下需要删点、拆段、缓冲。"),
            (6, "Q2交通费用", "乌鲁木齐起讫主方案 + 多口岸阈值1347.99元。"),
            (7, "Q2求解可信度", "当前为高质量可行解，DFJ精确证明未完成，给出效率升级方案。"),
            (8, "Q3文化考察", "MinMax均衡 + 5天缓冲；极端安全型用机动小组。"),
            (9, "Q4容量分配", "9条线路分配106358人，动态分流和分时预约缓解拥堵。"),
            (10, "数字孪生实验", "6类复合扰动情景检验路线和策略鲁棒性。"),
            (11, "CVaR策略选择", "不同风险偏好下给出可解释策略组合。"),
            (12, "结论与局限", "哪些是真实数据，哪些是仿真；后续接入12306/航班/实时预约。"),
        ],
        columns=["slide_no", "slide_title", "talk_track"],
    )


def plot_evidence_maturity(data_truth: pd.DataFrame) -> str:
    levels = {
        "题面Excel清洗": 4,
        "高德API": 5,
        "未接入实时数据；用阈值与仿真替代": 2,
        "校准模拟": 3,
        "规则校准+仿真": 3,
        "情景模拟": 3,
    }
    df = data_truth.copy()
    df["maturity"] = df["source_type"].map(levels).fillna(2)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(df["data_domain"], df["maturity"], color=["#2f6f9f", "#2a9d8f", "#e9c46a", "#f4a261", "#f4a261", "#6a994e"])
    ax.set_xlim(0, 5)
    ax.set_xlabel("证据成熟度：1=假设，5=真实API/权威数据")
    ax.set_title("数据证据成熟度审计")
    for i, v in enumerate(df["maturity"]):
        ax.text(v + 0.05, i, f"{int(v)}/5", va="center", fontsize=9)
    return save_fig(fig, "fig01_data_evidence_maturity.png")


def plot_model_stack(model_stack: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axis("off")
    colors = ["#2f6f9f", "#2a9d8f", "#6a994e", "#e9c46a", "#f4a261", "#c44536", "#6d597a"]
    y = list(range(len(model_stack)))[::-1]
    for idx, (_, row) in enumerate(model_stack.iterrows()):
        yy = y[idx]
        ax.add_patch(plt.Rectangle((0.03, yy), 0.94, 0.7, color=colors[idx], alpha=0.92))
        ax.text(0.07, yy + 0.36, row["layer"], va="center", ha="left", color="white", fontsize=15, weight="bold")
        ax.text(0.28, yy + 0.36, row["value"], va="center", ha="left", color="white", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, len(model_stack))
    ax.set_title("最终建模系统：从路线求解到风险决策", fontsize=17, weight="bold", pad=14)
    return save_fig(fig, "fig02_final_model_stack_v2.png")


def build_report(
    final_answers: pd.DataFrame,
    model_stack: pd.DataFrame,
    evidence: pd.DataFrame,
    data_truth: pd.DataFrame,
    paper_plan: pd.DataFrame,
    slides: pd.DataFrame,
    figures: list[str],
) -> str:
    missing = evidence[~evidence["exists"]]
    lines = [
        "# 新疆旅游线路安排最终科研交付总包 v2",
        "",
        "## 总结论",
        "",
        "本项目已经从“路径求解题”扩展为一个分层决策系统：先用真实/清洗数据构造交通与景点网络，再用确定性优化和元启发式得到可行路线，随后用游客画像、数字孪生仿真和CVaR风险策略把路线转成可执行、可答辩、可解释的旅游决策方案。",
        "",
        "最适合论文正文的主口径是“均衡稳健型”：第一问保留普通体力主线但对亲子路线拆段；第二问采用乌鲁木齐起讫主方案与多口岸实时比价策略；第三问采用5天项目缓冲；第四问采用分时预约+动态分流。",
        "",
        "## 四问最终答案",
        "",
        final_answers.to_markdown(index=False),
        "",
        "## 分层建模框架",
        "",
        model_stack.to_markdown(index=False),
        "",
        "## 数据真实性与仿真边界",
        "",
        data_truth.to_markdown(index=False),
        "",
        "## 证据链清单",
        "",
        evidence.to_markdown(index=False),
        "",
        "## 论文更新插入计划",
        "",
        paper_plan.to_markdown(index=False),
        "",
        "## 答辩页结构",
        "",
        slides.to_markdown(index=False),
        "",
        "## 图表输出",
        "",
    ]
    lines.extend(f"- {fig}" for fig in figures)
    lines += [
        "",
        "## 仍需谨慎表述的点",
        "",
        "1. 第二问当前主结果是高质量可行解，DFJ全局最优证明尚未完成；答辩时应称为限时MILP/局部搜索可行解与下界对比。",
        "2. 酒店房量、景区容量、预约余量、天气和道路封闭属于情景参数或校准仿真，不是实时数据。",
        "3. 12306与航班实时价格尚未直接接入，因此第二问多口岸策略应写成阈值决策和实时比价执行机制。",
        "4. 数字孪生和CVaR层的价值在于解释风险和策略，不替代原四问的确定性主模型。",
    ]
    if not missing.empty:
        lines += [
            "",
            "## 缺失证据提醒",
            "",
            missing.to_markdown(index=False),
        ]
    return "\n".join(lines)


def main() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_answers = build_final_answers()
    model_stack = build_model_stack()
    evidence = build_evidence_registry()
    data_truth = build_data_truth_table()
    paper_plan = build_paper_insert_plan()
    slides = build_slide_storyline()

    tables = {
        "final_answers": final_answers,
        "model_stack_v2": model_stack,
        "evidence_registry": evidence,
        "data_truth_table": data_truth,
        "paper_insert_plan": paper_plan,
        "slide_storyline": slides,
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")

    figures = [
        plot_evidence_maturity(data_truth),
        plot_model_stack(model_stack),
    ]

    with pd.ExcelWriter(OUTPUT_DIR / "新疆旅游最终科研交付总包v2.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    report = build_report(final_answers, model_stack, evidence, data_truth, paper_plan, slides, figures)
    report_path = OUTPUT_DIR / "新疆旅游最终科研交付总包v2.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "status": "success",
        "final_answers": len(final_answers),
        "model_layers": len(model_stack),
        "evidence_items": len(evidence),
        "missing_evidence_items": int((~evidence["exists"]).sum()),
        "figures": figures,
        "workbook": "outputs/新疆旅游最终科研交付总包v2.xlsx",
        "report": "outputs/新疆旅游最终科研交付总包v2.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
