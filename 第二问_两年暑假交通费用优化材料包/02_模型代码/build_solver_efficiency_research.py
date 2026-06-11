# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "solver_efficiency_outputs"
OUTPUT_DIR = ROOT / "outputs"


def clean_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(",", "")
    if text.lower() in {"", "inf", "infinite"}:
        return math.inf
    try:
        return float(text)
    except Exception:
        return math.nan


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def parse_highs_log(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "log_file": str(path),
            "available": False,
            "last_best_bound": math.nan,
            "last_best_solution": math.nan,
            "last_gap_percent": math.nan,
            "last_time_seconds": math.nan,
            "last_nodes_processed": math.nan,
            "valid_progress_lines": 0,
        }

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in {"L", "B", "T"}:
            parts = parts[1:]
        if len(parts) < 8:
            continue
        if not parts[0].isdigit() or not parts[3].endswith("%") or not parts[-1].endswith("s"):
            continue
        gap_text = parts[6].replace("%", "")
        rows.append(
            {
                "nodes_processed": int(parts[0]),
                "tree_explored_percent": clean_float(parts[3].replace("%", "")),
                "best_bound": clean_float(parts[4]),
                "best_solution": clean_float(parts[5]),
                "gap_percent": clean_float(gap_text),
                "time_seconds": clean_float(parts[-1].rstrip("s")),
            }
        )

    if not rows:
        return {
            "log_file": str(path),
            "available": True,
            "last_best_bound": math.nan,
            "last_best_solution": math.nan,
            "last_gap_percent": math.nan,
            "last_time_seconds": math.nan,
            "last_nodes_processed": math.nan,
            "valid_progress_lines": 0,
        }
    last = rows[-1]
    return {
        "log_file": str(path),
        "available": True,
        "last_best_bound": round(float(last["best_bound"]), 6) if math.isfinite(last["best_bound"]) else last["best_bound"],
        "last_best_solution": round(float(last["best_solution"]), 6) if math.isfinite(last["best_solution"]) else last["best_solution"],
        "last_gap_percent": round(float(last["gap_percent"]), 4) if math.isfinite(last["gap_percent"]) else last["gap_percent"],
        "last_time_seconds": round(float(last["time_seconds"]), 3) if math.isfinite(last["time_seconds"]) else last["time_seconds"],
        "last_nodes_processed": int(last["nodes_processed"]),
        "valid_progress_lines": len(rows),
    }


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "无可用数据。"
    shown = df.head(max_rows) if max_rows else df
    return shown.to_markdown(index=False)


def html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<p>无可用数据。</p>"
    shown = df.head(max_rows) if max_rows else df
    return shown.to_html(index=False, escape=True, border=0, classes="data-table")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def solver_availability() -> pd.DataFrame:
    rows = [
        ("SciPy/HiGHS", "scipy", "当前已用", "MILP 基线、下界、可行解", "可用，但缺少 lazy cuts callback"),
        ("NetworkX", "networkx", "当前辅助", "图结构检查、连通性/子回路识别", "可用"),
        ("OR-Tools", "ortools", "建议新增", "VRP/TSP 高效启发式、warm start", "当前缺包；脚本已预留"),
        ("Gurobi", "gurobipy", "建议新增", "精确 MILP、lazy constraints、MIP start、gap 证明", "当前缺包；若有学术许可优先"),
        ("CPLEX", "cplex", "备选新增", "精确 MILP、callback、工业级证明", "当前缺包"),
        ("Pyomo", "pyomo", "表达层可选", "更贴近数学公式的模型书写", "当前缺包；本身不提速"),
        ("PuLP", "pulp", "表达层可选", "轻量 MILP 表达层", "当前缺包；本身不提速"),
    ]
    out = []
    for solver, module, role, best_for, note in rows:
        out.append(
            {
                "solver_or_package": solver,
                "python_module": module,
                "available_now": module_available(module),
                "role_in_project": role,
                "best_for": best_for,
                "efficiency_note": note,
            }
        )
    return pd.DataFrame(out)


def build_current_status() -> pd.DataFrame:
    scenario_totals = read_csv(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    if scenario_totals.empty:
        return pd.DataFrame()
    cols = [
        "scenario_id",
        "covered_spots",
        "total_estimated_days",
        "max_year_days",
        "total_transport_cost_yuan_for_two",
        "total_travel_hours",
        "total_distance_km",
        "proved_optimal",
        "solver_status",
    ]
    return scenario_totals[[c for c in cols if c in scenario_totals.columns]].copy()


def build_log_status() -> pd.DataFrame:
    log_paths = [
        ("DFJ iterative current", ROOT / "logs" / "p2_exact_dfj.out.log"),
        ("DFJ static precuts", ROOT / "logs" / "p2_exact_dfj_static.out.log"),
        ("Single-commodity flow", ROOT / "logs" / "p2_exact_flow.out.log"),
        ("Rooted long MILP", ROOT / "logs" / "p2_rooted_long_milp.out.log"),
    ]
    rows = []
    for label, path in log_paths:
        item = parse_highs_log(path)
        item["attempt"] = label
        rows.append(item)
    cols = [
        "attempt",
        "available",
        "last_best_bound",
        "last_best_solution",
        "last_gap_percent",
        "last_time_seconds",
        "last_nodes_processed",
        "valid_progress_lines",
        "log_file",
    ]
    return pd.DataFrame(rows)[cols]


def build_solver_roadmap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "A",
                "method": "OR-Tools Routing",
                "time_budget": "10-120 秒",
                "target_output": "两年路线、费用、天数、可行性",
                "why_it_is_fast": "专门面向 VRP/TSP 的局部搜索与 Guided Local Search",
                "claim_allowed": "高质量可行解，不声明全局最优",
            },
            {
                "phase": "B",
                "method": "Gurobi/CPLEX branch-and-cut",
                "time_budget": "10-60 分钟",
                "target_output": "最优解或 gap <= 1%-3% 的证明",
                "why_it_is_fast": "lazy subtour cuts 在分支树中动态加入，支持 MIP start",
                "claim_allowed": "可证明最优或可证明近优",
            },
            {
                "phase": "C",
                "method": "日程-酒店-容量后验仿真",
                "time_budget": "1-5 分钟/场景",
                "target_output": "舒适度、风险、预约失败率、敏感性",
                "why_it_is_fast": "不把所有现实扰动塞进主 MILP，而是做场景压力测试",
                "claim_allowed": "路线鲁棒性与场景适配性",
            },
            {
                "phase": "D",
                "method": "联合优化小规模精确校验",
                "time_budget": "30-120 分钟",
                "target_output": "10-20 点子问题的精确对照",
                "why_it_is_fast": "用子集验证模型结构，不在全 38 点上硬求所有现实约束",
                "claim_allowed": "模型结构有效性/启发式误差校准",
            },
        ]
    )


def build_method_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "当前 SciPy/HiGHS MTZ",
                "speed": "中",
                "proof_strength": "弱-中",
                "engineering_cost": "低",
                "fit_to_q2": "适合保留为免费基线",
                "main_risk": "MTZ 松弛较弱，限时下通常只能给可行解",
            },
            {
                "method": "当前 SciPy/HiGHS iterative DFJ",
                "speed": "慢",
                "proof_strength": "中",
                "engineering_cost": "中",
                "fit_to_q2": "适合验证思路，不适合作为效率主线",
                "main_risk": "无法在分支过程中 lazy 加割，重解成本高",
            },
            {
                "method": "OR-Tools Routing",
                "speed": "很快",
                "proof_strength": "弱",
                "engineering_cost": "中",
                "fit_to_q2": "适合快速路线和 warm start",
                "main_risk": "不能作为全局最优证明",
            },
            {
                "method": "Gurobi/CPLEX DFJ lazy cuts",
                "speed": "快-很快",
                "proof_strength": "强",
                "engineering_cost": "中-高",
                "fit_to_q2": "最适合作为最终效率强化主模型",
                "main_risk": "需要安装与许可证",
            },
            {
                "method": "Pyomo/PuLP + CBC/HiGHS",
                "speed": "不一定提升",
                "proof_strength": "取决于后端",
                "engineering_cost": "低-中",
                "fit_to_q2": "适合写论文公式，不是效率核心",
                "main_risk": "换壳不换求解器，性能瓶颈仍在",
            },
        ]
    )


def build_report(
    availability: pd.DataFrame,
    current_status: pd.DataFrame,
    log_status: pd.DataFrame,
    roadmap: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    rooted = current_status[current_status["scenario_id"].eq("P2_ROOTED_URUMQI_MINCOST")] if not current_status.empty else pd.DataFrame()
    open_gateway = current_status[current_status["scenario_id"].eq("P2_OPEN_GATEWAY_LOWER_BOUND")] if not current_status.empty else pd.DataFrame()
    rooted_cost = float(rooted.iloc[0]["total_transport_cost_yuan_for_two"]) if not rooted.empty else math.nan
    open_cost = float(open_gateway.iloc[0]["total_transport_cost_yuan_for_two"]) if not open_gateway.empty else math.nan
    threshold = rooted_cost - open_cost if math.isfinite(rooted_cost) and math.isfinite(open_cost) else math.nan
    best_dfj = log_status[log_status["attempt"].eq("DFJ iterative current")]
    gap = best_dfj.iloc[0]["last_gap_percent"] if not best_dfj.empty else math.nan

    lines = [
        "# 第二问求解效率强化研究报告",
        "",
        "## 技术结论",
        "",
        f"- 当前第二问已经有可汇报的高质量可行解：乌鲁木齐起讫方案覆盖 38 个普通景点，合计 {rooted.iloc[0]['total_estimated_days'] if not rooted.empty else 'NA'} 天，境内交通费 {rooted_cost:.2f} 元/两人；开放式多口岸下界为 {open_cost:.2f} 元/两人。",
        f"- 现有 HiGHS/DFJ 精确证明链尚未完成；当前日志中 DFJ 迭代尝试的最近 gap 约为 {gap}%（以当前日志为准），因此不能把第二问称为全局最优。",
        f"- 效率优化的关键不是把模型换成 Pyomo/PuLP，而是把“路径启发式”和“精确证明”分层：OR-Tools 快速给路线与 warm start，Gurobi/CPLEX 用 lazy cuts 做 gap/最优性证明。",
        f"- 开放式方案与乌鲁木齐起讫方案的费用差为 {threshold:.2f} 元/两人，这个阈值应保留为第二问的重要政策敏感性结论。",
        "",
        "## 当前环境与求解器可用性",
        "",
        markdown_table(availability),
        "",
        "解释：当前环境具备 SciPy/HiGHS 与 NetworkX，足以继续做免费基线和图结构校验；但 OR-Tools、Gurobi、CPLEX 均未安装，所以效率强化需要新增依赖或许可证。",
        "",
        "## 当前第二问结果状态",
        "",
        markdown_table(current_status),
        "",
        "现有路线结果是可行且经过局部搜索改进的，但 `proved_optimal=False`。论文中应写成“限时 MILP + 局部搜索得到的高质量可行解”，不要写成“全局最优解”。",
        "",
        "## 精确求解日志暴露的瓶颈",
        "",
        markdown_table(log_status),
        "",
        "瓶颈判断：",
        "",
        "1. `SciPy/HiGHS iterative DFJ` 需要在求解结束后找子回路、加割、再重解，无法像 Gurobi/CPLEX 那样在分支定界过程中动态加 lazy cuts。",
        "2. 单商品流模型变量更多，连续流变量让 LP 规模变大；它能表达连通性，但在本实例上找可行解和缩 gap 都不够理想。",
        "3. 第二问节点数只有 38，但“两年分组 + 起讫 + 时间上限 + 全覆盖 + 子回路消除”会让分支树迅速变大；效率瓶颈主要来自证明最优性，不是构造可行路线。",
        "",
        "## 推荐的高效率求解栈",
        "",
        markdown_table(roadmap),
        "",
        "这个求解栈更贴合项目主旋律：主模型负责给严谨答案，启发式负责工程效率，仿真负责现实场景解释。三者不是互相替代，而是各自回答不同层次的问题。",
        "",
        "## 方法对比",
        "",
        markdown_table(comparison),
        "",
        "## 对代码和实验的直接落地",
        "",
        "- 已新增 `scripts/solve_problem2_ortools_routing.py`：装好 OR-Tools 后可直接跑第二问两车 VRP，输出路线、费用、天数和 Excel/Markdown 结果。",
        "- 建议后续新增 `scripts/solve_problem2_gurobi_lazy_dfj.py`：复用现有成本矩阵和路线评价函数，加入 Gurobi lazy subtour callback，并把 OR-Tools 或当前局部搜索路线作为 MIP start。",
        "- 每次实验记录 30 秒、2 分钟、10 分钟、60 分钟四个预算下的 incumbent、best bound、gap、路线天数和费用，用“anytime profile”证明效率提升。",
        "",
        "## 论文/答辩建议口径",
        "",
        "第二问可采用两层表述：",
        "",
        "1. 基准答案：在既定数据口径下，乌鲁木齐起讫两年完成 38 个普通景点，境内交通费用约 4340.68 元/两人。",
        "2. 条件答案：若两年外部大交通采用多口岸进出疆，且两人额外机票/火车票差价低于 1347.99 元，则开放式多口岸方案在总费用上更优。",
        "",
        "算法表述建议写成：`MILP 负责约束和可证明下界，OR-Tools/局部搜索负责快速构造高质量路线，场景仿真负责票价与出入口选择的不确定性评估`。",
    ]
    return "\n".join(lines)


def build_html_report(markdown_text: str, tables: dict[str, pd.DataFrame]) -> str:
    title = "第二问求解效率强化研究报告"
    sections = []
    current_heading = None
    current_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)
    if current_heading is not None:
        sections.append((current_heading, current_lines))

    def render_lines(lines: list[str]) -> str:
        out = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                continue
            if stripped.startswith("|"):
                continue
            if stripped.startswith("- "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{html.escape(stripped[2:])}</li>")
            elif re.match(r"^\d+\. ", stripped):
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<p>{html.escape(stripped)}</p>")
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append(f"<p>{html.escape(stripped)}</p>")
        if in_list:
            out.append("</ul>")
        return "\n".join(out)

    table_map = {
        "当前环境与求解器可用性": "availability",
        "当前第二问结果状态": "current_status",
        "精确求解日志暴露的瓶颈": "log_status",
        "推荐的高效率求解栈": "roadmap",
        "方法对比": "comparison",
    }
    body = []
    for heading, lines in sections:
        body.append(f"<section><h2>{html.escape(heading)}</h2>")
        body.append(render_lines(lines))
        key = table_map.get(heading)
        if key:
            body.append(html_table(tables[key]))
        body.append("</section>")

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; color: #1f2937; background: #f8fafc; }
    main { max-width: 1080px; margin: 0 auto; padding: 40px 24px 64px; }
    header { margin-bottom: 28px; }
    h1 { font-size: 32px; margin: 0 0 10px; color: #111827; }
    h2 { font-size: 22px; margin: 32px 0 12px; color: #111827; }
    p, li { line-height: 1.72; font-size: 15px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px 22px; margin: 18px 0; }
    .data-table { border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; }
    .data-table th, .data-table td { border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }
    .data-table th { background: #f3f4f6; font-weight: 650; color: #111827; }
    .meta { color: #6b7280; font-size: 14px; }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="meta">基于当前项目脚本、第二问输出、HiGHS 求解日志与本机求解器环境生成。</p>
    </header>
    {''.join(body)}
  </main>
</body>
</html>
"""


def main() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    availability = solver_availability()
    current_status = build_current_status()
    log_status = build_log_status()
    roadmap = build_solver_roadmap()
    comparison = build_method_comparison()

    write_csv(availability, OUT_DIR / "solver_availability.csv")
    write_csv(current_status, OUT_DIR / "current_problem2_status.csv")
    write_csv(log_status, OUT_DIR / "highs_log_status.csv")
    write_csv(roadmap, OUT_DIR / "recommended_efficiency_pipeline.csv")
    write_csv(comparison, OUT_DIR / "solver_method_comparison.csv")

    with pd.ExcelWriter(OUTPUT_DIR / "新疆旅游第二问求解效率强化研究.xlsx", engine="openpyxl") as writer:
        availability.to_excel(writer, sheet_name="solver_availability", index=False)
        current_status.to_excel(writer, sheet_name="current_p2_status", index=False)
        log_status.to_excel(writer, sheet_name="highs_log_status", index=False)
        roadmap.to_excel(writer, sheet_name="recommended_pipeline", index=False)
        comparison.to_excel(writer, sheet_name="method_comparison", index=False)

    report_md = build_report(availability, current_status, log_status, roadmap, comparison)
    report_path = OUTPUT_DIR / "新疆旅游第二问求解效率强化研究报告.md"
    report_path.write_text(report_md, encoding="utf-8")

    html_report = build_html_report(
        report_md,
        {
            "availability": availability,
            "current_status": current_status,
            "log_status": log_status,
            "roadmap": roadmap,
            "comparison": comparison,
        },
    )
    html_path = OUTPUT_DIR / "新疆旅游第二问求解效率强化研究报告.html"
    html_path.write_text(html_report, encoding="utf-8")

    summary = {
        "status": "success",
        "availability_rows": len(availability),
        "current_status_rows": len(current_status),
        "log_status_rows": len(log_status),
        "report": str(report_path),
        "html_report": str(html_path),
        "workbook": str(OUTPUT_DIR / "新疆旅游第二问求解效率强化研究.xlsx"),
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
