# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "defense_package_outputs"
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


def build_slide_outline() -> pd.DataFrame:
    rows = [
        (1, "题目与核心难点", "新疆尺度大、景点分散、四问目标不同", "不要把四问都做成一个TSP；强调场景差异"),
        (2, "数据底座", "40景点、真实高德OD、住宿/容量/费用参数", "说明哪些是真实数据，哪些是校准/仿真"),
        (3, "统一建模框架", "有向加权图 + 节点权 + 边权 + 场景约束", "放六层模型栈图"),
        (4, "第一问主结果", "32景点、30天、总成本14548.53元", "展示30天主路线与成本构成"),
        (5, "第一问人性化强化", "普通体力型满足；亲子型删2点后满足；长者型建议拆期", "突出疲劳凸惩罚和长途转场拆日"),
        (6, "第二问重建逻辑", "两年路径覆盖，目标是新疆境内交通费最小", "解释为什么不是谱聚类/平均分点"),
        (7, "第二问结果与多口岸阈值", "乌鲁木齐起讫4340.68元；开放式下界2992.69元；阈值1347.99元", "说明开放式是条件方案"),
        (8, "第三问文化考察", "三组MinMax，最大98.70小时，时间差12.16小时", "解释楼兰/尼雅专项组为什么合理"),
        (9, "第三问风险缓冲", "4天缓冲使项目完成概率达91.84%", "把它包装为文化专项项目管理"),
        (10, "第四问五一容量", "9条路线，分配106358人，总容量111956人", "强调路线产品组合而非单条最优路线"),
        (11, "第四问需求冲击与预约策略", "需求上浮10%仍溢出5038人；95%预约上限用于服务水平控制", "说明运营策略价值"),
        (12, "创新点与局限", "真实OD、分层建模、画像、仿真、策略；实时票价/房态仍需补充", "主动承认边界，减少答辩风险"),
    ]
    return pd.DataFrame(rows, columns=["slide_no", "title", "main_message", "speaker_note"])


def build_chart_plan() -> pd.DataFrame:
    rows = [
        ("图1", "模型分层架构图", "outputs/新疆旅游四问Fancy建模科研总报告.md", "论文第2章/答辩第3页"),
        ("表1", "符号说明", "outputs/新疆旅游论文建模章节表格索引.xlsx", "论文模型章节"),
        ("表2", "四问主结论总览", "research_synthesis_outputs/master_metrics.csv", "答辩摘要页"),
        ("图2", "第一问30天路线示意", "hybrid_30day_outputs/hard30_route_days.csv", "第一问结果页"),
        ("表3", "游客画像路线对比", "adaptive_strategy_outputs/q1_variant_summary.csv", "第一问强化页"),
        ("图3", "第二问两年路线对比", "problem2_openpath_outputs/route_summary.csv", "第二问结果页"),
        ("图4", "多口岸价格胜率柱状图", "policy_simulation_outputs/q2_gateway_price_summary.csv", "第二问仿真页"),
        ("表4", "第三问三组考察结果", "enhanced_model_outputs/problem3_minmax_summary.csv", "第三问结果页"),
        ("图5", "文化考察缓冲可靠性曲线", "policy_simulation_outputs/q3_project_buffer_curve.csv", "第三问仿真页"),
        ("表5", "五一线路容量分配", "enhanced_model_outputs/problem4_capacity_flow.csv", "第四问结果页"),
        ("图6", "需求冲击与溢出人数", "adaptive_strategy_outputs/q4_reallocation_summary.csv", "第四问策略页"),
        ("表6", "预约上限策略对比", "policy_simulation_outputs/q4_reservation_policy_summary.csv", "第四问运营策略页"),
    ]
    return pd.DataFrame(rows, columns=["figure_id", "figure_name", "source", "placement"])


def build_qa_bank() -> pd.DataFrame:
    rows = [
        (
            "为什么不直接做TSP？",
            "TSP只适合固定全部节点、单一目标的最短路径；本题四问目标不同，第一问有30天和收益选择，第二问是两年覆盖，第三问是多团队均衡，第四问是容量接待，所以需要分场景建模。",
            "高",
        ),
        (
            "第二问为什么排除楼兰和尼雅？",
            "它们需审批、成本高、普通游客可达性特殊，更符合第三问文化专项考察。若题面要求全部点覆盖，可以作为审批扩展方案单列，但普通游客主线路不应硬纳入。",
            "高",
        ),
        (
            "第二问是否证明全局最优？",
            "当前主结果是可行且经局部搜索改进的高质量解，精确DFJ求证仍未最终完成，因此不声称全局最优。论文可称为限时MILP可行解和开放式下界对比。",
            "高",
        ),
        (
            "酒店、容量、预约余量是否真实？",
            "高德OD是真实接口数据；景点和票价来自原始Excel清洗；酒店房态、景区预约余量和部分容量是模型校准/仿真参数，报告中应明确为场景分析参数。",
            "高",
        ),
        (
            "为什么加入游客画像？",
            "同一路线对不同游客体验差异很大。画像把亲子、长者、探索型游客的活动阈值转化为约束，使模型从低成本可行路线升级为可执行旅行方案。",
            "中",
        ),
        (
            "Monte Carlo参数是否主观？",
            "当前是课程建模中的场景校准参数，用于做敏感性和鲁棒性分析；若获得实时天气、道路封闭、预约失败率，可直接替换扰动分布重新求解。",
            "中",
        ),
        (
            "第四问为什么不是一条路线？",
            "五一接待面对的是群体需求，单条路线无法服务全部游客；更合理的是生成多条线路产品，并在容量约束下分配游客。",
            "高",
        ),
        (
            "模型最fancy的地方是什么？",
            "不是堆算法，而是把确定性优化、人性化体验和不确定性仿真连接起来，形成主方案、条件方案和应急策略组成的决策系统。",
            "中",
        ),
    ]
    return pd.DataFrame(rows, columns=["question", "answer", "priority"])


def build_validity_audit() -> pd.DataFrame:
    rows = [
        ("高德驾车OD", "真实API数据", "强", "可作为交通时间/距离/费用主边权", "仍非实时拥堵全时段数据"),
        ("景点属性/票价/游览时长", "原始Excel清洗", "中强", "可作为课程建模基础数据", "需注明存在人工清洗和文本解析"),
        ("12306/航班", "公开时刻种子/说明数据", "中", "用于多方式交通拓展", "不等同实时余票/实时票价"),
        ("酒店房态/价格", "校准/仿真", "中弱", "可做住宿可行性和敏感性分析", "不能称为真实实时房态"),
        ("景区容量/预约余量", "校准/仿真", "中弱", "可做五一容量流实验", "需补文旅局或景区预约平台数据"),
        ("第一问主路线", "混合元启发式+逐日复核", "强", "满足30天硬约束和体验强化", "不是数学全局最优证明"),
        ("第二问主路线", "限时MILP+局部搜索", "中强", "贴合交通费最小目标", "DFJ全局最优仍未最终证明"),
        ("第三问文化考察", "MinMax+风险仿真", "强", "任务均衡和缓冲天解释清楚", "延期分布为场景校准"),
        ("第四问容量分配", "容量流+需求冲击仿真", "中强", "适合节假日运营方案", "容量参数需真实预约余量校验"),
    ]
    return pd.DataFrame(rows, columns=["item", "data_or_model_status", "credibility", "can_claim", "must_disclose"])


def write_markdown(
    slide_outline: pd.DataFrame,
    chart_plan: pd.DataFrame,
    qa: pd.DataFrame,
    audit: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# 新疆旅游建模答辩材料包",
        "",
        "## 1. 12页PPT建议结构",
        "",
        slide_outline.to_markdown(index=False),
        "",
        "## 2. 3分钟汇报稿",
        "",
        "我们把新疆旅游线路问题建模为一个有向加权图优化问题。不同于简单最短路，本题四个问题的决策对象不同：第一问是30天个人旅游路线，第二问是两年暑假覆盖，第三问是三组文化专项考察，第四问是五一群体接待容量。因此我们采用分层模型：先建立真实数据底座和高德OD交通图，再分别构建PCOP、两年路径覆盖、MinMax Multi-TSP和容量流模型。为了让模型更贴近真实旅游，我们进一步加入游客画像、长途转场拆日、价格不确定、文化考察延期和预约上限策略。最终第一问得到32景点30天路线；第二问得到乌鲁木齐起讫和开放式多口岸两类方案，并给出1347.99元的口岸差价阈值；第三问给出三组均衡考察和4天缓冲建议；第四问给出9条线路产品、容量分配和需求上浮下的分流策略。我们的核心贡献不是只求一条路线，而是形成主方案、条件方案和应急策略组成的旅游决策系统。",
        "",
        "## 3. 8分钟汇报节奏",
        "",
        "1. 第1分钟：说明题目四问不是同一个TSP问题，提出分层旅游决策系统。",
        "2. 第2分钟：说明数据底座，高德OD是真实道路数据，容量和预约为场景校准。",
        "3. 第3分钟：讲第一问30天主路线和游客画像强化。",
        "4. 第4分钟：讲第二问两年路径覆盖和多口岸阈值。",
        "5. 第5分钟：讲第三问文化专项MinMax和4天缓冲。",
        "6. 第6分钟：讲第四问线路产品容量流和需求冲击。",
        "7. 第7分钟：讲Monte Carlo、画像和预约策略的现实意义。",
        "8. 第8分钟：主动说明局限和后续数据接入。",
        "",
        "## 4. 图表放置建议",
        "",
        chart_plan.to_markdown(index=False),
        "",
        "## 5. 高频追问与回答",
        "",
        qa.to_markdown(index=False),
        "",
        "## 6. 模型可信度审计",
        "",
        audit.to_markdown(index=False),
        "",
        "## 7. 最稳妥答辩口径",
        "",
        "本案例的结果分为三类：一是确定性优化主解，用来回答题目；二是人性化和仿真强化，用来说明真实世界中的体验和风险；三是条件策略，用来处理不同票价、不同游客和节假日需求波动。我们不会声称所有模拟参数都是真实实时数据，也不会把限时MILP说成已证明全局最优，而是明确数据层级和模型边界。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables = {
        "slide_outline": build_slide_outline(),
        "chart_plan": build_chart_plan(),
        "qa_bank": build_qa_bank(),
        "validity_audit": build_validity_audit(),
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游答辩材料包.xlsx")
    write_markdown(
        tables["slide_outline"],
        tables["chart_plan"],
        tables["qa_bank"],
        tables["validity_audit"],
        OUTPUT_DIR / "新疆旅游答辩材料包.md",
    )
    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游答辩材料包.xlsx",
        "report": "outputs/新疆旅游答辩材料包.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
