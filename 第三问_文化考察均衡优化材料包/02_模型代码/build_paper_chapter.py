# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "paper_chapter_outputs"
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


def build_notation() -> pd.DataFrame:
    rows = [
        ("V", "景点集合", "40个原始景点；第二问普通游客集合为38个可达景点"),
        ("E", "交通边集合", "由高德OD、接驳OD、铁路/航班种子构成"),
        ("K", "年份/车辆/团队集合", "第二问K={1,2}；第三问为3个考察组"),
        ("c_ij", "景点i到j的交通费用", "两人新疆境内交通费用，单位元"),
        ("t_ij", "景点i到j的交通时间", "小时"),
        ("s_i", "景点i游览/考察时间", "小时"),
        ("v_i", "景点价值", "题面偏好、文化、自然、优先级加权"),
        ("x_i", "是否选择景点i", "第一问PCOP变量"),
        ("x_ik", "景点i是否分配给第k年/第k组", "第二/第三问分配变量"),
        ("y_ijk", "第k条路线是否从i到j", "路径变量"),
        ("u_ik", "访问顺序变量", "MTZ或顺序约束变量"),
        ("z_r", "分配到线路产品r的人数", "第四问容量流变量"),
        ("C_r", "线路r容量上限", "12天窗口内可接待人数"),
        ("F_d", "第d天疲劳指数", "由活动时长、长途转场、风险构成"),
        ("H_p", "画像p舒适度阈值", "亲子/长者/探索型不同"),
        ("B", "缓冲天数", "文化考察与不确定性仿真变量"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "meaning", "usage"])


def build_algorithm_index() -> pd.DataFrame:
    rows = [
        ("A1", "Dijkstra/OD闭包", "把原始交通边转为景点间可比较边权", "数据预处理"),
        ("A2", "MILP/HiGHS", "求解PCOP基准、第二问路径覆盖、容量流", "确定性主模型"),
        ("A3", "ACO初始化", "生成多条高质量初始旅游序列", "第一问大规模路线"),
        ("A4", "ALNS破坏-修复", "移除低收益或高负担片段并重插入", "复杂约束搜索"),
        ("A5", "SA接受准则", "允许暂时接受较差解以跳出局部最优", "元启发式"),
        ("A6", "2-opt/swap/relocate", "改善局部路段顺序和费用", "后处理修复"),
        ("A7", "游客画像排程器", "长途转场拆日，计算舒适度和红色压力日", "人性化层"),
        ("A8", "Monte Carlo", "扰动价格、延期、需求、预约/容量", "鲁棒性验证"),
        ("A9", "阈值策略", "多口岸差价阈值、缓冲天、预约上限", "政策解释"),
    ]
    return pd.DataFrame(rows, columns=["algorithm_id", "name", "role", "section"])


def build_experiment_matrix() -> pd.DataFrame:
    rows = [
        ("E1", "第一问主解", "30天硬约束混合元启发式", "景点数、天数、总成本、时间窗", "32景点/30天/14548.53元"),
        ("E2", "第一问画像适配", "普通/亲子/长者/探索型", "舒适度、红色压力日、删点数", "亲子型删2点后30景点30天满足"),
        ("E3", "第二问主解", "乌鲁木齐起讫两年路径覆盖", "覆盖数、交通费、最大单年天数", "38景点/32天/4340.68元"),
        ("E4", "第二问下界", "开放式多口岸路径覆盖", "交通费和天数下界", "38景点/28天/2992.69元"),
        ("E5", "第二问价格仿真", "多口岸外部差价三角分布", "开放式更便宜概率", "乐观98.79%，均衡70.01%，高峰30.38%"),
        ("E6", "第三问主解", "三组MinMax文化考察", "最大完成时间、任务均衡", "最大98.70小时，差12.16小时"),
        ("E7", "第三问缓冲仿真", "文化考察延期Monte Carlo", "项目按时概率", "4天缓冲达91.84%"),
        ("E8", "第四问容量流", "9条线路产品分配", "接待人数、容量利用率", "106358人/111956容量"),
        ("E9", "第四问需求冲击", "需求上浮下替代线路分流", "溢出人数、分流新增接待", "上浮10%仍溢出5038人"),
        ("E10", "第四问预约策略", "100%/95%/90%预约上限", "接待、等待、服务水平", "95%上限接待106358人"),
    ]
    return pd.DataFrame(rows, columns=["experiment_id", "topic", "design", "metrics", "headline_result"])


def write_chapter(path: Path) -> None:
    master = read_csv(ROOT / "research_synthesis_outputs" / "master_metrics.csv")
    notation = build_notation()
    algorithm = build_algorithm_index()
    experiments = build_experiment_matrix()

    lines = [
        "# 新疆旅游线路安排数学建模章节草案",
        "",
        "## 1. 建模思想",
        "",
        "本研究将新疆旅游线路安排抽象为带节点权重和边权重的有向加权图优化问题。节点表示景点、住宿锚点和交通枢纽，边表示景点间转场、起终点接驳以及可选铁路/航班连接。边权包含时间、费用和风险，节点权包含游览价值、文化价值、开放时间、预约要求和容量约束。",
        "",
        "与只求最短路不同，本文采用分层建模：确定性优化回答题目四问，人性化层刻画游客体验，仿真层刻画真实世界扰动，策略层给出条件方案和运营建议。",
        "",
        "## 2. 符号说明",
        "",
        notation.to_markdown(index=False),
        "",
        "## 3. 第一问模型：30天低成本高价值旅游路线",
        "",
        "第一问可视为带收益的路径规划问题，即 Prize-Collecting Orienteering Problem。令 `x_i` 表示是否访问景点 `i`，`y_ij` 表示路线是否从 `i` 到 `j`。基础目标为最大化景点价值并惩罚交通、门票、住宿和风险成本：",
        "",
        "```text",
        "max  sum_i v_i x_i",
        "     - alpha * sum_{i,j} c_ij y_ij",
        "     - beta  * ticket_cost",
        "     - gamma * hotel_cost",
        "     - eta   * risk_penalty",
        "```",
        "",
        "核心约束为：每个景点最多访问一次，路线连通，总天数不超过30天，开放时间不违约，住宿锚点可达，普通游客不可达或需特殊审批的点不纳入普通路线。",
        "",
        "为贴近真实游客体验，进一步引入画像舒适度约束。对第 `d` 天定义疲劳指数：",
        "",
        "```text",
        "F_d = a_p * max(0, A_d - H_p)^1.35",
        "    + b_p * max(0, T_d - L_p)^1.35",
        "    + r_p * R_d",
        "    + q_p * I(consecutive_long_transfer)",
        "```",
        "",
        "其中 `A_d` 为单日活动时间，`T_d` 为单日转场时间，`R_d` 为风险代理变量，`H_p` 和 `L_p` 由游客画像 `p` 决定。模型输出显示：普通体力型和探索型可保留32景点；亲子舒适型建议删去2个非核心低优先级点；长者慢游型应作为慢游拆期方案。",
        "",
        "## 4. 第二问模型：两年暑假路径覆盖",
        "",
        "第二问的题面目标是节省新疆境内交通费用，因此建模为两条暑假路线的路径覆盖问题。令 `K={1,2}` 表示今、明两年，`x_ik` 表示景点 `i` 是否分配给第 `k` 年，`y_ijk` 表示第 `k` 年是否从 `i` 到 `j`。",
        "",
        "```text",
        "min  sum_{k in K} sum_{i,j in V union {0}} c_ij y_ijk",
        "```",
        "",
        "约束包括：",
        "",
        "```text",
        "sum_k x_ik = 1                         每个普通游客可达景点恰好覆盖一次",
        "sum_j y_ijk = x_ik                    被分配景点出度为1",
        "sum_j y_jik = x_ik                    被分配景点入度为1",
        "D_k <= D_max                          每年暑假天数上限",
        "N_min <= sum_i x_ik <= N_max          两年负担不过度失衡",
        "subtour_elimination(y)                消除子回路",
        "```",
        "",
        "主模型固定乌鲁木齐起讫，得到38景点、32天、两人新疆境内交通费用4340.68元。开放式多口岸模型得到费用下界2992.69元。两者差额为1347.99元，因此当两人多口岸外部大交通额外差价低于该阈值时，开放式方案总费用更优。",
        "",
        "## 5. 第三问模型：三组文化专项考察",
        "",
        "第三问是多团队任务均衡问题，而不是普通旅游游览问题。令 `x_ig` 表示文化景点 `i` 是否分配给第 `g` 组，`T_g` 表示第 `g` 组完成时间。目标为最小化最大完成时间：",
        "",
        "```text",
        "min T_max",
        "s.t. T_g <= T_max, forall g",
        "     sum_g x_ig = 1, forall i",
        "```",
        "",
        "模型不按景点数平均，而按交通时间、考察时间、文化价值和远程风险均衡。主结果最大完成时间98.70小时，三组完成时间差12.16小时。进一步用Monte Carlo模拟审批、道路、天气和现场工作延期，项目级90%可靠性需要预留4天缓冲。",
        "",
        "## 6. 第四问模型：五一12天容量接待",
        "",
        "第四问不是单条路线优化，而是路线产品组合与容量流分配。先生成候选线路集合 `R`，再决定每条线路分配游客数 `z_r`：",
        "",
        "```text",
        "max  sum_{r in R} a_r z_r",
        "s.t. 0 <= z_r <= C_r",
        "     sum_r z_r <= Demand",
        "     sum_{r: i in r} z_r <= Capacity_i, forall scenic spot i",
        "```",
        "",
        "基准方案生成9条线路，12天总容量111956人，分配106358人。需求上浮10%时，替代线路分流新增接待4628人，但仍溢出5038人，说明节假日运营必须结合预约上限、分流宣传和新增运力。",
        "",
        "## 7. 算法流程",
        "",
        algorithm.to_markdown(index=False),
        "",
        "综合算法流程如下：",
        "",
        "```text",
        "Step 1  清洗景点、交通、住宿、费用、容量数据",
        "Step 2  调用高德OD构造景点间真实道路时间/费用矩阵",
        "Step 3  按四问分别构建PCOP、路径覆盖、MinMax Multi-TSP、容量流模型",
        "Step 4  使用MILP、ACO-ALNS-SA和局部搜索得到确定性主解",
        "Step 5  使用逐日排程器检验开放时间、住宿、活动时长和转场可行性",
        "Step 6  引入游客画像、疲劳指数和长途转场拆日规则",
        "Step 7  使用Monte Carlo模拟价格、延期、需求、预约和容量扰动",
        "Step 8  输出主方案、条件方案和应急策略",
        "```",
        "",
        "## 8. 实验设计矩阵",
        "",
        experiments.to_markdown(index=False),
        "",
        "## 9. 当前结果总览",
        "",
        master.to_markdown(index=False),
        "",
        "## 10. 答辩口径",
        "",
        "本文不是简单地把全部景点串成一条路线，而是针对四个问题分别建立与场景匹配的模型。第一问强调30天个人旅游体验，第二问强调两年覆盖与新疆境内交通费，第三问强调文化考察任务均衡，第四问强调节假日群体接待和容量管理。进一步地，本文把游客画像、长途转场、价格不确定、延期风险和预约限流纳入模型，使结果从“能算”提升到“可解释、可执行、可应对扰动”。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables = {
        "notation": build_notation(),
        "algorithm_index": build_algorithm_index(),
        "experiment_matrix": build_experiment_matrix(),
        "master_metrics": read_csv(ROOT / "research_synthesis_outputs" / "master_metrics.csv"),
    }
    for name, df in tables.items():
        write_csv(df, OUT_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游论文建模章节表格索引.xlsx")
    write_chapter(OUTPUT_DIR / "新疆旅游论文建模章节草案.md")
    summary = {
        "tables": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in tables.items()},
        "workbook": "outputs/新疆旅游论文建模章节表格索引.xlsx",
        "chapter": "outputs/新疆旅游论文建模章节草案.md",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
