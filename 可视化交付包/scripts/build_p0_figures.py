from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "01_visual_data"
PAPER = PACKAGE / "06_paper_ready"
PPT = PACKAGE / "07_ppt_ready"
P0 = PACKAGE / "05_p0_figures"

SIMSUN = Path("C:/Windows/Fonts/simsun.ttc")
TIMES = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")

FONT_CN = FontProperties(fname=str(SIMSUN), weight="bold")
FONT_CN_REG = FontProperties(fname=str(SIMSUN))
FONT_EN = FontProperties(fname=str(TIMES))
FONT_EN_BOLD = FontProperties(fname=str(TIMES_BOLD), weight="bold")

COLORS = {
    "model_blue": "#1F4E79",
    "xinjiang_gold": "#D9A441",
    "low_risk_green": "#2E7D32",
    "medium_risk_orange": "#F59E0B",
    "high_risk_red": "#C62828",
    "culture_purple": "#6A4C93",
    "capacity_teal": "#008B8B",
    "neutral_gray": "#D9D9D9",
    "ink": "#222222",
    "muted": "#666666",
    "grid": "#E6E8F0",
    "light_background": "#F7F7F4",
    "panel": "#FFFFFF",
}

FIGURE_ROWS: list[dict[str, str]] = []


def ensure_dirs() -> None:
    for base in [PAPER / "png", PAPER / "svg", PAPER / "pdf", PPT, P0]:
        base.mkdir(parents=True, exist_ok=True)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["light_background"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Times New Roman",
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def wrap_cjk(text: str, width: int = 24) -> str:
    text = str(text)
    if len(text) <= width:
        return text
    lines: list[str] = []
    current = ""
    count = 0
    for ch in text:
        current += ch
        count += 1 if ord(ch) > 127 else 0.55
        if count >= width:
            lines.append(current)
            current = ""
            count = 0
    if current:
        lines.append(current)
    return "\n".join(lines)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(
        0.06,
        0.965,
        wrap_cjk(title, 34),
        ha="left",
        va="top",
        fontproperties=FONT_CN,
        fontsize=18,
        color=COLORS["ink"],
        linespacing=1.1,
    )
    fig.text(
        0.06,
        0.905,
        wrap_cjk(subtitle, 56),
        ha="left",
        va="top",
        fontproperties=FONT_CN_REG,
        fontsize=10.5,
        color=COLORS["muted"],
        linespacing=1.18,
    )


def style_axes(ax: plt.Axes, x_cn: bool = False, y_cn: bool = False, grid_axis: str = "y") -> None:
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, linestyle="-", alpha=0.9)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_CN_REG if x_cn else FONT_EN)
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_CN_REG if y_cn else FONT_EN)
        label.set_fontsize(9)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])


def set_cn_axis_labels(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FONT_CN, fontsize=10, color=COLORS["ink"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FONT_CN, fontsize=10, color=COLORS["ink"], labelpad=8)


def annotate_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "center",
    va: str = "center",
    color: str = COLORS["ink"],
    size: float = 9,
    cn: bool = False,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        color=color,
        fontsize=size,
        fontproperties=FONT_CN if cn else FONT_EN,
    )


def save_figure(fig: plt.Figure, stem: str, title: str, data_file: str, chart_type: str) -> None:
    png = PAPER / "png" / f"{stem}.png"
    svg = PAPER / "svg" / f"{stem}.svg"
    pdf = PAPER / "pdf" / f"{stem}.pdf"
    ppt_png = PPT / f"{stem}.png"
    p0_png = P0 / f"{stem}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(ppt_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(p0_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    FIGURE_ROWS.append(
        {
            "figure_id": stem,
            "title": title,
            "chart_type": chart_type,
            "data_file": data_file,
            "paper_png": str(png.relative_to(ROOT)),
            "paper_svg": str(svg.relative_to(ROOT)),
            "paper_pdf": str(pdf.relative_to(ROOT)),
            "ppt_png": str(ppt_png.relative_to(ROOT)),
        }
    )


def fig_00_model_evolution() -> None:
    df = read("overview_model_evolution.csv")
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_axis_off()
    add_header(fig, "四问从个人路线优化递进到节假日容量运营", "决策主体、时间尺度和目标函数逐层升级，四问模型均已冻结并进入可视化阶段。")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_w = 0.21
    xs = [0.04, 0.285, 0.53, 0.775]
    colors = [COLORS["model_blue"], COLORS["low_risk_green"], COLORS["culture_purple"], COLORS["capacity_teal"]]
    model_labels = [
        "Q1-V3 鲁棒多目标\n多模式定向游",
        "Q2-V3 两年路径覆盖\n与境内交通费最小化",
        "Q3-V2 三团队文化\nMinMax调度",
        "Q4-V2 线路产品组合\n与分时容量优化",
    ]
    takeaway_labels = [
        "32景点是覆盖上界，\n24景点是运营鲁棒主推",
        "开放多口岸可降境内费用，\n但需外部差价阈值判断",
        "文化考察按完成时间均衡，\n不按景点数量均分",
        "容量够不等于质量好，\n必须看分时与多资源瓶颈",
    ]
    metric_labels = [
        "24景点 / 5天缓冲 / 运营成功率94.0%",
        "鲁棒主推3099.91元 / 阈值1170.91元",
        "最大99.19小时 / 固定政策空间gap=0",
        "63候选 / 18入选 / 160190人容量",
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        x = xs[i]
        y = 0.21
        box = FancyBboxPatch(
            (x, y),
            box_w,
            0.55,
            boxstyle="round,pad=0.014,rounding_size=0.012",
            facecolor="#FFFFFF",
            edgecolor=colors[i],
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.02, y + 0.51, row["question"], fontproperties=FONT_EN_BOLD, fontsize=18, color=colors[i])
        ax.text(x + 0.02, y + 0.45, wrap_cjk(row["decision_subject"], 12), fontproperties=FONT_CN, fontsize=12, color=COLORS["ink"])
        ax.text(x + 0.02, y + 0.37, wrap_cjk(row["time_scale"], 12), fontproperties=FONT_CN_REG, fontsize=10, color=COLORS["muted"])
        ax.text(x + 0.02, y + 0.29, model_labels[i], fontproperties=FONT_CN_REG, fontsize=9.8, color=COLORS["ink"], linespacing=1.15)
        ax.text(x + 0.02, y + 0.17, takeaway_labels[i], fontproperties=FONT_CN, fontsize=10.4, color=colors[i], linespacing=1.15)
        ax.text(x + 0.02, y + 0.055, metric_labels[i], fontproperties=FONT_CN_REG, fontsize=8.7, color=COLORS["muted"], linespacing=1.1)
        if i < 3:
            arrow = FancyArrowPatch(
                (x + box_w + 0.008, y + 0.285),
                (xs[i + 1] - 0.012, y + 0.285),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=COLORS["neutral_gray"],
            )
            ax.add_patch(arrow)
    save_figure(fig, "fig_00_model_evolution", "四问从个人路线优化递进到节假日容量运营", "overview_model_evolution.csv", "流程卡片")


def fig_q1_route_funnel() -> None:
    df = read("q1_visual_route_tiers.csv").query("tier_order <= 4").sort_values("tier_order", ascending=False)
    colors = {
        "极限覆盖版": COLORS["medium_risk_orange"],
        "30景点均衡覆盖候选版": COLORS["medium_risk_orange"],
        "运营鲁棒主推版": COLORS["model_blue"],
        "严格舒适主推备选": COLORS["low_risk_green"],
    }
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    add_header(fig, "32景点是覆盖上界，24景点才是运营鲁棒主推", "漏斗按现实可执行性收束：覆盖越高不代表越稳健，严格舒适口径需另行展示。")
    fig.subplots_adjust(top=0.80, left=0.18, right=0.95)
    y = np.arange(len(df))
    bars = ax.barh(
        y,
        df["spots_count"],
        color=[colors[r] for r in df["selected_role"]],
        edgecolor=COLORS["ink"],
        linewidth=0.8,
        height=0.62,
    )
    ax.set_yticks(y, [wrap_cjk(x, 12) for x in df["selected_role"]])
    ax.set_xlim(0, 36)
    set_cn_axis_labels(ax, "景点数量", None)
    style_axes(ax, x_cn=False, y_cn=True, grid_axis="x")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"{int(row['spots_count'])}景点 / 缓冲{int(row['buffer_days'])}天 / 运营{row['operational_success_probability']:.1%}"
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontproperties=FONT_CN_REG,
            fontsize=9.2,
            color=COLORS["ink"],
        )
        ax.text(
            1.0,
            bar.get_y() + bar.get_height() / 2,
            wrap_cjk(row["visual_role_note"], 16),
            va="center",
            ha="left",
            fontproperties=FONT_CN_REG,
            fontsize=8.6,
            color="#FFFFFF",
        )
    save_figure(fig, "fig_q1_route_funnel", "32景点是覆盖上界，24景点才是运营鲁棒主推", "q1_visual_route_tiers.csv", "阶梯漏斗")


def fig_q1_success_comparison() -> None:
    tiers = read("q1_visual_route_tiers.csv").query("tier_order <= 4")
    df = read("q1_visual_success_comparison.csv")
    df = df[df["route_id"].isin(tiers["route_id"])]
    order = tiers.sort_values("tier_order")["selected_role"].tolist()
    metrics = ["运营成功率", "严格舒适成功率"]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    add_header(fig, "运营可完成不等于严格舒适", "Q1必须同时展示两类成功率：24景点主推是运营鲁棒方案，但严格舒适成功率并未达标。")
    fig.subplots_adjust(top=0.79, left=0.09, right=0.97, bottom=0.22)
    palette = {"运营成功率": COLORS["model_blue"], "严格舒适成功率": COLORS["low_risk_green"]}
    for i, metric in enumerate(metrics):
        vals = [
            df[(df["selected_role"] == role) & (df["metric_label"] == metric)]["probability"].iloc[0]
            for role in order
        ]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width=width,
            label=metric,
            color=palette[metric],
            edgecolor=COLORS["ink"],
            linewidth=0.7,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.025,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontproperties=FONT_EN,
                fontsize=8.6,
                color=COLORS["ink"],
            )
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x, [wrap_cjk(r, 9) for r in order])
    set_cn_axis_labels(ax, None, "成功率")
    style_axes(ax, x_cn=True, y_cn=False, grid_axis="y")
    ax.legend(prop=FONT_CN_REG, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    save_figure(fig, "fig_q1_success_comparison", "运营可完成不等于严格舒适", "q1_visual_success_comparison.csv", "分组柱状")


def fig_q2_cost_threshold() -> None:
    df = read("q2_visual_plan_compare.csv")
    df = df[df["selected_role"].isin(["ROOTED_URUMQI_MAIN", "OPEN_GATEWAY_MIN_INTRA_COST", "ROBUST_TWO_YEAR_MAIN"])].sort_values("plot_order")
    threshold = read("q2_visual_gateway_threshold.csv")["break_even_external_premium_yuan_for_two"].iloc[0]
    fig = plt.figure(figsize=(13.8, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.2], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    add_header(fig, "开放多口岸显著降低新疆境内交通费", "境内费用最低方案约2856.80元；若外部大交通差价低于1170.91元，开放多口岸总体更划算。")
    fig.subplots_adjust(top=0.80, left=0.11, right=0.95, bottom=0.11)
    colors = [COLORS["neutral_gray"], COLORS["low_risk_green"], COLORS["model_blue"]]
    bars = ax.bar(df["visual_role_label"], df["total_intra_transport_cost"], color=colors, edgecolor=COLORS["ink"], linewidth=0.7)
    for bar, val in zip(bars, df["total_intra_transport_cost"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 80, f"{val:,.2f}元", ha="center", va="bottom", fontproperties=FONT_CN_REG, fontsize=9)
    set_cn_axis_labels(ax, None, "新疆境内交通费（元/两人）")
    style_axes(ax, x_cn=True, y_cn=False, grid_axis="y")
    ax.set_ylim(0, max(df["total_intra_transport_cost"]) * 1.25)

    ax2.set_xlim(0, 1500)
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color=COLORS["grid"], linewidth=8, solid_capstyle="round")
    ax2.axvspan(0, threshold, color="#E9F5EC", alpha=1)
    ax2.axvspan(threshold, 1500, color="#F8E9E7", alpha=1)
    ax2.axvline(threshold, color=COLORS["high_risk_red"], linestyle="--", linewidth=1.5)
    ax2.text(threshold, 0.74, f"临界点 {threshold:,.2f}元", ha="center", va="bottom", fontproperties=FONT_CN, fontsize=10, color=COLORS["high_risk_red"])
    ax2.text(threshold / 2, 0.25, "开放多口岸更优", ha="center", va="center", fontproperties=FONT_CN, fontsize=10, color=COLORS["low_risk_green"])
    ax2.text((1500 + threshold) / 2, 0.25, "固定乌鲁木齐更稳妥", ha="center", va="center", fontproperties=FONT_CN, fontsize=10, color=COLORS["high_risk_red"])
    ax2.set_yticks([])
    set_cn_axis_labels(ax2, "外部大交通额外差价（元/两人）", None)
    style_axes(ax2, x_cn=False, y_cn=False, grid_axis="")
    save_figure(fig, "fig_q2_cost_threshold", "开放多口岸显著降低新疆境内交通费", "q2_visual_plan_compare.csv; q2_visual_gateway_threshold.csv", "柱状+阈值标尺")


def fig_q3_group_completion() -> None:
    df = read("q3_visual_group_summary.csv").sort_values("total_hours", ascending=True)
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    add_header(fig, "Q3-V2最大完成时间为99.19小时，组间差距为9.99小时", "三组按任务完成时间MinMax均衡；楼兰/尼雅专项组点数少但审批和安全成本高。")
    fig.subplots_adjust(top=0.80, left=0.20, right=0.95)
    y = np.arange(len(df))
    colors = [COLORS["model_blue"], COLORS["low_risk_green"], COLORS["culture_purple"]]
    bars = ax.barh(y, df["total_hours"], color=colors, edgecolor=COLORS["ink"], linewidth=0.8, height=0.55)
    ax.set_yticks(y, df["group_label"])
    max_hours = df["max_completion_hours"].max()
    ax.axvline(max_hours, color=COLORS["high_risk_red"], linestyle="--", linewidth=1.2)
    ax.text(max_hours + 0.5, len(df) - 0.35, "max=99.19h", fontproperties=FONT_EN_BOLD, fontsize=9, color=COLORS["high_risk_red"])
    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"{row['total_hours']:.2f}h / {int(row['spots_count'])}点"
        ax.text(bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontproperties=FONT_CN_REG, fontsize=9)
    set_cn_axis_labels(ax, "完成时间（小时）", None)
    style_axes(ax, x_cn=False, y_cn=True, grid_axis="x")
    ax.set_xlim(0, 110)
    save_figure(fig, "fig_q3_group_completion", "Q3-V2最大完成时间为99.19小时，组间差距为9.99小时", "q3_visual_group_summary.csv", "横向条形")


def fig_q3_exact_check_card() -> None:
    row = read("q3_visual_exact_check_card.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_axis_off()
    add_header(fig, "固定特殊准入政策空间下最优性缺口为0", "该最优性只在楼兰/尼雅固定专项组、三组均乌鲁木齐起讫、资源不跨组共享的政策空间内成立。")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cards = [
        ("文化候选点", f"{int(row['candidate_spots'])}", "candidate spots"),
        ("枚举可行分配", f"{int(row['feasible_partitions']):,}", "feasible partitions"),
        ("最大完成时间", f"{row['exact_max_completion_hours']:.2f}h", "exact max completion"),
        ("最优性缺口", f"{row['optimality_gap_under_constraints']:.0f}", "gap under fixed policy"),
    ]
    xs = [0.07, 0.30, 0.53, 0.76]
    for x, (label, value, note) in zip(xs, cards):
        rect = FancyBboxPatch((x, 0.36), 0.17, 0.25, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor="#FFFFFF", edgecolor=COLORS["culture_purple"], linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.085, 0.54, value, ha="center", va="center", fontproperties=FONT_EN_BOLD, fontsize=22, color=COLORS["culture_purple"])
        ax.text(x + 0.085, 0.45, label, ha="center", va="center", fontproperties=FONT_CN, fontsize=11, color=COLORS["ink"])
        ax.text(x + 0.085, 0.39, note, ha="center", va="center", fontproperties=FONT_EN, fontsize=8, color=COLORS["muted"])
    ax.text(
        0.07,
        0.23,
        "政策空间：楼兰古城和尼雅遗址固定由专项审批组执行；三组均乌鲁木齐起讫；资源不跨组共享。",
        ha="left",
        va="top",
        fontproperties=FONT_CN_REG,
        fontsize=10.5,
        color=COLORS["ink"],
    )
    save_figure(fig, "fig_q3_exact_gap_card", "固定特殊准入政策空间下最优性缺口为0", "q3_visual_exact_check_card.csv", "信息卡")


def fig_q4_capacity_compare() -> None:
    df = read("q4_visual_capacity_compare.csv")
    bars_df = df[df["item_type"].isin(["legacy_capacity", "q4v2_capacity"])].copy()
    demand = df.loc[df["item_type"] == "baseline_demand", "capacity_persons"].iloc[0]
    extreme = df.loc[df["item_type"] == "extreme_add_capacity_served", "capacity_persons"].iloc[0]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    add_header(fig, "Q4-V2将五一可投放容量从111956提升到160190", "18条12天线路产品扩展容量；基准需求106358人可被接收，但仍需分时和多资源审计。")
    fig.subplots_adjust(top=0.80, left=0.13, right=0.92, bottom=0.18)
    colors = [COLORS["neutral_gray"], COLORS["capacity_teal"]]
    bars = ax.bar(bars_df["item"], bars_df["capacity_persons"], color=colors, edgecolor=COLORS["ink"], linewidth=0.8, width=0.55)
    ax.axhline(demand, color=COLORS["medium_risk_orange"], linestyle="--", linewidth=1.4, label=f"基准需求 {demand:,.0f}")
    ax.axhline(extreme, color=COLORS["model_blue"], linestyle=":", linewidth=1.4, label=f"复合极端补运力可接待 {extreme:,.0f}")
    for bar, val in zip(bars, bars_df["capacity_persons"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3000, f"{val:,.0f}", ha="center", va="bottom", fontproperties=FONT_EN, fontsize=9)
    set_cn_axis_labels(ax, None, "12天容量/需求（人）")
    style_axes(ax, x_cn=True, y_cn=False, grid_axis="y")
    ax.legend(prop=FONT_CN_REG, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_ylim(0, max(df["capacity_persons"]) * 1.18)
    save_figure(fig, "fig_q4_capacity_compare", "Q4-V2将五一可投放容量从111956提升到160190", "q4_visual_capacity_compare.csv", "双柱+需求线")


def fig_q4_quality_audit() -> None:
    df = read("q4_visual_quality_audit.csv").copy()
    df = df.sort_values(["quality_pass", "max_day_active_hours"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(14.5, 10.5))
    add_header(fig, "线路产品还需质量审计：8条可直接投放，10条需微调或储备", "横轴为单日最大活动强度；绿色为可直接投放，橙色为建议储备或人工微调。")
    fig.subplots_adjust(top=0.84, left=0.33, right=0.94, bottom=0.08)
    labels = [wrap_cjk(x, 18) for x in df["route_theme"]]
    y = np.arange(len(df))
    colors = np.where(df["quality_pass"], COLORS["low_risk_green"], COLORS["medium_risk_orange"])
    bars = ax.barh(y, df["max_day_active_hours"], color=colors, edgecolor=COLORS["ink"], linewidth=0.6, height=0.58)
    ax.set_yticks(y, labels)
    ax.axvline(9, color=COLORS["high_risk_red"], linestyle="--", linewidth=1.2)
    ax.text(9.03, len(df) - 0.2, "9h参考线", fontproperties=FONT_CN_REG, fontsize=8.5, color=COLORS["high_risk_red"], ha="left", va="top")
    for bar, (_, row) in zip(bars, df.iterrows()):
        text = f"{row['max_day_active_hours']:.2f}h  红日{int(row['red_days'])}  长转{int(row['long_transfer_days'])}"
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, text, va="center", ha="left", fontproperties=FONT_CN_REG, fontsize=8.1)
    set_cn_axis_labels(ax, "最大单日活动强度（小时）", None)
    style_axes(ax, x_cn=False, y_cn=True, grid_axis="x")
    ax.set_xlim(0, max(df["max_day_active_hours"]) + 2.0)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["low_risk_green"], edgecolor=COLORS["ink"], label="可直接投放"),
            Patch(facecolor=COLORS["medium_risk_orange"], edgecolor=COLORS["ink"], label="需微调或储备"),
        ],
        prop=FONT_CN_REG,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.025),
        ncol=2,
    )
    save_figure(fig, "fig_q4_quality_audit", "线路产品还需质量审计：8条可直接投放，10条需微调或储备", "q4_visual_quality_audit.csv", "横向条形")


def fig_q4_timeslot_heatmap() -> None:
    df = read("q4_visual_timeslot_pressure.csv").copy()
    top_spots = (
        df.groupby("spot_name")["utilization"].max().sort_values(ascending=False).head(18).index.tolist()
    )
    df = df[df["spot_name"].isin(top_spots)]
    slot_order = {"morning": 0, "afternoon": 1, "evening": 2}
    df["col_order"] = df["day_index"] * 3 + df["time_slot"].map(slot_order)
    col_order = df[["day_slot", "col_order"]].drop_duplicates().sort_values("col_order")["day_slot"].tolist()
    pivot = df.pivot_table(index="spot_name", columns="day_slot", values="utilization", aggfunc="max").reindex(index=top_spots, columns=col_order)
    values = pivot.fillna(0).to_numpy()
    cmap = ListedColormap(["#E9F5EC", "#FFF4C2", "#FFD9B3", "#F2A6A6"])
    norm = BoundaryNorm([0, 0.75, 0.90, 1.0, max(1.8, np.nanmax(values) + 0.01)], cmap.N)
    fig, ax = plt.subplots(figsize=(15.8, 9.2))
    add_header(fig, "总量不超载不代表某日某时段不拥挤", "热力图显示各景区分时段利用率；超过100%的红色单元是分时预约和补运力优先处理对象。")
    fig.subplots_adjust(top=0.82, left=0.19, right=0.89, bottom=0.20)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(pivot.index)), [wrap_cjk(x, 11) for x in pivot.index])
    xticks = np.arange(0, len(col_order), 3)
    ax.set_xticks(xticks, [col_order[i].split("_")[0] for i in xticks], rotation=0)
    style_axes(ax, x_cn=False, y_cn=True, grid_axis="")
    ax.set_xlabel("日期（每列组含早/午/晚三个时段）", fontproperties=FONT_CN, fontsize=10, labelpad=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if val > 1.0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontproperties=FONT_EN_BOLD, fontsize=7.2, color=COLORS["high_risk_red"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0.375, 0.825, 0.95, 1.18])
    cbar.set_ticklabels(["<75%", "75-90%", "90-100%", ">100%"])
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT_EN)
        label.set_fontsize(8.5)
    save_figure(fig, "fig_q4_timeslot_heatmap", "总量不超载不代表某日某时段不拥挤", "q4_visual_timeslot_pressure.csv", "热力图")


def build_index() -> None:
    df = pd.DataFrame(FIGURE_ROWS)
    df.to_csv(PACKAGE / "figure_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    setup_theme()
    fig_00_model_evolution()
    fig_q1_route_funnel()
    fig_q1_success_comparison()
    fig_q2_cost_threshold()
    fig_q3_group_completion()
    fig_q3_exact_check_card()
    fig_q4_capacity_compare()
    fig_q4_quality_audit()
    fig_q4_timeslot_heatmap()
    build_index()
    print(f"P0 figures exported to {PAPER} and {PPT}")


if __name__ == "__main__":
    main()
