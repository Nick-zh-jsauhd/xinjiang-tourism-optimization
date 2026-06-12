from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


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
    "q1": "#1F4E79",
    "q2": "#2E7D32",
    "q3": "#6A4C93",
    "q4": "#008B8B",
    "orange": "#D9822B",
    "red": "#B23A35",
    "green": "#2E7D32",
    "ink": "#222222",
    "muted": "#666666",
    "axis": "#333333",
    "light": "#D9D9D9",
    "very_light": "#F2F2F2",
    "white": "#FFFFFF",
}

FIGURE_ROWS: list[dict[str, str]] = []


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


def ensure_dirs(clean: bool = True) -> None:
    targets = [PAPER / "png", PAPER / "svg", PAPER / "pdf", PPT, P0]
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)
        if clean:
            for suffix in ("*.png", "*.svg", "*.pdf"):
                for file in folder.glob(suffix):
                    file.unlink()


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Times New Roman",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def wrap_cjk(text: object, width: int = 14) -> str:
    if pd.isna(text):
        return ""
    value = str(text)
    lines: list[str] = []
    current = ""
    weight = 0.0
    for ch in value:
        current += ch
        weight += 1.0 if ord(ch) > 127 else 0.55
        if weight >= width:
            lines.append(current)
            current = ""
            weight = 0.0
    if current:
        lines.append(current)
    return "\n".join(lines)


def compact_label(text: object, limit: int = 15) -> str:
    if pd.isna(text):
        return ""
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def yuan(value: float) -> str:
    return f"{value:,.0f}"


def persons(value: float) -> str:
    return f"{value:,.0f}"


def clean_axis(ax: plt.Axes, *, x_cn: bool = False, y_cn: bool = False) -> None:
    ax.grid(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_CN_REG if x_cn else FONT_EN)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_CN_REG if y_cn else FONT_EN)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_EN_BOLD,
        fontsize=13,
        color=COLORS["ink"],
    )


def figure_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(
        0.02,
        0.985,
        title,
        ha="left",
        va="top",
        fontproperties=FONT_CN,
        fontsize=12,
        color=COLORS["ink"],
    )
    if subtitle:
        fig.text(
            0.02,
            0.935,
            subtitle,
            ha="left",
            va="top",
            fontproperties=FONT_CN_REG,
            fontsize=8.5,
            color=COLORS["muted"],
        )


def save_figure(fig: plt.Figure, stem: str, title: str, data_file: str, chart_type: str, note: str) -> None:
    paths = {
        "paper_png": PAPER / "png" / f"{stem}.png",
        "paper_svg": PAPER / "svg" / f"{stem}.svg",
        "paper_pdf": PAPER / "pdf" / f"{stem}.pdf",
        "ppt_png": PPT / f"{stem}.png",
        "p0_png": P0 / f"{stem}.png",
    }
    fig.savefig(paths["paper_png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["paper_svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["paper_pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["ppt_png"], dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["p0_png"], dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURE_ROWS.append(
        {
            "figure_id": stem,
            "title": title,
            "chart_type": chart_type,
            "data_file": data_file,
            "design_note": note,
            "paper_png": str(paths["paper_png"].relative_to(ROOT)),
            "paper_svg": str(paths["paper_svg"].relative_to(ROOT)),
            "paper_pdf": str(paths["paper_pdf"].relative_to(ROOT)),
            "ppt_png": str(paths["ppt_png"].relative_to(ROOT)),
        }
    )


def fig_q1_route_tradeoff() -> None:
    df = read("q1_visual_route_tiers.csv").query("tier_order <= 4").copy()
    df = df.sort_values("tier_order", ascending=False)
    labels = [
        "32景点覆盖上界",
        "30景点覆盖候选",
        "24景点运营主推",
        "20景点舒适备选",
    ][::-1]
    y = np.arange(len(df))

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(9.8, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.15], "wspace": 0.28},
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.13, right=0.98)
    figure_title(
        fig,
        "Q1：覆盖越高并不等于可执行，24景点是运营鲁棒主推",
        "左图同时显示覆盖、缓冲与红色压力日；右图拆分运营成功率和严格舒适成功率。",
    )

    colors = []
    for _, row in df.iterrows():
        if not bool(row["schedule_strict_feasible"]):
            colors.append("#BFBFBF")
        elif int(row["red_days"]) <= 1:
            colors.append(COLORS["green"])
        else:
            colors.append(COLORS["q1"])

    sizes = 75 + df["buffer_days"].to_numpy() * 38
    ax_l.scatter(
        df["spots_count"],
        y,
        s=sizes,
        c=colors,
        edgecolors=COLORS["ink"],
        linewidths=0.8,
        zorder=3,
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        ax_l.text(
            row["spots_count"] + 0.35,
            yi,
            f"缓冲{int(row['buffer_days'])}天 / 红日{int(row['red_days'])}",
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=8.3,
            color=COLORS["muted"] if not bool(row["schedule_strict_feasible"]) else COLORS["ink"],
        )
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontproperties=FONT_CN_REG)
    ax_l.set_xlim(18, 34)
    ax_l.set_ylim(-0.65, len(df) - 0.35)
    ax_l.set_xlabel("覆盖景点数", fontproperties=FONT_CN, fontsize=10)
    panel_label(ax_l, "a")
    clean_axis(ax_l, y_cn=True)

    for yi, (_, row) in zip(y, df.iterrows()):
        strict = float(row["strict_comfort_success_probability"])
        oper = float(row["operational_success_probability"])
        ax_r.plot([strict, oper], [yi, yi], color=COLORS["light"], linewidth=1.6, zorder=1)
        ax_r.scatter(strict, yi, s=52, color=COLORS["orange"], edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
        ax_r.scatter(oper, yi, s=52, color=COLORS["q1"], edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
        ax_r.text(oper + 0.025, yi + 0.08, percent(oper), ha="left", va="center", fontproperties=FONT_EN, fontsize=8)
        ax_r.text(strict + 0.025, yi - 0.10, percent(strict), ha="left", va="center", fontproperties=FONT_EN, fontsize=8, color=COLORS["orange"])
    ax_r.axvline(0.80, color=COLORS["axis"], linewidth=0.8, linestyle=(0, (3, 3)))
    ax_r.text(0.805, len(df) - 0.35, "80%阈值", ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["muted"])
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlim(-0.03, 1.05)
    ax_r.set_ylim(-0.65, len(df) - 0.35)
    ax_r.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_r.set_xlabel("仿真成功率", fontproperties=FONT_CN, fontsize=10)
    panel_label(ax_r, "b")
    clean_axis(ax_r)
    ax_r.scatter([], [], s=52, color=COLORS["q1"], edgecolor=COLORS["ink"], linewidth=0.5, label="运营成功率")
    ax_r.scatter([], [], s=52, color=COLORS["orange"], edgecolor=COLORS["ink"], linewidth=0.5, label="严格舒适成功率")
    leg = ax_r.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.15),
        ncol=2,
        prop=FONT_CN_REG,
        fontsize=8,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        text.set_fontproperties(FONT_CN_REG)

    save_figure(
        fig,
        "fig_q1_route_tradeoff",
        "Q1覆盖、缓冲与两类成功率取舍",
        "q1_visual_route_tiers.csv",
        "paired dot plot",
        "撤掉漏斗图，以同一张图展示覆盖上界、运营主推和严格舒适不足。",
    )


def fig_q2_gateway_cost_rule() -> None:
    plans = read("q2_visual_plan_compare.csv").sort_values("plot_order")
    threshold = read("q2_visual_gateway_threshold.csv").iloc[0]
    y = np.arange(len(plans))[::-1]
    labels = [compact_label(v, 11) for v in plans["visual_role_label"]]

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.7),
        gridspec_kw={"width_ratios": [1.25, 0.9], "wspace": 0.33},
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.15, right=0.98)
    figure_title(
        fig,
        "Q2：开放多口岸可降低境内交通费，但要用外部差价阈值决策",
        "费用只统计新疆境内交通；多口岸是否划算取决于两人大交通额外差价。",
    )

    costs = plans["total_intra_transport_cost"].to_numpy()
    colors = [COLORS["q2"] if "鲁棒" in v else "#A6A6A6" for v in plans["visual_role_label"]]
    for yi, cost, color in zip(y, costs, colors):
        ax_l.hlines(yi, xmin=2500, xmax=cost, color=COLORS["light"], linewidth=1.3)
        ax_l.scatter(cost, yi, s=58, color=color, edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
        ax_l.text(cost + 40, yi, f"{cost:.0f}元", ha="left", va="center", fontproperties=FONT_CN_REG, fontsize=8.6)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontproperties=FONT_CN_REG)
    ax_l.set_xlim(2500, 4250)
    ax_l.set_xlabel("两人境内交通费用（元）", fontproperties=FONT_CN, fontsize=10)
    ax_l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    panel_label(ax_l, "a")
    clean_axis(ax_l, y_cn=True)

    rooted = float(threshold["rooted_cost_yuan_for_two"])
    open_cost = float(threshold["open_gateway_cost_yuan_for_two"])
    be = float(threshold["break_even_external_premium_yuan_for_two"])
    ax_r.set_xlim(0, 1500)
    ax_r.set_ylim(-0.5, 1.0)
    ax_r.hlines(0, 0, 1500, color=COLORS["axis"], linewidth=0.9)
    ax_r.scatter([be], [0], s=70, color=COLORS["q2"], edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
    ax_r.vlines(be, -0.08, 0.08, color=COLORS["q2"], linewidth=1.2)
    ax_r.text(be, 0.22, f"阈值 {be:.0f} 元", ha="center", va="bottom", fontproperties=FONT_CN, fontsize=9)
    ax_r.text(170, -0.26, "低于阈值\n开放多口岸总费用更优", ha="left", va="top", fontproperties=FONT_CN_REG, fontsize=8.4, color=COLORS["q2"])
    ax_r.text(1030, -0.26, "高于阈值\n固定起讫更稳妥", ha="left", va="top", fontproperties=FONT_CN_REG, fontsize=8.4, color=COLORS["muted"])
    ax_r.text(
        0.02,
        0.88,
        f"计算：{rooted:.0f} - {open_cost:.0f} = {be:.0f}",
        transform=ax_r.transAxes,
        ha="left",
        va="center",
        fontproperties=FONT_CN_REG,
        fontsize=8.6,
        color=COLORS["muted"],
    )
    ax_r.set_yticks([])
    ax_r.set_xlabel("两人多口岸外部大交通额外差价（元）", fontproperties=FONT_CN, fontsize=10)
    ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    panel_label(ax_r, "b")
    clean_axis(ax_r)

    save_figure(
        fig,
        "fig_q2_gateway_cost_rule",
        "Q2多口岸费用与差价阈值",
        "q2_visual_plan_compare.csv; q2_visual_gateway_threshold.csv",
        "lollipop plus threshold axis",
        "用阈值轴替代大色块，突出境内费用节省的适用条件。",
    )


def fig_q3_minmax_evidence() -> None:
    group = read("q3_visual_group_summary.csv").sort_values("group_id")
    check = read("q3_visual_exact_check_card.csv").iloc[0]
    y = np.arange(len(group))[::-1]

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.8),
        gridspec_kw={"width_ratios": [1.35, 0.85], "wspace": 0.28},
    )
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.14, right=0.98)
    figure_title(
        fig,
        "Q3：三组按完成时间均衡，固定特殊准入政策空间下 exact gap = 0",
        "楼兰古城和尼雅遗址固定由专项审批组执行，三组均乌鲁木齐起讫。",
    )

    totals = group["total_hours"].to_numpy()
    max_hours = float(group["max_completion_hours"].iloc[0])
    labels = group["group_label"].tolist()
    for yi, (_, row) in zip(y, group.iterrows()):
        color = COLORS["q3"] if row["group_id"] == 1 else "#777777"
        ax_l.hlines(yi, xmin=84, xmax=row["total_hours"], color=COLORS["light"], linewidth=1.4)
        ax_l.scatter(row["total_hours"], yi, s=64, color=color, edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
        ax_l.text(row["total_hours"] + 0.6, yi, f"{row['total_hours']:.2f}h", ha="left", va="center", fontproperties=FONT_EN, fontsize=8.8)
        detail_y = yi - 0.23
        detail_va = "top"
        if yi == y.min():
            detail_y = yi + 0.25
            detail_va = "bottom"
        ax_l.text(
            84.2,
            detail_y,
            f"{int(row['spots_count'])}点；考察{row['fieldwork_hours']:.0f}h",
            ha="left",
            va=detail_va,
            fontproperties=FONT_CN_REG,
            fontsize=8,
            color=COLORS["muted"],
        )
    ax_l.axvline(max_hours, color=COLORS["q3"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax_l.text(max_hours + 0.4, y.max() + 0.2, "MinMax目标", ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["q3"])
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontproperties=FONT_CN_REG)
    ax_l.set_xlim(84, 103)
    ax_l.set_xlabel("完成时间（小时）", fontproperties=FONT_CN, fontsize=10)
    panel_label(ax_l, "a")
    clean_axis(ax_l, y_cn=True)

    ax_r.set_axis_off()
    panel_label(ax_r, "b")
    rows = [
        ("文化候选点", f"{int(check['candidate_spots'])}"),
        ("枚举可行分配", f"{int(check['feasible_partitions']):,}"),
        ("精确最大完成时间", f"{float(check['exact_max_completion_hours']):.2f} h"),
        ("固定政策空间 gap", f"{float(check['optimality_gap_under_constraints']):.1f}"),
    ]
    ax_r.text(0.02, 0.82, "最优性审计", transform=ax_r.transAxes, ha="left", va="center", fontproperties=FONT_CN, fontsize=10.5, color=COLORS["ink"])
    y0 = 0.66
    for idx, (name, value) in enumerate(rows):
        yy = y0 - idx * 0.15
        ax_r.hlines(yy - 0.055, 0.02, 0.98, transform=ax_r.transAxes, color=COLORS["light"], linewidth=0.7)
        ax_r.text(0.02, yy, name, transform=ax_r.transAxes, ha="left", va="center", fontproperties=FONT_CN_REG, fontsize=8.7, color=COLORS["muted"])
        ax_r.text(0.98, yy, value, transform=ax_r.transAxes, ha="right", va="center", fontproperties=FONT_EN_BOLD, fontsize=10, color=COLORS["q3"])
    ax_r.text(
        0.02,
        0.04,
        "该 gap 仅在固定专项审批组、资源不跨组共享等政策空间内成立。",
        transform=ax_r.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_CN_REG,
        fontsize=8,
        color=COLORS["muted"],
        wrap=True,
    )

    save_figure(
        fig,
        "fig_q3_minmax_evidence",
        "Q3三组完成时间与最优性审计",
        "q3_visual_group_summary.csv; q3_visual_exact_check_card.csv",
        "lollipop plus audit table",
        "合并原完成时间图和低信息卡片，用完成时间与审计指标支撑结论。",
    )


def fig_q4_capacity_evidence() -> None:
    df = read("q4_visual_capacity_compare.csv")
    order = ["基准需求", "旧9线路容量", "复合极端补运力可接待", "Q4-V2 18线路容量"]
    df["order"] = df["item"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("order")
    y = np.arange(len(df))[::-1]
    color_map = {
        "baseline_demand": COLORS["axis"],
        "legacy_capacity": "#8C8C8C",
        "extreme_add_capacity_served": COLORS["orange"],
        "q4v2_capacity": COLORS["q4"],
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.18, right=0.98)
    figure_title(
        fig,
        "Q4：18条12天产品把可投放容量从旧9线路上移到16.0万人",
        "基准需求、旧方案容量、极端策略可接待量与Q4-V2投放容量放在同一尺度比较。",
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        value = float(row["capacity_persons"])
        ax.hlines(yi, xmin=100000, xmax=value, color=COLORS["light"], linewidth=1.3)
        marker = "^" if row["item_type"] == "baseline_demand" else "o"
        ax.scatter(value, yi, s=68, marker=marker, color=color_map.get(row["item_type"], COLORS["q4"]), edgecolor=COLORS["ink"], linewidth=0.5, zorder=3)
        ax.text(value + 1700, yi, persons(value), ha="left", va="center", fontproperties=FONT_EN, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["item"].tolist(), fontproperties=FONT_CN_REG)
    ax.set_xlim(100000, 168000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/10000:.1f}"))
    ax.set_xlabel("可接待人数 / 需求人数（万人）", fontproperties=FONT_CN, fontsize=10)
    clean_axis(ax, y_cn=True)

    save_figure(
        fig,
        "fig_q4_capacity_evidence",
        "Q4容量提升证据图",
        "q4_visual_capacity_compare.csv",
        "lollipop",
        "替代双柱图，用同尺度容量证据支撑18条线路投放价值。",
    )


def fig_q4_quality_audit_dotplot() -> None:
    df = read("q4_visual_quality_audit.csv").copy()
    df = df.sort_values(["quality_pass", "max_day_active_hours"], ascending=[True, False])
    y = np.arange(len(df))[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 7.1))
    fig.subplots_adjust(top=0.82, bottom=0.11, left=0.36, right=0.98)
    figure_title(
        fig,
        "Q4：线路产品质量审计显示8条可直接投放，10条需要微调或储备",
        "按最大单日强度排序；9小时线用于识别普通游客产品的强度风险。",
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        pass_flag = bool(row["quality_pass"])
        color = COLORS["green"] if pass_flag else COLORS["orange"]
        marker = "o" if pass_flag else "X"
        ax.scatter(row["max_day_active_hours"], yi, s=45 if pass_flag else 55, marker=marker, color=color, edgecolor=COLORS["ink"], linewidth=0.45, zorder=3)
        ax.text(
            row["max_day_active_hours"] + 0.12,
            yi,
            f"{row['max_day_active_hours']:.1f}h / 红{int(row['red_days'])} / 长转{int(row['long_transfer_days'])}",
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=7.2,
            color=COLORS["ink"] if not pass_flag else COLORS["muted"],
        )
    labels = [wrap_cjk(v, 18) for v in df["route_theme"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=FONT_CN_REG)
    ax.axvline(9.0, color=COLORS["red"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(9.05, y.max() + 0.45, "9小时审计线", ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["red"])
    ax.set_xlim(4.0, max(11.5, float(df["max_day_active_hours"].max()) + 1.2))
    ax.set_xlabel("最大单日活动强度代理值（小时）", fontproperties=FONT_CN, fontsize=10)
    clean_axis(ax, y_cn=True)
    ax.scatter([], [], s=45, marker="o", color=COLORS["green"], edgecolor=COLORS["ink"], linewidth=0.45, label="可直接投放")
    ax.scatter([], [], s=55, marker="X", color=COLORS["orange"], edgecolor=COLORS["ink"], linewidth=0.45, label="需微调/储备")
    leg = ax.legend(frameon=False, loc="lower right", prop=FONT_CN_REG, fontsize=8)
    for text in leg.get_texts():
        text.set_fontproperties(FONT_CN_REG)

    save_figure(
        fig,
        "fig_q4_quality_audit_dotplot",
        "Q4线路质量审计点图",
        "q4_visual_quality_audit.csv",
        "ranked dot plot",
        "撤掉大色块横条，突出哪些线路因单日强度需要微调。",
    )


def fig_q4_bottleneck_shadow_price() -> None:
    df = read("q4_visual_shadow_top10.csv").copy()
    df = df.sort_values("shadow_price_index", ascending=True)
    y = np.arange(len(df))

    def label(row: pd.Series) -> str:
        resource = str(row["resource_or_slot"])
        cn_resource = {
            "morning": "上午预约",
            "midday": "午间预约",
            "afternoon": "下午预约",
            "licensed_guide": "持证导游",
            "coach_bus": "旅游车辆",
            "scenic_shuttle_or_parking": "摆渡/停车",
        }.get(resource, resource)
        return wrap_cjk(f"D{int(row['day_index'])} {row['location']} {cn_resource}", 17)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.28, right=0.98)
    figure_title(
        fig,
        "Q4：瓶颈不在总容量，而在少数分时预约与资源约束",
        "影子价格越高，增加一个容量单位对系统拥挤罚项的边际缓解越大。",
    )
    for yi, (_, row) in zip(y, df.iterrows()):
        color = COLORS["red"] if float(row["overload"]) > 0 else COLORS["q4"]
        ax.hlines(yi, xmin=0, xmax=row["shadow_price_index"], color=COLORS["light"], linewidth=0.8, alpha=0.55)
        ax.scatter(row["shadow_price_index"], yi, s=52, color=color, edgecolor=COLORS["ink"], linewidth=0.45, zorder=3)
        note = f"{row['shadow_price_index']:.1f}"
        if float(row["overload"]) > 0:
            note += f" / 超{row['overload']:.0f}"
        ax.text(row["shadow_price_index"] + 1.2, yi, note, ha="left", va="center", fontproperties=FONT_CN_REG, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(row) for _, row in df.iterrows()], fontproperties=FONT_CN_REG)
    ax.set_xlim(0, max(80, float(df["shadow_price_index"].max()) + 8))
    ax.set_xlabel("瓶颈影子价格指数", fontproperties=FONT_CN, fontsize=10)
    clean_axis(ax, y_cn=True)

    save_figure(
        fig,
        "fig_q4_bottleneck_shadow_price",
        "Q4瓶颈影子价格排序",
        "q4_visual_shadow_top10.csv",
        "ranked lollipop",
        "用Top瓶颈替代大面积空白热力图，解释为什么服务率满分仍需分时与资源控制。",
    )


def main() -> None:
    ensure_dirs(clean=True)
    setup_theme()
    fig_q1_route_tradeoff()
    fig_q2_gateway_cost_rule()
    fig_q3_minmax_evidence()
    fig_q4_capacity_evidence()
    fig_q4_quality_audit_dotplot()
    fig_q4_bottleneck_shadow_price()
    index = PACKAGE / "figure_index.csv"
    pd.DataFrame(FIGURE_ROWS).to_csv(index, index=False, encoding="utf-8-sig")
    print(f"Exported {len(FIGURE_ROWS)} Nature-style figures")
    print(index)


if __name__ == "__main__":
    main()
