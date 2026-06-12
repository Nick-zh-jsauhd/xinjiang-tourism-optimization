from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "08_q1_result_figures"
PNG = OUT / "png"
SVG = OUT / "svg"
PDF = OUT / "pdf"

SIMSUN = Path("C:/Windows/Fonts/simsun.ttc")
TIMES = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")

FONT_CN = FontProperties(fname=str(SIMSUN), weight="bold")
FONT_CN_REG = FontProperties(fname=str(SIMSUN))
FONT_EN = FontProperties(fname=str(TIMES))
FONT_EN_BOLD = FontProperties(fname=str(TIMES_BOLD), weight="bold")

COLORS = {
    "q1": "#1F4E79",
    "q1_light": "#8EA9C1",
    "green": "#4E7D5A",
    "green_light": "#D7E5D8",
    "orange": "#D9822B",
    "orange_light": "#E9C58F",
    "red": "#B23A35",
    "red_light": "#E9C8C4",
    "ink": "#222222",
    "muted": "#666666",
    "axis": "#333333",
    "light": "#D9D9D9",
    "very_light": "#F4F4F4",
    "white": "#FFFFFF",
}

FIGURE_ROWS: list[dict[str, str]] = []


def locate_q1_outputs() -> Path:
    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue
        hits = list(folder.glob("09_Q1_V3*/outputs"))
        if hits:
            return hits[0]
    raise FileNotFoundError("Cannot locate Q1-V3 outputs folder.")


def locate_q1_coordinates() -> Path:
    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue
        hits = list(folder.glob("01_*/enhanced_data/spot_coordinates_amap.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError("Cannot locate Q1 spot_coordinates_amap.csv.")


Q1_OUT = locate_q1_outputs()
Q1_COORDS = locate_q1_coordinates()


def read_q1(name: str) -> pd.DataFrame:
    return pd.read_csv(Q1_OUT / name, encoding="utf-8-sig")


def ensure_dirs(clean: bool = True) -> None:
    for folder in [PNG, SVG, PDF]:
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


def clean_axis(ax: plt.Axes, *, x_cn: bool = False, y_cn: bool = False) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", length=3, width=0.8, labelsize=8.5)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLORS["axis"])
        ax.spines[spine].set_linewidth(0.8)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FONT_CN_REG if x_cn else FONT_EN)
    for label in ax.get_yticklabels():
        label.set_fontproperties(FONT_CN_REG if y_cn else FONT_EN)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_EN_BOLD,
        fontsize=12,
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
            0.94,
            subtitle,
            ha="left",
            va="top",
            fontproperties=FONT_CN_REG,
            fontsize=8.4,
            color=COLORS["muted"],
        )


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def wrap_cjk(text: object, width: int = 12) -> str:
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


def short_role(role: object) -> str:
    value = str(role)
    mapping = {
        "极限覆盖版": "32覆盖上界",
        "30景点均衡覆盖候选版": "30覆盖候选",
        "运营鲁棒主推版": "24运营主推",
        "严格舒适主推备选": "20舒适备选",
        "亲子舒适版": "20亲子",
        "长者慢游版": "22长者",
    }
    return mapping.get(value, value)


def save_figure(fig: plt.Figure, stem: str, title: str, data_file: str, chart_type: str, note: str) -> None:
    paths = {
        "png": PNG / f"{stem}.png",
        "svg": SVG / f"{stem}.svg",
        "pdf": PDF / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURE_ROWS.append(
        {
            "figure_id": stem,
            "title": title,
            "chart_type": chart_type,
            "data_file": data_file,
            "design_note": note,
            "png": str(paths["png"].relative_to(ROOT)),
            "svg": str(paths["svg"].relative_to(ROOT)),
            "pdf": str(paths["pdf"].relative_to(ROOT)),
        }
    )


def red_band(red_days: pd.Series) -> pd.Series:
    return pd.cut(
        red_days,
        bins=[-0.1, 1, 3, 99],
        labels=["≤1红日", "2-3红日", "≥4红日"],
    )


def red_band_color(label: str) -> str:
    return {
        "≤1红日": COLORS["green"],
        "2-3红日": COLORS["orange"],
        "≥4红日": COLORS["red"],
    }.get(label, COLORS["muted"])


def fig_q1_01_candidate_space_legacy() -> None:
    candidates = read_q1("q1_v3_candidate_routes_enriched.csv")
    feasible = read_q1("q1_v3_feasible_robust_pareto_front.csv")
    selected = read_q1("q1_v3_selected_routes.csv")
    strict_front = read_q1("q1_v3_strict_comfort_pareto_front.csv")

    candidates = candidates.copy()
    candidates["red_band"] = red_band(candidates["red_days"])
    candidates["strict_flag"] = candidates["schedule_strict_feasible"].astype(bool)
    candidates["x_plot"] = candidates["spots_count"] + ((candidates.index % 7) - 3) * 0.035

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.14)
    figure_title(
        fig,
        "Q1：高覆盖路线在小时级排程和鲁棒仿真下快速收缩",
        "点为126条候选路线；实心为严格小时级排程可行，空心为不可严格排程；颜色表示基准红色压力日数量。",
    )

    for strict_flag in [False, True]:
        for band in ["≤1红日", "2-3红日", "≥4红日"]:
            sub = candidates[(candidates["strict_flag"] == strict_flag) & (candidates["red_band"].astype(str) == band)]
            if sub.empty:
                continue
            color = red_band_color(band)
            ax.scatter(
                sub["x_plot"],
                sub["operational_success_probability"],
                s=24 + sub["buffer_days"].fillna(0).to_numpy() * 9,
                facecolors=color if strict_flag else "none",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.72 if strict_flag else 0.46,
                zorder=2 if strict_flag else 1,
            )

    key_ids = ["Q1V3_Q32_340", "Q1V3_Q30_279", "Q1V3_Q24_120", "Q1V3_Q20_011"]
    offsets = {
        "Q1V3_Q32_340": (-0.95, 0.06),
        "Q1V3_Q30_279": (0.25, 0.055),
        "Q1V3_Q24_120": (0.30, -0.065),
        "Q1V3_Q20_011": (0.30, -0.085),
    }
    for _, row in selected[selected["route_id"].isin(key_ids)].iterrows():
        x = float(row["spots_count"])
        y = float(row["operational_success_probability"])
        ax.scatter(
            [x],
            [y],
            s=132,
            facecolors=COLORS["white"],
            edgecolors=COLORS["ink"],
            linewidths=1.5,
            zorder=5,
        )
        ax.scatter(
            [x],
            [y],
            s=76,
            facecolors=red_band_color(str(red_band(pd.Series([row["red_days"]])).iloc[0])),
            edgecolors=COLORS["ink"],
            linewidths=0.7,
            zorder=6,
        )
        dx, dy = offsets.get(str(row["route_id"]), (0.2, 0.04))
        ax.annotate(
            short_role(row["selected_role"]),
            xy=(x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=8.4,
            color=COLORS["ink"],
            arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6, shrinkA=5, shrinkB=5),
        )

    ax.axhline(0.8, color=COLORS["axis"], lw=0.85, ls=(0, (4, 3)), zorder=0)
    ax.text(
        32.55,
        0.815,
        "运营成功率阈值 80%",
        ha="right",
        va="bottom",
        fontproperties=FONT_CN_REG,
        fontsize=8,
        color=COLORS["muted"],
    )
    for xpos, label in [(24, "主推覆盖"), (30, "高覆盖候选")]:
        ax.axvline(xpos, color=COLORS["light"], lw=0.8, ls=(0, (2, 4)), zorder=0)
        ax.text(xpos + 0.1, 0.04, label, rotation=90, ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=7.5, color=COLORS["muted"])

    summary = (
        f"候选 {len(candidates)} 条  |  严格排程 {int(candidates['strict_flag'].sum())} 条  |  "
        f"运营可行前沿 {len(feasible)} 条  |  严格舒适前沿 {len(strict_front)} 条"
    )
    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontproperties=FONT_CN_REG,
        fontsize=8.2,
        color=COLORS["muted"],
    )

    ax.set_xlim(19, 33.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("覆盖景点数", fontproperties=FONT_CN_REG, fontsize=9.5)
    ax.set_ylabel("运营成功率", fontproperties=FONT_CN_REG, fontsize=9.5)
    ax.set_xticks([20, 22, 24, 26, 28, 30, 32])
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([percent(x) for x in np.linspace(0, 1.0, 6)])
    clean_axis(ax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["green"], markersize=6, label="≤1红日"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["orange"], markersize=6, label="2-3红日"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["red"], markeredgecolor=COLORS["red"], markersize=6, label="≥4红日"),
        Line2D([0], [0], marker="o", color=COLORS["axis"], markerfacecolor=COLORS["white"], markeredgecolor=COLORS["axis"], markersize=6, lw=0, label="空心=非严格排程"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, prop=FONT_CN_REG, fontsize=8, handletextpad=0.4, borderpad=0.2)

    save_figure(
        fig,
        "fig_q1_01_candidate_space",
        "Q1候选解空间与可行性收缩",
        "q1_v3_candidate_routes_enriched.csv; q1_v3_selected_routes.csv; q1_v3_feasible_robust_pareto_front.csv",
        "scatter",
        "展示126条候选路线在覆盖、运营成功率、红日和严格排程维度上的收缩。",
    )


def build_screening_counts(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for q, group in candidates.groupby("spots_count"):
        strict_schedule = group["schedule_strict_feasible"].astype(bool)
        operational = strict_schedule & (group["operational_success_probability"] >= 0.8)
        strict_comfort = operational & (group["strict_comfort_success_probability"] >= 0.8)
        rows.append(
            {
                "spots_count": int(q),
                "candidate": int(len(group)),
                "strict_schedule": int(strict_schedule.sum()),
                "operational_robust": int(operational.sum()),
                "strict_comfort_robust": int(strict_comfort.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("spots_count")


def fig_q1_01_screening_contraction() -> None:
    candidates = read_q1("q1_v3_candidate_routes_enriched.csv")
    selected = read_q1("q1_v3_selected_routes.csv")
    counts = build_screening_counts(candidates)

    fig = plt.figure(figsize=(11.8, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.38], wspace=0.23)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(top=0.78, left=0.07, right=0.985, bottom=0.15)
    figure_title(
        fig,
        "Q1：30天路线由最大覆盖收缩到鲁棒可执行方案",
        "候选路线先按覆盖数生成，再依次经过严格小时级排程、运营成功率≥80%、严格舒适成功率≥80%筛选。",
    )

    q = counts["spots_count"].to_numpy()
    series = [
        ("原始候选", "candidate", COLORS["muted"], "o"),
        ("严格小时排程", "strict_schedule", COLORS["q1"], "o"),
        ("运营鲁棒", "operational_robust", COLORS["orange"], "s"),
        ("严格舒适鲁棒", "strict_comfort_robust", COLORS["red"], "D"),
    ]
    for label, col, color, marker in series:
        ax_l.plot(
            q,
            counts[col],
            color=color,
            lw=1.8,
            marker=marker,
            markersize=5.4,
            markerfacecolor=COLORS["white"] if label == "原始候选" else color,
            markeredgecolor=color,
            markeredgewidth=1.1,
            label=label,
            zorder=3,
        )

    for q_focus, note, y_note in [(24, "24景点：运营主推可行档", 13.5), (30, "30景点：严格排程已归零", 2.5)]:
        ax_l.axvline(q_focus, color=COLORS["light"], lw=0.85, ls=(0, (3, 4)), zorder=0)
        ax_l.text(q_focus + 0.15, y_note, note, ha="left", va="center", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["muted"])

    for _, row in counts.iterrows():
        if int(row["spots_count"]) in [20, 24, 28, 30, 32]:
            ax_l.text(
                float(row["spots_count"]),
                float(row["operational_robust"]) + 0.52,
                str(int(row["operational_robust"])),
                ha="center",
                va="bottom",
                fontproperties=FONT_EN,
                fontsize=7.8,
                color=COLORS["orange"],
            )

    ax_l.set_xlim(19.4, 32.8)
    ax_l.set_ylim(-0.6, 19.1)
    ax_l.set_xticks([20, 22, 24, 26, 28, 30, 32])
    ax_l.set_yticks([0, 6, 12, 18])
    ax_l.set_xlabel("覆盖景点数", fontproperties=FONT_CN_REG, fontsize=9.2)
    ax_l.set_ylabel("存活候选路线数", fontproperties=FONT_CN_REG, fontsize=9.2)
    panel_label(ax_l, "A")
    clean_axis(ax_l)
    ax_l.legend(loc="lower left", frameon=False, prop=FONT_CN_REG, fontsize=8, handlelength=1.6)

    core_ids = ["Q1V3_Q32_340", "Q1V3_Q30_279", "Q1V3_Q24_120", "Q1V3_Q20_011"]
    rows = selected[selected["route_id"].isin(core_ids)].copy()
    order = {rid: i for i, rid in enumerate(core_ids)}
    rows["order"] = rows["route_id"].map(order)
    rows = rows.sort_values("order")
    role_map = {
        "极限覆盖版": "覆盖上界",
        "30景点均衡覆盖候选版": "高覆盖候选",
        "运营鲁棒主推版": "运营主推",
        "严格舒适主推备选": "舒适备选",
    }
    conclusion = {
        "Q1V3_Q32_340": "仅作上界",
        "Q1V3_Q30_279": "不作主推",
        "Q1V3_Q24_120": "推荐",
        "Q1V3_Q20_011": "保守备选",
    }
    table_rows = []
    for _, row in rows.iterrows():
        table_rows.append(
            [
                role_map.get(str(row["selected_role"]), str(row["selected_role"])),
                int(row["spots_count"]),
                int(row["buffer_days"]),
                int(row["red_days"]),
                int(row["time_window_violations"]),
                "是" if bool(row["schedule_strict_feasible"]) else "否",
                percent(float(row["operational_success_probability"])),
                percent(float(row["strict_comfort_success_probability"])),
                conclusion.get(str(row["route_id"]), ""),
            ]
        )

    ax_t.axis("off")
    panel_label(ax_t, "B")
    ax_t.text(
        0.0,
        1.02,
        "代表方案证据表",
        transform=ax_t.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_CN,
        fontsize=10,
        color=COLORS["ink"],
    )
    ax_t.text(
        0.0,
        0.965,
        "表格用于精确判断路线定位；图形用于展示候选解随约束增强而收缩。",
        transform=ax_t.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_CN_REG,
        fontsize=7.8,
        color=COLORS["muted"],
    )

    headers = ["定位", "景点", "缓冲", "红日", "时窗", "严排", "运营", "严舒", "判定"]
    table = ax_t.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.10, 1.0, 0.76],
        colWidths=[0.16, 0.075, 0.075, 0.065, 0.065, 0.065, 0.115, 0.115, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)

    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor(COLORS["light"])
        cell.set_linewidth(0.75)
        cell.get_text().set_fontproperties(FONT_CN_REG if c in [0, 5, 8] or r == 0 else FONT_EN)
        cell.get_text().set_color(COLORS["ink"])
        if r == 0:
            cell.set_facecolor("#EEF3F7")
            cell.get_text().set_fontproperties(FONT_CN)
            cell.set_height(0.145)
        else:
            cell.set_height(0.155)
            role = table_rows[r - 1][0]
            if role == "运营主推":
                cell.set_facecolor("#EAF1F6")
            elif role == "舒适备选":
                cell.set_facecolor("#EEF5EE")
            elif role == "覆盖上界":
                cell.set_facecolor("#F7F7F7")
            else:
                cell.set_facecolor(COLORS["white"])

    ax_t.text(
        0.0,
        0.02,
        "结论：32景点覆盖最高但不可执行；30景点仍非严格排程；24景点以5天缓冲和0时间窗违规换取运营鲁棒性。",
        transform=ax_t.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FONT_CN_REG,
        fontsize=8,
        color=COLORS["muted"],
    )

    save_figure(
        fig,
        "fig_q1_01_screening_contraction",
        "Q1最大覆盖到鲁棒可执行方案的筛选收缩",
        "q1_v3_candidate_routes_enriched.csv; q1_v3_selected_routes.csv",
        "line-dot plus evidence table",
        "用聚合筛选曲线和代表方案表替代拥挤散点，直接支撑24景点运营主推的筛选逻辑。",
    )


def fig_q1_02_route_tiers() -> None:
    selected = read_q1("q1_v3_selected_routes.csv")
    core_ids = ["Q1V3_Q32_340", "Q1V3_Q30_279", "Q1V3_Q24_120", "Q1V3_Q20_011"]
    df = selected[selected["route_id"].isin(core_ids)].copy()
    order = {rid: i for i, rid in enumerate(core_ids)}
    df["order"] = df["route_id"].map(order)
    df = df.sort_values("order", ascending=False)
    y = np.arange(len(df))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.34})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.13, right=0.98)
    figure_title(
        fig,
        "Q1：代表方案不是同一条路线的美化，而是四类不同决策角色",
        "32景点为覆盖上界，30景点为高覆盖候选，24景点为运营鲁棒主推，20景点为严格舒适方向备选。",
    )

    labels = [short_role(r) for r in df["selected_role"]]
    marker_colors = [COLORS["q1"] if bool(v) else COLORS["light"] for v in df["schedule_strict_feasible"]]
    sizes = 70 + df["buffer_days"].to_numpy() * 34
    ax_l.hlines(y, 20, df["spots_count"], color=COLORS["light"], lw=1.0, zorder=1)
    ax_l.scatter(df["spots_count"], y, s=sizes, c=marker_colors, edgecolors=COLORS["ink"], linewidths=0.7, zorder=3)
    for yi, (_, row) in zip(y, df.iterrows()):
        text = f"缓冲{int(row['buffer_days'])}天 / 红日{int(row['red_days'])} / 时间窗{int(row['time_window_violations'])}"
        ax_l.text(
            row["spots_count"] + 0.35,
            yi,
            text,
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=8,
            color=COLORS["muted"],
        )
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontproperties=FONT_CN_REG)
    ax_l.set_xlim(19, 34)
    ax_l.set_xticks([20, 24, 28, 32])
    ax_l.set_xlabel("覆盖景点数；点越大表示缓冲日越多", fontproperties=FONT_CN_REG, fontsize=9)
    panel_label(ax_l, "A")
    clean_axis(ax_l, y_cn=True)

    for yi, (_, row) in zip(y, df.iterrows()):
        op = float(row["operational_success_probability"])
        strict = float(row["strict_comfort_success_probability"])
        ax_r.plot([strict, op], [yi, yi], color=COLORS["light"], lw=1.2, zorder=1)
        ax_r.scatter([op], [yi], s=62, color=COLORS["q1"], edgecolors=COLORS["ink"], linewidths=0.6, zorder=3)
        ax_r.scatter([strict], [yi], s=62, color=COLORS["orange"], edgecolors=COLORS["ink"], linewidths=0.6, zorder=3)
        ax_r.text(op + 0.025, yi + 0.11, percent(op), ha="left", va="bottom", fontproperties=FONT_EN, fontsize=8, color=COLORS["q1"])
        ax_r.text(strict + 0.025, yi - 0.12, percent(strict), ha="left", va="top", fontproperties=FONT_EN, fontsize=8, color=COLORS["orange"])
    ax_r.axvline(0.8, color=COLORS["axis"], lw=0.85, ls=(0, (4, 3)))
    ax_r.text(0.805, len(df) - 0.52, "80%阈值", ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["muted"])
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([""] * len(y))
    ax_r.set_xlim(-0.03, 1.08)
    ax_r.set_xticks(np.linspace(0, 1, 6))
    ax_r.set_xticklabels([percent(x) for x in np.linspace(0, 1, 6)])
    ax_r.set_xlabel("成功率", fontproperties=FONT_CN_REG, fontsize=9)
    panel_label(ax_r, "B")
    clean_axis(ax_r)
    ax_r.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["q1"], markeredgecolor=COLORS["ink"], label="运营成功率"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["orange"], markeredgecolor=COLORS["ink"], label="严格舒适成功率"),
        ],
        loc="lower right",
        frameon=False,
        prop=FONT_CN_REG,
        fontsize=8,
    )

    save_figure(
        fig,
        "fig_q1_02_route_tiers",
        "Q1代表方案层级与两类成功率",
        "q1_v3_selected_routes.csv",
        "paired dot plot",
        "拆分覆盖上界、覆盖候选、运营主推与舒适备选，避免把运营成功率误读为完全舒适。",
    )


def fig_q1_03_main_route_map() -> None:
    selected = read_q1("q1_v3_selected_routes.csv")
    coords = pd.read_csv(Q1_COORDS, encoding="utf-8-sig")
    main = selected[selected["route_id"].eq("Q1V3_Q24_120")].iloc[0]
    route_ids = [x.strip() for x in str(main["spot_id_sequence"]).split("->")]
    route = pd.DataFrame({"spot_id": route_ids, "seq": np.arange(1, len(route_ids) + 1)}).merge(coords, on="spot_id", how="left")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.11)
    figure_title(
        fig,
        "Q1：24景点运营主推路线形成跨东疆、南疆、伊犁、北疆的可执行环线",
        "经纬度图只表达路线空间结构，不模拟真实道路轨迹；灰点为未入选景点，蓝线为访问顺序。",
    )

    ax.scatter(coords["longitude"], coords["latitude"], s=20, color=COLORS["light"], edgecolors="none", zorder=1)
    ax.plot(route["longitude"], route["latitude"], color=COLORS["q1"], lw=1.4, alpha=0.88, zorder=2)
    ax.scatter(route["longitude"], route["latitude"], s=42, color=COLORS["q1"], edgecolors=COLORS["white"], linewidths=0.7, zorder=3)
    ax.scatter(route.iloc[0]["longitude"], route.iloc[0]["latitude"], s=92, color=COLORS["green"], edgecolors=COLORS["ink"], linewidths=0.7, zorder=4)
    ax.scatter(route.iloc[-1]["longitude"], route.iloc[-1]["latitude"], s=92, color=COLORS["red"], edgecolors=COLORS["ink"], linewidths=0.7, zorder=4)

    key_labels = {
        "P001": (0.25, 0.20),
        "P034": (-1.05, -0.18),
        "P025": (-1.25, -0.10),
        "P009": (-0.95, 0.22),
        "P020": (-1.05, 0.20),
        "P013": (0.28, -0.58),
    }
    for sid, (dx, dy) in key_labels.items():
        row = route[route["spot_id"].eq(sid)]
        if row.empty:
            continue
        r = row.iloc[0]
        ax.text(
            float(r["longitude"]) + dx,
            float(r["latitude"]) + dy,
            f"{int(r['seq'])}. {r['spot_name']}",
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=8,
            color=COLORS["ink"],
            zorder=5,
        )

    metrics = (
        f"覆盖{int(main['spots_count'])}景点 / {int(main['region_count'])}区域 / {int(main['theme_count'])}主题\n"
        f"文化{int(main['cultural_spots'])}处，自然{int(main['natural_spots'])}处；缓冲{int(main['buffer_days'])}天\n"
        f"时间窗违规{int(main['time_window_violations'])}，运营成功率{percent(float(main['operational_success_probability']))}"
    )
    ax.text(
        0.02,
        0.98,
        metrics,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontproperties=FONT_CN_REG,
        fontsize=8.4,
        color=COLORS["ink"],
    )

    ax.set_xlim(74.4, 90.4)
    ax.set_ylim(36.6, 49.2)
    ax.set_xlabel("经度", fontproperties=FONT_CN_REG, fontsize=9)
    ax.set_ylabel("纬度", fontproperties=FONT_CN_REG, fontsize=9)
    clean_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["ink"], markersize=7, label="起点"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["red"], markeredgecolor=COLORS["ink"], markersize=7, label="终点"),
            Line2D([0], [0], color=COLORS["q1"], lw=1.4, label="访问顺序"),
        ],
        loc="lower left",
        frameon=False,
        prop=FONT_CN_REG,
        fontsize=8,
    )

    save_figure(
        fig,
        "fig_q1_03_main_route_map",
        "Q1运营鲁棒主推路线空间结构",
        "q1_v3_selected_routes.csv; spot_coordinates_amap.csv",
        "route map",
        "用经纬度和关键节点标注说明24景点主推路线的跨区域覆盖。",
    )


def compute_daily_load(route_id: str) -> pd.DataFrame:
    hourly = read_q1("q1_v3_hourly_itinerary.csv")
    route = hourly[hourly["route_id"].eq(route_id)].copy()
    rows: list[dict[str, object]] = []
    for day in range(1, 31):
        sub = route[route["day"].eq(day)]
        has_buffer = bool((sub["activity_type"] == "buffer_day").any()) or sub.empty
        travel = float(sub["travel_hours"].fillna(0).sum()) if not sub.empty else 0.0
        service = float(sub["service_hours"].fillna(0).sum()) if not sub.empty else 0.0
        lunch = float(sub.loc[sub["activity_type"].eq("lunch_break"), "duration_hours"].fillna(0).sum()) if not sub.empty else 0.0
        active = travel + service + lunch
        late = bool(sub["late_hotel"].fillna(False).astype(bool).any()) if not sub.empty else False
        heat = bool(sub["heat_avoidance_applied"].fillna(False).astype(bool).any()) or bool(sub["activity_type"].eq("heat_avoidance_wait").any()) if not sub.empty else False
        if has_buffer:
            level = "buffer"
        elif active > 10.0 or travel > 7.5 or late:
            level = "red"
        elif active > 8.5 or travel > 5.5 or heat:
            level = "yellow"
        else:
            level = "green"
        spot_names = "、".join([str(x) for x in sub.loc[sub["activity_type"].eq("visit"), "spot_name"].dropna().tolist()])
        rows.append(
            {
                "day": day,
                "travel": travel,
                "service": service,
                "lunch": lunch,
                "active": active,
                "level": level,
                "spot_names": spot_names,
            }
        )
    return pd.DataFrame(rows)


def fig_q1_04_itinerary_pressure() -> None:
    selected = read_q1("q1_v3_selected_routes.csv")
    main = selected[selected["route_id"].eq("Q1V3_Q24_120")].iloc[0]
    daily = compute_daily_load("Q1V3_Q24_120")

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    fig.subplots_adjust(top=0.78, left=0.08, right=0.98, bottom=0.17)
    figure_title(
        fig,
        "Q1：主推路线可在30天内排完，但高压集中在两次超长转场",
        "柱形分解每日交通、游览与午休时间；红/黄判定复用V3小时级排程规则，缓冲日不计活动强度。",
    )

    x = daily["day"].to_numpy()
    travel = daily["travel"].to_numpy()
    service = daily["service"].to_numpy()
    lunch = daily["lunch"].to_numpy()
    ax.bar(x, travel, color=COLORS["q1"], width=0.72, edgecolor=COLORS["white"], linewidth=0.25, label="交通/转场")
    ax.bar(x, service, bottom=travel, color="#9A9A9A", width=0.72, edgecolor=COLORS["white"], linewidth=0.25, label="游览")
    ax.bar(x, lunch, bottom=travel + service, color="#D6D6D6", width=0.72, edgecolor=COLORS["white"], linewidth=0.25, label="午休")

    level_colors = {"red": COLORS["red"], "yellow": COLORS["orange"], "green": COLORS["green"], "buffer": COLORS["light"]}
    for _, row in daily.iterrows():
        day = int(row["day"])
        if row["level"] == "buffer":
            ax.add_patch(plt.Rectangle((day - 0.36, 0), 0.72, 0.18, facecolor=COLORS["green_light"], edgecolor="none", zorder=4))
        else:
            ax.add_patch(plt.Rectangle((day - 0.36, -0.45), 0.72, 0.25, facecolor=level_colors[str(row["level"])], edgecolor="none", zorder=4))
        if row["level"] == "red":
            ax.add_patch(plt.Rectangle((day - 0.39, 0), 0.78, float(row["active"]), fill=False, edgecolor=COLORS["red"], lw=1.1, zorder=5))

    for yref, label in [(8.5, "标准活动阈值8.5h"), (10.0, "红色压力阈值10h")]:
        ax.axhline(yref, color=COLORS["axis"], lw=0.75, ls=(0, (4, 4)), zorder=0)
        ax.text(30.5, yref + 0.08, label, ha="right", va="bottom", fontproperties=FONT_CN_REG, fontsize=7.8, color=COLORS["muted"])

    red_days = daily[daily["level"].eq("red")]
    for _, row in red_days.iterrows():
        ax.text(
            int(row["day"]),
            float(row["active"]) + 0.42,
            f"D{int(row['day'])}",
            ha="center",
            va="bottom",
            fontproperties=FONT_EN_BOLD,
            fontsize=8,
            color=COLORS["red"],
        )

    summary = (
        f"模型汇总：活动/转场{int(main['active_or_transfer_days'])}天，缓冲{int(main['buffer_days'])}天，"
        f"红日{int(main['red_days'])}天，黄日{int(main['yellow_days'])}天，时间窗违规{int(main['time_window_violations'])}"
    )
    fig.text(0.08, 0.835, summary, ha="left", va="top", fontproperties=FONT_CN_REG, fontsize=8.2, color=COLORS["muted"])

    ax.set_xlim(0.3, 30.7)
    ax.set_ylim(-0.65, max(12.6, float(daily["active"].max()) + 1.15))
    ax.set_xticks(np.arange(1, 31, 1))
    ax.set_xlabel("行程日", fontproperties=FONT_CN_REG, fontsize=9)
    ax.set_ylabel("小时", fontproperties=FONT_CN_REG, fontsize=9)
    clean_axis(ax)
    ax.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
        prop=FONT_CN_REG,
        fontsize=8,
        handlelength=1.2,
        columnspacing=0.9,
    )

    save_figure(
        fig,
        "fig_q1_04_itinerary_pressure",
        "Q1主推路线逐日小时级强度条带",
        "q1_v3_hourly_itinerary.csv; q1_v3_selected_routes.csv",
        "stacked bar",
        "复算V3小时级排程红黄绿规则，展示主推路线的高压日和缓冲日。",
    )


def fig_q1_05_monte_carlo_distribution() -> None:
    trials = read_q1("q1_v3_simulation_trials.csv")
    selected = read_q1("q1_v3_selected_routes.csv")
    route_ids = ["Q1V3_Q30_279", "Q1V3_Q24_120", "Q1V3_Q20_011"]
    labels = {
        "Q1V3_Q30_279": "30覆盖候选",
        "Q1V3_Q24_120": "24运营主推",
        "Q1V3_Q20_011": "20舒适备选",
    }
    colors = {
        "Q1V3_Q30_279": "#7A7A7A",
        "Q1V3_Q24_120": COLORS["q1"],
        "Q1V3_Q20_011": COLORS["green"],
    }
    subset = trials[trials["route_id"].isin(route_ids)].copy()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 4.8), gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.32})
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.09, right=0.98)
    figure_title(
        fig,
        "Q1：运营鲁棒性已改善，但严格舒适风险主要来自红色压力日",
        "每条路线500个route-specific Monte Carlo样本；运营成功不限制红日，严格舒适要求红日≤1。",
    )

    for rid in route_ids:
        values = np.sort(subset[subset["route_id"].eq(rid)]["simulated_days"].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax_l.step(values, y, where="post", lw=1.8, color=colors[rid], label=labels[rid])
    ax_l.axvline(30, color=COLORS["axis"], lw=0.85, ls=(0, (4, 3)))
    ax_l.text(30.15, 0.08, "30天", ha="left", va="bottom", fontproperties=FONT_CN_REG, fontsize=8, color=COLORS["muted"])
    ax_l.set_xlabel("仿真完成天数", fontproperties=FONT_CN_REG, fontsize=9)
    ax_l.set_ylabel("累计概率", fontproperties=FONT_CN_REG, fontsize=9)
    ax_l.set_xlim(19, 41)
    ax_l.set_ylim(0, 1.02)
    ax_l.set_yticks(np.linspace(0, 1, 6))
    ax_l.set_yticklabels([percent(x) for x in np.linspace(0, 1, 6)])
    panel_label(ax_l, "A")
    clean_axis(ax_l)
    ax_l.legend(loc="lower right", frameon=False, prop=FONT_CN_REG, fontsize=8)

    bar_y = np.arange(len(route_ids))[::-1]
    category_colors = [COLORS["green"], COLORS["orange"], COLORS["red"]]
    category_labels = ["红日≤1", "红日2-3", "红日≥4"]
    left = np.zeros(len(route_ids))
    category_values: list[list[float]] = [[], [], []]
    for rid in route_ids:
        vals = subset[subset["route_id"].eq(rid)]["red_days"]
        category_values[0].append(float((vals <= 1).mean()))
        category_values[1].append(float(((vals >= 2) & (vals <= 3)).mean()))
        category_values[2].append(float((vals >= 4).mean()))
    for idx, (vals, color, label) in enumerate(zip(category_values, category_colors, category_labels)):
        vals_arr = np.array(vals)
        ax_r.barh(bar_y, vals_arr, left=left, height=0.52, color=color, edgecolor=COLORS["white"], linewidth=0.5, label=label)
        for yi, lv, vv in zip(bar_y, left, vals_arr):
            if vv >= 0.08:
                ax_r.text(lv + vv / 2, yi, percent(vv), ha="center", va="center", fontproperties=FONT_EN, fontsize=8, color="white" if idx == 2 else COLORS["ink"])
        left += vals_arr

    summary = selected[selected["route_id"].isin(route_ids)].set_index("route_id")
    for yi, rid in zip(bar_y, route_ids):
        row = summary.loc[rid]
        ax_r.text(
            1.02,
            yi,
            f"运营{percent(float(row['operational_success_probability']))} / 严舒{percent(float(row['strict_comfort_success_probability']))}",
            ha="left",
            va="center",
            fontproperties=FONT_CN_REG,
            fontsize=8,
            color=COLORS["muted"],
        )
    ax_r.set_yticks(bar_y)
    ax_r.set_yticklabels([labels[rid] for rid in route_ids], fontproperties=FONT_CN_REG)
    ax_r.set_xlim(0, 1.42)
    ax_r.set_xticks(np.linspace(0, 1, 6))
    ax_r.set_xticklabels([percent(x) for x in np.linspace(0, 1, 6)])
    ax_r.set_xlabel("仿真样本占比", fontproperties=FONT_CN_REG, fontsize=9)
    panel_label(ax_r, "B")
    clean_axis(ax_r, y_cn=True)
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.50, -0.22), ncol=3, frameon=False, prop=FONT_CN_REG, fontsize=8, handlelength=1.4, columnspacing=0.9)

    save_figure(
        fig,
        "fig_q1_05_monte_carlo_distribution",
        "Q1 Monte Carlo鲁棒性分布",
        "q1_v3_simulation_trials.csv; q1_v3_selected_routes.csv",
        "ECDF and stacked bar",
        "对比30景点候选、24景点主推和20景点舒适备选的完成天数与红日风险分布。",
    )


def fig_q1_06_exact_check() -> None:
    checks = read_q1("q1_v3_small_exact_check.csv")
    solved = checks[checks["status"].eq("solved_exact_ordering")].copy()
    solved["label"] = [
        "主推前15节点",
        "随机15节点子集1",
        "随机15节点子集2",
        "随机15节点子集3",
    ][: len(solved)]
    solved = solved.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(solved))

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    fig.subplots_adjust(top=0.74, left=0.23, right=0.97, bottom=0.28)
    figure_title(
        fig,
        "Q1：混合启发式排序经小规模精确校验，局部gap控制在2.65%以内",
        "Held-Karp只校验主推路线的局部排序质量，不等同于完整40景点鲁棒定向游全局最优证明。",
    )

    ax.hlines(y, 0, solved["relative_gap"], color=COLORS["light"], lw=1.3)
    colors = [COLORS["q1"] if v <= 0.02 else COLORS["orange"] for v in solved["relative_gap"]]
    ax.scatter(solved["relative_gap"], y, s=72, color=colors, edgecolors=COLORS["ink"], linewidths=0.6, zorder=3)
    for yi, (_, row) in zip(y, solved.iterrows()):
        ax.text(
            float(row["relative_gap"]) + 0.0015,
            yi,
            f"{float(row['relative_gap']) * 100:.2f}%",
            ha="left",
            va="center",
            fontproperties=FONT_EN,
            fontsize=8.4,
            color=COLORS["ink"],
        )
    missing = checks[checks["status"].ne("solved_exact_ordering")]
    note = "20节点prize-collecting MILP因当前环境缺少OR-Tools/Gurobi/CPLEX未运行，保留为后续精确校验。"
    if not missing.empty:
        note = str(missing.iloc[0]["check_scope"])
    fig.text(0.23, 0.08, note, ha="left", va="center", fontproperties=FONT_CN_REG, fontsize=7.8, color=COLORS["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels(solved["label"], fontproperties=FONT_CN_REG)
    ax.set_xlim(0, max(0.032, float(solved["relative_gap"].max()) + 0.008))
    ax.set_xlabel("相对gap", fontproperties=FONT_CN_REG, fontsize=9)
    ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xticklabels(["0%", "1%", "2%", "3%"])
    clean_axis(ax, y_cn=True)

    save_figure(
        fig,
        "fig_q1_06_exact_check",
        "Q1小规模精确校验审计",
        "q1_v3_small_exact_check.csv",
        "lollipop",
        "说明启发式主推路线通过Held-Karp局部排序校验，但不夸大全局最优性。",
    )


def write_index_and_readme() -> None:
    index = pd.DataFrame(FIGURE_ROWS)
    index_path = OUT / "q1_figure_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 第一问结果分析与可视化图组",
        "",
        "本文件夹为 Q1-V3 鲁棒多目标多模式定向游模型的论文图组。图形采用白底、无网格线、低饱和蓝灰主色，并优先表达建模结论而非装饰性总览。",
        "",
        "## 图表清单",
        "",
        "| 图号 | 文件 | 论文作用 |",
        "|---|---|---|",
    ]
    role = {
        "fig_q1_01_screening_contraction": "筛选收缩证据：用聚合图和代表方案表解释为何高覆盖路线不能直接作为推荐。",
        "fig_q1_02_route_tiers": "代表方案层级：拆分覆盖上界、覆盖候选、运营主推和舒适备选。",
        "fig_q1_03_main_route_map": "空间结构：展示24景点运营主推路线的跨区域覆盖。",
        "fig_q1_04_itinerary_pressure": "小时级可执行性：展示每日交通、游览、午休与缓冲日。",
        "fig_q1_05_monte_carlo_distribution": "鲁棒仿真：展示完成天数和红色压力日分布。",
        "fig_q1_06_exact_check": "算法可信度：展示小规模Held-Karp精确校验结果。",
    }
    for _, row in index.iterrows():
        lines.append(f"| {row['figure_id']} | `png/{row['figure_id']}.png` | {role.get(row['figure_id'], row['design_note'])} |")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- `operational_success_probability` 表示运营完成成功率，不限制红色压力日。",
            "- `strict_comfort_success_probability` 在运营成功基础上要求红色压力日不超过1天。",
            "- 主推路线 `Q1V3_Q24_120` 是运营鲁棒主推，不是完全舒适主推；其严格舒适成功率仍为0。",
            "- `fig_q1_04_itinerary_pressure` 的红/黄/绿判定复用 V3 小时级排程阈值：活动时长>10小时、交通>7.5小时或晚到为红日；活动>8.5小时、交通>5.5小时或高温避让为黄日。",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs(clean=True)
    setup_theme()
    fig_q1_01_screening_contraction()
    fig_q1_02_route_tiers()
    fig_q1_03_main_route_map()
    fig_q1_04_itinerary_pressure()
    fig_q1_05_monte_carlo_distribution()
    fig_q1_06_exact_check()
    write_index_and_readme()
    print(f"Wrote {len(FIGURE_ROWS)} Q1 figures to {OUT}")


if __name__ == "__main__":
    main()
