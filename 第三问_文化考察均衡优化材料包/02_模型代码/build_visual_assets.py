# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
OUT_DIR = ROOT / "visual_assets_outputs"
FIG_DIR = OUT_DIR / "figures"
OUTPUT_DIR = ROOT / "outputs"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


COLORS = {
    "blue": "#2f6f9f",
    "teal": "#2a9d8f",
    "green": "#6a994e",
    "yellow": "#e9c46a",
    "orange": "#f4a261",
    "red": "#c44536",
    "gray": "#6c757d",
    "light": "#f3f6f8",
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
    return pd.read_csv(path, encoding="utf-8-sig")


def save_fig(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_value_labels(ax, values, fmt="{:.0f}", dy=0.01):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for i, value in enumerate(values):
        ax.text(i, value + span * dy, fmt.format(value), ha="center", va="bottom", fontsize=9, color="#263238")


def plot_model_stack() -> Path:
    layers = [
        ("数据层", "景点/交通/住宿/容量/费用"),
        ("确定性优化", "PCOP / 路径覆盖 / MinMax / 容量流"),
        ("元启发式", "ACO + ALNS + SA + 2-opt"),
        ("人性化层", "疲劳凸惩罚 / 游客画像 / 转场拆日"),
        ("仿真层", "价格 / 延期 / 需求 / 预约扰动"),
        ("策略层", "多口岸阈值 / 缓冲天 / 预约上限 / 分流"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.axis("off")
    y_positions = np.arange(len(layers))[::-1]
    palette = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["yellow"], COLORS["orange"], COLORS["red"]]
    for i, ((title, text), y) in enumerate(zip(layers, y_positions)):
        ax.add_patch(plt.Rectangle((0.05, y), 0.9, 0.72, color=palette[i], alpha=0.88, transform=ax.transData))
        ax.text(0.09, y + 0.46, title, color="white", fontsize=15, fontweight="bold", va="center")
        ax.text(0.28, y + 0.46, text, color="white", fontsize=12, va="center")
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.5, y - 0.03), xytext=(0.5, y - 0.27), arrowprops=dict(arrowstyle="->", color="#455a64", lw=1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, len(layers))
    ax.set_title("新疆旅游线路安排：分层决策系统", fontsize=18, fontweight="bold", pad=16)
    return save_fig(fig, "fig01_model_stack.png")


def plot_q1_persona() -> Path:
    df = read_csv(ROOT / "adaptive_strategy_outputs" / "q1_variant_summary.csv")
    labels = {
        "explorer_dense": "探索型",
        "standard_active": "普通型",
        "family_comfort": "亲子型",
        "senior_slow": "长者型",
    }
    df["label"] = df["persona_id"].map(labels)
    df["mean_comfort_score"] = df["mean_comfort_score"].map(num)
    df["red_days"] = df["red_days"].map(num)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(df))
    bars = ax1.bar(x, df["mean_comfort_score"], color=[COLORS["green"], COLORS["teal"], COLORS["yellow"], COLORS["orange"]], width=0.58)
    ax1.axhline(85, color=COLORS["gray"], linestyle="--", lw=1.2, label="舒适度参考线 85")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("平均舒适度")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"])
    add_value_labels(ax1, df["mean_comfort_score"], "{:.1f}", 0.015)
    ax2 = ax1.twinx()
    ax2.plot(x, df["red_days"], color=COLORS["red"], marker="o", lw=2.2, label="红色压力日")
    ax2.set_ylim(0, max(6, df["red_days"].max() + 1))
    ax2.set_ylabel("红色压力日")
    for i, value in enumerate(df["red_days"]):
        ax2.text(i, value + 0.15, f"{int(value)}天", ha="center", color=COLORS["red"], fontsize=9)
    ax1.set_title("第一问：不同游客画像下的路线舒适度", fontsize=16, fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.22)
    return save_fig(fig, "fig02_q1_persona_comfort.png")


def plot_q2_gateway() -> Path:
    df = read_csv(ROOT / "policy_simulation_outputs" / "q2_gateway_price_summary.csv")
    label_map = {
        "optimistic_multicity": "乐观多口岸",
        "balanced_multicity": "均衡情景",
        "peak_summer_multicity": "暑期高峰",
    }
    df["label"] = df["scenario_id"].map(label_map)
    df["prob"] = df["prob_open_gateway_cheaper"].map(num) * 100
    df["savings"] = df["expected_savings_if_choose_open_yuan_for_two"].map(num)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(df))
    ax1.bar(x, df["prob"], color=[COLORS["green"], COLORS["yellow"], COLORS["red"]], width=0.58)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("开放式多口岸更便宜概率(%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"])
    ax1.axhline(50, color=COLORS["gray"], linestyle="--", lw=1)
    add_value_labels(ax1, df["prob"], "{:.1f}%", 0.015)
    ax2 = ax1.twinx()
    ax2.plot(x, df["savings"], color=COLORS["blue"], marker="s", lw=2.2)
    ax2.axhline(0, color="#263238", lw=1)
    ax2.set_ylabel("选择开放式的期望节省(元/两人)")
    for i, value in enumerate(df["savings"]):
        ax2.text(i, value + (60 if value >= 0 else -120), f"{value:.0f}", ha="center", color=COLORS["blue"], fontsize=9)
    ax1.set_title("第二问：多口岸外部票价不确定性", fontsize=16, fontweight="bold")
    ax1.grid(axis="y", alpha=0.22)
    return save_fig(fig, "fig03_q2_gateway_probability.png")


def plot_q3_buffer() -> Path:
    df = read_csv(ROOT / "policy_simulation_outputs" / "q3_project_buffer_curve.csv")
    df["buffer_days"] = df["buffer_days"].map(num)
    df["prob"] = df["all_groups_on_time_probability"].map(num) * 100
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(df["buffer_days"], df["prob"], color=COLORS["teal"], lw=2.6, marker="o")
    ax.axhline(90, color=COLORS["red"], linestyle="--", lw=1.4, label="90%可靠性目标")
    ax.axvline(4, color=COLORS["orange"], linestyle="--", lw=1.4, label="建议缓冲4天")
    ax.set_xlabel("缓冲天数")
    ax.set_ylabel("三组同时按期完成概率(%)")
    ax.set_ylim(0, 105)
    ax.set_title("第三问：文化专项考察缓冲天数仿真", fontsize=16, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend()
    for _, row in df.iterrows():
        if row["buffer_days"] in [0, 2, 4, 5]:
            ax.text(row["buffer_days"], row["prob"] + 3, f"{row['prob']:.1f}%", ha="center", fontsize=9)
    return save_fig(fig, "fig04_q3_buffer_curve.png")


def plot_q4_demand() -> Path:
    df = read_csv(ROOT / "adaptive_strategy_outputs" / "q4_reallocation_summary.csv")
    for col in ["demand_multiplier", "requested_visitors", "served_after_reallocation", "unresolved_overflow_visitors"]:
        df[col] = df[col].map(num)
    labels = [f"{m:.0%}" for m in df["demand_multiplier"]]
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width / 2, df["served_after_reallocation"], width, color=COLORS["teal"], label="分流后接待")
    ax.bar(x + width / 2, df["unresolved_overflow_visitors"], width, color=COLORS["red"], label="未消化溢出")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("游客人数")
    ax.set_title("第四问：需求上浮后的接待与溢出", fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    for i, value in enumerate(df["unresolved_overflow_visitors"]):
        ax.text(i + width / 2, value + 700, f"{int(value)}", ha="center", fontsize=9, color=COLORS["red"])
    return save_fig(fig, "fig05_q4_demand_reallocation.png")


def plot_q4_reservation() -> Path:
    df = read_csv(ROOT / "policy_simulation_outputs" / "q4_reservation_policy_summary.csv")
    df = df[df["demand_multiplier"].map(num) == 1.10].copy()
    label_map = {
        "full_capacity": "100%满容量",
        "safety_cap_95pct": "95%安全上限",
        "comfort_cap_90pct": "90%舒适上限",
    }
    df["label"] = df["policy_id"].map(label_map)
    for col in ["served_visitors", "waitlist_or_rejected_visitors"]:
        df[col] = df[col].map(num)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x, df["served_visitors"], color=COLORS["blue"], label="接待")
    ax.bar(x, df["waitlist_or_rejected_visitors"], bottom=df["served_visitors"], color=COLORS["orange"], label="等待/拒绝")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("游客人数")
    ax.set_title("第四问：需求上浮10%时的预约上限策略", fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        ax.text(idx, row["served_visitors"] / 2, f"{int(row['served_visitors'])}", ha="center", va="center", color="white", fontsize=10)
        ax.text(idx, row["served_visitors"] + row["waitlist_or_rejected_visitors"] + 900, f"{int(row['waitlist_or_rejected_visitors'])}", ha="center", fontsize=9, color=COLORS["orange"])
    return save_fig(fig, "fig06_q4_reservation_policy.png")


def plot_q2_cost_days() -> Path:
    df = read_csv(ROOT / "problem2_openpath_outputs" / "scenario_totals.csv")
    df["cost"] = df["total_transport_cost_yuan_for_two"].map(num)
    df["days"] = df["total_estimated_days"].map(num)
    labels = ["乌鲁木齐起讫", "开放式下界"]
    fig, ax1 = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(len(df))
    ax1.bar(x, df["cost"], color=[COLORS["blue"], COLORS["green"]], width=0.5)
    ax1.set_ylabel("新疆境内交通费(元/两人)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    add_value_labels(ax1, df["cost"], "{:.0f}", 0.015)
    ax2 = ax1.twinx()
    ax2.plot(x, df["days"], color=COLORS["red"], marker="o", lw=2.2)
    ax2.set_ylabel("估算总天数")
    for i, value in enumerate(df["days"]):
        ax2.text(i, value + 0.6, f"{int(value)}天", ha="center", color=COLORS["red"], fontsize=10)
    ax1.set_title("第二问：乌鲁木齐起讫与开放式下界对比", fontsize=16, fontweight="bold")
    ax1.grid(axis="y", alpha=0.22)
    return save_fig(fig, "fig07_q2_cost_days_comparison.png")


def build_visual_index(paths: list[tuple[str, str, Path, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "title": title,
                "path": str(path).replace("\\", "/"),
                "recommended_use": use,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for figure_id, title, path, use in paths
        ]
    )


def write_report(index: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 新疆旅游PPT可视化图表包",
        "",
        "本包将前面模型和仿真实验转成可直接放进PPT/论文的PNG图。所有图由当前CSV结果自动生成，避免手工改数。",
        "",
        index.to_markdown(index=False),
        "",
        "建议使用方式：",
        "",
        "1. 答辩第3页放模型分层图。",
        "2. 第一问页放游客画像舒适度图。",
        "3. 第二问页放费用-天数对比图和多口岸胜率图。",
        "4. 第三问页放缓冲可靠性曲线。",
        "5. 第四问页放需求冲击图和预约上限图。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    figures = [
        ("图1", "模型分层架构", plot_model_stack(), "PPT第3页/论文方法框架"),
        ("图2", "第一问游客画像舒适度", plot_q1_persona(), "第一问强化结果"),
        ("图3", "第二问多口岸胜率", plot_q2_gateway(), "第二问政策仿真"),
        ("图4", "第三问缓冲可靠性", plot_q3_buffer(), "第三问风险仿真"),
        ("图5", "第四问需求冲击", plot_q4_demand(), "第四问容量策略"),
        ("图6", "第四问预约上限策略", plot_q4_reservation(), "第四问运营策略"),
        ("图7", "第二问费用天数对比", plot_q2_cost_days(), "第二问主结果对比"),
    ]
    index = build_visual_index(figures)
    write_path = OUT_DIR / "visual_index.csv"
    index.to_csv(write_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_DIR / "新疆旅游PPT可视化图表索引.xlsx", engine="openpyxl") as writer:
        index.to_excel(writer, index=False, sheet_name="visual_index")
    write_report(index, OUTPUT_DIR / "新疆旅游PPT可视化图表包.md")
    summary = {
        "figures": len(index),
        "index": "visual_assets_outputs/visual_index.csv",
        "report": "outputs/新疆旅游PPT可视化图表包.md",
        "workbook": "outputs/新疆旅游PPT可视化图表索引.xlsx",
    }
    (OUT_DIR / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
