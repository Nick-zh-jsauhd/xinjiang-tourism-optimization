from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "01_visual_data"


def find_file(name: str) -> Path:
    matches = [p for p in ROOT.rglob(name) if ".git" not in p.parts]
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(find_file(name), encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def pick_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def parse_mode_hours(value: object) -> dict[str, float]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    result: dict[str, float] = {}
    for part in text.split(";"):
        if ":" in part:
            key, raw = part.split(":", 1)
            try:
                result[key.strip()] = float(raw)
            except ValueError:
                continue
    return result


def build_overview() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "Q1",
                "decision_subject": "王先生夫妇",
                "time_scale": "30天暑期自由行",
                "final_model": "Q1-V3 鲁棒多目标多模式定向游",
                "visual_takeaway": "32景点是覆盖上界，24景点是运营鲁棒主推",
                "core_metric": "24景点、5天缓冲、运营成功率0.940",
            },
            {
                "question": "Q2",
                "decision_subject": "王先生夫妇两年暑假",
                "time_scale": "两个暑假",
                "final_model": "Q2-V3 两年鲁棒多模式路径覆盖与境内交通费用最小化",
                "visual_takeaway": "开放多口岸可降境内费用，但需看外部大交通差价阈值",
                "core_metric": "鲁棒主推约3099.91元，阈值1170.91元",
            },
            {
                "question": "Q3",
                "decision_subject": "少数民族研究所三组考察队",
                "time_scale": "并行文化考察",
                "final_model": "Q3-V2 鲁棒三团队文化考察MinMax调度",
                "visual_takeaway": "文化考察按任务完成时间均衡，不按景点数均分",
                "core_metric": "最大99.19小时，固定政策空间exact gap=0",
            },
            {
                "question": "Q4",
                "decision_subject": "自治区文旅部门",
                "time_scale": "五一12天",
                "final_model": "Q4-V2 线路产品组合与分时容量优化",
                "visual_takeaway": "容量够不等于质量好，必须看分时与多资源瓶颈",
                "core_metric": "63候选、18入选、160190人投放容量",
            },
        ]
    )


def build_q1() -> None:
    selected = read_csv("q1_v3_selected_routes.csv")
    hourly = read_csv("q1_v3_hourly_itinerary.csv")

    role_order = {
        "极限覆盖版": 1,
        "30景点均衡覆盖候选版": 2,
        "运营鲁棒主推版": 3,
        "严格舒适主推备选": 4,
        "亲子舒适版": 5,
        "长者慢游版": 6,
    }
    tiers = pick_columns(
        selected,
        [
            "selected_role",
            "route_id",
            "spots_count",
            "planned_trip_days",
            "active_or_transfer_days",
            "buffer_days",
            "red_days",
            "yellow_days",
            "time_window_violations",
            "schedule_strict_feasible",
            "operational_success_probability",
            "strict_comfort_success_probability",
            "mean_comfort_score",
            "total_cost_yuan_excluding_meals",
            "selection_status",
        ],
    )
    tiers["tier_order"] = tiers["selected_role"].map(role_order).fillna(99).astype(int)
    tiers["visual_role_note"] = tiers["selected_role"].map(
        {
            "极限覆盖版": "覆盖上界，不作为现实执行方案",
            "30景点均衡覆盖候选版": "高覆盖候选，未达到严格鲁棒阈值",
            "运营鲁棒主推版": "运营鲁棒主推，不等于严格舒适主推",
            "严格舒适主推备选": "更保守，严格舒适成功率仍未达80%",
            "亲子舒适版": "亲子扩展方案",
            "长者慢游版": "长者扩展方案",
        }
    )
    tiers = tiers.sort_values("tier_order")
    write_csv(tiers, "q1_visual_route_tiers.csv")

    success_rows = []
    for _, row in tiers.iterrows():
        for key, label in [
            ("operational_success_probability", "运营成功率"),
            ("strict_comfort_success_probability", "严格舒适成功率"),
        ]:
            success_rows.append(
                {
                    "selected_role": row["selected_role"],
                    "route_id": row["route_id"],
                    "spots_count": row["spots_count"],
                    "buffer_days": row["buffer_days"],
                    "metric_key": key,
                    "metric_label": label,
                    "probability": row[key],
                }
            )
    write_csv(pd.DataFrame(success_rows), "q1_visual_success_comparison.csv")

    selected_ids = set(tiers["route_id"])
    daily = hourly[hourly["route_id"].isin(selected_ids)].copy()
    daily["travel_hours"] = pd.to_numeric(daily.get("travel_hours", 0), errors="coerce").fillna(0)
    daily["service_hours"] = pd.to_numeric(daily.get("service_hours", 0), errors="coerce").fillna(0)
    daily["duration_hours"] = pd.to_numeric(daily.get("duration_hours", 0), errors="coerce").fillna(0)
    grouped = (
        daily.groupby(["route_id", "day"], as_index=False)
        .agg(
            travel_hours=("travel_hours", "sum"),
            service_hours=("service_hours", "sum"),
            total_duration_hours=("duration_hours", "sum"),
            has_buffer=("activity_type", lambda s: int((s == "buffer_day").any())),
            has_late_hotel=("late_hotel", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() > 0)),
            has_heat_avoidance=("heat_avoidance_applied", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() > 0)),
            has_recovery=("recovery_after_long_transfer", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).max() > 0)),
            spot_names=("spot_name", lambda s: "、".join([x for x in s.dropna().astype(str).unique() if x and x != "nan"][:4])),
        )
    )
    grouped["active_hours_proxy"] = grouped["travel_hours"] + grouped["service_hours"]

    def pressure(row: pd.Series) -> str:
        if row["has_buffer"] and row["active_hours_proxy"] <= 0.5:
            return "buffer"
        if row["active_hours_proxy"] >= 9 or row["travel_hours"] >= 7 or row["has_late_hotel"]:
            return "red"
        if row["active_hours_proxy"] >= 7 or row["travel_hours"] >= 5 or row["has_heat_avoidance"] or row["has_recovery"]:
            return "yellow"
        return "green"

    grouped["pressure_level"] = grouped.apply(pressure, axis=1)
    grouped = grouped.merge(tiers[["route_id", "selected_role", "spots_count"]], on="route_id", how="left")
    write_csv(grouped.sort_values(["selected_role", "day"]), "q1_visual_daily_pressure.csv")


def build_q2() -> None:
    selected = read_csv("q2_v3_selected_plans.csv")
    segments = read_csv("q2_v3_route_segments.csv")
    thresholds = read_csv("q2_v3_gateway_thresholds.csv")

    roles = [
        "ROOTED_URUMQI_MAIN",
        "OPEN_GATEWAY_MIN_INTRA_COST",
        "ROBUST_TWO_YEAR_MAIN",
        "COMFORT_BALANCED_BACKUP",
    ]
    compare = selected[selected["selected_role"].isin(roles)].copy()
    compare["visual_role_label"] = compare["selected_role"].map(
        {
            "ROOTED_URUMQI_MAIN": "固定乌鲁木齐起讫",
            "OPEN_GATEWAY_MIN_INTRA_COST": "开放多口岸最低费用",
            "ROBUST_TWO_YEAR_MAIN": "开放多口岸鲁棒主推",
            "COMFORT_BALANCED_BACKUP": "舒适均衡备选",
        }
    )
    compare["plot_order"] = compare["selected_role"].map({role: i + 1 for i, role in enumerate(roles)})
    write_csv(
        pick_columns(
            compare,
            [
                "plot_order",
                "selected_role",
                "visual_role_label",
                "plan_id",
                "gateway_policy",
                "covered_spots",
                "year1_days",
                "year2_days",
                "year_day_difference",
                "total_intra_transport_cost",
                "external_premium_break_even",
                "operational_success_probability",
                "strict_comfort_success_probability",
                "red_days",
                "time_window_violations",
                "year1_entry_gateway",
                "year1_exit_gateway",
                "year2_entry_gateway",
                "year2_exit_gateway",
            ],
        ).sort_values("plot_order"),
        "q2_visual_plan_compare.csv",
    )

    balance = []
    for _, row in compare.iterrows():
        for year in [1, 2]:
            balance.append(
                {
                    "selected_role": row["selected_role"],
                    "visual_role_label": row["visual_role_label"],
                    "plan_id": row["plan_id"],
                    "year": f"Year {year}",
                    "days": row[f"year{year}_days"],
                    "spots": row[f"year{year}_spots"],
                    "transport_cost": row[f"year{year}_transport_cost"],
                }
            )
    write_csv(pd.DataFrame(balance), "q2_visual_year_balance.csv")
    write_csv(thresholds, "q2_visual_gateway_threshold.csv")

    keep_roles = roles[:3]
    plan_role = compare[compare["selected_role"].isin(keep_roles)][["selected_role", "visual_role_label", "plan_id"]]
    mode_rows = []
    for _, plan in plan_role.iterrows():
        subset = segments[segments["plan_id"] == plan["plan_id"]]
        by_mode = (
            subset.groupby(["year", "mode"], as_index=False)
            .agg(
                segment_count=("segment_order", "count"),
                time_hours=("time_hours", "sum"),
                cost_yuan_for_two=("cost_yuan_for_two", "sum"),
            )
        )
        by_mode["selected_role"] = plan["selected_role"]
        by_mode["visual_role_label"] = plan["visual_role_label"]
        by_mode["plan_id"] = plan["plan_id"]
        mode_rows.append(by_mode)
    mode_mix = pd.concat(mode_rows, ignore_index=True) if mode_rows else pd.DataFrame()
    write_csv(mode_mix, "q2_visual_mode_mix.csv")


def build_q3() -> None:
    routes = read_csv("q3_v2_assignment_routes.csv")
    buffer = read_csv("q3_v2_buffer_policy.csv")
    exact = read_csv("q3_v2_exact_check.csv")
    resources = read_csv("q3_v2_resource_usage_calendar.csv")

    group_summary = pick_columns(
        routes,
        [
            "group_id",
            "group_role",
            "spots_count",
            "approval_spots_count",
            "remote_or_approval_count",
            "travel_hours",
            "fieldwork_hours",
            "safety_buffer_hours",
            "service_hours",
            "total_hours",
            "days",
            "risk_score",
            "culture_value_total",
            "route_sequence",
            "max_completion_hours",
        ],
    )
    group_summary["group_label"] = group_summary["group_id"].map(
        {1: "Group1 专项审批组", 2: "Group2 标准文化线", 3: "Group3 标准文化线"}
    )
    group_summary["balance_gap_to_max_hours"] = group_summary["max_completion_hours"] - group_summary["total_hours"]
    write_csv(group_summary, "q3_visual_group_summary.csv")

    policy_order = {
        "buffer_4d": 1,
        "buffer_5d": 2,
        "mobile_team_4d": 3,
        "mobile_team_5d": 4,
        "split_or_postpone_7d": 5,
    }
    buffer["policy_order"] = buffer["policy_id"].map(policy_order).fillna(99).astype(int)
    write_csv(buffer.sort_values("policy_order"), "q3_visual_buffer_policy.csv")
    long_rows = []
    for _, row in buffer.iterrows():
        for key, label in [
            ("weighted_success_probability", "加权成功率"),
            ("worst_success_probability", "最弱场景成功率"),
        ]:
            long_rows.append(
                {
                    "policy_id": row["policy_id"],
                    "policy_name": row["policy_name"],
                    "strategy_tier": row["strategy_tier"],
                    "selection_status": row["selection_status"],
                    "policy_order": row["policy_order"],
                    "metric_key": key,
                    "metric_label": label,
                    "success_probability": row[key],
                }
            )
    write_csv(pd.DataFrame(long_rows), "q3_visual_buffer_policy_long.csv")
    write_csv(exact, "q3_visual_exact_check_card.csv")

    resource_calendar = (
        resources.groupby(["day", "resource_type", "group_id"], as_index=False)
        .agg(resource_count=("resource_id", "nunique"), spot_names=("spot_name", lambda s: "、".join(s.dropna().astype(str).unique())))
    )
    write_csv(resource_calendar, "q3_visual_resource_calendar.csv")


def build_q4() -> None:
    summary = json.loads(find_file("q4_v2_run_summary.json").read_text(encoding="utf-8"))
    quality = read_csv("q4_v2_route_quality_audit.csv")
    load = read_csv("q4_v2_spot_timeslot_load.csv")
    shadow = read_csv("q4_v2_bottleneck_shadow_prices.csv")
    scenario = read_csv("q4_v2_scenario_simulation_summary.csv")
    allocation = read_csv("q4_v2_capacity_ratio_allocation.csv")

    capacity = pd.DataFrame(
        [
            {
                "item": "旧9线路容量",
                "capacity_persons": summary["legacy_9route_capacity_persons_12day"],
                "item_type": "legacy_capacity",
            },
            {
                "item": "Q4-V2 18线路容量",
                "capacity_persons": summary["selected_capacity_persons_12day"],
                "item_type": "q4v2_capacity",
            },
            {
                "item": "基准需求",
                "capacity_persons": summary["baseline_demand"],
                "item_type": "baseline_demand",
            },
            {
                "item": "复合极端补运力可接待",
                "capacity_persons": summary["extreme_add_capacity_served"],
                "item_type": "extreme_add_capacity_served",
            },
        ]
    )
    write_csv(capacity, "q4_visual_capacity_compare.csv")

    quality = quality.copy()
    quality["quality_status"] = quality["quality_pass"].map({True: "可直接投放", False: "需微调或储备"})
    quality["quality_order"] = quality["quality_pass"].map({True: 1, False: 2})
    quality = quality.sort_values(["quality_order", "max_day_active_hours", "red_days"], ascending=[True, False, False])
    write_csv(quality, "q4_visual_quality_audit.csv")

    load = load.copy()
    slot_order = {"morning": 1, "afternoon": 2, "evening": 3}
    load["slot_order"] = load["time_slot"].map(slot_order).fillna(99).astype(int)
    load["day_slot"] = "D" + load["day_index"].astype(str).str.zfill(2) + "_" + load["time_slot"].astype(str)
    load["pressure_bucket"] = pd.cut(
        load["utilization"],
        bins=[-1, 0.75, 0.90, 1.0, 10],
        labels=["低于75%", "75%-90%", "90%-100%", "超过100%"],
    )
    write_csv(load.sort_values(["day_index", "slot_order", "spot_name"]), "q4_visual_timeslot_pressure.csv")

    scenario = scenario.copy()
    scenario["strict_policy_label"] = scenario["strict_policy_pass"].map({True: "严格通过", False: "严格未通过"})
    scenario["scenario_order"] = scenario["demand_multiplier"].rank(method="dense").astype(int)
    policy_order = {
        "legacy_9_full_release": 1,
        "q4v2_full_release": 2,
        "q4v2_safety_cap_95": 3,
        "q4v2_comfort_cap_90": 4,
        "q4v2_staggered_prebooking": 5,
        "q4v2_add_capacity_plus_stagger": 6,
    }
    scenario["policy_order"] = scenario["policy_id"].map(policy_order).fillna(99).astype(int)
    write_csv(scenario.sort_values(["scenario_order", "policy_order"]), "q4_visual_strategy_matrix.csv")

    shadow_top = shadow.sort_values("shadow_price_index", ascending=False).head(10).copy()
    write_csv(shadow_top, "q4_visual_shadow_top10.csv")

    allocation_cols = [
        "strict_capacity_ratio_visitors",
        "preference_elastic_visitors",
        "balanced_optimized_visitors",
    ]
    alloc_long = allocation.melt(
        id_vars=[
            "route_id",
            "route_theme",
            "product_type",
            "route_capacity_persons_12day",
            "comfort_score",
            "low_pressure_score",
        ],
        value_vars=allocation_cols,
        var_name="allocation_metric",
        value_name="visitors",
    )
    alloc_long["allocation_label"] = alloc_long["allocation_metric"].map(
        {
            "strict_capacity_ratio_visitors": "题面容量比例分配",
            "preference_elastic_visitors": "现实偏好弹性分配",
            "balanced_optimized_visitors": "运营平衡优化分配",
        }
    )
    write_csv(alloc_long, "q4_visual_allocation_comparison_long.csv")


def build_manifest() -> None:
    rows = []
    for p in sorted(OUT.glob("*.csv")):
        if p.name == "visual_data_manifest.csv":
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        rows.append(
            {
                "file_name": p.name,
                "rows": len(df),
                "columns": len(df.columns),
                "path": str(p.relative_to(ROOT)),
            }
        )
    write_csv(pd.DataFrame(rows), "visual_data_manifest.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(build_overview(), "overview_model_evolution.csv")
    build_q1()
    build_q2()
    build_q3()
    build_q4()
    build_manifest()
    print(f"visual data built: {OUT}")


if __name__ == "__main__":
    main()

