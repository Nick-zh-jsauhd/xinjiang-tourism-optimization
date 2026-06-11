from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR.resolve()))

import build_daily_itinerary_layer as base  # noqa: E402


DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
MODEL_DIR = ROOT / "enhanced_model_outputs"
ITINERARY_DIR = ROOT / "itinerary_outputs"
REPAIR_DIR = ROOT / "itinerary_repair_outputs"
OUTPUT_DIR = ROOT / "outputs"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def route_score(route_id: str, sequence: list[str], ctx: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    seg = base.schedule_route(
        route_id,
        " -> ".join(sequence),
        ctx["spots"],
        ctx["time_windows"],
        ctx["special"],
        ctx["od_map"],
        ctx["depot_map"],
    )
    days = base.build_day_summary(seg, ctx["hotels"])
    timew = base.build_time_window_feasibility(seg)
    lodging = base.build_lodging_feasibility(days)
    time_window_violations = int((timew["time_window_feasible"] == False).sum())  # noqa: E712
    over_budget_days = int(days["active_budget_status"].eq("over_budget").sum())
    long_transfer_days = int(days["notes"].fillna("").str.contains("跨区长距离|超过8小时", regex=True).sum())
    limited_lodging = int(((~lodging["lodging_feasible"]) & lodging["requires_hotel_night"]).sum())
    scheduled_days = int(days["day"].max()) if not days.empty else 0
    active_hours = float(days["day_active_hours"].sum()) if not days.empty else 0.0
    score = (
        time_window_violations * 10000
        + over_budget_days * 400
        + long_transfer_days * 120
        + limited_lodging * 80
        + scheduled_days * 10
        + active_hours * 0.1
    )
    return score, {
        "segments": seg,
        "days": days,
        "timew": timew,
        "lodging": lodging,
        "time_window_violations": time_window_violations,
        "over_budget_days": over_budget_days,
        "long_transfer_days": long_transfer_days,
        "limited_lodging": limited_lodging,
        "scheduled_days": scheduled_days,
        "active_hours": active_hours,
    }


def local_order_repair(route_id: str, original_sequence: list[str], ctx: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    sequence = original_sequence[:]
    actions: list[dict[str, Any]] = []
    current_score, current_eval = route_score(route_id, sequence, ctx)
    spot_by_name = {r["spot_name"]: r.to_dict() for _, r in ctx["spots"].iterrows()}

    def region_compatible(candidate_sequence: list[str], target_name: str, target_index: int) -> bool:
        target = spot_by_name.get(target_name)
        if target is None:
            return False
        neighbors = []
        if target_index > 0:
            neighbors.append(candidate_sequence[target_index - 1])
        if target_index + 1 < len(candidate_sequence):
            neighbors.append(candidate_sequence[target_index + 1])
        for name in neighbors:
            neighbor = spot_by_name.get(name)
            if neighbor is None:
                continue
            if neighbor.get("hub_name") == target.get("hub_name"):
                return True
            if neighbor.get("region_cluster") == target.get("region_cluster"):
                return True
        return False

    for _ in range(6):
        violations = current_eval["timew"][current_eval["timew"]["time_window_feasible"] == False]  # noqa: E712
        if violations.empty:
            break
        best_candidate = None
        for _, issue in violations.iterrows():
            target_name = issue["to_name"]
            if target_name not in sequence:
                continue
            idx = sequence.index(target_name)
            for new_idx in range(max(0, idx - 2), min(len(sequence), idx + 3)):
                if new_idx == idx:
                    continue
                candidate = sequence[:]
                item = candidate.pop(idx)
                candidate.insert(new_idx, item)
                if not region_compatible(candidate, target_name, new_idx):
                    continue
                cand_score, cand_eval = route_score(route_id, candidate, ctx)
                if cand_score + 1e-9 < current_score:
                    best_candidate = (cand_score, cand_eval, candidate, target_name, idx, new_idx)
        if best_candidate is None:
            break
        new_score, new_eval, new_sequence, target_name, old_idx, new_idx = best_candidate
        actions.append(
            {
                "route_id": route_id,
                "repair_type": "local_order_move",
                "target_spot": target_name,
                "from_position": old_idx + 1,
                "to_position": new_idx + 1,
                "score_before": round(current_score, 3),
                "score_after": round(new_score, 3),
                "time_window_violations_before": current_eval["time_window_violations"],
                "time_window_violations_after": new_eval["time_window_violations"],
                "over_budget_days_before": current_eval["over_budget_days"],
                "over_budget_days_after": new_eval["over_budget_days"],
                "action_status": "applied",
                "rationale": "局部移动闭园较早或到达过晚的景点，优先消除时间窗冲突。",
            }
        )
        sequence = new_sequence
        current_score, current_eval = new_score, new_eval
    return sequence, pd.DataFrame(actions)


def build_spot_context(spots: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {r["spot_id"]: r.to_dict() for _, r in spots.iterrows()}


def candidate_hotel_repairs(
    route_id: str,
    segments: pd.DataFrame,
    lodging: pd.DataFrame,
    spots: pd.DataFrame,
    hotels: pd.DataFrame,
    od_map: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    spot_info = build_spot_context(spots)
    hub_spots = spots.groupby("hub_name")["spot_id"].apply(list).to_dict()
    standard_hubs = hotels[hotels["is_hotel_hub"].astype(bool)]["hub_name"].tolist()
    standard_hubs = [h for h in standard_hubs if h in hub_spots]
    rows: list[dict[str, Any]] = []

    for _, l in lodging.iterrows():
        if not bool(l["requires_hotel_night"]) or bool(l["lodging_feasible"]):
            continue
        day = int(l["day"])
        day_seg = segments[(segments["route_id"].eq(route_id)) & (segments["day"].eq(day))]
        next_seg = segments[(segments["route_id"].eq(route_id)) & (segments["day"].eq(day + 1))]
        if day_seg.empty:
            continue
        current_sid = str(day_seg.iloc[-1]["spot_id"]).strip()
        next_sid = str(next_seg.iloc[0]["spot_id"]).strip() if not next_seg.empty else ""
        if not current_sid or current_sid == "nan" or current_sid not in spot_info:
            continue
        current_hub = spot_info[current_sid]["hub_name"]
        current_region = spot_info[current_sid]["region_cluster"]

        candidates = []
        for hub in standard_hubs:
            best_rep = None
            for rep_sid in hub_spots[hub]:
                to_rep = od_map.get((current_sid, rep_sid))
                if to_rep is None:
                    continue
                from_rep = od_map.get((rep_sid, next_sid), {"time": 0.0, "cost": 0.0, "risk": 0.0}) if next_sid and next_sid in spot_info else {"time": 0.0, "cost": 0.0, "risk": 0.0}
                direct = od_map.get((current_sid, next_sid), {"time": 0.0, "cost": 0.0, "risk": 0.0}) if next_sid and next_sid in spot_info else {"time": 0.0, "cost": 0.0, "risk": 0.0}
                detour_time = base.num(to_rep["time"]) + base.num(from_rep["time"]) - base.num(direct["time"])
                detour_cost = base.num(to_rep["cost"]) + base.num(from_rep["cost"]) - base.num(direct["cost"])
                same_region_bonus = -0.75 if spot_info[rep_sid]["region_cluster"] == current_region else 0.0
                score = detour_time + max(0.0, detour_cost) / 500.0 + same_region_bonus
                cand = {
                    "hub": hub,
                    "representative_spot_id": rep_sid,
                    "representative_spot_name": spot_info[rep_sid]["spot_name"],
                    "transfer_to_hotel_hours": base.num(to_rep["time"]),
                    "transfer_to_hotel_cost_yuan_per_two": base.num(to_rep["cost"]),
                    "next_day_return_or_detour_hours": max(0.0, detour_time),
                    "detour_cost_yuan_per_two": max(0.0, detour_cost),
                    "score": score,
                }
                if best_rep is None or cand["score"] < best_rep["score"]:
                    best_rep = cand
            if best_rep is not None:
                candidates.append(best_rep)
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda x: x["score"])[0]
        rows.append(
            {
                "route_id": route_id,
                "day": day,
                "base_lodging_hub": current_hub,
                "recommended_lodging_hub": chosen["hub"],
                "representative_hotel_area_spot": chosen["representative_spot_name"],
                "transfer_to_hotel_hours": round(chosen["transfer_to_hotel_hours"], 3),
                "transfer_to_hotel_cost_yuan_per_two": round(chosen["transfer_to_hotel_cost_yuan_per_two"], 2),
                "estimated_extra_detour_hours": round(chosen["next_day_return_or_detour_hours"], 3),
                "estimated_extra_detour_cost_yuan_per_two": round(chosen["detour_cost_yuan_per_two"], 2),
                "repair_status": "recommended",
                "rationale": "将非标准住宿点替换为最近且绕行成本较低的标准酒店枢纽。",
            }
        )
    return pd.DataFrame(rows)


def build_operational_actions(route_id: str, days: pd.DataFrame, segments: pd.DataFrame, timew: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, d in days[days["active_budget_status"].eq("over_budget")].iterrows():
        day = int(d["day"])
        day_seg = segments[(segments["route_id"].eq(route_id)) & (segments["day"].eq(day))]
        long_transfer = day_seg[day_seg["activity_type"].eq("long_transfer_to_region")]
        if not long_transfer.empty:
            seg = long_transfer.iloc[0]
            action = "insert_time_expanded_rail_or_air"
            rationale = "跨区转移超过8小时，应优先用夜间火车/航班真实班次替换，或拆成中停过夜。"
        else:
            seg = day_seg.sort_values("active_increment_hours", ascending=False).iloc[0]
            if base.num(seg["service_hours"]) >= 8:
                action = "split_long_visit_over_two_days"
                rationale = "单景区游览时间过长，应拆成两天或降低游览时长参数。"
            else:
                action = "move_previous_evening_or_relax_budget"
                rationale = "交通+游览略超8小时，可前一晚靠近目的地住宿，或将该日定义为9-10小时高强度日。"
        rows.append(
            {
                "route_id": route_id,
                "day": day,
                "issue_type": "daily_active_budget_overrun",
                "issue_value": round(base.num(d["day_active_hours"]), 2),
                "recommended_action": action,
                "target_segment": f"{seg['from_name']} -> {seg['to_name']}",
                "expected_effect": "减少或解释超过8小时的日程压力",
                "action_status": "requires_real_schedule_or_preference",
                "rationale": rationale,
            }
        )
    for _, t in timew[timew["time_window_feasible"] == False].iterrows():  # noqa: E712
        rows.append(
            {
                "route_id": route_id,
                "day": int(t["day"]),
                "issue_type": "time_window_violation",
                "issue_value": f"{t['service_end_time']} > {t['close_time']}",
                "recommended_action": "local_reorder_or_next_morning_visit",
                "target_segment": t["to_name"],
                "expected_effect": "消除闭园后游览冲突",
                "action_status": "auto_repair_attempted",
                "rationale": "优先在相邻区域内前移该景点；若仍不可行，则改为次日上午游览。",
            }
        )
    return pd.DataFrame(rows)


def export_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])


def write_report(summary: pd.DataFrame, actions: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "新疆旅游逐日行程二次修复报告.md"
    lines = [
        "# 新疆旅游逐日行程二次修复报告",
        "",
        "## 1. 修复策略",
        "",
        "本轮在逐日排程层基础上做局部修复，不改变景点集合。自动修复部分包括时间窗冲突的局部顺序调整；运营建议部分包括超8小时高压日、超长跨区转移和非标准住宿点替换。",
        "",
        "## 2. 修复前后指标",
        "",
        "| 路线 | 原排程天数 | 修复后天数 | 原时间窗冲突 | 修复后时间窗冲突 | 原超8小时天数 | 修复后超8小时天数 | 非标准住宿建议数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['route_id']} | {int(r['baseline_days'])} | {int(r['repaired_days'])} | {int(r['baseline_time_window_violations'])} | {int(r['repaired_time_window_violations'])} | {int(r['baseline_over_budget_days'])} | {int(r['repaired_over_budget_days'])} | {int(r['lodging_repair_recommendations'])} |"
        )
    lines += [
        "",
        "## 3. 仍需人工确认的运营动作",
        "",
        "超8小时高压日不是简单排序问题，通常需要真实夜间火车/航班、司机轮换、团队偏好或酒店库存来最终确定。本轮把这些动作保留为 `operational_repair_actions`，便于在论文中解释“模型可行性检查 -> 运营修复建议”的闭环。",
        "",
        f"本轮共生成 {len(actions)} 条修复/运营动作，其中自动顺序修复和住宿替换建议均写入 Excel 工作簿。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    REPAIR_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    spots = read_csv(DATA_DIR / "spot_clean.csv")
    time_windows = read_csv(ENHANCED_DIR / "spot_time_windows.csv")
    hotels = read_csv(ENHANCED_DIR / "hotel_hub_constraints.csv")
    special = read_csv(ENHANCED_DIR / "special_access_constraints.csv")
    od = read_csv(ENHANCED_DIR / "enhanced_od_matrix.csv")
    depot = read_csv(ENHANCED_DIR / "depot_access_matrix.csv")
    route_model = read_csv(MODEL_DIR / "problem1_pcop_summary.csv")
    baseline_summary = read_csv(ITINERARY_DIR / "route_schedule_summary.csv")

    od_map = {
        (r["from_spot_id"], r["to_spot_id"]): {
            "time": base.num(r["shortest_time_hours"]),
            "cost": base.num(r["shortest_cost_yuan_per_two"]),
            "risk": base.num(r["path_risk"]),
            "modes": base.clean(r["path_modes"]),
        }
        for _, r in od.iterrows()
    }
    depot_map = {
        r["spot_id"]: {
            "depot_to_spot_time": base.num(r["depot_to_spot_time"]),
            "spot_to_depot_time": base.num(r["spot_to_depot_time"]),
            "depot_to_spot_cost": base.num(r["depot_to_spot_cost"]),
            "spot_to_depot_cost": base.num(r["spot_to_depot_cost"]),
            "depot_to_spot_risk": base.num(r["depot_to_spot_risk"]),
            "spot_to_depot_risk": base.num(r["spot_to_depot_risk"]),
        }
        for _, r in depot.iterrows()
    }
    ctx = {
        "spots": spots,
        "time_windows": time_windows,
        "hotels": hotels,
        "special": special,
        "od_map": od_map,
        "depot_map": depot_map,
    }

    repaired_rows = []
    order_actions = []
    lodging_actions = []
    operational_actions = []

    for _, r in route_model.iterrows():
        rid = r["route_id"]
        original_sequence = base.route_names(r["route_sequence"])
        repaired_sequence, actions = local_order_repair(rid, original_sequence, ctx)
        if not actions.empty:
            order_actions.append(actions)
        seg = base.schedule_route(rid, " -> ".join(repaired_sequence), spots, time_windows, special, od_map, depot_map)
        days = base.build_day_summary(seg, hotels)
        timew = base.build_time_window_feasibility(seg)
        lodging = base.build_lodging_feasibility(days)
        lodging_rec = candidate_hotel_repairs(rid, seg, lodging, spots, hotels, od_map)
        ops = build_operational_actions(rid, days, seg, timew)
        if not lodging_rec.empty:
            lodging_actions.append(lodging_rec)
        if not ops.empty:
            operational_actions.append(ops)
        repaired_rows.append(
            {
                "route_id": rid,
                "original_sequence": " -> ".join(original_sequence),
                "repaired_sequence": " -> ".join(repaired_sequence),
                "sequence_changed": original_sequence != repaired_sequence,
                "repaired_days": int(days["day"].max()),
                "repaired_over_budget_days": int(days["active_budget_status"].eq("over_budget").sum()),
                "repaired_time_window_violations": int((timew["time_window_feasible"] == False).sum()),  # noqa: E712
                "repaired_limited_lodging_nights": int(((~lodging["lodging_feasible"]) & lodging["requires_hotel_night"]).sum()),
                "lodging_repair_recommendations": len(lodging_rec),
                "operational_actions": len(ops),
            }
        )
        for table_name, df in {
            f"{rid}_segments": seg,
            f"{rid}_days": days,
            f"{rid}_time_windows": timew,
            f"{rid}_lodging": lodging,
        }.items():
            write_csv(df, REPAIR_DIR / f"{table_name}.csv")

    repair_summary = pd.DataFrame(repaired_rows)
    baseline = baseline_summary.rename(
        columns={
            "scheduled_days": "baseline_days",
            "over_budget_days": "baseline_over_budget_days",
            "time_window_violations": "baseline_time_window_violations",
            "limited_lodging_nights": "baseline_limited_lodging_nights",
        }
    )[
        [
            "route_id",
            "baseline_days",
            "baseline_over_budget_days",
            "baseline_time_window_violations",
            "baseline_limited_lodging_nights",
        ]
    ]
    repair_summary = baseline.merge(repair_summary, on="route_id", how="right")
    order_actions_df = pd.concat(order_actions, ignore_index=True) if order_actions else pd.DataFrame()
    lodging_actions_df = pd.concat(lodging_actions, ignore_index=True) if lodging_actions else pd.DataFrame()
    operational_actions_df = pd.concat(operational_actions, ignore_index=True) if operational_actions else pd.DataFrame()
    all_actions = pd.concat(
        [
            order_actions_df.assign(action_group="order_repair") if not order_actions_df.empty else pd.DataFrame(),
            lodging_actions_df.assign(action_group="lodging_repair") if not lodging_actions_df.empty else pd.DataFrame(),
            operational_actions_df.assign(action_group="operational_repair") if not operational_actions_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )

    tables = {
        "repair_summary": repair_summary,
        "order_repair_actions": order_actions_df,
        "lodging_repair_candidates": lodging_actions_df,
        "operational_repair_actions": operational_actions_df,
        "all_repair_actions": all_actions,
    }
    for name, df in tables.items():
        write_csv(df, REPAIR_DIR / f"{name}.csv")
    workbook_path = OUTPUT_DIR / "新疆旅游逐日行程二次修复结果.xlsx"
    export_workbook(tables, workbook_path)
    report_path = write_report(repair_summary, all_actions)
    summary = {
        "route_count": len(repair_summary),
        "order_repair_actions": len(order_actions_df),
        "lodging_repair_recommendations": len(lodging_actions_df),
        "operational_repair_actions": len(operational_actions_df),
        "routes": repair_summary.to_dict("records"),
        "workbook": str(workbook_path),
        "report": str(report_path),
    }
    (REPAIR_DIR / "repair_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
