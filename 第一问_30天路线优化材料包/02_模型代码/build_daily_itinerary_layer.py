from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
MODEL_DIR = ROOT / "enhanced_model_outputs"
OUTPUT_DIR = ROOT / "outputs"
ITINERARY_DIR = ROOT / "itinerary_outputs"

DAY_START_HOUR = 8.5
DAILY_ACTIVE_BUDGET_HOURS = 8.0
LONG_TRANSFER_THRESHOLD_HOURS = 5.5
START_HUB = "乌鲁木齐市"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def num(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_clock(value: Any, default: float) -> float:
    text = clean(value)
    if not text:
        return default
    if ":" not in text:
        return default
    h, m = text.split(":", 1)
    try:
        return int(h) + int(m[:2]) / 60.0
    except Exception:
        return default


def fmt_clock(hours: float | None) -> str:
    if hours is None or not math.isfinite(hours):
        return ""
    day_offset = int(hours // 24)
    h = int(hours % 24)
    m = int(round((hours - math.floor(hours)) * 60))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    prefix = f"+{day_offset} " if day_offset else ""
    return f"{prefix}{h:02d}:{m:02d}"


def route_names(sequence: str) -> list[str]:
    return [x.strip() for x in clean(sequence).split(" -> ") if x.strip()]


def service_hours(spot: pd.Series, special: pd.DataFrame) -> float:
    row = special[special["spot_id"].eq(spot["spot_id"])]
    buffer = num(row.iloc[0]["safety_buffer_hours"], 0.0) if not row.empty else 0.0
    return num(spot["visit_hours_mid"], 0.0) + buffer


def segment_metric(
    current_spot_id: str,
    target_spot_id: str,
    od_map: dict[tuple[str, str], dict[str, Any]],
    depot_map: dict[str, dict[str, float]],
    return_to_depot: bool = False,
) -> dict[str, Any]:
    if return_to_depot:
        m = depot_map[target_spot_id]
        return {
            "time": m["spot_to_depot_time"],
            "cost": m["spot_to_depot_cost"],
            "risk": m["spot_to_depot_risk"],
            "modes": "return_to_urumqi",
        }
    if current_spot_id == "DEPOT":
        m = depot_map[target_spot_id]
        return {
            "time": m["depot_to_spot_time"],
            "cost": m["depot_to_spot_cost"],
            "risk": m["depot_to_spot_risk"],
            "modes": "depot_to_spot",
        }
    if current_spot_id == target_spot_id:
        return {"time": 0.0, "cost": 0.0, "risk": 0.0, "modes": "overnight_near_spot"}
    return od_map[(current_spot_id, target_spot_id)].copy()


def planning_values(clock: float, active_used: float, travel_h: float, service_h: float, open_h: float, close_h: float) -> dict[str, float | bool]:
    depart = clock
    arrival = depart + travel_h
    start = max(arrival, open_h)
    wait = max(0.0, open_h - arrival)
    end = start + service_h
    active_increment = travel_h + wait + service_h
    return {
        "depart": depart,
        "arrival": arrival,
        "service_start": start,
        "service_end": end,
        "wait": wait,
        "active_increment": active_increment,
        "budget_ok": active_used + active_increment <= DAILY_ACTIVE_BUDGET_HOURS + 1e-9,
        "time_window_ok": end <= close_h + 1e-9,
    }


def add_row(
    rows: list[dict[str, Any]],
    *,
    route_id: str,
    day: int,
    step_no: int,
    activity_type: str,
    from_name: str,
    to_name: str,
    spot_id: str,
    hub_name: str,
    metric: dict[str, Any],
    depart_h: float,
    arrival_h: float,
    service_start_h: float | None,
    service_end_h: float | None,
    open_h: float | None,
    close_h: float | None,
    wait_h: float,
    service_h: float,
    ticket_cost: float,
    active_increment_h: float,
    day_active_after_h: float,
    budget_ok: bool,
    time_window_ok: bool | None,
    note: str,
) -> None:
    rows.append(
        {
            "route_id": route_id,
            "day": day,
            "step_no": step_no,
            "activity_type": activity_type,
            "from_name": from_name,
            "to_name": to_name,
            "spot_id": spot_id,
            "destination_hub": hub_name,
            "transport_modes": metric.get("modes", ""),
            "depart_time": fmt_clock(depart_h),
            "arrival_time": fmt_clock(arrival_h),
            "service_start_time": fmt_clock(service_start_h),
            "service_end_time": fmt_clock(service_end_h),
            "open_time": fmt_clock(open_h),
            "close_time": fmt_clock(close_h),
            "travel_hours": round(num(metric.get("time")), 3),
            "wait_hours": round(wait_h, 3),
            "service_hours": round(service_h, 3),
            "active_increment_hours": round(active_increment_h, 3),
            "day_active_after_hours": round(day_active_after_h, 3),
            "transport_cost_yuan_per_two": round(num(metric.get("cost")), 2),
            "ticket_cost_yuan_per_two": round(ticket_cost, 2),
            "risk_score": round(num(metric.get("risk")), 3),
            "daily_budget_feasible": bool(budget_ok),
            "time_window_feasible": "" if time_window_ok is None else bool(time_window_ok),
            "lodging_hub_after_step": hub_name,
            "note": note,
        }
    )


def schedule_route(
    route_id: str,
    sequence: str,
    spots: pd.DataFrame,
    time_windows: pd.DataFrame,
    special: pd.DataFrame,
    od_map: dict[tuple[str, str], dict[str, Any]],
    depot_map: dict[str, dict[str, float]],
) -> pd.DataFrame:
    spots_by_name = {r["spot_name"]: r for _, r in spots.iterrows()}
    windows = time_windows.set_index("spot_id").to_dict("index")
    names = route_names(sequence)
    rows: list[dict[str, Any]] = []
    day = 1
    step_no = 1
    clock = DAY_START_HOUR
    active = 0.0
    current_spot_id = "DEPOT"
    current_name = START_HUB

    i = 0
    while i < len(names):
        spot = spots_by_name[names[i]]
        sid = clean(spot["spot_id"])
        hub = clean(spot["hub_name"])
        metric = segment_metric(current_spot_id, sid, od_map, depot_map)
        window = windows.get(sid, {})
        open_h = parse_clock(window.get("open_time"), 9.0)
        close_h = parse_clock(window.get("close_time"), 19.0)
        svc_h = service_hours(spot, special)
        ticket = num(spot["ticket_high_total_yuan_per_person"], 0.0) * 2
        plan = planning_values(clock, active, num(metric["time"]), svc_h, open_h, close_h)

        if active > 0 and (not plan["budget_ok"] or not plan["time_window_ok"]):
            day += 1
            clock = DAY_START_HOUR
            active = 0.0
            continue

        if active == 0 and num(metric["time"]) >= LONG_TRANSFER_THRESHOLD_HOURS and num(metric["time"]) + svc_h > DAILY_ACTIVE_BUDGET_HOURS:
            depart = clock
            arrival = depart + num(metric["time"])
            active_increment = num(metric["time"])
            budget_ok = active_increment <= DAILY_ACTIVE_BUDGET_HOURS + 1e-9
            add_row(
                rows,
                route_id=route_id,
                day=day,
                step_no=step_no,
                activity_type="long_transfer_to_region",
                from_name=current_name,
                to_name=f"{names[i]}所在区域",
                spot_id=sid,
                hub_name=hub,
                metric=metric,
                depart_h=depart,
                arrival_h=arrival,
                service_start_h=None,
                service_end_h=None,
                open_h=None,
                close_h=None,
                wait_h=0.0,
                service_h=0.0,
                ticket_cost=0.0,
                active_increment_h=active_increment,
                day_active_after_h=active_increment,
                budget_ok=budget_ok,
                time_window_ok=None,
                note="跨区长距离转移；到达后次日游览目标景区" if budget_ok else "跨区长距离转移超过8小时，建议改为夜间火车/航班或拆分中停",
            )
            step_no += 1
            current_spot_id = sid
            current_name = f"{names[i]}所在区域"
            day += 1
            clock = DAY_START_HOUR
            active = 0.0
            continue

        note_parts: list[str] = []
        if not plan["budget_ok"]:
            note_parts.append("当天活动超过8小时预算")
        if not plan["time_window_ok"]:
            note_parts.append("服务结束晚于闭园时间")
        if num(metric["time"]) >= LONG_TRANSFER_THRESHOLD_HOURS:
            note_parts.append("长距离转移压力")
        add_row(
            rows,
            route_id=route_id,
            day=day,
            step_no=step_no,
            activity_type="visit",
            from_name=current_name,
            to_name=names[i],
            spot_id=sid,
            hub_name=hub,
            metric=metric,
            depart_h=float(plan["depart"]),
            arrival_h=float(plan["arrival"]),
            service_start_h=float(plan["service_start"]),
            service_end_h=float(plan["service_end"]),
            open_h=open_h,
            close_h=close_h,
            wait_h=float(plan["wait"]),
            service_h=svc_h,
            ticket_cost=ticket,
            active_increment_h=float(plan["active_increment"]),
            day_active_after_h=active + float(plan["active_increment"]),
            budget_ok=bool(plan["budget_ok"]),
            time_window_ok=bool(plan["time_window_ok"]),
            note="；".join(note_parts),
        )
        step_no += 1
        clock = float(plan["service_end"])
        active += float(plan["active_increment"])
        current_spot_id = sid
        current_name = names[i]
        i += 1

    if current_spot_id != "DEPOT":
        metric = segment_metric(current_spot_id, current_spot_id, od_map, depot_map, return_to_depot=True)
        travel_h = num(metric["time"])
        if active > 0 and active + travel_h > DAILY_ACTIVE_BUDGET_HOURS + 1e-9:
            day += 1
            clock = DAY_START_HOUR
            active = 0.0
        depart = clock
        arrival = depart + travel_h
        budget_ok = active + travel_h <= DAILY_ACTIVE_BUDGET_HOURS + 1e-9
        add_row(
            rows,
            route_id=route_id,
            day=day,
            step_no=step_no,
            activity_type="return_to_urumqi",
            from_name=current_name,
            to_name=START_HUB,
            spot_id="",
            hub_name=START_HUB,
            metric=metric,
            depart_h=depart,
            arrival_h=arrival,
            service_start_h=None,
            service_end_h=None,
            open_h=None,
            close_h=None,
            wait_h=0.0,
            service_h=0.0,
            ticket_cost=0.0,
            active_increment_h=travel_h,
            day_active_after_h=active + travel_h,
            budget_ok=budget_ok,
            time_window_ok=None,
            note="" if budget_ok else "返程交通超过当天8小时预算",
        )

    return pd.DataFrame(rows)


def build_day_summary(segments: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    hotel_map = hotels.set_index("hub_name").to_dict("index")
    rows = []
    for (route_id, day), g in segments.groupby(["route_id", "day"], sort=True):
        last = g.iloc[-1]
        hub = clean(last["lodging_hub_after_step"])
        hotel = hotel_map.get(hub, {})
        visits = g[g["activity_type"].eq("visit")]
        active = num(g["active_increment_hours"].sum())
        lodging_status = "standard_hotel_hub" if bool(hotel.get("is_hotel_hub", False)) else "limited_or_simulated_lodging"
        rows.append(
            {
                "route_id": route_id,
                "day": day,
                "start_time": g["depart_time"].iloc[0],
                "end_time": g["service_end_time"].replace("", pd.NA).dropna().iloc[-1] if not g["service_end_time"].replace("", pd.NA).dropna().empty else g["arrival_time"].iloc[-1],
                "visit_spots_count": len(visits),
                "visit_spots": " -> ".join(visits["to_name"].tolist()),
                "day_travel_hours": round(num(g["travel_hours"].sum()), 2),
                "day_wait_hours": round(num(g["wait_hours"].sum()), 2),
                "day_service_hours": round(num(g["service_hours"].sum()), 2),
                "day_active_hours": round(active, 2),
                "active_budget_status": "ok" if active <= DAILY_ACTIVE_BUDGET_HOURS + 1e-9 else "over_budget",
                "over_budget_hours": round(max(0.0, active - DAILY_ACTIVE_BUDGET_HOURS), 2),
                "transport_cost_yuan_per_two": round(num(g["transport_cost_yuan_per_two"].sum()), 2),
                "ticket_cost_yuan_per_two": round(num(g["ticket_cost_yuan_per_two"].sum()), 2),
                "risk_score": round(num(g["risk_score"].sum()), 3),
                "lodging_hub": hub,
                "lodging_status": lodging_status,
                "room_price_yuan": num(hotel.get("default_room_price_yuan_per_night"), 260.0),
                "time_window_violations": int((visits["time_window_feasible"] == False).sum()),  # noqa: E712
                "notes": "；".join([clean(x) for x in g["note"].tolist() if clean(x)]),
            }
        )
    return pd.DataFrame(rows)


def build_time_window_feasibility(segments: pd.DataFrame) -> pd.DataFrame:
    visits = segments[segments["activity_type"].eq("visit")].copy()
    visits["feasibility_status"] = visits.apply(
        lambda r: "ok" if bool(r["time_window_feasible"]) and bool(r["daily_budget_feasible"]) else "needs_adjustment",
        axis=1,
    )
    return visits[
        [
            "route_id",
            "day",
            "step_no",
            "to_name",
            "spot_id",
            "arrival_time",
            "service_start_time",
            "service_end_time",
            "open_time",
            "close_time",
            "wait_hours",
            "service_hours",
            "daily_budget_feasible",
            "time_window_feasible",
            "feasibility_status",
            "note",
        ]
    ]


def build_lodging_feasibility(day_summary: pd.DataFrame) -> pd.DataFrame:
    out = day_summary.copy()
    last_day = out.groupby("route_id")["day"].transform("max")
    out["requires_hotel_night"] = out["day"] < last_day
    out["lodging_feasible"] = (~out["requires_hotel_night"]) | out["lodging_status"].eq("standard_hotel_hub")
    out["lodging_note"] = out.apply(
        lambda r: "final return day" if not r["requires_hotel_night"] else ("standard lodging hub" if r["lodging_feasible"] else "limited/simulated lodging; should verify hotel availability"),
        axis=1,
    )
    return out[
        [
            "route_id",
            "day",
            "requires_hotel_night",
            "lodging_hub",
            "lodging_status",
            "room_price_yuan",
            "lodging_feasible",
            "lodging_note",
        ]
    ]


def build_route_summary(route_model: pd.DataFrame, day_summary: pd.DataFrame, lodging: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in route_model.iterrows():
        rid = r["route_id"]
        days = day_summary[day_summary["route_id"].eq(rid)]
        lodg = lodging[lodging["route_id"].eq(rid)]
        hotel_nights = max(0, int(days["day"].max()) - 1) if not days.empty else 0
        hotel_cost = days[days["day"].lt(days["day"].max())]["room_price_yuan"].sum() if not days.empty else 0.0
        rows.append(
            {
                "route_id": rid,
                "optimized_spots_count": int(r["spots_count"]),
                "optimized_days_proxy": int(r["days"]),
                "scheduled_days": int(days["day"].max()) if not days.empty else 0,
                "hotel_nights": hotel_nights,
                "scheduled_active_hours": round(num(days["day_active_hours"].sum()), 2),
                "scheduled_travel_hours": round(num(days["day_travel_hours"].sum()), 2),
                "scheduled_service_hours": round(num(days["day_service_hours"].sum()), 2),
                "transport_cost_yuan_per_two": round(num(days["transport_cost_yuan_per_two"].sum()), 2),
                "ticket_cost_yuan_per_two": round(num(days["ticket_cost_yuan_per_two"].sum()), 2),
                "hotel_cost_yuan_room": round(num(hotel_cost), 2),
                "itinerary_proxy_cost_yuan_excluding_meals": round(num(days["transport_cost_yuan_per_two"].sum()) + num(days["ticket_cost_yuan_per_two"].sum()) + num(hotel_cost), 2),
                "over_budget_days": int(days["active_budget_status"].eq("over_budget").sum()),
                "time_window_violations": int(days["time_window_violations"].sum()),
                "limited_lodging_nights": int(((~lodg["lodging_feasible"]) & lodg["requires_hotel_night"]).sum()) if not lodg.empty else 0,
                "long_transfer_steps": int(day_summary[day_summary["route_id"].eq(rid)]["notes"].str.contains("长距离|跨区", regex=True, na=False).sum()),
                "route_sequence": r["route_sequence"],
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_summary(compare: pd.DataFrame, current_route_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current_label = {
        "previous_enhanced_multimodal": "原增强多方式模型",
        "amap_road_multimodal": "高德道路+12306+航班+接驳下界模型",
        "amap_selfdrive_only": "纯高德自驾基线",
    }
    for _, r in compare.iterrows():
        route_id = r["route_id"]
        if route_id == "P1_AMAP_30day_selfdrive_OP":
            anchor = compare[(compare["experiment"].eq("previous_enhanced_multimodal")) & (compare["route_id"].eq("PCOP_ALNS"))]
        else:
            anchor = compare[(compare["experiment"].eq("previous_enhanced_multimodal")) & (compare["route_id"].eq(route_id))]
        base = anchor.iloc[0] if not anchor.empty else None
        scheduled = current_route_summary[current_route_summary["route_id"].eq(route_id)] if r["experiment"] == "amap_road_multimodal" else pd.DataFrame()
        rows.append(
            {
                "experiment": r["experiment"],
                "experiment_label": current_label.get(r["experiment"], r["experiment"]),
                "route_id": route_id,
                "spots_count": num(r["spots_count"]),
                "model_total_hours": num(r["total_hours"]),
                "scheduled_days_if_available": int(scheduled.iloc[0]["scheduled_days"]) if not scheduled.empty else None,
                "transport_cost_yuan": num(r["transport_cost_yuan"]),
                "objective_or_proxy_cost_yuan": num(r["objective_or_proxy_cost_yuan"]),
                "delta_hours_vs_anchor": round(num(r["total_hours"]) - num(base["total_hours"]), 2) if base is not None else None,
                "delta_cost_vs_anchor": round(num(r["objective_or_proxy_cost_yuan"]) - num(base["objective_or_proxy_cost_yuan"]), 2) if base is not None else None,
                "anchor_route_id": base["route_id"] if base is not None else "",
                "interpretation": "接入真实道路/班次/接驳下界后的当前方案" if r["experiment"] == "amap_road_multimodal" else ("保留用于消融对比" if r["experiment"] == "previous_enhanced_multimodal" else "纯自驾可作为实际落地备选，但成本和长途驾驶压力更高"),
            }
        )
    return pd.DataFrame(rows)


def export_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])


def write_report(route_summary: pd.DataFrame, sensitivity: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "新疆旅游逐日行程与可行性强化报告.md"
    lines = [
        "# 新疆旅游逐日行程与可行性强化报告",
        "",
        "## 1. 本轮强化内容",
        "",
        "本轮在既有图论/运筹优化结果之上，新增逐日排程层。排程层不重新选择景点，而是把 `PCOP_MILP` 与 `PCOP_ALNS` 的路线顺序转化为可执行行程，检查每日8小时活动预算、景区开放时间窗、住宿落点和长距离转移压力。",
        "",
        "核心规则：每天08:30出发；每日交通+等待+游览预算为8小时；无法放入当日的下一个景点顺延到次日；跨区长距离交通可形成单独转移日；夜间住宿优先落在已有酒店枢纽，非酒店枢纽标记为需要复核。",
        "",
        "## 2. 路线排程结果",
        "",
        "| 路线 | 景点数 | 优化天数代理 | 排程天数 | 超8小时天数 | 时间窗冲突 | 需复核住宿夜数 | 行程代理费用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in route_summary.iterrows():
        lines.append(
            f"| {r['route_id']} | {int(r['optimized_spots_count'])} | {int(r['optimized_days_proxy'])} | {int(r['scheduled_days'])} | {int(r['over_budget_days'])} | {int(r['time_window_violations'])} | {int(r['limited_lodging_nights'])} | {r['itinerary_proxy_cost_yuan_excluding_meals']:.2f} |"
        )
    lines += [
        "",
        "## 3. 敏感性与消融对比",
        "",
        "| 实验 | 路线 | 景点数 | 模型总小时 | 相对锚点小时变化 | 相对锚点成本变化 | 说明 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in sensitivity.iterrows():
        dh = "" if pd.isna(r["delta_hours_vs_anchor"]) else f"{r['delta_hours_vs_anchor']:.2f}"
        dc = "" if pd.isna(r["delta_cost_vs_anchor"]) else f"{r['delta_cost_vs_anchor']:.2f}"
        lines.append(
            f"| {r['experiment_label']} | {r['route_id']} | {int(r['spots_count'])} | {r['model_total_hours']:.2f} | {dh} | {dc} | {r['interpretation']} |"
        )
    lines += [
        "",
        "## 4. 结果解释",
        "",
        "逐日排程层暴露的是“数学路径”到“真实旅行计划”之间的差距：如果某日出现长距离转移或非标准住宿点，并不代表路径优化错误，而是说明下一轮应进一步引入 time-expanded 火车/航班边、酒店库存和更精细的夜间落点约束。",
        "",
        "本轮输出的 `time_window_feasibility`、`lodging_feasibility` 和 `daily_itinerary_segments` 可直接作为论文附录中的可行性检查表。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ITINERARY_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    spots = read_csv(DATA_DIR / "spot_clean.csv")
    time_windows = read_csv(ENHANCED_DIR / "spot_time_windows.csv")
    hotels = read_csv(ENHANCED_DIR / "hotel_hub_constraints.csv")
    special = read_csv(ENHANCED_DIR / "special_access_constraints.csv")
    od = read_csv(ENHANCED_DIR / "enhanced_od_matrix.csv")
    depot = read_csv(ENHANCED_DIR / "depot_access_matrix.csv")
    route_model = read_csv(MODEL_DIR / "problem1_pcop_summary.csv")
    compare = read_csv(MODEL_DIR / "transport_experiment_comparison.csv")

    od_map = {
        (r["from_spot_id"], r["to_spot_id"]): {
            "time": num(r["shortest_time_hours"]),
            "cost": num(r["shortest_cost_yuan_per_two"]),
            "risk": num(r["path_risk"]),
            "modes": clean(r["path_modes"]),
        }
        for _, r in od.iterrows()
    }
    depot_map = {
        r["spot_id"]: {
            "depot_to_spot_time": num(r["depot_to_spot_time"]),
            "spot_to_depot_time": num(r["spot_to_depot_time"]),
            "depot_to_spot_cost": num(r["depot_to_spot_cost"]),
            "spot_to_depot_cost": num(r["spot_to_depot_cost"]),
            "depot_to_spot_risk": num(r["depot_to_spot_risk"]),
            "spot_to_depot_risk": num(r["spot_to_depot_risk"]),
        }
        for _, r in depot.iterrows()
    }

    segments = pd.concat(
        [
            schedule_route(r["route_id"], r["route_sequence"], spots, time_windows, special, od_map, depot_map)
            for _, r in route_model.iterrows()
        ],
        ignore_index=True,
    )
    day_summary = build_day_summary(segments, hotels)
    time_window_feasibility = build_time_window_feasibility(segments)
    lodging_feasibility = build_lodging_feasibility(day_summary)
    route_summary = build_route_summary(route_model, day_summary, lodging_feasibility)
    sensitivity = build_sensitivity_summary(compare, route_summary)

    tables = {
        "route_schedule_summary": route_summary,
        "daily_itinerary_segments": segments,
        "daily_itinerary_days": day_summary,
        "time_window_feasibility": time_window_feasibility,
        "lodging_feasibility": lodging_feasibility,
        "sensitivity_ablation": sensitivity,
    }
    for name, df in tables.items():
        write_csv(df, ITINERARY_DIR / f"{name}.csv")

    workbook_path = OUTPUT_DIR / "新疆旅游逐日行程与可行性强化结果.xlsx"
    export_workbook(tables, workbook_path)
    report_path = write_report(route_summary, sensitivity)
    summary = {
        "route_count": len(route_summary),
        "segment_rows": len(segments),
        "day_rows": len(day_summary),
        "time_window_rows": len(time_window_feasibility),
        "lodging_rows": len(lodging_feasibility),
        "routes": route_summary[["route_id", "scheduled_days", "over_budget_days", "time_window_violations", "limited_lodging_nights"]].to_dict("records"),
        "workbook": str(workbook_path),
        "report": str(report_path),
    }
    (ITINERARY_DIR / "daily_itinerary_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
