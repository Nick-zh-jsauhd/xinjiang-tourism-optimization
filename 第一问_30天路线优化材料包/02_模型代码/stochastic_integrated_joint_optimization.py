from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
REPAIR_DIR = ROOT / "itinerary_repair_outputs"
ADVANCED_DIR = ROOT / "advanced_model_outputs"
OUTPUT_DIR = ROOT / "outputs"

MONTE_CARLO_TRIALS = 260
RNG_SEED = 20260610
INTEGRATED_ALNS_ITERATIONS = 95
JOINT_MAX_DAYS = 30
DAY_ACTIVE_HOURS = 8.0

sys.path.insert(0, str(ROOT / "scripts"))
import build_daily_itinerary_layer as itinerary  # noqa: E402


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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean(value).lower()
    return text in {"true", "1", "yes", "y"}


def parse_time(value: Any, default: float = 0.0) -> float:
    text = clean(value)
    if not text:
        return default
    if text.startswith("+") and " " in text:
        day, rest = text.split(" ", 1)
        return 24 * int(day[1:]) + parse_time(rest, default)
    if ":" not in text:
        return default
    h, m = text.split(":", 1)
    try:
        return int(h) + int(m[:2]) / 60.0
    except Exception:
        return default


def load_context() -> dict[str, Any]:
    spots = read_csv(DATA_DIR / "spot_clean.csv")
    cultural = read_csv(ENHANCED_DIR / "cultural_tags.csv")
    time_windows = read_csv(ENHANCED_DIR / "spot_time_windows.csv")
    hotels = read_csv(ENHANCED_DIR / "hotel_hub_constraints.csv")
    special = read_csv(ENHANCED_DIR / "special_access_constraints.csv")
    od = read_csv(ENHANCED_DIR / "enhanced_od_matrix.csv")
    depot = read_csv(ENHANCED_DIR / "depot_access_matrix.csv")
    capacity = read_csv(ENHANCED_DIR / "capacity_by_spot.csv")
    repaired = read_csv(REPAIR_DIR / "repair_summary.csv")

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
    culture_map = cultural.set_index("spot_id")["culture_value_score"].to_dict()
    capacity_map = capacity.set_index("spot_id").to_dict("index")
    hotel_map = hotels.set_index("hub_name").to_dict("index")
    special_map = special.set_index("spot_id").to_dict("index")
    window_map = time_windows.set_index("spot_id").to_dict("index")
    spot_map = spots.set_index("spot_id").to_dict("index")
    hotel_hub_set = {clean(r["hub_name"]) for _, r in hotels.iterrows() if truthy(r.get("is_hotel_hub"))}
    value_map = {sid: spot_value(pd.Series(row, name=sid), culture_map) for sid, row in spot_map.items()}
    service_map = {
        sid: num(row.get("visit_hours_mid"), 0.0) + num(special_map.get(sid, {}).get("safety_buffer_hours"), 0.0)
        for sid, row in spot_map.items()
    }
    ticket_map = {sid: num(row.get("ticket_high_total_yuan_per_person"), 0.0) * 2 for sid, row in spot_map.items()}
    return {
        "spots": spots,
        "cultural": cultural,
        "time_windows": time_windows,
        "hotels": hotels,
        "special": special,
        "od": od,
        "depot": depot,
        "capacity": capacity,
        "repaired": repaired,
        "od_map": od_map,
        "depot_map": depot_map,
        "culture_map": culture_map,
        "capacity_map": capacity_map,
        "hotel_map": hotel_map,
        "special_map": special_map,
        "window_map": window_map,
        "spot_map": spot_map,
        "hotel_hub_set": hotel_hub_set,
        "value_map": value_map,
        "service_map": service_map,
        "ticket_map": ticket_map,
    }


def route_ids_from_sequence(sequence: str, spots: pd.DataFrame) -> list[str]:
    name_to_id = spots.set_index("spot_name")["spot_id"].to_dict()
    ids = []
    for name in itinerary.route_names(sequence):
        sid = name_to_id.get(name)
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def route_sequence_from_ids(route_ids: list[str], spots: pd.DataFrame) -> str:
    id_to_name = spots.set_index("spot_id")["spot_name"].to_dict()
    return " -> ".join(id_to_name[sid] for sid in route_ids if sid in id_to_name)


def schedule_route_ids(route_id: str, route_ids: list[str], ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    sequence = route_sequence_from_ids(route_ids, ctx["spots"])
    segments = itinerary.schedule_route(
        route_id,
        sequence,
        ctx["spots"],
        ctx["time_windows"],
        ctx["special"],
        ctx["od_map"],
        ctx["depot_map"],
    )
    days = itinerary.build_day_summary(segments, ctx["hotels"])
    time_windows = itinerary.build_time_window_feasibility(segments)
    lodging = itinerary.build_lodging_feasibility(days)
    summary = itinerary.build_route_summary(
        pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "spots_count": len(route_ids),
                    "days": int(days["day"].max()) if not days.empty else 0,
                    "route_sequence": sequence,
                }
            ]
        ),
        days,
        lodging,
    )
    return {"segments": segments, "days": days, "time_windows": time_windows, "lodging": lodging, "summary": summary}


def spot_value(spot: pd.Series, culture_map: dict[str, float]) -> float:
    sid = clean(spot.get("spot_id", spot.name))
    base = num(spot.get("priority_score_for_op"), 1.0)
    culture = num(culture_map.get(sid), 0.0)
    topic = 1.0 if truthy(spot.get("is_topic_preference")) else 0.0
    natural = 0.35 if truthy(spot.get("is_natural")) else 0.0
    cultural_flag = 0.35 if truthy(spot.get("is_cultural")) else 0.0
    return base + 0.85 * culture + topic + natural + cultural_flag


def score_scheduled_route(route_ids: list[str], ctx: dict[str, Any], route_label: str = "candidate") -> dict[str, Any]:
    scheduled = schedule_route_ids(route_label, route_ids, ctx)
    summary = scheduled["summary"].iloc[0].to_dict()
    spot_index = ctx["spots"].set_index("spot_id")
    total_value = sum(spot_value(spot_index.loc[sid], ctx["culture_map"]) for sid in route_ids if sid in spot_index.index)
    over_budget_hours = scheduled["days"]["over_budget_hours"].sum() if not scheduled["days"].empty else 0.0
    objective = (
        total_value * 1000.0
        - num(summary.get("transport_cost_yuan_per_two")) * 0.35
        - num(summary.get("ticket_cost_yuan_per_two")) * 0.25
        - num(summary.get("hotel_cost_yuan_room")) * 0.20
        - num(summary.get("scheduled_days")) * 95.0
        - num(summary.get("over_budget_days")) * 900.0
        - over_budget_hours * 420.0
        - num(summary.get("time_window_violations")) * 6000.0
        - num(summary.get("limited_lodging_nights")) * 1600.0
        - max(0.0, num(summary.get("scheduled_days")) - JOINT_MAX_DAYS) * 1800.0
    )
    return {
        "objective": objective,
        "total_value": total_value,
        "summary": summary,
        "scheduled": scheduled,
    }


def fast_score_route(route_ids: list[str], ctx: dict[str, Any]) -> dict[str, Any]:
    day = 1
    clock = itinerary.DAY_START_HOUR
    active = 0.0
    current = "DEPOT"
    current_hub = itinerary.START_HUB
    day_rows: list[dict[str, Any]] = []
    transport_cost = 0.0
    ticket_cost = 0.0
    total_travel = 0.0
    total_service = 0.0
    time_window_violations = 0
    over_budget_days = 0
    long_transfer_steps = 0

    def close_day(day_no: int, active_hours: float, hub: str) -> None:
        day_rows.append(
            {
                "day": day_no,
                "active_hours": active_hours,
                "hub": hub,
                "over_budget": active_hours > DAY_ACTIVE_HOURS + 1e-9,
            }
        )

    i = 0
    while i < len(route_ids):
        sid = route_ids[i]
        spot = ctx["spot_map"].get(sid)
        if not spot:
            i += 1
            continue
        if current == "DEPOT":
            dep = ctx["depot_map"][sid]
            metric = {
                "time": num(dep["depot_to_spot_time"]),
                "cost": num(dep["depot_to_spot_cost"]),
                "risk": num(dep.get("depot_to_spot_risk")),
            }
        elif current == sid:
            metric = {"time": 0.0, "cost": 0.0, "risk": 0.0}
        else:
            metric = ctx["od_map"].get((current, sid), {"time": 999.0, "cost": 999999.0, "risk": 1.0})
        travel_h = num(metric["time"])
        service_h = num(ctx["service_map"].get(sid))
        window = ctx["window_map"].get(sid, {})
        open_h = parse_time(window.get("open_time"), 9.0)
        close_h = parse_time(window.get("close_time"), 19.0)
        arrival = clock + travel_h
        service_start = max(arrival, open_h)
        wait_h = max(0.0, open_h - arrival)
        service_end = service_start + service_h
        active_inc = travel_h + wait_h + service_h
        budget_ok = active + active_inc <= DAY_ACTIVE_HOURS + 1e-9
        tw_ok = service_end <= close_h + 1e-9

        if active > 0 and (not budget_ok or not tw_ok):
            close_day(day, active, current_hub)
            day += 1
            clock = itinerary.DAY_START_HOUR
            active = 0.0
            continue

        if active == 0 and travel_h >= itinerary.LONG_TRANSFER_THRESHOLD_HOURS and travel_h + service_h > DAY_ACTIVE_HOURS:
            transport_cost += num(metric["cost"])
            total_travel += travel_h
            long_transfer_steps += 1
            close_day(day, travel_h, clean(spot.get("hub_name")))
            day += 1
            clock = itinerary.DAY_START_HOUR
            active = 0.0
            current = sid
            current_hub = clean(spot.get("hub_name"))
            continue

        transport_cost += num(metric["cost"])
        ticket_cost += num(ctx["ticket_map"].get(sid))
        total_travel += travel_h
        total_service += service_h
        active += active_inc
        clock = service_end
        current = sid
        current_hub = clean(spot.get("hub_name"))
        if not budget_ok:
            over_budget_days += 1
        if not tw_ok:
            time_window_violations += 1
        i += 1

    if route_ids:
        dep = ctx["depot_map"][route_ids[-1]]
        return_time = num(dep["spot_to_depot_time"])
        return_cost = num(dep["spot_to_depot_cost"])
        if active > 0 and active + return_time > DAY_ACTIVE_HOURS + 1e-9:
            close_day(day, active, current_hub)
            day += 1
            active = return_time
            current_hub = itinerary.START_HUB
        else:
            active += return_time
            current_hub = itinerary.START_HUB
        transport_cost += return_cost
        total_travel += return_time
    close_day(day, active, current_hub)

    hotel_cost = 0.0
    limited_lodging = 0
    for row in day_rows[:-1]:
        hub = clean(row["hub"])
        hotel = ctx["hotel_map"].get(hub, {})
        hotel_cost += num(hotel.get("default_room_price_yuan_per_night"), 260.0)
        if hub not in ctx["hotel_hub_set"]:
            limited_lodging += 1
    over_budget_days = max(over_budget_days, sum(1 for row in day_rows if row["over_budget"]))
    scheduled_days = len(day_rows)
    total_value = sum(num(ctx["value_map"].get(sid)) for sid in route_ids)
    objective = (
        total_value * 1000.0
        - transport_cost * 0.35
        - ticket_cost * 0.25
        - hotel_cost * 0.20
        - scheduled_days * 95.0
        - over_budget_days * 900.0
        - sum(max(0.0, num(row["active_hours"]) - DAY_ACTIVE_HOURS) for row in day_rows) * 420.0
        - time_window_violations * 6000.0
        - limited_lodging * 1600.0
        - max(0, scheduled_days - JOINT_MAX_DAYS) * 1800.0
    )
    return {
        "objective": objective,
        "total_value": total_value,
        "summary": {
            "optimized_spots_count": len(route_ids),
            "scheduled_days": scheduled_days,
            "hotel_nights": max(0, scheduled_days - 1),
            "scheduled_active_hours": round(sum(num(row["active_hours"]) for row in day_rows), 3),
            "scheduled_travel_hours": round(total_travel, 3),
            "scheduled_service_hours": round(total_service, 3),
            "transport_cost_yuan_per_two": round(transport_cost, 2),
            "ticket_cost_yuan_per_two": round(ticket_cost, 2),
            "hotel_cost_yuan_room": round(hotel_cost, 2),
            "itinerary_proxy_cost_yuan_excluding_meals": round(transport_cost + ticket_cost + hotel_cost, 2),
            "over_budget_days": int(over_budget_days),
            "time_window_violations": int(time_window_violations),
            "limited_lodging_nights": int(limited_lodging),
            "long_transfer_steps": int(long_transfer_steps),
        },
    }


def travel_time_between(current: str, target: str, ctx: dict[str, Any]) -> float:
    if current == "DEPOT":
        return num(ctx["depot_map"].get(target, {}).get("depot_to_spot_time"), 999.0)
    return num(ctx["od_map"].get((current, target), {}).get("time"), 999.0)


def greedy_nearest_order(selected: list[str], ctx: dict[str, Any]) -> list[str]:
    remaining = selected[:]
    route: list[str] = []
    current = "DEPOT"
    while remaining:
        nxt = min(remaining, key=lambda sid: travel_time_between(current, sid, ctx))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return two_opt_order(route, ctx)


def two_opt_order(route: list[str], ctx: dict[str, Any], max_passes: int = 4) -> list[str]:
    def total_time(candidate: list[str]) -> float:
        if not candidate:
            return 0.0
        value = travel_time_between("DEPOT", candidate[0], ctx)
        for a, b in zip(candidate[:-1], candidate[1:]):
            value += travel_time_between(a, b, ctx)
        value += num(ctx["depot_map"].get(candidate[-1], {}).get("spot_to_depot_time"), 0.0)
        return value

    best = route[:]
    best_val = total_time(best)
    for _ in range(max_passes):
        improved = False
        for i in range(0, max(0, len(best) - 3)):
            for j in range(i + 2, min(len(best), i + 12)):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                val = total_time(cand)
                if val + 1e-9 < best_val:
                    best, best_val, improved = cand, val, True
        if not improved:
            break
    return best


def scenario_multiplier(rng: np.random.Generator, scenario: str, modes: str, base_risk: float, base_time: float) -> tuple[float, float, bool, str]:
    scenario_base = {
        "normal_weekday": (1.00, 1.00),
        "summer_peak": (1.15, 1.08),
        "holiday_peak": (1.36, 1.18),
        "adverse_weather": (1.55, 1.10),
        "road_closure": (1.85, 1.05),
    }[scenario]
    mode_sigma = 0.10
    delay_prob = 0.0
    if "air" in modes:
        mode_sigma += 0.16
        delay_prob += 0.08
    if "rail" in modes:
        mode_sigma += 0.07
        delay_prob += 0.03
    if "self_drive" in modes:
        mode_sigma += 0.10
        delay_prob += 0.03 + min(0.15, base_risk * 0.25)
    if "scenic_shuttle" in modes:
        mode_sigma += 0.05
        delay_prob += 0.02
    if scenario == "adverse_weather":
        delay_prob += 0.08 + min(0.12, base_risk * 0.22)
    if scenario == "road_closure":
        delay_prob += 0.18 + min(0.20, base_risk * 0.35)
    noise = float(rng.lognormal(mean=-0.5 * mode_sigma**2, sigma=mode_sigma))
    delay = 0.0
    severe = False
    issue = ""
    if rng.random() < delay_prob:
        delay = float(rng.uniform(0.4, 2.6))
        issue = "transport_delay"
        if scenario == "road_closure" and ("self_drive" in modes or base_time >= 5.5) and rng.random() < 0.18:
            delay += float(rng.uniform(3.0, 7.0))
            severe = True
            issue = "severe_road_delay"
    return scenario_base[0] * noise + delay / max(base_time, 0.5), scenario_base[1], severe, issue


def simulate_single_route(
    base_route_id: str,
    route_ids: list[str],
    ctx: dict[str, Any],
    trials: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = ["normal_weekday", "summer_peak", "holiday_peak", "adverse_weather", "road_closure"]
    probabilities = np.array([0.55, 0.22, 0.13, 0.07, 0.03])
    spot_index = ctx["spots"].set_index("spot_id")
    hotel_map = ctx["hotel_map"]
    trial_rows: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []

    for trial in range(1, trials + 1):
        scenario = str(rng.choice(scenarios, p=probabilities))
        od_map = {k: dict(v) for k, v in ctx["od_map"].items()}
        depot_map = {k: dict(v) for k, v in ctx["depot_map"].items()}
        severe_transport_events = 0
        delay_events = 0

        leg_pairs = list(zip(route_ids[:-1], route_ids[1:]))
        for a, b in leg_pairs:
            metric = od_map[(a, b)]
            mult, cost_mult, severe, issue = scenario_multiplier(
                rng,
                scenario,
                clean(metric.get("modes")),
                num(metric.get("risk")),
                num(metric.get("time")),
            )
            metric["time"] = max(0.05, num(metric["time"]) * mult)
            metric["cost"] = max(0.0, num(metric["cost"]) * cost_mult)
            metric["risk"] = min(1.0, num(metric["risk"]) * (1.0 + max(0.0, mult - 1.0) * 0.2))
            if issue:
                delay_events += 1
            if severe:
                severe_transport_events += 1

        for sid in route_ids[:1]:
            metric = depot_map[sid]
            mult, cost_mult, severe, issue = scenario_multiplier(
                rng,
                scenario,
                "depot_to_spot",
                num(metric.get("depot_to_spot_risk")),
                num(metric.get("depot_to_spot_time")),
            )
            metric["depot_to_spot_time"] = max(0.05, num(metric["depot_to_spot_time"]) * mult)
            metric["depot_to_spot_cost"] = max(0.0, num(metric["depot_to_spot_cost"]) * cost_mult)
            if issue:
                delay_events += 1
            if severe:
                severe_transport_events += 1
        for sid in route_ids[-1:]:
            metric = depot_map[sid]
            mult, cost_mult, severe, issue = scenario_multiplier(
                rng,
                scenario,
                "return_to_depot",
                num(metric.get("spot_to_depot_risk")),
                num(metric.get("spot_to_depot_time")),
            )
            metric["spot_to_depot_time"] = max(0.05, num(metric["spot_to_depot_time"]) * mult)
            metric["spot_to_depot_cost"] = max(0.0, num(metric["spot_to_depot_cost"]) * cost_mult)
            if issue:
                delay_events += 1
            if severe:
                severe_transport_events += 1

        sim_ctx = dict(ctx)
        sim_ctx["od_map"] = od_map
        sim_ctx["depot_map"] = depot_map
        scheduled = schedule_route_ids(f"{base_route_id}_MC_{trial:04d}", route_ids, sim_ctx)
        days = scheduled["days"]
        lodging = scheduled["lodging"]
        timew = scheduled["time_windows"]
        summary = scheduled["summary"].iloc[0].to_dict()

        reservation_failures = 0
        crowd_penalty = 0.0
        for sid in route_ids:
            spot = spot_index.loc[sid]
            cap = ctx["capacity_map"].get(sid, {})
            cap_source = clean(cap.get("capacity_source_type"))
            reservation_prob = 0.0
            if truthy(spot.get("requires_reservation")):
                reservation_prob = {"normal_weekday": 0.02, "summer_peak": 0.07, "holiday_peak": 0.18, "adverse_weather": 0.05, "road_closure": 0.04}[scenario]
                if cap_source == "simulated":
                    reservation_prob += 0.02
            if rng.random() < reservation_prob:
                reservation_failures += 1
            demand_factor = {"normal_weekday": 0.35, "summer_peak": 0.62, "holiday_peak": 0.92, "adverse_weather": 0.42, "road_closure": 0.38}[scenario]
            if cap_source == "simulated":
                demand_factor += 0.08
            crowd_penalty += max(0.0, demand_factor - 0.85) * 0.15

        hotel_full_nights = 0
        hotel_extra_cost = 0.0
        hotel_extra_hours = 0.0
        for _, row in lodging.iterrows():
            if not bool(row["requires_hotel_night"]):
                continue
            hub = clean(row["lodging_hub"])
            source_type = clean(hotel_map.get(hub, {}).get("source_type"))
            base_prob = {"normal_weekday": 0.02, "summer_peak": 0.08, "holiday_peak": 0.20, "adverse_weather": 0.04, "road_closure": 0.05}[scenario]
            if source_type == "simulated_limited_lodging":
                base_prob += 0.18
            if rng.random() < base_prob:
                hotel_full_nights += 1
                hotel_extra_cost += float(rng.uniform(80, 260))
                hotel_extra_hours += float(rng.uniform(0.5, 2.5))

        total_cost = (
            num(summary.get("itinerary_proxy_cost_yuan_excluding_meals"))
            + hotel_extra_cost
            + num(summary.get("transport_cost_yuan_per_two")) * max(0.0, {"normal_weekday": 0.00, "summer_peak": 0.03, "holiday_peak": 0.10, "adverse_weather": 0.04, "road_closure": 0.06}[scenario])
        )
        over_budget_days = int(num(summary.get("over_budget_days")))
        time_window_violations = int(num(summary.get("time_window_violations")))
        limited_lodging = int(num(summary.get("limited_lodging_nights")))
        route_feasible = (
            time_window_violations == 0
            and reservation_failures == 0
            and severe_transport_events == 0
            and hotel_full_nights <= 1
            and over_budget_days <= max(3, math.ceil(len(route_ids) * 0.12))
        )

        trial_rows.append(
            {
                "route_id": base_route_id,
                "trial": trial,
                "scenario": scenario,
                "spots_count": len(route_ids),
                "scheduled_days": int(num(summary.get("scheduled_days"))),
                "active_hours": round(num(summary.get("scheduled_active_hours")) + hotel_extra_hours, 3),
                "over_budget_days": over_budget_days,
                "time_window_violations": time_window_violations,
                "limited_lodging_nights": limited_lodging,
                "hotel_full_nights": hotel_full_nights,
                "reservation_failures": reservation_failures,
                "transport_delay_events": delay_events,
                "severe_transport_events": severe_transport_events,
                "crowd_penalty": round(crowd_penalty, 4),
                "total_proxy_cost_yuan": round(total_cost, 2),
                "route_feasible": bool(route_feasible),
            }
        )
        if not route_feasible:
            for issue_name, issue_value in [
                ("reservation_failure", reservation_failures),
                ("hotel_full", hotel_full_nights),
                ("severe_transport", severe_transport_events),
                ("time_window_violation", time_window_violations),
                ("active_budget_pressure", over_budget_days),
            ]:
                if issue_value:
                    driver_rows.append(
                        {
                            "route_id": base_route_id,
                            "trial": trial,
                            "scenario": scenario,
                            "risk_driver": issue_name,
                            "driver_count": int(issue_value),
                        }
                    )

    return pd.DataFrame(trial_rows), pd.DataFrame(driver_rows)


def monte_carlo_evaluation(ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    all_trials = []
    all_drivers = []
    for _, row in ctx["repaired"].iterrows():
        route_id = clean(row["route_id"])
        route_ids = route_ids_from_sequence(row["repaired_sequence"], ctx["spots"])
        trials, drivers = simulate_single_route(route_id, route_ids, ctx, MONTE_CARLO_TRIALS, rng)
        all_trials.append(trials)
        all_drivers.append(drivers)
    trial_df = pd.concat(all_trials, ignore_index=True)
    driver_df = pd.concat(all_drivers, ignore_index=True) if all_drivers else pd.DataFrame()
    summary_rows = []
    for route_id, g in trial_df.groupby("route_id"):
        summary_rows.append(
            {
                "route_id": route_id,
                "trials": len(g),
                "spots_count": int(g["spots_count"].median()),
                "feasibility_rate": round(float(g["route_feasible"].mean()), 4),
                "mean_days": round(float(g["scheduled_days"].mean()), 2),
                "p95_days": round(float(g["scheduled_days"].quantile(0.95)), 2),
                "mean_cost_yuan": round(float(g["total_proxy_cost_yuan"].mean()), 2),
                "p95_cost_yuan": round(float(g["total_proxy_cost_yuan"].quantile(0.95)), 2),
                "mean_over_budget_days": round(float(g["over_budget_days"].mean()), 2),
                "prob_any_reservation_failure": round(float((g["reservation_failures"] > 0).mean()), 4),
                "prob_any_hotel_full": round(float((g["hotel_full_nights"] > 0).mean()), 4),
                "prob_severe_transport": round(float((g["severe_transport_events"] > 0).mean()), 4),
                "mean_crowd_penalty": round(float(g["crowd_penalty"].mean()), 4),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["feasibility_rate", "p95_cost_yuan"], ascending=[False, True])
    if not driver_df.empty:
        risk_summary = (
            driver_df.groupby(["route_id", "risk_driver"], as_index=False)["driver_count"]
            .sum()
            .sort_values(["route_id", "driver_count"], ascending=[True, False])
        )
    else:
        risk_summary = pd.DataFrame(columns=["route_id", "risk_driver", "driver_count"])
    return {"monte_carlo_trials": trial_df, "monte_carlo_summary": summary_df, "monte_carlo_risk_drivers": risk_summary}


@dataclass
class AlnsState:
    route_ids: list[str]
    objective: float
    summary: dict[str, Any]


def integrated_alns(ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = random.Random(RNG_SEED + 17)
    spots = ctx["spots"]
    spot_index = spots.set_index("spot_id")
    candidates = [
        sid
        for sid, r in spot_index.iterrows()
        if not truthy(r.get("ordinary_tourist_restricted")) and sid in ctx["depot_map"]
    ]
    base_row = ctx["repaired"].sort_values(["repaired_time_window_violations", "repaired_days"]).iloc[0]
    current_route = route_ids_from_sequence(base_row["repaired_sequence"], spots)
    if len(current_route) < 10:
        current_route = candidates[:]

    cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def evaluate(route: list[str]) -> dict[str, Any]:
        route = list(dict.fromkeys(route))
        key = tuple(route)
        if key not in cache:
            cache[key] = fast_score_route(route, ctx)
        return cache[key]

    def repair_insert(route: list[str], removed: list[str]) -> list[str]:
        route = route[:]
        seen_pool: set[str] = set()
        candidate_pool = []
        for x in removed + [c for c in candidates if c not in route]:
            if x not in route and x not in seen_pool:
                seen_pool.add(x)
                candidate_pool.append(x)
        candidate_pool = sorted(candidate_pool, key=lambda sid: spot_value(spot_index.loc[sid], ctx["culture_map"]), reverse=True)
        for sid in candidate_pool[:18]:
            if len(route) >= len(candidates):
                break
            best_pos = None
            best_eval = None
            positions = list(range(len(route) + 1))
            rng.shuffle(positions)
            for pos in positions[: min(len(positions), 14)]:
                trial = route[:pos] + [sid] + route[pos:]
                ev = evaluate(trial)
                if best_eval is None or ev["objective"] > best_eval["objective"]:
                    best_eval = ev
                    best_pos = pos
            if best_eval and best_pos is not None:
                baseline = evaluate(route)
                summary = best_eval["summary"]
                if (
                    best_eval["objective"] >= baseline["objective"] - 250
                    and num(summary.get("scheduled_days")) <= JOINT_MAX_DAYS + 8
                    and num(summary.get("time_window_violations")) <= 1
                ):
                    route = route[:best_pos] + [sid] + route[best_pos:]
        return route

    def remove_problem_segment(route: list[str]) -> tuple[list[str], list[str]]:
        ev = evaluate(route)
        if num(ev["summary"].get("over_budget_days")) <= 0:
            count = max(1, int(len(route) * rng.choice([0.08, 0.12, 0.18])))
            removed = rng.sample(route, min(count, len(route)))
            return [x for x in route if x not in set(removed)], removed
        arc_pressure: list[tuple[float, str]] = []
        for a, b in zip(route[:-1], route[1:]):
            m = ctx["od_map"].get((a, b), {})
            pressure = num(m.get("time")) + 2.0 * num(m.get("risk"))
            arc_pressure.append((pressure, b))
        if not arc_pressure:
            removed = rng.sample(route, min(2, len(route)))
        else:
            removed = [sid for _, sid in sorted(arc_pressure, reverse=True)[: max(1, min(3, len(arc_pressure)))]]
        return [x for x in route if x not in set(removed)], removed

    current_eval = evaluate(current_route)
    current = AlnsState(current_route, current_eval["objective"], current_eval["summary"])
    best = current
    history = []

    for iteration in range(1, INTEGRATED_ALNS_ITERATIONS + 1):
        route = current.route_ids[:]
        removed: list[str] = []
        op = rng.choice(["problem_removal", "random_removal", "swap", "two_opt", "regional_shuffle"])
        if op == "problem_removal":
            route, removed = remove_problem_segment(route)
            route = repair_insert(route, removed)
        elif op == "random_removal":
            count = max(1, int(len(route) * rng.choice([0.08, 0.14, 0.22])))
            removed = rng.sample(route, min(count, len(route)))
            route = [x for x in route if x not in set(removed)]
            route = repair_insert(route, removed)
        elif op == "swap" and len(route) >= 2:
            i, j = sorted(rng.sample(range(len(route)), 2))
            route[i], route[j] = route[j], route[i]
        elif op == "two_opt" and len(route) >= 4:
            i, j = sorted(rng.sample(range(len(route)), 2))
            if j - i >= 2:
                route = route[:i] + list(reversed(route[i:j])) + route[j:]
        elif op == "regional_shuffle" and len(route) >= 5:
            regions = {sid: clean(spot_index.loc[sid]["region_cluster"]) for sid in route}
            region = rng.choice(list(set(regions.values())))
            idxs = [i for i, sid in enumerate(route) if regions[sid] == region]
            if len(idxs) >= 2:
                vals = [route[i] for i in idxs]
                rng.shuffle(vals)
                for k, i in enumerate(idxs):
                    route[i] = vals[k]

        route = list(dict.fromkeys(route))
        ev = evaluate(route)
        cand = AlnsState(route, ev["objective"], ev["summary"])
        temp = max(80.0, 1400.0 * (1.0 - iteration / INTEGRATED_ALNS_ITERATIONS))
        accept = cand.objective >= current.objective or rng.random() < math.exp((cand.objective - current.objective) / temp)
        if accept:
            current = cand
        if cand.objective > best.objective:
            best = cand
        history.append(
            {
                "iteration": iteration,
                "operator": op,
                "accepted": bool(accept),
                "candidate_objective": round(cand.objective, 3),
                "current_objective": round(current.objective, 3),
                "best_objective": round(best.objective, 3),
                "candidate_spots": len(cand.route_ids),
                "candidate_days": int(num(cand.summary.get("scheduled_days"))),
                "candidate_time_window_violations": int(num(cand.summary.get("time_window_violations"))),
                "candidate_limited_lodging_nights": int(num(cand.summary.get("limited_lodging_nights"))),
            }
        )

    best_eval = score_scheduled_route(best.route_ids, ctx, "INTEGRATED_ALNS")
    best_summary = dict(best_eval["summary"])
    best_summary.update(
        {
            "route_id": "INTEGRATED_ALNS",
            "solver_status": "schedule_aware_alns_completed",
            "objective": round(best.objective, 3),
            "total_value_score": round(best_eval["total_value"], 3),
            "route_sequence": route_sequence_from_ids(best.route_ids, spots),
        }
    )
    detail = best_eval["scheduled"]["segments"].copy()
    detail["route_id"] = "INTEGRATED_ALNS"
    return {
        "integrated_alns_summary": pd.DataFrame([best_summary]),
        "integrated_alns_segments": detail,
        "integrated_alns_history": pd.DataFrame(history),
    }


def build_joint_milp(ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    spots = ctx["spots"].copy().reset_index(drop=True)
    spot_ids = spots["spot_id"].tolist()
    n = len(spot_ids)
    days = JOINT_MAX_DAYS
    spot_pos = {sid: i for i, sid in enumerate(spot_ids)}
    culture_map = ctx["culture_map"]

    service = np.array([itinerary.service_hours(spots.iloc[i], ctx["special"]) for i in range(n)], dtype=float)
    ticket = np.array([num(spots.iloc[i]["ticket_high_total_yuan_per_person"]) * 2 for i in range(n)], dtype=float)
    values = np.array([spot_value(spots.iloc[i], culture_map) for i in range(n)], dtype=float)
    restricted = np.array([truthy(spots.iloc[i].get("ordinary_tourist_restricted")) for i in range(n)], dtype=bool)
    hotel_hubs = {
        clean(row["hub_name"])
        for _, row in ctx["hotels"].iterrows()
        if truthy(row.get("is_hotel_hub"))
    }
    spot_has_hotel_anchor = np.array([clean(spots.iloc[i]["hub_name"]) in hotel_hubs for i in range(n)], dtype=bool)

    open_hours = np.zeros(n)
    close_hours = np.zeros(n)
    tw_map = ctx["time_windows"].set_index("spot_id").to_dict("index")
    for i, sid in enumerate(spot_ids):
        tw = tw_map.get(sid, {})
        open_hours[i] = parse_time(tw.get("open_time"), 9.0)
        close_hours[i] = parse_time(tw.get("close_time"), 19.0)
        if close_hours[i] >= 24.0:
            close_hours[i] = 23.99

    travel = np.full((n, n), 999.0)
    cost = np.full((n, n), 999999.0)
    risk = np.zeros((n, n))
    for i, a in enumerate(spot_ids):
        for j, b in enumerate(spot_ids):
            if i == j:
                travel[i, j] = 0.0
                cost[i, j] = 0.0
                continue
            m = ctx["od_map"].get((a, b))
            if m:
                travel[i, j] = num(m["time"], 999.0)
                cost[i, j] = num(m["cost"], 999999.0)
                risk[i, j] = num(m["risk"], 0.0)
    depot_out = np.array([num(ctx["depot_map"][sid]["depot_to_spot_time"], 999.0) for sid in spot_ids])
    depot_in = np.array([num(ctx["depot_map"][sid]["spot_to_depot_time"], 999.0) for sid in spot_ids])
    depot_out_cost = np.array([num(ctx["depot_map"][sid]["depot_to_spot_cost"], 999999.0) for sid in spot_ids])
    depot_in_cost = np.array([num(ctx["depot_map"][sid]["spot_to_depot_cost"], 999999.0) for sid in spot_ids])

    idx: dict[str, Any] = {}
    offset = 0
    idx["x"] = np.arange(offset, offset + n)
    offset += n
    idx["z"] = np.arange(offset, offset + n * days).reshape(n, days)
    offset += n * days

    arc_pairs: list[tuple[int, int]] = []
    # Node 0 is start, 1..n are spots, n+1 is end.
    for j in range(1, n + 1):
        arc_pairs.append((0, j))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                arc_pairs.append((i, j))
    for i in range(1, n + 1):
        arc_pairs.append((i, n + 1))
    y_index = {arc: offset + k for k, arc in enumerate(arc_pairs)}
    idx["y"] = y_index
    offset += len(arc_pairs)
    q_arc_pairs: set[tuple[int, int]] = set()
    regions = [clean(spots.iloc[i]["region_cluster"]) for i in range(n)]
    for i in range(n):
        ranked = sorted([j for j in range(n) if j != i], key=lambda j: travel[i, j])
        for j in ranked[:12]:
            if travel[i, j] <= 6.5 or regions[i] == regions[j]:
                q_arc_pairs.add((i, j))

    q_index: dict[tuple[int, int, int], int] = {}
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) not in q_arc_pairs:
                continue
            for d in range(days):
                q_index[(i, j, d)] = offset
                offset += 1
    idx["q"] = q_index
    idx["t"] = np.arange(offset, offset + n * days).reshape(n, days)
    offset += n * days
    idx["u"] = np.arange(offset, offset + n)
    offset += n
    idx["v"] = np.arange(offset, offset + days)
    offset += days
    idx["slack_active"] = np.arange(offset, offset + days)
    offset += days
    idx["slack_hotel"] = np.arange(offset, offset + days)
    offset += days
    var_count = offset

    c = np.zeros(var_count)
    c[idx["x"]] = ticket * 0.15 - values * 900.0
    for (a, b), y_idx in y_index.items():
        if a == 0:
            j = b - 1
            c[y_idx] = depot_out[j] * 35.0 + depot_out_cost[j] * 0.28
        elif b == n + 1:
            i = a - 1
            c[y_idx] = depot_in[i] * 35.0 + depot_in_cost[i] * 0.28
        else:
            i, j = a - 1, b - 1
            c[y_idx] = travel[i, j] * 35.0 + cost[i, j] * 0.28 + risk[i, j] * 250.0
    c[idx["v"]] = 120.0
    c[idx["slack_active"]] = 7000.0
    c[idx["slack_hotel"]] = 4500.0

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    integrality = np.zeros(var_count)
    binary_blocks = [idx["x"], idx["z"].ravel(), np.fromiter(y_index.values(), dtype=int), np.fromiter(q_index.values(), dtype=int), idx["v"]]
    for block in binary_blocks:
        integrality[block] = 1
    ub[idx["t"].ravel()] = 24.0
    ub[idx["u"]] = n
    ub[idx["slack_active"]] = 10.0
    ub[idx["slack_hotel"]] = 1.0
    for i, is_restricted in enumerate(restricted):
        if is_restricted or service[i] > max(0.0, close_hours[i] - open_hours[i]) + 1e-6:
            ub[idx["x"][i]] = 0.0
            ub[idx["z"][i, :]] = 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(terms: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(lower)
        for col, val in terms:
            rows.append(r)
            cols.append(col)
            data.append(val)
        lower.append(lo)
        upper.append(hi)

    for i in range(n):
        add_constraint([(idx["x"][i], 1.0)] + [(idx["z"][i, d], -1.0) for d in range(days)], 0.0, 0.0)
    for i in range(n):
        for d in range(days):
            add_constraint([(idx["z"][i, d], 1.0), (idx["v"][d], -1.0)], -math.inf, 0.0)
    for d in range(days - 1):
        add_constraint([(idx["v"][d], 1.0), (idx["v"][d + 1], -1.0)], 0.0, math.inf)

    add_constraint([(y_index[(0, j)], 1.0) for j in range(1, n + 1)], 1.0, 1.0)
    add_constraint([(y_index[(i, n + 1)], 1.0) for i in range(1, n + 1)], 1.0, 1.0)
    for i in range(n):
        node = i + 1
        out_terms = [(y_index[(node, j)], 1.0) for j in range(1, n + 2) if j != node and (node, j) in y_index]
        in_terms = [(y_index[(j, node)], 1.0) for j in range(0, n + 1) if j != node and (j, node) in y_index]
        add_constraint(out_terms + [(idx["x"][i], -1.0)], 0.0, 0.0)
        add_constraint(in_terms + [(idx["x"][i], -1.0)], 0.0, 0.0)

    add_constraint([(idx["x"][i], 1.0) for i in range(n)], 20.0, n)
    for i in range(n):
        add_constraint([(idx["u"][i], 1.0), (idx["x"][i], -1.0)], 0.0, math.inf)
        add_constraint([(idx["u"][i], 1.0), (idx["x"][i], -float(n))], -math.inf, 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y_idx = y_index[(i + 1, j + 1)]
            add_constraint([(idx["u"][i], 1.0), (idx["u"][j], -1.0), (y_idx, float(n))], -math.inf, n - 1)

    for i in range(n):
        day_i_terms = [(idx["z"][i, d], float(d + 1)) for d in range(days)]
        for j in range(n):
            if i == j:
                continue
            day_j_terms = [(idx["z"][j, d], float(d + 1)) for d in range(days)]
            y_idx = y_index[(i + 1, j + 1)]
            add_constraint(day_j_terms + [(col, -val) for col, val in day_i_terms] + [(y_idx, -float(days))], -float(days), math.inf)

    big_m = 30.0
    for i in range(n):
        for d in range(days):
            z = idx["z"][i, d]
            t = idx["t"][i, d]
            add_constraint([(t, 1.0), (z, -open_hours[i])], 0.0, math.inf)
            add_constraint([(t, 1.0), (z, -(close_hours[i] - service[i])),], -math.inf, 0.0)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            y_idx = y_index[(i + 1, j + 1)]
            for d in range(days):
                zi = idx["z"][i, d]
                zj = idx["z"][j, d]
                if (i, j) not in q_arc_pairs:
                    add_constraint([(y_idx, 1.0), (zi, 1.0), (zj, 1.0)], -math.inf, 2.0)
                    continue
                q = q_index[(i, j, d)]
                add_constraint([(q, 1.0), (y_idx, -1.0)], -math.inf, 0.0)
                add_constraint([(q, 1.0), (zi, -1.0)], -math.inf, 0.0)
                add_constraint([(q, 1.0), (zj, -1.0)], -math.inf, 0.0)
                add_constraint([(q, 1.0), (y_idx, -1.0), (zi, -1.0), (zj, -1.0)], -2.0, math.inf)
                add_constraint(
                    [(idx["t"][j, d], 1.0), (idx["t"][i, d], -1.0), (q, -big_m)],
                    service[i] + travel[i, j] - big_m,
                    math.inf,
                )

    for d in range(days):
        terms: list[tuple[int, float]] = [(idx["slack_active"][d], -1.0)]
        for i in range(n):
            terms.append((idx["z"][i, d], service[i]))
        for (i, j, q_day), q_var in q_index.items():
            if q_day == d:
                terms.append((q_var, travel[i, j]))
        add_constraint(terms, -math.inf, DAY_ACTIVE_HOURS)
        if d < days - 1:
            terms = [(idx["slack_hotel"][d], 1.0), (idx["v"][d + 1], -1.0)]
            for i in range(n):
                if spot_has_hotel_anchor[i]:
                    terms.append((idx["z"][i, d], 1.0))
            add_constraint(terms, 0.0, math.inf)

    a = coo_matrix((data, (rows, cols)), shape=(len(lower), var_count)).tocsr()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(a, np.array(lower), np.array(upper)),
        options={"time_limit": 18, "mip_rel_gap": 0.10, "disp": False},
    )

    if res.x is None:
        return build_relaxed_joint_milp(ctx, strict_status=f"strict_sparse_time_model_failed:{res.message}")

    sol = res.x
    arcs = [(a0, b0) for (a0, b0), y_idx in y_index.items() if sol[y_idx] > 0.5]
    next_map = {a0: b0 for a0, b0 in arcs}
    route_nodes = []
    cur = 0
    seen = set()
    while cur in next_map:
        nxt = next_map[cur]
        if nxt == n + 1 or nxt in seen:
            break
        route_nodes.append(nxt)
        seen.add(nxt)
        cur = nxt
    route_ids = [spot_ids[node - 1] for node in route_nodes]
    scheduled = schedule_route_ids("FULL_JOINT_MILP", route_ids, ctx)
    score = score_scheduled_route(route_ids, ctx, "FULL_JOINT_MILP")

    assignment_rows = []
    for i, sid in enumerate(spot_ids):
        if sol[idx["x"][i]] <= 0.5:
            continue
        assigned_day = int(np.argmax(sol[idx["z"][i, :]]) + 1)
        assignment_rows.append(
            {
                "spot_id": sid,
                "spot_name": spots.iloc[i]["spot_name"],
                "assigned_day": assigned_day,
                "model_service_start": round(float(sol[idx["t"][i, assigned_day - 1]]), 3),
                "open_time": open_hours[i],
                "close_time": close_hours[i],
                "service_hours": round(float(service[i]), 3),
                "hub_name": spots.iloc[i]["hub_name"],
                "hotel_anchor_available": bool(spot_has_hotel_anchor[i]),
                "value_score": round(float(values[i]), 3),
            }
        )
    joint_summary = dict(score["summary"])
    joint_summary.update(
        {
            "route_id": "FULL_JOINT_MILP",
            "solver_status": str(res.message),
            "mip_objective": round(float(res.fun), 3),
            "integrated_objective": round(float(score["objective"]), 3),
            "selected_value_score": round(float(score["total_value"]), 3),
            "selected_spots_by_milp": len(route_ids),
            "model_active_slack_hours": round(float(sol[idx["slack_active"]].sum()), 3),
            "model_hotel_anchor_slack_days": round(float(sol[idx["slack_hotel"]].sum()), 3),
            "route_sequence": route_sequence_from_ids(route_ids, spots),
        }
    )
    seg = scheduled["segments"].copy()
    seg["route_id"] = "FULL_JOINT_MILP"
    return {
        "joint_optimization_summary": pd.DataFrame([joint_summary]),
        "joint_optimization_day_assignment": pd.DataFrame(assignment_rows),
        "joint_optimization_segments": seg,
    }


def build_relaxed_joint_milp(ctx: dict[str, Any], strict_status: str) -> dict[str, pd.DataFrame]:
    spots = ctx["spots"].copy().reset_index(drop=True)
    spot_ids = spots["spot_id"].tolist()
    n = len(spot_ids)
    days = JOINT_MAX_DAYS
    service = np.array([num(ctx["service_map"].get(sid)) for sid in spot_ids], dtype=float)
    ticket = np.array([num(ctx["ticket_map"].get(sid)) for sid in spot_ids], dtype=float)
    values = np.array([num(ctx["value_map"].get(sid)) for sid in spot_ids], dtype=float)
    restricted = np.array([truthy(spots.iloc[i].get("ordinary_tourist_restricted")) for i in range(n)], dtype=bool)
    hotel_anchor = np.array([clean(spots.iloc[i]["hub_name"]) in ctx["hotel_hub_set"] for i in range(n)], dtype=bool)

    open_hours = np.zeros(n)
    close_hours = np.zeros(n)
    for i, sid in enumerate(spot_ids):
        tw = ctx["window_map"].get(sid, {})
        open_hours[i] = parse_time(tw.get("open_time"), 9.0)
        close_hours[i] = parse_time(tw.get("close_time"), 19.0)
        if close_hours[i] >= 24.0:
            close_hours[i] = 23.99

    travel = np.full((n, n), 999.0)
    cost = np.full((n, n), 999999.0)
    risk = np.zeros((n, n))
    for i, a in enumerate(spot_ids):
        for j, b in enumerate(spot_ids):
            if i == j:
                travel[i, j] = 0.0
                cost[i, j] = 0.0
                continue
            m = ctx["od_map"].get((a, b), {})
            travel[i, j] = num(m.get("time"), 999.0)
            cost[i, j] = num(m.get("cost"), 999999.0)
            risk[i, j] = num(m.get("risk"), 0.0)
    depot_out = np.array([num(ctx["depot_map"][sid]["depot_to_spot_time"], 999.0) for sid in spot_ids])
    depot_in = np.array([num(ctx["depot_map"][sid]["spot_to_depot_time"], 999.0) for sid in spot_ids])
    depot_out_cost = np.array([num(ctx["depot_map"][sid]["depot_to_spot_cost"], 999999.0) for sid in spot_ids])
    depot_in_cost = np.array([num(ctx["depot_map"][sid]["spot_to_depot_cost"], 999999.0) for sid in spot_ids])
    nearest_buffer = np.array([min(1.6, max(0.25, np.partition(travel[i][travel[i] > 0], 0)[0] * 0.35)) for i in range(n)])

    offset = 0
    x = np.arange(offset, offset + n)
    offset += n
    z = np.arange(offset, offset + n * days).reshape(n, days)
    offset += n * days
    arc_pairs: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        arc_pairs.append((0, j))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                arc_pairs.append((i, j))
    for i in range(1, n + 1):
        arc_pairs.append((i, n + 1))
    y = {arc: offset + k for k, arc in enumerate(arc_pairs)}
    offset += len(arc_pairs)
    u = np.arange(offset, offset + n)
    offset += n
    v = np.arange(offset, offset + days)
    offset += days
    slack_active = np.arange(offset, offset + days)
    offset += days
    slack_hotel = np.arange(offset, offset + days)
    offset += days
    var_count = offset

    c = np.zeros(var_count)
    c[x] = ticket * 0.20 - values * 1000.0
    for (a, b), y_idx in y.items():
        if a == 0:
            j = b - 1
            c[y_idx] = depot_out[j] * 85.0 + depot_out_cost[j] * 0.42
        elif b == n + 1:
            i = a - 1
            c[y_idx] = depot_in[i] * 85.0 + depot_in_cost[i] * 0.42
        else:
            i, j = a - 1, b - 1
            c[y_idx] = travel[i, j] * 85.0 + cost[i, j] * 0.42 + risk[i, j] * 900.0
    c[v] = 180.0
    c[slack_active] = 18000.0
    c[slack_hotel] = 16000.0

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    integrality = np.zeros(var_count)
    for block in [x, z.ravel(), np.fromiter(y.values(), dtype=int), v]:
        integrality[block] = 1
    ub[u] = n
    ub[slack_active] = 10.0
    ub[slack_hotel] = 1.0
    for i in range(n):
        if restricted[i] or service[i] > max(0.0, close_hours[i] - open_hours[i]) + 1e-9:
            ub[x[i]] = 0.0
            ub[z[i, :]] = 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(terms: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(lower)
        for col, val in terms:
            rows.append(r)
            cols.append(col)
            data.append(val)
        lower.append(lo)
        upper.append(hi)

    for i in range(n):
        add([(x[i], 1.0)] + [(z[i, d], -1.0) for d in range(days)], 0.0, 0.0)
    for i in range(n):
        for d in range(days):
            add([(z[i, d], 1.0), (v[d], -1.0)], -math.inf, 0.0)
    for d in range(days - 1):
        add([(v[d], 1.0), (v[d + 1], -1.0)], 0.0, math.inf)

    add([(y[(0, j)], 1.0) for j in range(1, n + 1)], 1.0, 1.0)
    add([(y[(i, n + 1)], 1.0) for i in range(1, n + 1)], 1.0, 1.0)
    for i in range(n):
        node = i + 1
        out_terms = [(y[(node, j)], 1.0) for j in range(1, n + 2) if j != node and (node, j) in y]
        in_terms = [(y[(j, node)], 1.0) for j in range(0, n + 1) if j != node and (j, node) in y]
        add(out_terms + [(x[i], -1.0)], 0.0, 0.0)
        add(in_terms + [(x[i], -1.0)], 0.0, 0.0)
    add([(x[i], 1.0) for i in range(n)], 24.0, n)

    for i in range(n):
        add([(u[i], 1.0), (x[i], -1.0)], 0.0, math.inf)
        add([(u[i], 1.0), (x[i], -float(n))], -math.inf, 0.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                add([(u[i], 1.0), (u[j], -1.0), (y[(i + 1, j + 1)], float(n))], -math.inf, n - 1)
    for i in range(n):
        day_i = [(z[i, d], float(d + 1)) for d in range(days)]
        for j in range(n):
            if i != j:
                day_j = [(z[j, d], float(d + 1)) for d in range(days)]
                add(day_j + [(col, -val) for col, val in day_i] + [(y[(i + 1, j + 1)], -float(days))], -float(days), math.inf)

    for d in range(days):
        active_terms = [(slack_active[d], -1.0)]
        count_terms = []
        for i in range(n):
            active_terms.append((z[i, d], service[i] + nearest_buffer[i]))
            count_terms.append((z[i, d], 1.0))
        add(active_terms, -math.inf, DAY_ACTIVE_HOURS)
        add(count_terms, -math.inf, 3.0)
        if d < days - 1:
            hotel_terms = [(slack_hotel[d], 1.0), (v[d + 1], -1.0)]
            for i in range(n):
                if hotel_anchor[i]:
                    hotel_terms.append((z[i, d], 1.0))
            add(hotel_terms, 0.0, math.inf)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(lower), var_count)).tocsr()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"time_limit": 35, "mip_rel_gap": 0.08, "disp": False},
    )

    if res.x is None:
        return build_assignment_joint_milp(ctx, strict_status=f"relaxed_sequence_model_failed:{res.message}; {strict_status}")

    sol = res.x
    arcs = [(a, b) for (a, b), y_idx in y.items() if sol[y_idx] > 0.5]
    next_map = {a: b for a, b in arcs}
    route_nodes = []
    cur = 0
    seen = set()
    while cur in next_map:
        nxt = next_map[cur]
        if nxt == n + 1 or nxt in seen:
            break
        route_nodes.append(nxt)
        seen.add(nxt)
        cur = nxt
    route_ids = [spot_ids[node - 1] for node in route_nodes]
    score = score_scheduled_route(route_ids, ctx, "FULL_JOINT_MILP")
    assignment_rows = []
    for i, sid in enumerate(spot_ids):
        if sol[x[i]] <= 0.5:
            continue
        assigned_day = int(np.argmax(sol[z[i, :]]) + 1)
        assignment_rows.append(
            {
                "spot_id": sid,
                "spot_name": spots.iloc[i]["spot_name"],
                "assigned_day": assigned_day,
                "service_hours": round(float(service[i]), 3),
                "service_plus_buffer_hours": round(float(service[i] + nearest_buffer[i]), 3),
                "hub_name": spots.iloc[i]["hub_name"],
                "hotel_anchor_available": bool(hotel_anchor[i]),
                "value_score": round(float(values[i]), 3),
            }
        )
    summary = dict(score["summary"])
    summary.update(
        {
            "route_id": "FULL_JOINT_MILP",
            "solver_status": f"{res.message}; {strict_status}",
            "formulation_level": "full_40_spot_day_assignment_milp_with_sequence_lodging_budget",
            "selected_spots_by_milp": len(route_ids),
            "mip_objective": round(float(res.fun), 3),
            "integrated_objective": round(float(score["objective"]), 3),
            "selected_value_score": round(float(score["total_value"]), 3),
            "model_active_slack_hours": round(float(sol[slack_active].sum()), 3),
            "model_hotel_anchor_slack_days": round(float(sol[slack_hotel].sum()), 3),
            "route_sequence": route_sequence_from_ids(route_ids, ctx["spots"]),
        }
    )
    seg = score["scheduled"]["segments"].copy()
    seg["route_id"] = "FULL_JOINT_MILP"
    return {
        "joint_optimization_summary": pd.DataFrame([summary]),
        "joint_optimization_day_assignment": pd.DataFrame(assignment_rows),
        "joint_optimization_segments": seg,
    }


def build_assignment_joint_milp(ctx: dict[str, Any], strict_status: str) -> dict[str, pd.DataFrame]:
    spots = ctx["spots"].copy().reset_index(drop=True)
    spot_ids = spots["spot_id"].tolist()
    n = len(spot_ids)
    days = JOINT_MAX_DAYS
    service = np.array([num(ctx["service_map"].get(sid)) for sid in spot_ids], dtype=float)
    ticket = np.array([num(ctx["ticket_map"].get(sid)) for sid in spot_ids], dtype=float)
    values = np.array([num(ctx["value_map"].get(sid)) for sid in spot_ids], dtype=float)
    restricted = np.array([truthy(spots.iloc[i].get("ordinary_tourist_restricted")) for i in range(n)], dtype=bool)
    hotel_anchor = np.array([clean(spots.iloc[i]["hub_name"]) in ctx["hotel_hub_set"] for i in range(n)], dtype=bool)
    nearest_buffer = []
    for sid in spot_ids:
        vals = [num(m.get("time")) for (a, _), m in ctx["od_map"].items() if a == sid and num(m.get("time")) > 0]
        nearest_buffer.append(min(1.6, max(0.25, min(vals) * 0.35 if vals else 0.5)))
    nearest_buffer = np.array(nearest_buffer)

    offset = 0
    x = np.arange(offset, offset + n)
    offset += n
    z = np.arange(offset, offset + n * days).reshape(n, days)
    offset += n * days
    v = np.arange(offset, offset + days)
    offset += days
    slack_active = np.arange(offset, offset + days)
    offset += days
    slack_hotel = np.arange(offset, offset + days)
    offset += days
    var_count = offset

    c = np.zeros(var_count)
    c[x] = ticket * 0.20 - values * 1000.0
    c[v] = 140.0
    c[slack_active] = 12000.0
    c[slack_hotel] = 9500.0

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    integrality = np.zeros(var_count)
    for block in [x, z.ravel(), v]:
        integrality[block] = 1
    ub[slack_active] = 10.0
    ub[slack_hotel] = 1.0
    for i in range(n):
        if restricted[i]:
            ub[x[i]] = 0.0
            ub[z[i, :]] = 0.0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(terms: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(lower)
        for col, val in terms:
            rows.append(r)
            cols.append(col)
            data.append(val)
        lower.append(lo)
        upper.append(hi)

    for i in range(n):
        add([(x[i], 1.0)] + [(z[i, d], -1.0) for d in range(days)], 0.0, 0.0)
    for i in range(n):
        for d in range(days):
            add([(z[i, d], 1.0), (v[d], -1.0)], -math.inf, 0.0)
    for d in range(days - 1):
        add([(v[d], 1.0), (v[d + 1], -1.0)], 0.0, math.inf)
    add([(x[i], 1.0) for i in range(n)], 24.0, n)

    for d in range(days):
        active_terms = [(slack_active[d], -1.0)]
        count_terms = []
        for i in range(n):
            active_terms.append((z[i, d], service[i] + nearest_buffer[i]))
            count_terms.append((z[i, d], 1.0))
        add(active_terms, -math.inf, DAY_ACTIVE_HOURS)
        add(count_terms, -math.inf, 3.0)
        if d < days - 1:
            hotel_terms = [(slack_hotel[d], 1.0), (v[d + 1], -1.0)]
            for i in range(n):
                if hotel_anchor[i]:
                    hotel_terms.append((z[i, d], 1.0))
            add(hotel_terms, 0.0, math.inf)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(lower), var_count)).tocsr()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"time_limit": 20, "mip_rel_gap": 0.03, "disp": False},
    )

    if res.x is None:
        fallback_row = ctx["repaired"].sort_values(["repaired_time_window_violations", "repaired_days"]).iloc[0]
        route_ids = route_ids_from_sequence(fallback_row["repaired_sequence"], ctx["spots"])
        assignment = pd.DataFrame()
        status = f"assignment_failed:{res.message}; {strict_status}"
        route_id = "FULL_JOINT_FALLBACK"
    else:
        sol = res.x
        assignment_rows = []
        day_to_spots: dict[int, list[str]] = {}
        for i, sid in enumerate(spot_ids):
            if sol[x[i]] <= 0.5:
                continue
            d = int(np.argmax(sol[z[i, :]]) + 1)
            day_to_spots.setdefault(d, []).append(sid)
            assignment_rows.append(
                {
                    "spot_id": sid,
                    "spot_name": spots.iloc[i]["spot_name"],
                    "assigned_day": d,
                    "service_hours": round(float(service[i]), 3),
                    "service_plus_buffer_hours": round(float(service[i] + nearest_buffer[i]), 3),
                    "hub_name": spots.iloc[i]["hub_name"],
                    "hotel_anchor_available": bool(hotel_anchor[i]),
                    "value_score": round(float(values[i]), 3),
                }
            )
        assignment = pd.DataFrame(assignment_rows)
        selected_ids = []
        for d in sorted(day_to_spots):
            selected_ids.extend(day_to_spots[d])
        route_ids = greedy_nearest_order(selected_ids, ctx)
        status = f"{res.message}; {strict_status}"
        route_id = "FULL_JOINT_ASSIGNMENT_MILP"

    score = score_scheduled_route(route_ids, ctx, route_id)
    summary = dict(score["summary"])
    summary.update(
        {
            "route_id": route_id,
            "solver_status": status,
            "formulation_level": "full_40_spot_selection_day_assignment_milp_plus_nearest_sequence",
            "selected_spots_by_milp": len(route_ids),
            "mip_objective": round(float(res.fun), 3) if res.x is not None else math.nan,
            "integrated_objective": round(float(score["objective"]), 3),
            "selected_value_score": round(float(score["total_value"]), 3),
            "model_active_slack_hours": round(float(res.x[slack_active].sum()), 3) if res.x is not None else math.nan,
            "model_hotel_anchor_slack_days": round(float(res.x[slack_hotel].sum()), 3) if res.x is not None else math.nan,
            "route_sequence": route_sequence_from_ids(route_ids, ctx["spots"]),
        }
    )
    seg = score["scheduled"]["segments"].copy()
    seg["route_id"] = route_id
    return {
        "joint_optimization_summary": pd.DataFrame([summary]),
        "joint_optimization_day_assignment": assignment,
        "joint_optimization_segments": seg,
    }


def build_comparison(ctx: dict[str, Any], alns: dict[str, pd.DataFrame], joint: dict[str, pd.DataFrame], mc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, row in ctx["repaired"].iterrows():
        rid = clean(row["route_id"])
        route_ids = route_ids_from_sequence(row["repaired_sequence"], ctx["spots"])
        ev = score_scheduled_route(route_ids, ctx, f"{rid}_REPAIRED")
        s = ev["summary"]
        rows.append({"model": rid + "_REPAIRED", "method": "baseline_repaired", "objective": round(ev["objective"], 3), **s})
    for name, df in [
        ("INTEGRATED_ALNS", alns["integrated_alns_summary"]),
        ("FULL_JOINT_MILP", joint["joint_optimization_summary"]),
    ]:
        if not df.empty:
            row = df.iloc[0].to_dict()
            rows.append({"model": name, "method": "advanced", **row})
    out = pd.DataFrame(rows)
    if not mc["monte_carlo_summary"].empty:
        mc_s = mc["monte_carlo_summary"].rename(columns={"route_id": "mc_route_id"})
        out = out.merge(mc_s, how="left", left_on=out["model"].str.replace("_REPAIRED", "", regex=False), right_on="mc_route_id")
        if "key_0" in out.columns:
            out = out.drop(columns=["key_0"])
    return out


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    mc = tables["monte_carlo_summary"]
    alns = tables["integrated_alns_summary"]
    joint = tables["joint_optimization_summary"]
    lines = [
        "# 新疆旅游随机仿真与联合优化强化报告",
        "",
        "本轮新增三类强化：Monte Carlo 扰动仿真、逐日排程感知 ALNS、全量景点联合 MILP。目标是把原有路线从静态最优扩展为可解释、可检验、可抗扰动的旅游调度原型。",
        "",
        "## 1. Monte Carlo 稳健性评价",
        "",
    ]
    if not mc.empty:
        lines.append(mc.to_markdown(index=False))
    lines += [
        "",
        "仿真扰动覆盖普通日、暑期高峰、节假日高峰、恶劣天气和道路封闭五类场景；扰动对象包括交通时间、交通费用、酒店满房、预约失败和拥挤惩罚。",
        "",
        "## 2. 集成式 ALNS",
        "",
    ]
    if not alns.empty:
        keep = [
            "route_id",
            "solver_status",
            "objective",
            "optimized_spots_count",
            "scheduled_days",
            "over_budget_days",
            "time_window_violations",
            "limited_lodging_nights",
            "route_sequence",
        ]
        lines.append(alns[[c for c in keep if c in alns.columns]].to_markdown(index=False))
    lines += [
        "",
        "该 ALNS 不再只按总时长修路，而是在每次候选变动后调用逐日排程器，直接把开放时间、住宿可行性和日活动强度纳入评价函数。",
        "",
        "## 3. 全量联合优化",
        "",
    ]
    if not joint.empty:
        keep = [
            "route_id",
            "solver_status",
            "mip_objective",
            "integrated_objective",
            "selected_spots_by_milp",
            "scheduled_days",
            "over_budget_days",
            "time_window_violations",
            "limited_lodging_nights",
            "model_active_slack_hours",
            "model_hotel_anchor_slack_days",
            "route_sequence",
        ]
        lines.append(joint[[c for c in keep if c in joint.columns]].to_markdown(index=False))
    lines += [
        "",
        "本轮尝试了严格的顺序-时间联合 MILP；在全 40 景点、30 天尺度下，严格稀疏时间模型和带顺序的日分配模型均触及时限。最终可求解版本为全量候选集上的“景点选择-天分配 MILP + 最近邻/2-opt 串接”，显式处理景点选择、日分配、每日活动预算、住宿锚点和特殊准入限制；全局访问顺序由后续串接启发式生成，并继续通过逐日排程器复核。",
        "",
        "## 4. 汇报口径",
        "",
        "建议将本轮称为“随机仿真与联合调度强化模型”。它比原模型更贴近真实场景，但仍不是实时商用系统；实时票价、实时房态、预约余量和道路管制应作为后续在线数据接口。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ADVANCED_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ctx = load_context()

    print("running_monte_carlo", flush=True)
    mc_tables = monte_carlo_evaluation(ctx)
    print("running_integrated_alns", flush=True)
    alns_tables = integrated_alns(ctx)
    print("running_full_joint_milp", flush=True)
    joint_tables = build_joint_milp(ctx)
    print("writing_outputs", flush=True)
    comparison = build_comparison(ctx, alns_tables, joint_tables, mc_tables)

    tables: dict[str, pd.DataFrame] = {
        **mc_tables,
        **alns_tables,
        **joint_tables,
        "advanced_model_comparison": comparison,
    }
    for name, df in tables.items():
        write_csv(df, ADVANCED_DIR / f"{name}.csv")
    summary = {
        "monte_carlo_trials": int(len(tables["monte_carlo_trials"])),
        "routes_simulated": int(tables["monte_carlo_summary"]["route_id"].nunique()) if not tables["monte_carlo_summary"].empty else 0,
        "integrated_alns_spots": int(tables["integrated_alns_summary"].iloc[0].get("optimized_spots_count", 0)) if not tables["integrated_alns_summary"].empty else 0,
        "integrated_alns_days": int(tables["integrated_alns_summary"].iloc[0].get("scheduled_days", 0)) if not tables["integrated_alns_summary"].empty else 0,
        "joint_solver_status": clean(tables["joint_optimization_summary"].iloc[0].get("solver_status", "")) if not tables["joint_optimization_summary"].empty else "",
        "joint_selected_spots": int(tables["joint_optimization_summary"].iloc[0].get("selected_spots_by_milp", 0)) if not tables["joint_optimization_summary"].empty and "selected_spots_by_milp" in tables["joint_optimization_summary"].columns else 0,
    }
    (ADVANCED_DIR / "advanced_solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游随机仿真与联合优化结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游随机仿真与联合优化强化报告.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
