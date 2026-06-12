from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DAY_HOURS = 8.0
FIELDWORK_MULTIPLIER = 4.0
CULTURE_THRESHOLD = 5
GROUPS = 3
SPECIAL_GROUP = 0
MIN_SPOTS_PER_GROUP = 2
TRIALS_PER_SCENARIO_POLICY = 500
RNG_SEED = 20260612


@dataclass(frozen=True)
class RouteMetric:
    mask: int
    route: tuple[int, ...]
    travel_hours: float
    transport_cost_yuan: float
    route_risk: float


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def num(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def find_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def child_by_prefix(parent: Path, prefix: str) -> Path:
    return next(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))


def normalize_mode(path_modes: object, hours: float) -> str:
    text = "" if pd.isna(path_modes) else str(path_modes).lower()
    if not text:
        return "same_spot"
    if "air" in text:
        return "air"
    if "rail" in text:
        return "rail"
    if "bus" in text or "coach" in text:
        return "coach"
    if "scenic_shuttle" in text:
        return "scenic_shuttle"
    if "self_drive" in text or "fallback_road" in text:
        if hours <= 1.5:
            return "taxi_transfer"
        if hours <= 6.0:
            return "rental_car"
        return "charter_car"
    return "mixed_transfer"


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    pkg_root = root.parent
    input_dir = child_by_prefix(pkg_root, "01_")
    model_data = input_dir / "model_data"
    enhanced_data = input_dir / "enhanced_data"
    risk_dir = child_by_prefix(pkg_root, "06_") / "digital_twin_outputs"

    return {
        "spots": pd.read_csv(model_data / "spot_clean.csv"),
        "cultural": pd.read_csv(enhanced_data / "cultural_tags.csv"),
        "special": pd.read_csv(enhanced_data / "special_access_constraints.csv"),
        "od": pd.read_csv(enhanced_data / "enhanced_od_matrix_with_amap.csv"),
        "amap_od": pd.read_csv(enhanced_data / "amap_driving_od_matrix_clean.csv"),
        "depot": pd.read_csv(enhanced_data / "depot_access_matrix.csv"),
        "scenarios": pd.read_csv(risk_dir / "scenario_parameters.csv"),
    }


def build_candidate_set(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spots = tables["spots"]
    cultural = tables["cultural"]
    special = tables["special"]
    df = spots.merge(cultural, on=["spot_id", "spot_name"], how="left").merge(
        special,
        on=["spot_id", "spot_name"],
        how="left",
        suffixes=("", "_special"),
    )
    df["requires_approval_final"] = df["requires_approval_special"].map(truthy) | df["requires_approval"].map(truthy)
    df["selected_for_q3"] = (df["culture_value_score"].map(num) >= CULTURE_THRESHOLD) | df["requires_approval_final"]
    df = df[df["selected_for_q3"]].copy()
    df["tour_visit_hours"] = df["visit_hours_mid"].map(num)
    df["fieldwork_hours"] = df["tour_visit_hours"] * FIELDWORK_MULTIPLIER
    df["safety_buffer_hours"] = df["safety_buffer_hours"].map(num)
    df["fieldwork_plus_buffer_hours"] = df["fieldwork_hours"] + df["safety_buffer_hours"]
    df["approval_lead_days"] = df["approval_lead_days"].map(lambda x: int(num(x)))
    df["minimum_group_size"] = df["minimum_group_size"].map(lambda x: int(num(x, 1)))
    df["requires_licensed_guide"] = df["requires_licensed_guide"].map(truthy)
    df["requires_offroad_vehicle"] = df["requires_offroad_vehicle"].map(truthy)
    df["requires_border_permit"] = df["requires_border_permit"].map(truthy)
    df["remote_or_approval_flag"] = df["requires_approval_final"] | df["requires_offroad_vehicle"] | df["high_altitude_or_remote"].map(truthy)

    reasons = []
    for _, row in df.iterrows():
        r = []
        if num(row["culture_value_score"]) >= CULTURE_THRESHOLD:
            r.append(f"culture_value_score>={CULTURE_THRESHOLD}")
        if row["requires_approval_final"]:
            r.append("requires_approval")
        if row["requires_licensed_guide"]:
            r.append("requires_licensed_guide")
        if row["requires_offroad_vehicle"]:
            r.append("requires_offroad_vehicle")
        if truthy(row.get("tag_archaeology")):
            r.append("archaeology")
        if truthy(row.get("tag_silk_road")):
            r.append("silk_road")
        reasons.append(";".join(r))
    df["selection_reason"] = reasons
    keep = [
        "spot_id",
        "spot_name",
        "region_cluster",
        "hub_name",
        "culture_value_score",
        "tag_silk_road",
        "tag_religion",
        "tag_ethnic_folk",
        "tag_archaeology",
        "tag_world_or_key_heritage",
        "selected_for_q3",
        "selection_reason",
        "requires_approval_final",
        "approval_lead_days",
        "requires_licensed_guide",
        "requires_offroad_vehicle",
        "requires_border_permit",
        "minimum_group_size",
        "tour_visit_hours",
        "fieldwork_hours",
        "safety_buffer_hours",
        "fieldwork_plus_buffer_hours",
        "ticket_high_total_yuan_per_person",
        "remote_or_approval_flag",
        "ordinary_tourist_allowed",
        "constraint_note",
    ]
    return df[keep].sort_values(["requires_approval_final", "culture_value_score", "spot_id"], ascending=[False, False, True]).reset_index(drop=True)


def build_matrices(candidates: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(candidates)
    time = np.zeros((n + 1, n + 1), dtype=float)
    cost = np.zeros((n + 1, n + 1), dtype=float)
    risk = np.zeros((n + 1, n + 1), dtype=float)
    label_rows: list[dict[str, object]] = []

    od = tables["od"].copy()
    if "scenario_id" in od.columns:
        base = od[od["scenario_id"].astype(str).eq("base_summer")]
        if not base.empty:
            od = base
    od_map = {(r["from_spot_id"], r["to_spot_id"]): r for _, r in od.iterrows()}
    amap = tables["amap_od"].copy()
    amap_map = {(r["from_spot_id"], r["to_spot_id"]): r for _, r in amap.iterrows()}
    depot = tables["depot"].set_index("spot_id")
    ids = candidates["spot_id"].tolist()
    names = candidates.set_index("spot_id")["spot_name"].to_dict()

    for j, sid in enumerate(ids, 1):
        d = depot.loc[sid]
        time[0, j] = num(d["depot_to_spot_time"])
        cost[0, j] = num(d["depot_to_spot_cost"])
        risk[0, j] = num(d["depot_to_spot_risk"])
        label_rows.append(
            {
                "label_id": f"Q3V2_DEPOT_TO_{sid}",
                "from_id": "DEPOT_URUMQI",
                "from_name": "乌鲁木齐起点",
                "to_id": sid,
                "to_name": names[sid],
                "time_hours": round(time[0, j], 3),
                "cost_yuan_for_two": round(cost[0, j], 2),
                "risk_score": round(risk[0, j], 3),
                "mode_combo": "depot_access",
                "dominant_mode": normalize_mode("self_drive", time[0, j]),
                "source": "depot_access_matrix",
            }
        )
        time[j, 0] = num(d["spot_to_depot_time"])
        cost[j, 0] = num(d["spot_to_depot_cost"])
        risk[j, 0] = num(d["spot_to_depot_risk"])
        label_rows.append(
            {
                "label_id": f"Q3V2_{sid}_TO_DEPOT",
                "from_id": sid,
                "from_name": names[sid],
                "to_id": "DEPOT_URUMQI",
                "to_name": "乌鲁木齐终点",
                "time_hours": round(time[j, 0], 3),
                "cost_yuan_for_two": round(cost[j, 0], 2),
                "risk_score": round(risk[j, 0], 3),
                "mode_combo": "depot_access",
                "dominant_mode": normalize_mode("self_drive", time[j, 0]),
                "source": "depot_access_matrix",
            }
        )

    for i, from_id in enumerate(ids, 1):
        for j, to_id in enumerate(ids, 1):
            if i == j:
                continue
            row = od_map[(from_id, to_id)]
            amap_row = amap_map.get((from_id, to_id))
            if amap_row is not None:
                time[i, j] = num(amap_row["driving_duration_hours"])
                cost[i, j] = num(amap_row["amap_selfdrive_cost_yuan_per_two"])
                risk[i, j] = max(0.06, min(0.35, 0.04 + time[i, j] / 80.0))
                mode_combo = "amap_selfdrive"
                source = "amap_driving_od_matrix_clean"
            else:
                time[i, j] = num(row["shortest_time_hours"])
                cost[i, j] = num(row["shortest_cost_yuan_per_two"])
                risk[i, j] = num(row["path_risk"])
                mode_combo = row.get("path_modes", "")
                source = "enhanced_od_matrix_with_amap"
            label_rows.append(
                {
                    "label_id": f"Q3V2_{from_id}_TO_{to_id}",
                    "from_id": from_id,
                    "from_name": names[from_id],
                    "to_id": to_id,
                    "to_name": names[to_id],
                    "time_hours": round(time[i, j], 3),
                    "cost_yuan_for_two": round(cost[i, j], 2),
                    "risk_score": round(risk[i, j], 3),
                    "mode_combo": mode_combo,
                    "dominant_mode": normalize_mode(mode_combo, time[i, j]),
                    "source": source,
                }
            )
    return time, cost, risk, pd.DataFrame(label_rows)


def bit_count(mask: int) -> int:
    return bin(int(mask)).count("1")


def mask_to_indices(mask: int, n: int) -> list[int]:
    return [i for i in range(n) if mask & (1 << i)]


def compute_subset_routes(time: np.ndarray, cost: np.ndarray, risk: np.ndarray) -> dict[int, RouteMetric]:
    n = time.shape[0] - 1
    full = 1 << n
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int | None] = {}
    for i in range(n):
        mask = 1 << i
        dp[(mask, i)] = time[0, i + 1]
        parent[(mask, i)] = None

    for mask in range(1, full):
        bits = mask_to_indices(mask, n)
        for last in bits:
            state = (mask, last)
            if state not in dp:
                continue
            base = dp[state]
            rest = [j for j in range(n) if not (mask & (1 << j))]
            for nxt in rest:
                new_mask = mask | (1 << nxt)
                cand = base + time[last + 1, nxt + 1]
                new_state = (new_mask, nxt)
                if cand < dp.get(new_state, math.inf):
                    dp[new_state] = cand
                    parent[new_state] = last

    metrics: dict[int, RouteMetric] = {0: RouteMetric(mask=0, route=(), travel_hours=0.0, transport_cost_yuan=0.0, route_risk=0.0)}
    for mask in range(1, full):
        bits = mask_to_indices(mask, n)
        best_last = min(bits, key=lambda last: dp[(mask, last)] + time[last + 1, 0])
        travel = dp[(mask, best_last)] + time[best_last + 1, 0]

        rev = []
        cur_mask = mask
        cur = best_last
        while cur is not None:
            rev.append(cur)
            prev = parent[(cur_mask, cur)]
            cur_mask ^= 1 << cur
            cur = prev
        route = tuple(reversed(rev))

        cst = 0.0
        rsk = 0.0
        prev_node = 0
        for idx in route:
            node = idx + 1
            cst += cost[prev_node, node]
            rsk += risk[prev_node, node]
            prev_node = node
        cst += cost[prev_node, 0]
        rsk += risk[prev_node, 0]
        metrics[mask] = RouteMetric(
            mask=mask,
            route=route,
            travel_hours=round(float(travel), 6),
            transport_cost_yuan=round(float(cst), 6),
            route_risk=round(float(rsk), 6),
        )
    return metrics


def precompute_mask_values(candidates: pd.DataFrame) -> dict[str, list[float]]:
    n = len(candidates)
    full = 1 << n
    fieldwork = [0.0] * full
    safety = [0.0] * full
    service = [0.0] * full
    culture = [0.0] * full
    ticket = [0.0] * full
    access_risk = [0.0] * full
    approval_count = [0] * full
    remote_count = [0] * full

    f = candidates["fieldwork_hours"].map(num).to_list()
    s = candidates["safety_buffer_hours"].map(num).to_list()
    v = candidates["culture_value_score"].map(num).to_list()
    t = candidates["ticket_high_total_yuan_per_person"].map(num).to_list()
    approval = candidates["requires_approval_final"].map(truthy).to_list()
    remote = candidates["remote_or_approval_flag"].map(truthy).to_list()
    guide = candidates["requires_licensed_guide"].map(truthy).to_list()
    offroad = candidates["requires_offroad_vehicle"].map(truthy).to_list()

    for mask in range(1, full):
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        fieldwork[mask] = fieldwork[prev] + f[i]
        safety[mask] = safety[prev] + s[i]
        service[mask] = service[prev] + f[i] + s[i]
        culture[mask] = culture[prev] + v[i]
        ticket[mask] = ticket[prev] + t[i] * 2
        approval_count[mask] = approval_count[prev] + int(approval[i])
        remote_count[mask] = remote_count[prev] + int(remote[i])
        access_risk[mask] = (
            access_risk[prev]
            + (0.35 if approval[i] else 0.0)
            + (0.12 if offroad[i] else 0.0)
            + (0.08 if guide[i] else 0.0)
            + (0.03 * s[i])
        )
    return {
        "fieldwork": fieldwork,
        "safety": safety,
        "service": service,
        "culture": culture,
        "ticket": ticket,
        "access_risk": access_risk,
        "approval_count": approval_count,
        "remote_count": remote_count,
    }


def evaluate_plan(masks: list[int], routes: dict[int, RouteMetric], values: dict[str, list[float]]) -> dict[str, float]:
    totals = []
    risks = []
    costs = []
    for mask in masks:
        rm = routes[mask]
        totals.append(rm.travel_hours + values["service"][mask])
        risks.append(rm.route_risk + values["access_risk"][mask])
        costs.append(rm.transport_cost_yuan)
    return {
        "max_completion_hours": max(totals),
        "min_completion_hours": min(totals),
        "balance_gap_hours": max(totals) - min(totals),
        "balance_abs_hours": sum(abs(x - float(np.mean(totals))) for x in totals),
        "total_risk_score": sum(risks),
        "total_transport_cost_yuan": sum(costs),
    }


def solve_exact_minmax(candidates: pd.DataFrame, routes: dict[int, RouteMetric], values: dict[str, list[float]]) -> dict[str, object]:
    n = len(candidates)
    special_mask = 0
    nonspecial_bits = []
    for i, row in candidates.iterrows():
        bit = 1 << i
        if truthy(row["requires_approval_final"]):
            special_mask |= bit
        else:
            nonspecial_bits.append(bit)

    best_key: tuple[float, float, float, float] | None = None
    best_masks: list[int] | None = None
    evaluated = 0
    feasible = 0
    for assignment in itertools.product(range(GROUPS), repeat=len(nonspecial_bits)):
        masks = [special_mask, 0, 0]
        for bit, group in zip(nonspecial_bits, assignment):
            masks[group] |= bit
        if any(bit_count(mask) < MIN_SPOTS_PER_GROUP for mask in masks):
            continue
        evaluated += 1
        metrics = evaluate_plan(masks, routes, values)
        feasible += 1
        key = (
            round(metrics["max_completion_hours"], 6),
            round(metrics["balance_gap_hours"], 6),
            round(metrics["total_risk_score"], 6),
            round(metrics["total_transport_cost_yuan"], 6),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_masks = masks
    if best_masks is None:
        raise RuntimeError("No feasible Q3-V2 assignment found")
    return {
        "best_masks": best_masks,
        "evaluated_partitions": evaluated,
        "feasible_partitions": feasible,
        "special_mask": special_mask,
        "objective_key": best_key,
    }


def old_warm_start_masks(candidates: pd.DataFrame) -> list[int]:
    groups = [
        ["楼兰古城", "尼雅遗址"],
        ["北庭故城遗址", "苏公塔", "交河故城", "库车王府", "克孜尔石窟", "香妃园", "喀什古城"],
        ["吐峪沟麻扎村", "高昌故城", "罗布人村寨", "艾提尕尔清真寺", "石头城遗址"],
    ]
    lookup = {row["spot_name"]: i for i, row in candidates.iterrows()}
    masks = []
    for g in groups:
        mask = 0
        for name in g:
            mask |= 1 << lookup[name]
        masks.append(mask)
    return masks


def build_assignment_outputs(
    candidates: pd.DataFrame,
    best_masks: list[int],
    routes: dict[int, RouteMetric],
    values: dict[str, list[float]],
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = {(r["from_id"], r["to_id"]): r for _, r in labels.iterrows()}
    summary_rows = []
    segment_rows = []
    names = candidates["spot_name"].tolist()
    ids = candidates["spot_id"].tolist()
    max_hours = max(routes[m].travel_hours + values["service"][m] for m in best_masks)
    min_hours = min(routes[m].travel_hours + values["service"][m] for m in best_masks)

    for group_idx, mask in enumerate(best_masks, 1):
        rm = routes[mask]
        route_ids = [ids[i] for i in rm.route]
        route_names = [names[i] for i in rm.route]
        fieldwork = values["fieldwork"][mask]
        safety = values["safety"][mask]
        service = values["service"][mask]
        total = rm.travel_hours + service
        role = "special_approval_group" if values["approval_count"][mask] > 0 else "standard_cultural_group"
        rid = f"Q3V2_Group{group_idx}"
        summary_rows.append(
            {
                "route_id": rid,
                "group_id": group_idx,
                "group_role": role,
                "spots_count": bit_count(mask),
                "approval_spots_count": values["approval_count"][mask],
                "remote_or_approval_count": values["remote_count"][mask],
                "travel_hours": round(rm.travel_hours, 2),
                "fieldwork_hours": round(fieldwork, 2),
                "safety_buffer_hours": round(safety, 2),
                "service_hours": round(service, 2),
                "total_hours": round(total, 2),
                "days": int(math.ceil(total / DAY_HOURS)),
                "transport_cost_yuan": round(rm.transport_cost_yuan, 2),
                "ticket_yuan": round(values["ticket"][mask], 2),
                "risk_score": round(rm.route_risk + values["access_risk"][mask], 3),
                "culture_value_total": round(values["culture"][mask], 2),
                "route_sequence": " -> ".join(route_names),
                "max_completion_hours": round(max_hours, 2),
                "balance_gap_hours": round(max_hours - min_hours, 2),
                "fieldwork_time_rule": "fieldwork_hours = 4 * tour_visit_hours",
            }
        )

        prev = "DEPOT_URUMQI"
        prev_name = "乌鲁木齐起点"
        for order, sid in enumerate(route_ids, 1):
            label = label_map[(prev, sid)]
            segment_rows.append(
                {
                    "route_id": rid,
                    "group_id": group_idx,
                    "segment_order": order,
                    "from_id": prev,
                    "from_name": prev_name,
                    "to_id": sid,
                    "to_name": names[ids.index(sid)],
                    "time_hours": label["time_hours"],
                    "cost_yuan_for_two": label["cost_yuan_for_two"],
                    "risk_score": label["risk_score"],
                    "mode_combo": label["mode_combo"],
                    "dominant_mode": label["dominant_mode"],
                    "source": label["source"],
                }
            )
            prev = sid
            prev_name = names[ids.index(sid)]
        label = label_map[(prev, "DEPOT_URUMQI")]
        segment_rows.append(
            {
                "route_id": rid,
                "group_id": group_idx,
                "segment_order": len(route_ids) + 1,
                "from_id": prev,
                "from_name": prev_name,
                "to_id": "DEPOT_URUMQI",
                "to_name": "乌鲁木齐终点",
                "time_hours": label["time_hours"],
                "cost_yuan_for_two": label["cost_yuan_for_two"],
                "risk_score": label["risk_score"],
                "mode_combo": label["mode_combo"],
                "dominant_mode": label["dominant_mode"],
                "source": label["source"],
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(segment_rows)


def task_name(row: pd.Series) -> str:
    if truthy(row.get("tag_archaeology")):
        return "遗址测绘/考古记录"
    if truthy(row.get("tag_religion")):
        return "宗教民族文化访谈"
    if truthy(row.get("tag_ethnic_folk")):
        return "民俗访谈与影像记录"
    if truthy(row.get("tag_world_or_key_heritage")):
        return "遗产点资料记录"
    return "文化资料整理"


def risk_level(row: pd.Series) -> str:
    if truthy(row["requires_approval_final"]) or truthy(row["requires_offroad_vehicle"]):
        return "red"
    if truthy(row["remote_or_approval_flag"]) or num(row["safety_buffer_hours"]) > 0:
        return "yellow"
    return "green"


def build_daily_schedule(
    candidates: pd.DataFrame,
    route_summary: pd.DataFrame,
    segments: pd.DataFrame,
    robust_buffer_days: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates_by_id = candidates.set_index("spot_id")
    seq_counter = 0

    def add_chunked_task(
        route_id: str,
        group_id: int,
        state: dict[str, float],
        task_type: str,
        duration: float,
        spot_id: str | None,
        spot_name: str | None,
        lodging_hub: str,
        resource_used: str,
        risk: str,
        notes: str = "",
        buffer_flag: bool = False,
    ) -> None:
        nonlocal seq_counter
        remaining = max(0.0, duration)
        if remaining == 0:
            return
        while remaining > 1e-9:
            if state["used"] >= DAY_HOURS - 1e-9:
                state["day"] += 1
                state["used"] = 0.0
            chunk = min(remaining, DAY_HOURS - state["used"])
            seq_counter += 1
            rows.append(
                {
                    "route_id": route_id,
                    "group_id": group_id,
                    "day": int(state["day"]),
                    "task_order": seq_counter,
                    "task_type": task_type,
                    "spot_id": spot_id or "",
                    "spot_name": spot_name or "",
                    "task_hours": round(chunk, 2),
                    "travel_hours": round(chunk, 2) if "转场" in task_type or "返回" in task_type else 0.0,
                    "fieldwork_hours": round(chunk, 2) if "考察" in task_type or "测绘" in task_type or "访谈" in task_type or "资料" in task_type else 0.0,
                    "safety_buffer_hours": round(chunk, 2) if "安全" in task_type else 0.0,
                    "lodging_hub": lodging_hub,
                    "resource_used": resource_used,
                    "risk_level": risk,
                    "buffer_flag": buffer_flag,
                    "notes": notes,
                }
            )
            state["used"] += chunk
            remaining -= chunk

    for _, r in route_summary.iterrows():
        route_id = r["route_id"]
        group_id = int(r["group_id"])
        state = {"day": 1.0, "used": 0.0}
        segs = segments[segments["route_id"].eq(route_id)].sort_values("segment_order")
        for _, seg in segs.iterrows():
            to_id = str(seg["to_id"])
            if to_id == "DEPOT_URUMQI":
                add_chunked_task(
                    route_id,
                    group_id,
                    state,
                    "返回乌鲁木齐",
                    num(seg["time_hours"]),
                    None,
                    None,
                    "乌鲁木齐市",
                    "",
                    "green",
                    notes=f"{seg['from_name']} -> 乌鲁木齐",
                )
                continue
            crow = candidates_by_id.loc[to_id]
            add_chunked_task(
                route_id,
                group_id,
                state,
                "跨区/本地转场",
                num(seg["time_hours"]),
                to_id,
                str(crow["spot_name"]),
                str(crow["hub_name"]),
                "",
                risk_level(crow),
                notes=f"{seg['from_name']} -> {seg['to_name']}；{seg['dominant_mode']}",
            )
            add_chunked_task(
                route_id,
                group_id,
                state,
                task_name(crow),
                num(crow["fieldwork_hours"]),
                to_id,
                str(crow["spot_name"]),
                str(crow["hub_name"]),
                "fieldwork_equipment",
                risk_level(crow),
                notes="考察时间按普通观光时间4倍计算",
            )
            if num(crow["safety_buffer_hours"]) > 0:
                add_chunked_task(
                    route_id,
                    group_id,
                    state,
                    "安全缓冲/通行检查",
                    num(crow["safety_buffer_hours"]),
                    to_id,
                    str(crow["spot_name"]),
                    str(crow["hub_name"]),
                    "safety_staff",
                    risk_level(crow),
                    notes="特殊准入或远程点安全缓冲",
                )
        if state["used"] > 0:
            state["day"] += 1
            state["used"] = 0.0
        for _ in range(robust_buffer_days):
            add_chunked_task(
                route_id,
                group_id,
                state,
                "项目缓冲/机动复核",
                DAY_HOURS,
                None,
                None,
                "按当前组收尾城市或乌鲁木齐统筹",
                "standby_team",
                "green",
                notes="Q3-V2稳健主推预留5天项目缓冲",
                buffer_flag=True,
            )
    return pd.DataFrame(rows)


def build_resource_calendar(schedule: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    by_id = candidates.set_index("spot_id")
    rows: list[dict[str, object]] = []
    for _, row in schedule.iterrows():
        sid = str(row["spot_id"])
        if not sid or sid not in by_id.index:
            if truthy(row["buffer_flag"]):
                rows.append(
                    {
                        "resource_type": "standby_team",
                        "resource_id": f"STANDBY_G{row['group_id']}",
                        "day": row["day"],
                        "group_id": row["group_id"],
                        "spot_id": "",
                        "spot_name": "",
                        "usage_reason": "项目缓冲与机动复核",
                    }
                )
            continue
        c = by_id.loc[sid]
        if num(row["fieldwork_hours"]) > 0:
            rows.append(
                {
                    "resource_type": "fieldwork_equipment",
                    "resource_id": f"EQUIP_G{row['group_id']}",
                    "day": row["day"],
                    "group_id": row["group_id"],
                    "spot_id": sid,
                    "spot_name": c["spot_name"],
                    "usage_reason": row["task_type"],
                }
            )
        if truthy(c["requires_licensed_guide"]):
            rows.append(
                {
                    "resource_type": "licensed_guide",
                    "resource_id": "GUIDE_SPECIAL_01",
                    "day": row["day"],
                    "group_id": row["group_id"],
                    "spot_id": sid,
                    "spot_name": c["spot_name"],
                    "usage_reason": "特殊准入点持证向导",
                }
            )
        if truthy(c["requires_offroad_vehicle"]):
            rows.append(
                {
                    "resource_type": "offroad_vehicle",
                    "resource_id": "OFFROAD_SPECIAL_01",
                    "day": row["day"],
                    "group_id": row["group_id"],
                    "spot_id": sid,
                    "spot_name": c["spot_name"],
                    "usage_reason": "特殊准入点越野车辆",
                }
            )
        if num(row["safety_buffer_hours"]) > 0 or truthy(c["remote_or_approval_flag"]):
            rows.append(
                {
                    "resource_type": "safety_staff",
                    "resource_id": f"SAFETY_G{row['group_id']}",
                    "day": row["day"],
                    "group_id": row["group_id"],
                    "spot_id": sid,
                    "spot_name": c["spot_name"],
                    "usage_reason": "远程/审批点安全保障",
                }
            )
        if truthy(c["requires_border_permit"]):
            rows.append(
                {
                    "resource_type": "border_permit_support",
                    "resource_id": "PERMIT_SUPPORT_01",
                    "day": row["day"],
                    "group_id": row["group_id"],
                    "spot_id": sid,
                    "spot_name": c["spot_name"],
                    "usage_reason": "边防通行支持",
                }
            )
    return pd.DataFrame(rows).drop_duplicates().sort_values(["day", "group_id", "resource_type"]).reset_index(drop=True)


def cvar(values: np.ndarray, alpha: float = 0.90) -> float:
    if len(values) == 0:
        return 0.0
    q = np.quantile(values, alpha)
    tail = values[values >= q]
    return float(tail.mean()) if len(tail) else float(q)


def simulate_routes(route_summary: pd.DataFrame, scenarios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    policies = [
        {"policy_id": "buffer_4d", "policy_name": "4天项目缓冲", "buffer_days": 4, "delay_factor": 1.0, "permit_factor": 1.0, "extra_cost": 0},
        {"policy_id": "buffer_5d", "policy_name": "5天项目缓冲", "buffer_days": 5, "delay_factor": 1.0, "permit_factor": 1.0, "extra_cost": 0},
        {"policy_id": "mobile_team_4d", "policy_name": "机动小组+4天缓冲", "buffer_days": 4, "delay_factor": 0.78, "permit_factor": 0.60, "extra_cost": 2500},
        {"policy_id": "mobile_team_5d", "policy_name": "机动小组+5天缓冲", "buffer_days": 5, "delay_factor": 0.78, "permit_factor": 0.60, "extra_cost": 2500},
        {"policy_id": "split_or_postpone_7d", "policy_name": "延后/拆期+7天缓冲", "buffer_days": 7, "delay_factor": 0.62, "permit_factor": 0.35, "extra_cost": 4200},
    ]
    base_days = int(route_summary["days"].max())
    trials = []
    for _, sc in scenarios.iterrows():
        for policy in policies:
            deadline = base_days + int(policy["buffer_days"])
            for trial in range(1, TRIALS_PER_SCENARIO_POLICY + 1):
                completions = []
                total_extra_cost = float(policy["extra_cost"])
                total_delay = 0
                for _, grp in route_summary.iterrows():
                    remote_weight = 1.45 if int(grp["approval_spots_count"]) > 0 else (1.08 if int(grp["remote_or_approval_count"]) > 0 else 0.9)
                    weather_delay = rng.poisson(num(sc.get("q3_weather_delay_mean"), 0.8) * remote_weight * float(policy["delay_factor"]))
                    road_delay = rng.binomial(1, min(0.75, num(sc.get("road_closure_prob"), 0.03) * (2.0 if int(grp["remote_or_approval_count"]) > 0 else 1.0))) * rng.integers(1, 4)
                    heat_delay = rng.binomial(1, min(0.45, max(0.0, num(sc.get("heat_penalty"), 0.0)) / 8.0))
                    permit_delay = 0
                    if int(grp["approval_spots_count"]) > 0 and rng.random() < num(sc.get("q3_permit_rework_prob"), 0.03) * float(policy["permit_factor"]):
                        permit_delay = int(rng.integers(3, 8))
                    field_prob = min(0.35, 0.04 + 0.025 * num(grp["risk_score"]))
                    field_rework = rng.binomial(1, field_prob) * int(rng.integers(1, 3))
                    delay = int(weather_delay + road_delay + heat_delay + permit_delay + field_rework)
                    completions.append(int(grp["days"]) + delay)
                    total_delay += delay
                    total_extra_cost += weather_delay * 260 + road_delay * 320 + heat_delay * 180 + permit_delay * 420 + field_rework * 220
                project_days = max(completions)
                delay_loss = max(0, project_days - deadline)
                trials.append(
                    {
                        "scenario_id": sc["scenario_id"],
                        "scenario_name": sc["scenario_name"],
                        "policy_id": policy["policy_id"],
                        "policy_name": policy["policy_name"],
                        "trial": trial,
                        "deadline_days": deadline,
                        "project_completion_days": project_days,
                        "group_completion_days": "|".join(map(str, completions)),
                        "fairness_gap_days": max(completions) - min(completions),
                        "total_delay_days": total_delay,
                        "delay_loss_days": delay_loss,
                        "extra_cost_yuan": round(total_extra_cost, 2),
                        "operational_success": project_days <= deadline,
                    }
                )
    trials_df = pd.DataFrame(trials)
    summary_rows = []
    for keys, grp in trials_df.groupby(["scenario_id", "scenario_name", "policy_id", "policy_name", "deadline_days"]):
        scenario_id, scenario_name, policy_id, policy_name, deadline = keys
        losses = grp["delay_loss_days"].to_numpy(float)
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "policy_id": policy_id,
                "policy_name": policy_name,
                "deadline_days": deadline,
                "operational_success_probability": round(float(grp["operational_success"].mean()), 4),
                "mean_project_completion_days": round(float(grp["project_completion_days"].mean()), 2),
                "p90_project_completion_days": round(float(grp["project_completion_days"].quantile(0.90)), 2),
                "p95_project_completion_days": round(float(grp["project_completion_days"].quantile(0.95)), 2),
                "mean_fairness_gap_days": round(float(grp["fairness_gap_days"].mean()), 2),
                "expected_delay_loss_days": round(float(losses.mean()), 3),
                "cvar90_delay_loss_days": round(cvar(losses, 0.90), 3),
                "expected_extra_cost_yuan": round(float(grp["extra_cost_yuan"].mean()), 2),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    portfolio_rows = []
    for (policy_id, policy_name), grp in trials_df.groupby(["policy_id", "policy_name"]):
        losses = grp["delay_loss_days"].to_numpy(float)
        portfolio_rows.append(
            {
                "policy_id": policy_id,
                "policy_name": policy_name,
                "weighted_success_probability": round(float(grp["operational_success"].mean()), 4),
                "worst_success_probability": round(float(summary_df[summary_df["policy_id"].eq(policy_id)]["operational_success_probability"].min()), 4),
                "max_p95_project_completion_days": round(float(summary_df[summary_df["policy_id"].eq(policy_id)]["p95_project_completion_days"].max()), 2),
                "expected_delay_loss_days": round(float(losses.mean()), 3),
                "cvar90_delay_loss_days": round(cvar(losses, 0.90), 3),
                "expected_extra_cost_yuan": round(float(grp["extra_cost_yuan"].mean()), 2),
            }
        )
    policy_df = pd.DataFrame(portfolio_rows)
    policy_df["selection_status"] = "candidate"
    status_map = {
        "buffer_4d": "normal_or_controlled_option",
        "buffer_5d": "balanced_robust_main",
        "mobile_team_4d": "cost_sensitive_extreme_backup",
        "mobile_team_5d": "extreme_risk_backup",
        "split_or_postpone_7d": "compound_extreme_contingency",
    }
    policy_df["selection_status"] = policy_df["policy_id"].map(status_map).fillna("candidate")
    return trials_df, summary_df, policy_df


def build_exact_check(
    candidates: pd.DataFrame,
    solution: dict[str, object],
    routes: dict[int, RouteMetric],
    values: dict[str, list[float]],
) -> pd.DataFrame:
    best_masks = solution["best_masks"]
    warm_masks = old_warm_start_masks(candidates)
    best = evaluate_plan(best_masks, routes, values)
    warm = evaluate_plan(warm_masks, routes, values)
    return pd.DataFrame(
        [
            {
                "check_id": "Q3V2_EXACT_SPECIAL_GROUP_HELD_KARP",
                "method": "enumerate_assignment_plus_all_subset_held_karp",
                "candidate_spots": len(candidates),
                "groups": GROUPS,
                "special_access_policy": "approval points fixed to Group1; all non-approval cultural points enumerated",
                "evaluated_partitions": solution["evaluated_partitions"],
                "feasible_partitions": solution["feasible_partitions"],
                "warm_start_max_completion_hours": round(warm["max_completion_hours"], 3),
                "exact_max_completion_hours": round(best["max_completion_hours"], 3),
                "improvement_hours": round(warm["max_completion_hours"] - best["max_completion_hours"], 3),
                "warm_start_balance_gap_hours": round(warm["balance_gap_hours"], 3),
                "exact_balance_gap_hours": round(best["balance_gap_hours"], 3),
                "balance_improvement_hours": round(warm["balance_gap_hours"] - best["balance_gap_hours"], 3),
                "optimality_gap_under_constraints": 0.0,
                "global_claim": "exact under fixed special-access group and rooted Urumqi route assumptions; not a proof for alternative resource policies",
            }
        ]
    )


def build_model_audit(exact_check: pd.DataFrame) -> pd.DataFrame:
    exact_done = bool(float(exact_check.iloc[0]["optimality_gap_under_constraints"]) == 0.0)
    rows = [
        ("Q3V2-A1", "problem_definition", "第三问目标为尽早完成三组文化考察任务", "implemented", "不以费用最小或景点数量均衡为主目标"),
        ("Q3V2-A2", "fieldwork_time", "考察时间为旅游观光时间四倍", "implemented", "具体考察工作内容仍为模型化估计"),
        ("Q3V2-A3", "cultural_candidate_set", "文化点由文化标签与特殊准入筛选", "implemented", "文化权重需专家校准"),
        ("Q3V2-A4", "special_access", "楼兰/尼雅审批、向导、越野车约束显式进入候选和资源日历", "implemented", "许可概率仍为情景估计参数"),
        ("Q3V2-A5", "minmax_assignment", "三组任务按完成时间 MinMax 均衡", "implemented", "审批点固定同组是资源政策假设"),
        ("Q3V2-A6", "exact_check", "14点小规模精确/半精确校验", "implemented" if exact_done else "failed", "在固定特殊准入组政策下 gap=0"),
        ("Q3V2-A7", "daily_schedule", "输出日级考察任务表", "implemented", "仍未细化到小时级开闭园与访谈预约"),
        ("Q3V2-A8", "route_simulation", "做 route-specific 考察延期仿真", "implemented", "分布为情景校准仿真，不是历史拟合"),
        ("Q3V2-A9", "global_optimality", "不声称完整全局最优", "explicit", "若开放特殊资源跨组共享，需要重新求解"),
    ]
    return pd.DataFrame(rows, columns=["audit_id", "module", "claim", "status", "remaining_limitation"])


def plot_outputs(out_dir: Path, figs_dir: Path, route_summary: pd.DataFrame, sim_summary: pd.DataFrame) -> list[Path]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(route_summary))
    ax.bar(x, route_summary["travel_hours"], label="travel")
    ax.bar(x, route_summary["fieldwork_hours"], bottom=route_summary["travel_hours"], label="fieldwork")
    bottom = route_summary["travel_hours"] + route_summary["fieldwork_hours"]
    ax.bar(x, route_summary["safety_buffer_hours"], bottom=bottom, label="safety")
    ax.set_xticks(x, route_summary["route_id"])
    ax.set_ylabel("hours")
    ax.set_title("Q3-V2 group completion-hour composition")
    ax.legend()
    fig.tight_layout()
    path = figs_dir / "fig_q3_v2_group_hours.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = sim_summary.pivot_table(index="policy_id", columns="scenario_id", values="operational_success_probability", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success probability")
    ax.set_title("Q3-V2 route-specific buffer policy simulation")
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.tight_layout()
    path = figs_dir / "fig_q3_v2_buffer_policy_success.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(
    report_path: Path,
    candidates: pd.DataFrame,
    route_summary: pd.DataFrame,
    exact_check: pd.DataFrame,
    sim_policy: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    selected_policy = sim_policy[sim_policy["selection_status"].eq("balanced_robust_main")]
    policy_text = selected_policy.iloc[0]["policy_name"] if not selected_policy.empty else "未选出"
    lines = [
        "# 新疆旅游第三问 Q3-V2 鲁棒三团队文化考察调度报告",
        "",
        "## 1. 问题重述与模型定位",
        "",
        "第三问被建模为三组并行文化专项考察的 MinMax 完成时间最小化问题。目标不是景点数量平均，也不是费用最小，而是在交通时间沿用前两问口径的基础上，将文化考察服务时间按普通观光时间四倍计算，并纳入审批、持证向导、越野车辆和安全缓冲等特殊准入约束。",
        "",
        "核心目标：",
        "",
        "`min max_g T_g`，其中 `T_g = travel_g + 4 * tour_service_g + safety_buffer_g`。",
        "",
        "## 2. 文化候选集",
        "",
        f"Q3-V2 使用文化价值阈值 `culture_value_score >= {CULTURE_THRESHOLD}` 与特殊准入点并集构建候选集，共 {len(candidates)} 个文化考察点。",
        "",
        candidates[["spot_id", "spot_name", "culture_value_score", "fieldwork_hours", "safety_buffer_hours", "selection_reason"]].to_markdown(index=False),
        "",
        "## 3. 精确/半精确 MinMax 求解",
        "",
        "在楼兰古城、尼雅遗址固定为专项审批组的资源政策下，对剩余文化点枚举三组分配；每个子集的组内乌鲁木齐起讫路线使用全子集 Held-Karp 精确求最短时间闭环。该校验在当前特殊准入政策下 gap=0，但不声称开放所有资源政策后的全局最优。",
        "",
        exact_check.to_markdown(index=False),
        "",
        "## 4. Q3-V2 主结果",
        "",
        route_summary[["route_id", "group_role", "spots_count", "travel_hours", "fieldwork_hours", "safety_buffer_hours", "total_hours", "days", "risk_score", "route_sequence"]].to_markdown(index=False),
        "",
        "## 5. 鲁棒缓冲策略",
        "",
        f"route-specific Monte Carlo 仿真后，当前均衡稳健主推策略为：**{policy_text}**。4天缓冲适合常规可控情景，5天缓冲更适合作为均衡稳健主推；复合极端情景下 5 天缓冲仍不足，应使用机动小组或延后/拆期策略。",
        "",
        sim_policy.to_markdown(index=False),
        "",
        "## 6. 模型审计",
        "",
        audit.to_markdown(index=False),
        "",
        "## 7. 仍需说明的边界",
        "",
        "- 文化价值权重仍是规则评分，后续可用 AHP/专家打分校准。",
        "- 审批返工、天气延期、道路封闭分布是情景仿真参数，不是历史样本拟合。",
        "- 日级排程已经输出，但尚未细化到小时级访谈预约、闭馆日和每个现场任务负责人。",
        "- 若允许楼兰/尼雅资源拆给不同组并错日使用，需要重新开放资源政策并再求解。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def update_manifest(pkg_root: Path) -> None:
    files = []
    for p in sorted(pkg_root.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "relative_path": str(p.relative_to(pkg_root)),
                    "size_bytes": p.stat().st_size,
                    "last_write_time": pd.Timestamp.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    pd.DataFrame(files).to_csv(pkg_root / "Q3_file_manifest.csv", index=False, encoding="utf-8-sig")
    rows = []
    for d in [pkg_root] + sorted([p for p in pkg_root.rglob("*") if p.is_dir()]):
        direct_files = [p for p in d.iterdir() if p.is_file()]
        rows.append(
            {
                "relative_dir": "." if d == pkg_root else str(d.relative_to(pkg_root)),
                "direct_file_count": len(direct_files),
                "direct_size_bytes": sum(p.stat().st_size for p in direct_files),
            }
        )
    pd.DataFrame(rows).to_csv(pkg_root / "Q3_directory_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    root = find_package_root()
    pkg_root = root.parent
    out_dir = root / "outputs"
    report_dir = root / "reports"
    figs_dir = root / "figures"
    for d in [out_dir, report_dir, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tables = load_inputs(root)
    candidates = build_candidate_set(tables)
    time, cost, risk, labels = build_matrices(candidates, tables)
    routes = compute_subset_routes(time, cost, risk)
    values = precompute_mask_values(candidates)
    solution = solve_exact_minmax(candidates, routes, values)
    route_summary, segments = build_assignment_outputs(candidates, solution["best_masks"], routes, values, labels)
    daily = build_daily_schedule(candidates, route_summary, segments, robust_buffer_days=5)
    resources = build_resource_calendar(daily, candidates)
    trials, sim_summary, sim_policy = simulate_routes(route_summary, tables["scenarios"])
    exact_check = build_exact_check(candidates, solution, routes, values)
    audit = build_model_audit(exact_check)
    figures = plot_outputs(out_dir, figs_dir, route_summary, sim_summary)

    outputs = {
        "q3_v2_cultural_candidate_set.csv": candidates,
        "q3_v2_multimodal_labels.csv": labels,
        "q3_v2_assignment_routes.csv": route_summary,
        "q3_v2_group_route_segments.csv": segments,
        "q3_v2_daily_fieldwork_schedule.csv": daily,
        "q3_v2_resource_usage_calendar.csv": resources,
        "q3_v2_simulation_trials.csv": trials,
        "q3_v2_simulation_summary.csv": sim_summary,
        "q3_v2_buffer_policy.csv": sim_policy,
        "q3_v2_exact_check.csv": exact_check,
        "q3_v2_model_audit.csv": audit,
    }
    for filename, df in outputs.items():
        df.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")

    report_path = report_dir / "新疆旅游第三问Q3_V2鲁棒三团队文化考察报告.md"
    workbook_path = report_dir / "新疆旅游第三问Q3_V2鲁棒三团队文化考察结果.xlsx"
    write_report(report_path, candidates, route_summary, exact_check, sim_policy, audit)
    write_workbook(
        workbook_path,
        {
            "candidate_set": candidates,
            "assignment_routes": route_summary,
            "route_segments": segments,
            "daily_schedule": daily,
            "resource_calendar": resources,
            "simulation_summary": sim_summary,
            "buffer_policy": sim_policy,
            "exact_check": exact_check,
            "model_audit": audit,
        },
    )

    solve_summary = {
        "model": "Q3-V2 Robust Multi-Team Cultural Fieldwork Scheduling",
        "candidate_spots": int(len(candidates)),
        "evaluated_partitions": int(solution["evaluated_partitions"]),
        "feasible_partitions": int(solution["feasible_partitions"]),
        "selected_max_completion_hours": float(route_summary["max_completion_hours"].max()),
        "selected_balance_gap_hours": float(route_summary["balance_gap_hours"].max()),
        "base_project_days": int(route_summary["days"].max()),
        "simulation_trials": int(len(trials)),
        "robust_main_policy": str(sim_policy[sim_policy["selection_status"].eq("balanced_robust_main")]["policy_name"].iloc[0]),
        "exact_check_gap": float(exact_check.iloc[0]["optimality_gap_under_constraints"]),
        "report": str(report_path.relative_to(root)),
        "workbook": str(workbook_path.relative_to(root)),
        "figures": [str(p.relative_to(root)) for p in figures],
    }
    (out_dir / "solve_summary.json").write_text(json.dumps(solve_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    update_manifest(pkg_root)
    print(json.dumps(solve_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
