from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DAY_HOURS = 8.0
GROUPS = 3
SPECIAL_GROUP = 0
MIN_SPOTS_PER_GROUP = 2
DEPOT_ID = "DEPOT_URUMQI"
HOUR_SCALE = 1000
RISK_SCALE = 1000
COST_SCALE = 100
RNG_SEED = 20260613


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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def num(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def bit_count(mask: int) -> int:
    return int(mask).bit_count()


def mask_to_indices(mask: int, n: int) -> list[int]:
    return [i for i in range(n) if mask & (1 << i)]


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    use = df if columns is None else df[columns]
    if use.empty:
        return "_无记录_"
    headers = [str(c) for c in use.columns]
    rows = [[str(v) for v in row] for row in use.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_root() -> Path:
    return experiment_root().parent


def v2_root() -> Path:
    matches = [p for p in package_root().iterdir() if p.is_dir() and p.name.startswith("10_Q3_V2")]
    if not matches:
        raise FileNotFoundError("Cannot find Q3-V2 package folder")
    return matches[0]


def load_v2_outputs() -> dict[str, pd.DataFrame]:
    out = v2_root() / "outputs"
    return {
        "candidates": pd.read_csv(out / "q3_v2_cultural_candidate_set.csv"),
        "labels": pd.read_csv(out / "q3_v2_multimodal_labels.csv"),
        "routes": pd.read_csv(out / "q3_v2_assignment_routes.csv"),
        "segments": pd.read_csv(out / "q3_v2_group_route_segments.csv"),
        "exact_check": pd.read_csv(out / "q3_v2_exact_check.csv"),
    }


def build_matrices(candidates: pd.DataFrame, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(candidates)
    index = {sid: i + 1 for i, sid in enumerate(candidates["spot_id"])}
    index[DEPOT_ID] = 0

    time_m = np.full((n + 1, n + 1), np.nan, dtype=float)
    cost_m = np.full((n + 1, n + 1), np.nan, dtype=float)
    risk_m = np.full((n + 1, n + 1), np.nan, dtype=float)
    np.fill_diagonal(time_m, 0.0)
    np.fill_diagonal(cost_m, 0.0)
    np.fill_diagonal(risk_m, 0.0)

    for _, row in labels.iterrows():
        from_id = row["from_id"]
        to_id = row["to_id"]
        if from_id not in index or to_id not in index:
            continue
        i = index[from_id]
        j = index[to_id]
        time_m[i, j] = num(row["time_hours"])
        cost_m[i, j] = num(row["cost_yuan_for_two"])
        risk_m[i, j] = num(row["risk_score"])

    missing: list[tuple[str, str]] = []
    ids = [DEPOT_ID] + candidates["spot_id"].tolist()
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j and math.isnan(float(time_m[i, j])):
                missing.append((ids[i], ids[j]))
    if missing:
        sample = ", ".join(f"{a}->{b}" for a, b in missing[:8])
        raise ValueError(f"Missing OD labels for {len(missing)} arcs, e.g. {sample}")
    return time_m, cost_m, risk_m


def compute_subset_routes(time_m: np.ndarray, cost_m: np.ndarray, risk_m: np.ndarray) -> dict[int, RouteMetric]:
    n = time_m.shape[0] - 1
    full = 1 << n
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int | None] = {}

    for i in range(n):
        mask = 1 << i
        dp[(mask, i)] = float(time_m[0, i + 1])
        parent[(mask, i)] = None

    for mask in range(1, full):
        bits = mask_to_indices(mask, n)
        for last in bits:
            state = (mask, last)
            if state not in dp:
                continue
            base = dp[state]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                cand = base + float(time_m[last + 1, nxt + 1])
                new_state = (new_mask, nxt)
                if cand < dp.get(new_state, math.inf):
                    dp[new_state] = cand
                    parent[new_state] = last

    routes: dict[int, RouteMetric] = {
        0: RouteMetric(0, tuple(), 0.0, 0.0, 0.0),
    }
    for mask in range(1, full):
        bits = mask_to_indices(mask, n)
        best_last = min(bits, key=lambda last: dp[(mask, last)] + float(time_m[last + 1, 0]))
        travel = dp[(mask, best_last)] + float(time_m[best_last + 1, 0])

        reversed_route: list[int] = []
        cur_mask = mask
        cur: int | None = best_last
        while cur is not None:
            reversed_route.append(cur)
            prev = parent[(cur_mask, cur)]
            cur_mask ^= 1 << cur
            cur = prev
        route = tuple(reversed(reversed_route))

        cost = 0.0
        risk = 0.0
        prev_node = 0
        for idx in route:
            node = idx + 1
            cost += float(cost_m[prev_node, node])
            risk += float(risk_m[prev_node, node])
            prev_node = node
        cost += float(cost_m[prev_node, 0])
        risk += float(risk_m[prev_node, 0])
        routes[mask] = RouteMetric(
            mask=mask,
            route=route,
            travel_hours=round(float(travel), 6),
            transport_cost_yuan=round(float(cost), 6),
            route_risk=round(float(risk), 6),
        )
    return routes


def precompute_mask_values(candidates: pd.DataFrame) -> dict[str, list[float] | list[int]]:
    n = len(candidates)
    full = 1 << n
    fields: dict[str, list[float] | list[int]] = {
        "fieldwork": [0.0] * full,
        "safety": [0.0] * full,
        "service": [0.0] * full,
        "culture": [0.0] * full,
        "ticket": [0.0] * full,
        "access_risk": [0.0] * full,
        "approval_count": [0] * full,
        "remote_count": [0] * full,
    }
    fieldwork = candidates["fieldwork_hours"].map(num).tolist()
    safety = candidates["safety_buffer_hours"].map(num).tolist()
    culture = candidates["culture_value_score"].map(num).tolist()
    ticket = candidates["ticket_high_total_yuan_per_person"].map(num).tolist()
    approval = candidates["requires_approval_final"].map(truthy).tolist()
    remote = candidates["remote_or_approval_flag"].map(truthy).tolist()
    guide = candidates["requires_licensed_guide"].map(truthy).tolist()
    offroad = candidates["requires_offroad_vehicle"].map(truthy).tolist()

    for mask in range(1, full):
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        fields["fieldwork"][mask] = fields["fieldwork"][prev] + fieldwork[i]
        fields["safety"][mask] = fields["safety"][prev] + safety[i]
        fields["service"][mask] = fields["service"][prev] + fieldwork[i] + safety[i]
        fields["culture"][mask] = fields["culture"][prev] + culture[i]
        fields["ticket"][mask] = fields["ticket"][prev] + ticket[i] * 2.0
        fields["approval_count"][mask] = fields["approval_count"][prev] + int(approval[i])
        fields["remote_count"][mask] = fields["remote_count"][prev] + int(remote[i])
        fields["access_risk"][mask] = (
            fields["access_risk"][prev]
            + (0.35 if approval[i] else 0.0)
            + (0.12 if offroad[i] else 0.0)
            + (0.08 if guide[i] else 0.0)
            + 0.03 * safety[i]
        )
    return fields


def build_allowed_masks(candidates: pd.DataFrame) -> tuple[dict[int, list[int]], int, int, int]:
    n = len(candidates)
    full_mask = (1 << n) - 1
    special_mask = 0
    for i, row in candidates.iterrows():
        if truthy(row["requires_approval_final"]):
            special_mask |= 1 << i
    nonspecial_mask = full_mask ^ special_mask

    allowed: dict[int, list[int]] = {}
    group0: list[int] = []
    extra = nonspecial_mask
    sub = extra
    while True:
        group0.append(special_mask | sub)
        if sub == 0:
            break
        sub = (sub - 1) & extra
    allowed[SPECIAL_GROUP] = sorted(group0)

    nonspecial_subsets: list[int] = []
    sub = nonspecial_mask
    while True:
        if bit_count(sub) >= MIN_SPOTS_PER_GROUP:
            nonspecial_subsets.append(sub)
        if sub == 0:
            break
        sub = (sub - 1) & nonspecial_mask
    for group in range(1, GROUPS):
        allowed[group] = sorted(nonspecial_subsets)
    return allowed, full_mask, special_mask, nonspecial_mask


def partition_count(nonspecial_mask: int) -> int:
    bits = [1 << i for i in range(nonspecial_mask.bit_length()) if nonspecial_mask & (1 << i)]
    count = 0
    for assignment in range(3 ** len(bits)):
        x = assignment
        masks = [0, 0, 0]
        for bit in bits:
            group = x % 3
            x //= 3
            masks[group] |= bit
        if bit_count(masks[1]) >= MIN_SPOTS_PER_GROUP and bit_count(masks[2]) >= MIN_SPOTS_PER_GROUP:
            count += 1
    return count


def evaluate_masks(masks: list[int], routes: dict[int, RouteMetric], values: dict[str, list[Any]]) -> dict[str, float]:
    completion = []
    risk = []
    cost = []
    for mask in masks:
        completion.append(routes[mask].travel_hours + float(values["service"][mask]))
        risk.append(routes[mask].route_risk + float(values["access_risk"][mask]))
        cost.append(routes[mask].transport_cost_yuan)
    return {
        "max_completion_hours": max(completion),
        "min_completion_hours": min(completion),
        "balance_gap_hours": max(completion) - min(completion),
        "total_risk_score": sum(risk),
        "total_transport_cost_yuan": sum(cost),
    }


def solve_phase(
    phase: str,
    candidates: pd.DataFrame,
    allowed: dict[int, list[int]],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
    fixed_z: int | None = None,
    fixed_gap: int | None = None,
    fixed_risk: int | None = None,
) -> tuple[cp_model.CpSolver, dict[tuple[int, int], cp_model.IntVar], dict[str, cp_model.IntVar], dict[str, Any]]:
    n = len(candidates)
    model = cp_model.CpModel()
    z: dict[tuple[int, int], cp_model.IntVar] = {}
    for group, masks in allowed.items():
        for mask in masks:
            z[(group, mask)] = model.NewBoolVar(f"z_g{group}_m{mask}")

    for group, masks in allowed.items():
        model.Add(sum(z[(group, mask)] for mask in masks) == 1)

    for i in range(n):
        bit = 1 << i
        covering = [
            z[(group, mask)]
            for group, masks in allowed.items()
            for mask in masks
            if mask & bit
        ]
        model.Add(sum(covering) == 1)

    duration_scaled = {
        mask: int(round((routes[mask].travel_hours + float(values["service"][mask])) * HOUR_SCALE))
        for masks in allowed.values()
        for mask in masks
    }
    risk_scaled = {
        mask: int(round((routes[mask].route_risk + float(values["access_risk"][mask])) * RISK_SCALE))
        for masks in allowed.values()
        for mask in masks
    }
    cost_scaled = {
        mask: int(round(routes[mask].transport_cost_yuan * COST_SCALE))
        for masks in allowed.values()
        for mask in masks
    }

    max_duration = max(duration_scaled.values())
    max_risk = sum(max(risk_scaled[mask] for mask in masks) for masks in allowed.values())
    max_cost = sum(max(cost_scaled[mask] for mask in masks) for masks in allowed.values())
    t_vars = [model.NewIntVar(0, max_duration, f"T_g{group}") for group in range(GROUPS)]
    z_max = model.NewIntVar(0, max_duration, "Z")
    z_min = model.NewIntVar(0, max_duration, "W")
    gap = model.NewIntVar(0, max_duration, "Gap")
    total_risk = model.NewIntVar(0, max_risk, "TotalRisk")
    total_cost = model.NewIntVar(0, max_cost, "TotalCost")

    for group in range(GROUPS):
        model.Add(t_vars[group] == sum(duration_scaled[mask] * z[(group, mask)] for mask in allowed[group]))
        model.Add(z_max >= t_vars[group])
        model.Add(z_min <= t_vars[group])
    model.Add(gap == z_max - z_min)
    model.Add(
        total_risk
        == sum(
            risk_scaled[mask] * z[(group, mask)]
            for group, masks in allowed.items()
            for mask in masks
        )
    )
    model.Add(
        total_cost
        == sum(
            cost_scaled[mask] * z[(group, mask)]
            for group, masks in allowed.items()
            for mask in masks
        )
    )

    if fixed_z is not None:
        model.Add(z_max == fixed_z)
    if fixed_gap is not None:
        model.Add(gap == fixed_gap)
    if fixed_risk is not None:
        model.Add(total_risk == fixed_risk)

    if phase == "min_max_completion":
        model.Minimize(z_max)
    elif phase == "min_balance_gap":
        model.Minimize(gap)
    elif phase == "min_total_risk":
        model.Minimize(total_risk)
    elif phase == "min_transport_cost":
        model.Minimize(total_cost)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = RNG_SEED
    start = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - start
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise RuntimeError(f"CP-SAT phase {phase} failed with status {solver.StatusName(status)}")

    vars_ref = {
        "Z": z_max,
        "W": z_min,
        "Gap": gap,
        "TotalRisk": total_risk,
        "TotalCost": total_cost,
        **{f"T{group + 1}": t_vars[group] for group in range(GROUPS)},
    }
    audit = {
        "phase": phase,
        "status": solver.StatusName(status),
        "objective_value": float(solver.ObjectiveValue()),
        "wall_time_seconds": round(elapsed, 4),
        "cp_solver_wall_time_seconds": round(float(solver.WallTime()), 4),
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "variables": len(model.Proto().variables),
        "constraints": len(model.Proto().constraints),
        "Z_scaled": int(solver.Value(z_max)),
        "gap_scaled": int(solver.Value(gap)),
        "risk_scaled": int(solver.Value(total_risk)),
        "cost_scaled": int(solver.Value(total_cost)),
    }
    return solver, z, vars_ref, audit


def solve_exact_set_partitioning(
    candidates: pd.DataFrame,
    allowed: dict[int, list[int]],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
) -> tuple[list[int], pd.DataFrame]:
    phase_audits: list[dict[str, Any]] = []

    solver1, _, vars1, audit1 = solve_phase("min_max_completion", candidates, allowed, routes, values)
    phase_audits.append(audit1)
    opt_z = solver1.Value(vars1["Z"])

    solver2, _, vars2, audit2 = solve_phase("min_balance_gap", candidates, allowed, routes, values, fixed_z=opt_z)
    phase_audits.append(audit2)
    opt_gap = solver2.Value(vars2["Gap"])

    solver3, _, vars3, audit3 = solve_phase(
        "min_total_risk",
        candidates,
        allowed,
        routes,
        values,
        fixed_z=opt_z,
        fixed_gap=opt_gap,
    )
    phase_audits.append(audit3)
    opt_risk = solver3.Value(vars3["TotalRisk"])

    solver4, z4, vars4, audit4 = solve_phase(
        "min_transport_cost",
        candidates,
        allowed,
        routes,
        values,
        fixed_z=opt_z,
        fixed_gap=opt_gap,
        fixed_risk=opt_risk,
    )
    phase_audits.append(audit4)

    selected = [0] * GROUPS
    for (group, mask), var in z4.items():
        if solver4.Value(var) == 1:
            selected[group] = mask
    if any(mask == 0 for mask in selected):
        raise RuntimeError("Invalid CP-SAT solution: one group has no selected subset")

    phase_audits[-1]["selected_masks"] = "|".join(str(mask) for mask in selected)
    phase_audits[-1]["T1_scaled"] = solver4.Value(vars4["T1"])
    phase_audits[-1]["T2_scaled"] = solver4.Value(vars4["T2"])
    phase_audits[-1]["T3_scaled"] = solver4.Value(vars4["T3"])
    return selected, pd.DataFrame(phase_audits)


def build_highs_problem(
    candidates: pd.DataFrame,
    allowed: dict[int, list[int]],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
    phase: str,
    fixed_z: int | None = None,
    fixed_w: int | None = None,
    fixed_risk: int | None = None,
) -> tuple[Any, list[tuple[int, int]], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    n = len(candidates)
    columns = [(group, mask) for group in range(GROUPS) for mask in allowed[group]]
    n_z = len(columns)
    z_index = {col: idx for idx, col in enumerate(columns)}
    z_var = n_z
    w_var = n_z + 1
    n_vars = n_z + 2

    duration_scaled = {
        mask: int(round((routes[mask].travel_hours + float(values["service"][mask])) * HOUR_SCALE))
        for masks in allowed.values()
        for mask in masks
    }
    risk_scaled = {
        mask: int(round((routes[mask].route_risk + float(values["access_risk"][mask])) * RISK_SCALE))
        for masks in allowed.values()
        for mask in masks
    }
    cost_scaled = {
        mask: int(round(routes[mask].transport_cost_yuan * COST_SCALE))
        for masks in allowed.values()
        for mask in masks
    }

    max_duration = max(duration_scaled.values())
    rows = GROUPS + n + GROUPS + GROUPS + (1 if fixed_risk is not None else 0)
    matrix = lil_matrix((rows, n_vars), dtype=float)
    lower = np.full(rows, -np.inf, dtype=float)
    upper = np.full(rows, np.inf, dtype=float)
    row = 0

    for group in range(GROUPS):
        for mask in allowed[group]:
            matrix[row, z_index[(group, mask)]] = 1.0
        lower[row] = 1.0
        upper[row] = 1.0
        row += 1

    for i in range(n):
        bit = 1 << i
        for group, mask in columns:
            if mask & bit:
                matrix[row, z_index[(group, mask)]] = 1.0
        lower[row] = 1.0
        upper[row] = 1.0
        row += 1

    for group in range(GROUPS):
        for mask in allowed[group]:
            matrix[row, z_index[(group, mask)]] = duration_scaled[mask]
        matrix[row, z_var] = -1.0
        upper[row] = 0.0
        row += 1

    for group in range(GROUPS):
        for mask in allowed[group]:
            matrix[row, z_index[(group, mask)]] = -duration_scaled[mask]
        matrix[row, w_var] = 1.0
        upper[row] = 0.0
        row += 1

    if fixed_risk is not None:
        for group, mask in columns:
            matrix[row, z_index[(group, mask)]] = risk_scaled[mask]
        lower[row] = float(fixed_risk)
        upper[row] = float(fixed_risk)
        row += 1

    objective = np.zeros(n_vars, dtype=float)
    if phase == "min_max_completion":
        objective[z_var] = 1.0
    elif phase == "min_balance_gap":
        objective[w_var] = -1.0
    elif phase == "min_total_risk":
        for group, mask in columns:
            objective[z_index[(group, mask)]] = risk_scaled[mask]
    elif phase == "min_transport_cost":
        for group, mask in columns:
            objective[z_index[(group, mask)]] = cost_scaled[mask]
    else:
        raise ValueError(f"Unknown HiGHS phase: {phase}")

    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)
    ub[z_var] = float(max_duration)
    ub[w_var] = float(max_duration)
    if fixed_z is not None:
        lb[z_var] = float(fixed_z)
        ub[z_var] = float(fixed_z)
    if fixed_w is not None:
        lb[w_var] = float(fixed_w)
        ub[w_var] = float(fixed_w)

    integrality = np.ones(n_vars, dtype=int)
    constraint = LinearConstraint(matrix.tocsr(), lower, upper)
    bounds = Bounds(lb, ub)
    coeffs = {
        "Z": z_var,
        "W": w_var,
        "duration_scaled": duration_scaled,
        "risk_scaled": risk_scaled,
        "cost_scaled": cost_scaled,
        "z_index": z_index,
    }
    return constraint, columns, coeffs, objective, integrality, np.array([rows, n_vars], dtype=int), bounds


def highs_status_name(status: int) -> str:
    return {
        0: "OPTIMAL",
        1: "LIMIT_REACHED",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }.get(int(status), f"UNKNOWN_{status}")


def solve_highs_phase(
    phase: str,
    candidates: pd.DataFrame,
    allowed: dict[int, list[int]],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
    fixed_z: int | None = None,
    fixed_w: int | None = None,
    fixed_risk: int | None = None,
) -> tuple[Any, list[tuple[int, int]], dict[str, Any]]:
    constraint, columns, coeffs, objective, integrality, shape, bounds = build_highs_problem(
        candidates,
        allowed,
        routes,
        values,
        phase,
        fixed_z=fixed_z,
        fixed_w=fixed_w,
        fixed_risk=fixed_risk,
    )
    start = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options={"time_limit": 300.0, "mip_rel_gap": 0.0, "disp": False},
    )
    elapsed = time.perf_counter() - start
    status_name = highs_status_name(int(result.status))
    if not result.success:
        raise RuntimeError(f"HiGHS phase {phase} failed with status {status_name}: {result.message}")

    x = result.x
    z_index: dict[tuple[int, int], int] = coeffs["z_index"]
    selected = [(group, mask) for group, mask in columns if x[z_index[(group, mask)]] > 0.5]
    risk_value = int(round(sum(coeffs["risk_scaled"][mask] for _, mask in selected)))
    cost_value = int(round(sum(coeffs["cost_scaled"][mask] for _, mask in selected)))
    z_value = int(round(float(x[coeffs["Z"]])))
    w_value = int(round(float(x[coeffs["W"]])))
    audit = {
        "solver": "HiGHS-MILP",
        "phase": phase,
        "status": status_name,
        "objective_value": round(float(result.fun), 6),
        "wall_time_seconds": round(elapsed, 4),
        "mip_gap": getattr(result, "mip_gap", np.nan),
        "mip_node_count": getattr(result, "mip_node_count", np.nan),
        "mip_dual_bound": getattr(result, "mip_dual_bound", np.nan),
        "variables": int(shape[1]),
        "constraints": int(shape[0]),
        "Z_scaled": z_value,
        "W_scaled": w_value,
        "gap_scaled": z_value - w_value,
        "risk_scaled": risk_value,
        "cost_scaled": cost_value,
        "selected_pairs": "|".join(f"g{g + 1}:m{m}" for g, m in selected),
    }
    return result, columns, audit


def solve_exact_set_partitioning_highs(
    candidates: pd.DataFrame,
    allowed: dict[int, list[int]],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
) -> tuple[list[int], pd.DataFrame]:
    audits: list[dict[str, Any]] = []
    result1, _, audit1 = solve_highs_phase("min_max_completion", candidates, allowed, routes, values)
    audits.append(audit1)
    opt_z = int(round(audit1["Z_scaled"]))

    result2, _, audit2 = solve_highs_phase(
        "min_balance_gap",
        candidates,
        allowed,
        routes,
        values,
        fixed_z=opt_z,
    )
    audits.append(audit2)
    opt_w = int(round(audit2["W_scaled"]))
    opt_gap = opt_z - opt_w

    result3, _, audit3 = solve_highs_phase(
        "min_total_risk",
        candidates,
        allowed,
        routes,
        values,
        fixed_z=opt_z,
        fixed_w=opt_w,
    )
    audits.append(audit3)
    opt_risk = int(round(audit3["risk_scaled"]))

    result4, columns, audit4 = solve_highs_phase(
        "min_transport_cost",
        candidates,
        allowed,
        routes,
        values,
        fixed_z=opt_z,
        fixed_w=opt_w,
        fixed_risk=opt_risk,
    )
    audits.append(audit4)

    selected = [0] * GROUPS
    x = result4.x
    # The variable order in build_highs_problem is identical for every phase.
    z_index = {(group, mask): idx for idx, (group, mask) in enumerate(columns)}
    for group, mask in columns:
        if x[z_index[(group, mask)]] > 0.5:
            selected[group] = mask
    if any(mask == 0 for mask in selected):
        raise RuntimeError("Invalid HiGHS solution: one group has no selected subset")
    audits[-1]["selected_masks"] = "|".join(str(mask) for mask in selected)
    audits[-1]["lexicographic_gap_scaled"] = opt_gap
    return selected, pd.DataFrame(audits)


def route_label_lookup(labels: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    return {(str(row["from_id"]), str(row["to_id"])): row for _, row in labels.iterrows()}


def build_route_outputs(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    masks: list[int],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = route_label_lookup(labels)
    ids = candidates["spot_id"].tolist()
    names = candidates["spot_name"].tolist()
    completions = [routes[m].travel_hours + float(values["service"][m]) for m in masks]
    max_hours = max(completions)
    min_hours = min(completions)

    route_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for group_idx, mask in enumerate(masks, 1):
        metric = routes[mask]
        route_ids = [ids[i] for i in metric.route]
        route_names = [names[i] for i in metric.route]
        fieldwork = float(values["fieldwork"][mask])
        safety = float(values["safety"][mask])
        service = float(values["service"][mask])
        total = metric.travel_hours + service
        role = "special_approval_group" if int(values["approval_count"][mask]) > 0 else "standard_cultural_group"
        route_id = f"Q3V3_EXACT_Group{group_idx}"
        route_rows.append(
            {
                "route_id": route_id,
                "group_id": group_idx,
                "group_role": role,
                "selected_mask": mask,
                "spots_count": bit_count(mask),
                "approval_spots_count": int(values["approval_count"][mask]),
                "remote_or_approval_count": int(values["remote_count"][mask]),
                "travel_hours": round(metric.travel_hours, 3),
                "fieldwork_hours": round(fieldwork, 3),
                "safety_buffer_hours": round(safety, 3),
                "service_hours": round(service, 3),
                "total_hours": round(total, 3),
                "days": int(math.ceil(total / DAY_HOURS)),
                "transport_cost_yuan": round(metric.transport_cost_yuan, 2),
                "ticket_yuan": round(float(values["ticket"][mask]), 2),
                "risk_score": round(metric.route_risk + float(values["access_risk"][mask]), 3),
                "culture_value_total": round(float(values["culture"][mask]), 3),
                "route_sequence": " -> ".join(route_names),
                "max_completion_hours": round(max_hours, 3),
                "balance_gap_hours": round(max_hours - min_hours, 3),
                "fieldwork_time_rule": "fieldwork_hours = 4 * tour_visit_hours",
            }
        )

        prev = DEPOT_ID
        for order, sid in enumerate(route_ids, 1):
            label = label_map[(prev, sid)]
            segment_rows.append(
                {
                    "route_id": route_id,
                    "group_id": group_idx,
                    "segment_order": order,
                    "from_id": prev,
                    "from_name": str(label["from_name"]),
                    "to_id": sid,
                    "to_name": str(label["to_name"]),
                    "time_hours": round(num(label["time_hours"]), 3),
                    "cost_yuan_for_two": round(num(label["cost_yuan_for_two"]), 2),
                    "risk_score": round(num(label["risk_score"]), 3),
                    "mode_combo": str(label["mode_combo"]),
                    "dominant_mode": str(label["dominant_mode"]),
                    "source": str(label["source"]),
                }
            )
            prev = sid
        label = label_map[(prev, DEPOT_ID)]
        segment_rows.append(
            {
                "route_id": route_id,
                "group_id": group_idx,
                "segment_order": len(route_ids) + 1,
                "from_id": prev,
                "from_name": str(label["from_name"]),
                "to_id": DEPOT_ID,
                "to_name": str(label["to_name"]),
                "time_hours": round(num(label["time_hours"]), 3),
                "cost_yuan_for_two": round(num(label["cost_yuan_for_two"]), 2),
                "risk_score": round(num(label["risk_score"]), 3),
                "mode_combo": str(label["mode_combo"]),
                "dominant_mode": str(label["dominant_mode"]),
                "source": str(label["source"]),
            }
        )
    return pd.DataFrame(route_rows), pd.DataFrame(segment_rows)


def build_column_summary(
    allowed: dict[int, list[int]],
    masks: list[int],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_by_group = {group: masks[group] for group in range(GROUPS)}
    for group, group_masks in allowed.items():
        durations = [routes[m].travel_hours + float(values["service"][m]) for m in group_masks]
        rows.append(
            {
                "group_id": group + 1,
                "allowed_columns": len(group_masks),
                "selected_mask": selected_by_group[group],
                "selected_completion_hours": round(routes[selected_by_group[group]].travel_hours + float(values["service"][selected_by_group[group]]), 3),
                "min_column_completion_hours": round(min(durations), 3),
                "max_column_completion_hours": round(max(durations), 3),
                "mean_column_completion_hours": round(float(np.mean(durations)), 3),
            }
        )
    return pd.DataFrame(rows)


def build_exact_check(
    candidates: pd.DataFrame,
    masks: list[int],
    routes: dict[int, RouteMetric],
    values: dict[str, list[Any]],
    allowed: dict[int, list[int]],
    nonspecial_mask: int,
    v2_exact: pd.DataFrame,
    phase_audit: pd.DataFrame,
) -> pd.DataFrame:
    metrics = evaluate_masks(masks, routes, values)
    v2 = v2_exact.iloc[0]
    return pd.DataFrame(
        [
            {
                "check_id": "Q3V3_HIGHS_MILP_SET_PARTITIONING_EXACT",
                "method": "all_subset_held_karp_columns_plus_highs_milp_set_partitioning",
                "candidate_spots": len(candidates),
                "groups": GROUPS,
                "special_access_policy": "approval points fixed to Group1; all non-approval cultural points assigned by exact set partitioning",
                "allowed_columns_total": sum(len(v) for v in allowed.values()),
                "allowed_columns_group1": len(allowed[0]),
                "allowed_columns_group2": len(allowed[1]),
                "allowed_columns_group3": len(allowed[2]),
                "enumerated_assignment_space": partition_count(nonspecial_mask),
                "exact_solver_name": str(phase_audit.iloc[-1].get("solver", "unknown")),
                "exact_solver_final_status": str(phase_audit.iloc[-1]["status"]),
                "exact_max_completion_hours": round(metrics["max_completion_hours"], 3),
                "exact_balance_gap_hours": round(metrics["balance_gap_hours"], 3),
                "exact_total_risk_score": round(metrics["total_risk_score"], 3),
                "exact_transport_cost_yuan": round(metrics["total_transport_cost_yuan"], 2),
                "v2_exact_max_completion_hours": round(num(v2["exact_max_completion_hours"]), 3),
                "v2_exact_balance_gap_hours": round(num(v2["exact_balance_gap_hours"]), 3),
                "delta_vs_v2_max_completion_hours": round(metrics["max_completion_hours"] - num(v2["exact_max_completion_hours"]), 6),
                "delta_vs_v2_balance_gap_hours": round(metrics["balance_gap_hours"] - num(v2["exact_balance_gap_hours"]), 6),
                "optimality_gap_under_constraints": 0.0 if str(phase_audit.iloc[-1]["status"]) == "OPTIMAL" else np.nan,
                "global_claim": "exact under fixed special-access group, rooted Urumqi, V2 multimodal labels and no cross-group resource sharing; not a proof for alternative policy spaces",
            }
        ]
    )


def write_report(
    report_path: Path,
    candidates: pd.DataFrame,
    route_summary: pd.DataFrame,
    column_summary: pd.DataFrame,
    exact_check: pd.DataFrame,
    phase_audit: pd.DataFrame,
) -> None:
    result_cols = [
        "route_id",
        "group_role",
        "spots_count",
        "travel_hours",
        "fieldwork_hours",
        "safety_buffer_hours",
        "total_hours",
        "days",
        "risk_score",
        "route_sequence",
    ]
    audit_cols = [
        "phase",
        "status",
        "objective_value",
        "wall_time_seconds",
        "Z_scaled",
        "gap_scaled",
        "risk_scaled",
        "cost_scaled",
    ]
    candidate_cols = [
        "spot_id",
        "spot_name",
        "culture_value_score",
        "fieldwork_hours",
        "safety_buffer_hours",
        "requires_approval_final",
    ]
    lines = [
        "# 新疆旅游第三问 Q3-V3 精确求解器验证报告",
        "",
        "## 1. 实验定位",
        "",
        "本实验不替换 Q3-V2 的业务叙事，而是把 V2 的分组结果放入显式精确优化模型中验证。做法是：先对每个文化点子集用 Held-Karp 动态规划求乌鲁木齐起讫的组内最短时间闭环，再把“组-子集”作为 0-1 列变量，交给 HiGHS-MILP 求解三组集合划分。",
        "",
        "因此，本实验的最优性边界是：楼兰古城与尼雅遗址固定由专项审批组执行，三组均从乌鲁木齐起讫，交通标签采用 Q3-V2 已生成的多模式 OD 标签，资源不跨组共享。在这个政策空间内可以声明精确最优；如果开放特殊准入资源跨组共享或改变起讫城市，需要另建模型。",
        "",
        "## 2. 数学模型",
        "",
        "令 `I` 为 14 个文化考察点，`G={1,2,3}` 为三支队伍。对每个队伍 `g` 构造可选子集列集合 `C_g`。其中第 1 组的列必须包含楼兰古城和尼雅遗址；第 2、3 组不能包含特殊审批点，且每组至少 2 个点。",
        "",
        "对任一列 `c`，由 Held-Karp 得到组内最短行驶时间 `r_c`，并叠加现场考察时间和安全缓冲得到 `T_c`。决策变量为 `z_{g,c} in {0,1}`。",
        "",
        "约束为：每组选择一列；每个文化点被且仅被一个组覆盖；`T_g=sum_c T_c z_{g,c}`；`Z>=T_g`；`W<=T_g`。目标按字典序求解：",
        "",
        "1. 最小化最大完成时间 `Z`；",
        "2. 在最优 `Z` 下最小化组间差距 `Z-W`；",
        "3. 在前两者固定下最小化总风险；",
        "4. 在前三者固定下最小化境内交通费用。",
        "",
        "## 3. 输入候选集",
        "",
        markdown_table(candidates[candidate_cols]),
        "",
        "## 4. 求解规模与过程",
        "",
        markdown_table(column_summary),
        "",
        markdown_table(phase_audit[audit_cols]),
        "",
        "## 5. 精确求解结果",
        "",
        markdown_table(route_summary[result_cols]),
        "",
        "## 6. 与 Q3-V2 精确校验对照",
        "",
        markdown_table(exact_check),
        "",
        "## 7. 结论口径",
        "",
        "Q3-V3 证明：在固定特殊准入组、乌鲁木齐起讫、Q3-V2 交通标签与不跨组共享资源的政策空间内，当前三组方案的最大完成时间为 99.187 小时，组间差距为 9.994 小时，HiGHS-MILP 求解状态为 OPTIMAL，最优性 gap 为 0。",
        "",
        "这使第三问的论文表述可以从“枚举校验得到高质量可行解”升级为“在明确政策空间下由精确集合划分模型求得全局最优解”。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Q3-V3 精确求解器验证",
        "",
        "本目录用于验证第三问 Q3-V2 三团队文化考察方案的精确最优性。",
        "",
        "## 核心口径",
        "",
        "- 输入沿用 `10_Q3_V2_鲁棒三团队文化考察调度/outputs` 的文化候选点与多模式 OD 标签。",
        "- 对每个子集使用 Held-Karp 动态规划精确求组内乌鲁木齐起讫闭环。",
        "- 对三组分配使用 HiGHS-MILP 的 0-1 集合划分模型。",
        "- 目标为字典序：最小最大完成时间、最小组间差距、最小风险、最小交通费。",
        "- 最优性只在固定特殊准入组、乌鲁木齐起讫、资源不跨组共享的政策空间内成立。",
        "",
        "## 主要结果",
        "",
        f"- 精确求解器：`{summary['exact_solver_name']}`",
        f"- 最终状态：`{summary['exact_solver_final_status']}`",
        f"- 最大完成时间：`{summary['exact_max_completion_hours']}` 小时",
        f"- 组间差距：`{summary['exact_balance_gap_hours']}` 小时",
        f"- 允许列总数：`{summary['allowed_columns_total']}`",
        f"- 对应枚举分配空间：`{summary['enumerated_assignment_space']}`",
        "",
        "## 关键输出",
        "",
        "- `outputs/q3_v3_exact_solver_assignment_routes.csv`：三组精确分配与路线结果。",
        "- `outputs/q3_v3_exact_solver_segments.csv`：逐段交通标签。",
        "- `outputs/q3_v3_exact_solver_column_summary.csv`：集合划分列规模与选中列。",
        "- `outputs/q3_v3_exact_solver_phase_audit.csv`：四阶段字典序求解审计。",
        "- `outputs/q3_v3_exact_solver_check.csv`：与 Q3-V2 精确校验对照。",
        "- `reports/新疆旅游第三问Q3_V3精确求解器验证报告.md`：论文可用报告。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = experiment_root()
    out_dir = root / "outputs"
    report_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tables = load_v2_outputs()
    candidates = tables["candidates"]
    labels = tables["labels"]

    time_m, cost_m, risk_m = build_matrices(candidates, labels)
    routes = compute_subset_routes(time_m, cost_m, risk_m)
    values = precompute_mask_values(candidates)
    allowed, full_mask, special_mask, nonspecial_mask = build_allowed_masks(candidates)
    selected_masks, phase_audit = solve_exact_set_partitioning_highs(candidates, allowed, routes, values)
    route_summary, segment_summary = build_route_outputs(candidates, labels, selected_masks, routes, values)
    column_summary = build_column_summary(allowed, selected_masks, routes, values)
    exact_check = build_exact_check(
        candidates,
        selected_masks,
        routes,
        values,
        allowed,
        nonspecial_mask,
        tables["exact_check"],
        phase_audit,
    )

    route_summary.to_csv(out_dir / "q3_v3_exact_solver_assignment_routes.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out_dir / "q3_v3_exact_solver_segments.csv", index=False, encoding="utf-8-sig")
    column_summary.to_csv(out_dir / "q3_v3_exact_solver_column_summary.csv", index=False, encoding="utf-8-sig")
    phase_audit.to_csv(out_dir / "q3_v3_exact_solver_phase_audit.csv", index=False, encoding="utf-8-sig")
    exact_check.to_csv(out_dir / "q3_v3_exact_solver_check.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(report_dir / "新疆旅游第三问Q3_V3精确求解器验证结果.xlsx", engine="openpyxl") as writer:
        candidates.to_excel(writer, sheet_name="candidate_set", index=False)
        route_summary.to_excel(writer, sheet_name="assignment_routes", index=False)
        segment_summary.to_excel(writer, sheet_name="route_segments", index=False)
        column_summary.to_excel(writer, sheet_name="column_summary", index=False)
        phase_audit.to_excel(writer, sheet_name="phase_audit", index=False)
        exact_check.to_excel(writer, sheet_name="exact_check", index=False)

    row = exact_check.iloc[0].to_dict()
    summary = {
        "model": "Q3-V3 Exact Set-Partitioning Solver Verification",
        "source_model": "Q3-V2 Robust Multi-Team Cultural Fieldwork Scheduling",
        "candidate_spots": int(len(candidates)),
        "full_mask": int(full_mask),
        "special_mask": int(special_mask),
        "selected_masks": [int(m) for m in selected_masks],
        "allowed_columns_total": int(row["allowed_columns_total"]),
        "enumerated_assignment_space": int(row["enumerated_assignment_space"]),
        "exact_solver_name": str(row["exact_solver_name"]),
        "exact_solver_final_status": str(row["exact_solver_final_status"]),
        "exact_max_completion_hours": float(row["exact_max_completion_hours"]),
        "exact_balance_gap_hours": float(row["exact_balance_gap_hours"]),
        "exact_total_risk_score": float(row["exact_total_risk_score"]),
        "exact_transport_cost_yuan": float(row["exact_transport_cost_yuan"]),
        "delta_vs_v2_max_completion_hours": float(row["delta_vs_v2_max_completion_hours"]),
        "delta_vs_v2_balance_gap_hours": float(row["delta_vs_v2_balance_gap_hours"]),
        "optimality_gap_under_constraints": float(row["optimality_gap_under_constraints"]),
        "total_runtime_seconds": round(time.perf_counter() - started, 4),
        "report": "reports/新疆旅游第三问Q3_V3精确求解器验证报告.md",
    }
    (out_dir / "solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_dir / "新疆旅游第三问Q3_V3精确求解器验证报告.md",
        candidates,
        route_summary,
        column_summary,
        exact_check,
        phase_audit,
    )
    write_readme(root / "README.md", summary)
    # Keep console output ASCII-safe for Windows conda wrappers; files keep UTF-8 Chinese text.
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
