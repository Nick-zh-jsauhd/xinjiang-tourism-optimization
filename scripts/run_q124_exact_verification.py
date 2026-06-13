from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]

Q1_BASE = ROOT / "第一问_30天路线优化材料包" / "09_Q1_V3_鲁棒联合优化"
Q2_BASE = ROOT / "第二问_两年暑假交通费用优化材料包" / "11_Q2_V3_两年鲁棒多模式路径覆盖"
Q4_BASE = ROOT / "第四问_五一容量接待优化材料包" / "09_Q4_V2_12天线路产品组合与分时容量优化"

Q1_EXACT = ROOT / "第一问_30天路线优化材料包" / "10_Q1_V4_精确求解器验证"
Q2_EXACT = ROOT / "第二问_两年暑假交通费用优化材料包" / "12_Q2_V4_精确求解器验证"
Q4_EXACT = ROOT / "第四问_五一容量接待优化材料包" / "10_Q4_V3_精确求解器验证"


def ensure_dirs(base: Path) -> None:
    for child in ("outputs", "reports", "scripts"):
        (base / child).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_sequence(value: str) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split("->") if part.strip()]


def milp_status(result) -> str:
    if result.success:
        return "optimal"
    message = str(getattr(result, "message", "")).lower()
    if "time" in message or getattr(result, "status", None) == 1:
        return "time_limit_with_incumbent" if getattr(result, "x", None) is not None else "time_limit_no_incumbent"
    if getattr(result, "x", None) is not None:
        return "nonoptimal_with_incumbent"
    return f"failed_status_{getattr(result, 'status', 'unknown')}"


@dataclass
class MatrixData:
    matrix: np.ndarray
    node_ids: List[str]
    node_names: Dict[str, str]


def route_value(route: List[int], matrix: np.ndarray) -> float:
    value = 0.0
    for a, b in zip(route[:-1], route[1:]):
        value += float(matrix[a, b])
    return value


def solve_open_hamiltonian_path(
    matrix: np.ndarray,
    start: int,
    end: int,
    time_limit: float,
) -> Tuple[object, List[int], float]:
    """Exact open Hamiltonian path with fixed start/end and MTZ subtour cuts."""

    n = matrix.shape[0]
    x_vars: Dict[Tuple[int, int], int] = {}
    var_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_vars[(i, j)] = var_count
            var_count += 1
    u_offset = var_count
    var_count += n

    c = np.zeros(var_count)
    for (i, j), idx in x_vars.items():
        c[idx] = matrix[i, j]

    integrality = np.zeros(var_count, dtype=int)
    integrality[:u_offset] = 1

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    ub[u_offset:] = n - 1
    for i in range(n):
        if i == start:
            lb[u_offset + i] = 0
            ub[u_offset + i] = 0
        else:
            lb[u_offset + i] = 1

    rows: List[Tuple[Dict[int, float], float, float]] = []

    for i in range(n):
        coeff = {}
        for j in range(n):
            if i != j:
                coeff[x_vars[(i, j)]] = 1.0
        rhs = 1.0
        if i == end:
            rhs = 0.0
        rows.append((coeff, rhs, rhs))

    for j in range(n):
        coeff = {}
        for i in range(n):
            if i != j:
                coeff[x_vars[(i, j)]] = 1.0
        rhs = 1.0
        if j == start:
            rhs = 0.0
        rows.append((coeff, rhs, rhs))

    for i in range(n):
        if i == start:
            continue
        for j in range(n):
            if j == start or i == j:
                continue
            coeff = {
                u_offset + i: 1.0,
                u_offset + j: -1.0,
                x_vars[(i, j)]: float(n),
            }
            rows.append((coeff, -np.inf, n - 1.0))

    A = lil_matrix((len(rows), var_count), dtype=float)
    lb_cons = np.empty(len(rows))
    ub_cons = np.empty(len(rows))
    for r, (coeff, lo, hi) in enumerate(rows):
        for idx, val in coeff.items():
            A[r, idx] = val
        lb_cons[r] = lo
        ub_cons[r] = hi

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A.tocsr(), lb_cons, ub_cons),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )

    route: List[int] = []
    value = math.nan
    if getattr(result, "x", None) is not None:
        chosen = {(i, j): result.x[idx] for (i, j), idx in x_vars.items() if result.x[idx] > 0.5}
        route = [start]
        seen = {start}
        current = start
        for _ in range(n - 1):
            nxt = None
            for (i, j), val in chosen.items():
                if i == current and val > 0.5:
                    nxt = j
                    break
            if nxt is None or nxt in seen:
                break
            route.append(nxt)
            seen.add(nxt)
            current = nxt
        if len(route) == n and route[-1] == end:
            value = route_value(route, matrix)

    return result, route, value


def q1_exact_order() -> dict:
    ensure_dirs(Q1_EXACT)
    selected = read_csv(Q1_BASE / "outputs" / "q1_v3_selected_routes.csv")
    labels = read_csv(Q1_BASE / "outputs" / "q1_v3_multimodal_labels.csv")

    main = selected.loc[selected["route_id"] == "Q1V3_Q24_120"].iloc[0]
    node_ids = split_sequence(main["spot_id_sequence"])
    node_names = {}
    for _, row in labels.iterrows():
        node_names[str(row["from_spot_id"])] = str(row["from_spot_name"])
        node_names[str(row["to_spot_id"])] = str(row["to_spot_name"])

    n = len(node_ids)
    matrix = np.full((n, n), np.inf)
    chosen_label = {}
    label_subset = labels[
        labels["from_spot_id"].isin(node_ids) & labels["to_spot_id"].isin(node_ids)
    ].copy()
    label_subset["time_hours"] = pd.to_numeric(label_subset["time_hours"], errors="coerce")
    label_subset = label_subset.dropna(subset=["time_hours"])
    for (fr, to), group in label_subset.groupby(["from_spot_id", "to_spot_id"]):
        i = node_ids.index(fr)
        j = node_ids.index(to)
        best = group.sort_values(["time_hours", "cost_yuan_for_two"]).iloc[0]
        matrix[i, j] = float(best["time_hours"])
        chosen_label[(fr, to)] = {
            "label_id": best["label_id"],
            "mode_sequence": best["mode_sequence"],
            "time_hours": float(best["time_hours"]),
            "cost_yuan_for_two": float(best["cost_yuan_for_two"]),
        }
    np.fill_diagonal(matrix, 0.0)
    if not np.isfinite(matrix).all():
        missing = np.argwhere(~np.isfinite(matrix))
        raise ValueError(f"Q1 matrix has missing labels: {missing[:5].tolist()}")

    heuristic_route = list(range(n))
    heuristic_value = route_value(heuristic_route, matrix)
    result, exact_route, exact_value = solve_open_hamiltonian_path(
        matrix=matrix,
        start=0,
        end=n - 1,
        time_limit=300.0,
    )

    status = milp_status(result)
    gap_to_exact = math.nan
    if math.isfinite(exact_value) and exact_value > 0:
        gap_to_exact = (heuristic_value - exact_value) / exact_value

    exact_ids = [node_ids[i] for i in exact_route] if exact_route else []
    exact_names = [node_names.get(node_id, node_id) for node_id in exact_ids]
    route_rows = []
    for order, node_id in enumerate(exact_ids, start=1):
        route_rows.append(
            {
                "order": order,
                "spot_id": node_id,
                "spot_name": node_names.get(node_id, node_id),
            }
        )

    check = pd.DataFrame(
        [
            {
                "experiment_id": "Q1V4_FIXED_24_SPOT_OPEN_PATH_MILP",
                "solver": "HiGHS-MILP via scipy.optimize.milp",
                "scope": "Q1V3_Q24_120固定24景点集合，固定首尾点，精确重排交通时间最小开放路径",
                "spots_count": n,
                "start_spot": node_names.get(node_ids[0], node_ids[0]),
                "end_spot": node_names.get(node_ids[-1], node_ids[-1]),
                "solver_status": status,
                "proved_optimal": bool(result.success),
                "heuristic_travel_hours_on_best_labels": round(heuristic_value, 6),
                "exact_travel_hours_on_best_labels": round(exact_value, 6) if math.isfinite(exact_value) else "",
                "relative_gap_vs_exact": round(gap_to_exact, 8) if math.isfinite(gap_to_exact) else "",
                "mip_objective": round(float(result.fun), 6) if getattr(result, "fun", None) is not None else "",
                "mip_gap": getattr(result, "mip_gap", ""),
                "route_sequence_ids": " -> ".join(exact_ids),
                "route_sequence_names": " -> ".join(exact_names),
            }
        ]
    )
    write_csv(check, Q1_EXACT / "outputs" / "q1_v4_exact_order_check.csv")
    write_csv(pd.DataFrame(route_rows), Q1_EXACT / "outputs" / "q1_v4_exact_order_route.csv")

    report = f"""# Q1-V4 精确求解器验证

## 模型边界

本实验不重新选择景点，不替代 Q1-V3 的鲁棒仿真主模型。它只在 Q1-V3 运营鲁棒主推路线 `Q1V3_Q24_120` 的 24 个已选景点上，固定首点 `{node_names.get(node_ids[0], node_ids[0])}` 与尾点 `{node_names.get(node_ids[-1], node_ids[-1])}`，求解开放 Hamilton 路径的交通时间最小精确排序。

数学口径：

- 决策变量：`x_ij` 表示从景点 i 直接转至景点 j，`u_i` 为 MTZ 顺序变量；
- 约束：首点出度为 1 入度为 0，尾点入度为 1 出度为 0，其余景点入度/出度均为 1，并加入 MTZ 子回路消除约束；
- 目标：最小化所选 24 景点之间的多模式非支配标签最小交通时间之和；
- 求解器：HiGHS-MILP via `scipy.optimize.milp`。

## 结果

- 求解状态：`{status}`
- 是否证明最优：`{bool(result.success)}`
- V3 顺序在同一 best-label 时间矩阵下的交通时间：`{heuristic_value:.3f}` 小时
- 精确重排交通时间：`{exact_value:.3f}` 小时
- 相对差距：`{gap_to_exact:.4%}`（正值表示 V3 顺序仍有重排节省空间）

## 论文表述建议

Q1-V3 的主结论仍应表述为“鲁棒多目标多模式定向游的高质量可行解”。Q1-V4 可作为精确子问题校验：在固定主推 24 景点集合与固定首尾点下，路径排序子问题已由 MILP 精确求解，并可用于报告启发式排序与精确排序之间的差距。
"""
    write_text(report, Q1_EXACT / "reports" / "q1_v4_exact_solver_report.md")
    write_text(
        "# Q1-V4 精确求解器验证\n\n"
        "本文件夹用于验证第一问主推路线在固定24景点集合下的精确排序子问题。"
        "输出位于 `outputs/`，报告位于 `reports/`。\n",
        Q1_EXACT / "README.md",
    )
    return check.iloc[0].to_dict()


def q2_cost_matrix(
    spot_ids: List[str],
    labels: pd.DataFrame,
    gateway_labels: pd.DataFrame,
    gateway_name: str = "乌鲁木齐市",
) -> Tuple[np.ndarray, Dict[str, str]]:
    node_ids = [gateway_name] + spot_ids
    n = len(node_ids)
    matrix = np.full((n, n), np.inf)
    np.fill_diagonal(matrix, 0.0)

    labels = labels.copy()
    labels["cost_yuan_for_two"] = pd.to_numeric(labels["cost_yuan_for_two"], errors="coerce")
    for (fr, to), group in labels.groupby(["from_id", "to_id"]):
        if fr in spot_ids and to in spot_ids:
            i = node_ids.index(fr)
            j = node_ids.index(to)
            matrix[i, j] = float(group["cost_yuan_for_two"].min())

    gl = gateway_labels[gateway_labels["gateway_name"] == gateway_name].copy()
    gl["cost_yuan_for_two"] = pd.to_numeric(gl["cost_yuan_for_two"], errors="coerce")
    for _, row in gl.iterrows():
        fr = str(row["from_id"])
        to = str(row["to_id"])
        if fr in node_ids and to in node_ids:
            matrix[node_ids.index(fr), node_ids.index(to)] = float(row["cost_yuan_for_two"])

    if not np.isfinite(matrix).all():
        missing = np.argwhere(~np.isfinite(matrix))
        raise ValueError(f"Q2 cost matrix has missing entries: {missing[:10].tolist()}")

    return matrix, {node_id: node_id for node_id in node_ids}


def solve_two_route_mtz(
    matrix: np.ndarray,
    min_route_nodes: int,
    max_route_nodes: int,
    time_limit: float,
) -> Tuple[object, Dict[int, List[int]], float]:
    """Two rooted routes covering each customer exactly once."""

    n_customers = matrix.shape[0] - 1
    all_nodes = range(n_customers + 1)
    customers = range(1, n_customers + 1)
    groups = range(2)

    x_vars: Dict[Tuple[int, int, int], int] = {}
    var_count = 0
    for g in groups:
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    continue
                x_vars[(g, i, j)] = var_count
                var_count += 1
    y_offset = var_count
    y_vars: Dict[Tuple[int, int], int] = {}
    for g in groups:
        for i in customers:
            y_vars[(g, i)] = var_count
            var_count += 1
    u_offset = var_count
    u_vars: Dict[Tuple[int, int], int] = {}
    for g in groups:
        for i in customers:
            u_vars[(g, i)] = var_count
            var_count += 1

    c = np.zeros(var_count)
    for (g, i, j), idx in x_vars.items():
        c[idx] = matrix[i, j]

    integrality = np.zeros(var_count, dtype=int)
    integrality[:u_offset] = 1

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    ub[u_offset:] = n_customers

    rows: List[Tuple[Dict[int, float], float, float]] = []

    for i in customers:
        coeff = {y_vars[(g, i)]: 1.0 for g in groups}
        rows.append((coeff, 1.0, 1.0))

    for g in groups:
        out_depot = {x_vars[(g, 0, j)]: 1.0 for j in customers}
        in_depot = {x_vars[(g, i, 0)]: 1.0 for i in customers}
        rows.append((out_depot, 1.0, 1.0))
        rows.append((in_depot, 1.0, 1.0))

        size_coeff = {y_vars[(g, i)]: 1.0 for i in customers}
        rows.append((size_coeff, float(min_route_nodes), float(max_route_nodes)))

        for i in customers:
            out_coeff = {x_vars[(g, i, j)]: 1.0 for j in all_nodes if j != i}
            out_coeff[y_vars[(g, i)]] = -1.0
            rows.append((out_coeff, 0.0, 0.0))

            in_coeff = {x_vars[(g, j, i)]: 1.0 for j in all_nodes if j != i}
            in_coeff[y_vars[(g, i)]] = -1.0
            rows.append((in_coeff, 0.0, 0.0))

            rows.append(({u_vars[(g, i)]: 1.0, y_vars[(g, i)]: -float(n_customers)}, -np.inf, 0.0))
            rows.append(({u_vars[(g, i)]: 1.0, y_vars[(g, i)]: -1.0}, 0.0, np.inf))

        for i in customers:
            for j in customers:
                if i == j:
                    continue
                coeff = {
                    u_vars[(g, i)]: 1.0,
                    u_vars[(g, j)]: -1.0,
                    x_vars[(g, i, j)]: float(n_customers),
                }
                rows.append((coeff, -np.inf, float(n_customers - 1)))

    A = lil_matrix((len(rows), var_count), dtype=float)
    lb_cons = np.empty(len(rows))
    ub_cons = np.empty(len(rows))
    for r, (coeff, lo, hi) in enumerate(rows):
        for idx, val in coeff.items():
            A[r, idx] = val
        lb_cons[r] = lo
        ub_cons[r] = hi

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A.tocsr(), lb_cons, ub_cons),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )

    routes: Dict[int, List[int]] = {}
    value = math.nan
    if getattr(result, "x", None) is not None:
        value = float(result.fun) if getattr(result, "fun", None) is not None else math.nan
        for g in groups:
            selected_edges = {
                (i, j): result.x[idx]
                for (gg, i, j), idx in x_vars.items()
                if gg == g and result.x[idx] > 0.5
            }
            route = [0]
            current = 0
            seen = {0}
            for _ in range(n_customers + 1):
                nxt = None
                for (i, j), val in selected_edges.items():
                    if i == current and val > 0.5:
                        nxt = j
                        break
                if nxt is None:
                    break
                route.append(nxt)
                if nxt == 0:
                    break
                if nxt in seen:
                    break
                seen.add(nxt)
                current = nxt
            routes[g] = route

    return result, routes, value


def rooted_route_cost(order: List[int], matrix: np.ndarray) -> float:
    route = [0] + order + [0]
    return route_value(route, matrix)


def q2_exact_two_year_mip() -> dict:
    ensure_dirs(Q2_EXACT)
    plans = read_csv(Q2_BASE / "outputs" / "q2_v3_candidate_plans.csv")
    labels = read_csv(Q2_BASE / "outputs" / "q2_v3_multimodal_labels.csv")
    gateway_labels = read_csv(Q2_BASE / "outputs" / "q2_v3_gateway_labels.csv")

    rooted = plans.loc[plans["plan_id"] == "Q2V3_072_ROOTED_URUMQI"].iloc[0]
    y1_full = split_sequence(rooted["year1_spot_id_sequence"])
    y2_full = split_sequence(rooted["year2_spot_id_sequence"])
    y1 = y1_full[:10]
    y2 = y2_full[:10]
    spot_ids = y1 + y2
    matrix, _ = q2_cost_matrix(spot_ids, labels, gateway_labels)

    heuristic_y1 = list(range(1, 11))
    heuristic_y2 = list(range(11, 21))
    heuristic_cost = rooted_route_cost(heuristic_y1, matrix) + rooted_route_cost(heuristic_y2, matrix)

    result, exact_routes, exact_cost = solve_two_route_mtz(
        matrix=matrix,
        min_route_nodes=8,
        max_route_nodes=12,
        time_limit=300.0,
    )
    status = milp_status(result)
    gap = math.nan
    if math.isfinite(exact_cost) and exact_cost > 0:
        gap = (heuristic_cost - exact_cost) / exact_cost

    route_rows = []
    for g, route in exact_routes.items():
        clean = [node for node in route if node != 0]
        route_rows.append(
            {
                "year_group": g + 1,
                "spots_count": len(clean),
                "route_ids": "乌鲁木齐市 -> " + " -> ".join(spot_ids[i - 1] for i in clean) + " -> 乌鲁木齐市",
                "route_cost_yuan_for_two": round(route_value(route, matrix), 6) if len(route) >= 2 else "",
            }
        )

    check = pd.DataFrame(
        [
            {
                "experiment_id": "Q2V4_20_NODE_TWO_YEAR_ROOTED_MIP",
                "solver": "HiGHS-MILP via scipy.optimize.milp",
                "scope": "从Q2V3固定乌鲁木齐方案抽取Year1前10点与Year2前10点，精确求解两条乌鲁木齐起讫路线覆盖20节点",
                "spots_count": 20,
                "route_count": 2,
                "min_spots_per_year": 8,
                "max_spots_per_year": 12,
                "solver_status": status,
                "proved_optimal": bool(result.success),
                "heuristic_cost_yuan_for_two": round(heuristic_cost, 6),
                "exact_cost_yuan_for_two": round(exact_cost, 6) if math.isfinite(exact_cost) else "",
                "relative_gap_vs_exact": round(gap, 8) if math.isfinite(gap) else "",
                "mip_objective": round(float(result.fun), 6) if getattr(result, "fun", None) is not None else "",
                "mip_gap": getattr(result, "mip_gap", ""),
            }
        ]
    )
    write_csv(check, Q2_EXACT / "outputs" / "q2_v4_20node_two_year_mip_check.csv")
    write_csv(pd.DataFrame(route_rows), Q2_EXACT / "outputs" / "q2_v4_20node_exact_routes.csv")

    report = f"""# Q2-V4 精确求解器验证

## 模型边界

本实验补上 Q2-V3 中原先未运行的 20 节点两年覆盖 MILP。它不是完整 38 景点两年鲁棒模型的全局最优证明，而是一个代表性子问题的精确求解：

- 节点：从固定乌鲁木齐主方案 `Q2V3_072_ROOTED_URUMQI` 中抽取 Year1 前10个、Year2 前10个普通游客可达景点；
- 路线：两条路线，均从乌鲁木齐出发并返回乌鲁木齐；
- 覆盖：20个景点每个恰好访问一次；
- 均衡：每年 8-12 个景点；
- 目标：最小化两人新疆境内交通费用；
- 约束：使用 MTZ 子回路消除约束；
- 求解器：HiGHS-MILP via `scipy.optimize.milp`。

## 结果

- 求解状态：`{status}`
- 是否证明最优：`{bool(result.success)}`
- 原 Q2-V3 抽样顺序费用：`{heuristic_cost:.2f}` 元
- MILP 精确费用：`{exact_cost:.2f}` 元
- 相对差距：`{gap:.4%}`（正值表示原抽样顺序在该子问题上仍可节省费用）

## 论文表述建议

Q2-V3 仍应作为 38 景点两年鲁棒多模式覆盖主模型；Q2-V4 可作为“20节点两路径精确校验”，证明所用费用矩阵和两年覆盖约束能够被标准 MILP 精确求解，并给出启发式路线在代表性子问题上的最优性差距。
"""
    write_text(report, Q2_EXACT / "reports" / "q2_v4_exact_solver_report.md")
    write_text(
        "# Q2-V4 精确求解器验证\n\n"
        "本文件夹补充第二问的 20 节点两年覆盖 MILP 精确校验。输出位于 `outputs/`，报告位于 `reports/`。\n",
        Q2_EXACT / "README.md",
    )
    return check.iloc[0].to_dict()


def normalize_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return pd.Series(np.full(len(values), 50.0), index=series.index)
    return (values - lo) / (hi - lo) * 100.0


def solve_q4_portfolio(
    candidates: pd.DataFrame,
    selected_counts: Dict[str, int],
    baseline_demand: float,
    current_regions: List[str],
    current_quality_proxy_count: int,
    time_limit: float,
) -> Tuple[object, List[str], float]:
    n = len(candidates)
    c = -candidates["exact_score"].to_numpy(dtype=float)
    integrality = np.ones(n, dtype=int)
    lb = np.zeros(n)
    ub = np.ones(n)

    rows: List[Tuple[Dict[int, float], float, float]] = []
    rows.append(({i: 1.0 for i in range(n)}, 18.0, 18.0))

    capacity = candidates["route_capacity_persons_12day"].astype(float).to_numpy()
    rows.append(({i: capacity[i] for i in range(n)}, baseline_demand, np.inf))

    quality_proxy = candidates["quality_pass_proxy"].astype(float).to_numpy()
    rows.append(({i: quality_proxy[i] for i in range(n)}, float(current_quality_proxy_count), np.inf))

    for product_type, count in selected_counts.items():
        coeff = {i: 1.0 for i, value in enumerate(candidates["product_type"]) if value == product_type}
        rows.append((coeff, float(count), float(count)))

    for region in current_regions:
        coeff = {
            i: 1.0
            for i, value in enumerate(candidates["region_set"].fillna(""))
            if region in str(value).replace("；", ";").split(";")
        }
        if coeff:
            rows.append((coeff, 1.0, np.inf))

    A = lil_matrix((len(rows), n), dtype=float)
    lb_cons = np.empty(len(rows))
    ub_cons = np.empty(len(rows))
    for r, (coeff, lo, hi) in enumerate(rows):
        for idx, val in coeff.items():
            A[r, idx] = val
        lb_cons[r] = lo
        ub_cons[r] = hi

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A.tocsr(), lb_cons, ub_cons),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    chosen_ids: List[str] = []
    score = math.nan
    if getattr(result, "x", None) is not None:
        mask = result.x > 0.5
        chosen_ids = candidates.loc[mask, "route_id"].tolist()
        score = float(candidates.loc[mask, "exact_score"].sum())
    return result, chosen_ids, score


def q4_exact_portfolio() -> dict:
    ensure_dirs(Q4_EXACT)
    candidates = read_csv(Q4_BASE / "outputs" / "q4_v2_candidate_route_products.csv")
    selected = read_csv(Q4_BASE / "outputs" / "q4_v2_selected_route_portfolio.csv")
    summary = json.loads((Q4_BASE / "outputs" / "q4_v2_run_summary.json").read_text(encoding="utf-8"))

    numeric_cols = [
        "route_capacity_persons_12day",
        "portfolio_quality_score",
        "comfort_score",
        "diversity_score",
        "low_pressure_score",
        "attraction_score",
        "resource_intensity_score",
        "avg_interspot_travel_hours_proxy",
        "buffer_days",
        "long_transfer_days_proxy",
    ]
    for col in numeric_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0.0)

    candidates["quality_pass_proxy"] = (
        (candidates["comfort_score"] >= 75.0)
        & (candidates["buffer_days"] >= 2.0)
        & (candidates["long_transfer_days_proxy"] <= 3.0)
        & (candidates["avg_interspot_travel_hours_proxy"] <= 6.0)
    ).astype(int)

    candidates["exact_score"] = (
        0.25 * candidates["portfolio_quality_score"]
        + 0.20 * normalize_100(candidates["route_capacity_persons_12day"])
        + 0.15 * candidates["comfort_score"]
        + 0.15 * candidates["diversity_score"]
        + 0.10 * candidates["low_pressure_score"]
        + 0.10 * candidates["attraction_score"]
        - 0.03 * normalize_100(candidates["resource_intensity_score"])
        - 0.02 * normalize_100(candidates["avg_interspot_travel_hours_proxy"])
    )

    selected_ids = set(selected["route_id"].tolist())
    selected_counts = selected.groupby("product_type")["route_id"].count().to_dict()
    current_regions = sorted(
        {
            region.strip()
            for value in selected["region_set"].fillna("")
            for region in str(value).replace("；", ";").split(";")
            if region.strip()
        }
    )
    current_quality_proxy_count = int(candidates[candidates["route_id"].isin(selected_ids)]["quality_pass_proxy"].sum())
    baseline_demand = float(summary["baseline_demand"])

    result, chosen_ids, exact_score = solve_q4_portfolio(
        candidates=candidates,
        selected_counts=selected_counts,
        baseline_demand=baseline_demand,
        current_regions=current_regions,
        current_quality_proxy_count=current_quality_proxy_count,
        time_limit=120.0,
    )
    status = milp_status(result)

    exact = candidates[candidates["route_id"].isin(chosen_ids)].copy()
    current = candidates[candidates["route_id"].isin(selected_ids)].copy()

    current_score = float(current["exact_score"].sum())
    exact_capacity = float(exact["route_capacity_persons_12day"].sum())
    current_capacity = float(current["route_capacity_persons_12day"].sum())
    overlap_count = len(selected_ids.intersection(set(chosen_ids)))
    improvement = (exact_score - current_score) / abs(current_score) if current_score else math.nan

    exact = exact.sort_values(["product_type", "exact_score"], ascending=[True, False])
    output_cols = [
        "route_id",
        "route_theme",
        "product_type",
        "target_segment",
        "route_capacity_persons_12day",
        "portfolio_quality_score",
        "comfort_score",
        "diversity_score",
        "low_pressure_score",
        "quality_pass_proxy",
        "exact_score",
        "route_sequence",
    ]
    write_csv(exact[output_cols], Q4_EXACT / "outputs" / "q4_v3_exact_portfolio_routes.csv")

    comparison = pd.DataFrame(
        [
            {
                "experiment_id": "Q4V3_63_CANDIDATE_18_ROUTE_PORTFOLIO_MILP",
                "solver": "HiGHS-MILP via scipy.optimize.milp",
                "scope": "63条候选12天线路中，在Q4-V2相同产品类型配额、区域覆盖、容量下界和质量代理约束下，精确选择18条线路最大化综合投放分",
                "candidate_routes": len(candidates),
                "selected_routes": 18,
                "baseline_demand": baseline_demand,
                "solver_status": status,
                "proved_optimal": bool(result.success),
                "current_q4v2_exact_score": round(current_score, 6),
                "milp_exact_score": round(exact_score, 6) if math.isfinite(exact_score) else "",
                "relative_score_improvement": round(improvement, 8) if math.isfinite(improvement) else "",
                "current_q4v2_capacity": round(current_capacity, 6),
                "milp_capacity": round(exact_capacity, 6),
                "current_quality_proxy_count": current_quality_proxy_count,
                "milp_quality_proxy_count": int(exact["quality_pass_proxy"].sum()),
                "route_overlap_count": overlap_count,
                "mip_objective": round(float(result.fun), 6) if getattr(result, "fun", None) is not None else "",
                "mip_gap": getattr(result, "mip_gap", ""),
            }
        ]
    )
    write_csv(comparison, Q4_EXACT / "outputs" / "q4_v3_exact_portfolio_check.csv")

    report = f"""# Q4-V3 精确求解器验证

## 模型边界

本实验不重新生成线路，也不替代 Q4-V2 的容量仿真。它在 Q4-V2 已生成的 63 条 12天候选线路产品中，建立一个二进制 MILP，精确选择 18 条线路。

数学口径：

- 决策变量：`x_r` 表示候选线路 r 是否入选；
- 规模约束：入选线路数固定为 18；
- 容量约束：总 12天承载量不低于题面基准需求 `{baseline_demand:.0f}` 人；
- 产品约束：保持 Q4-V2 入选组合的产品类型配额 `{selected_counts}`；
- 区域约束：覆盖 Q4-V2 已覆盖的全部区域；
- 质量约束：候选表可计算的质量代理通过线路数不低于当前组合的 `{current_quality_proxy_count}` 条；
- 目标：最大化容量、组合质量、舒适度、多样性、低压力和吸引力组成的线性综合投放分；
- 求解器：HiGHS-MILP via `scipy.optimize.milp`。

## 结果

- 求解状态：`{status}`
- 是否证明最优：`{bool(result.success)}`
- Q4-V2 当前组合综合分：`{current_score:.3f}`
- MILP 精确组合综合分：`{exact_score:.3f}`
- 相对提升：`{improvement:.3%}`
- Q4-V2 当前组合容量：`{current_capacity:.0f}` 人
- MILP 精确组合容量：`{exact_capacity:.0f}` 人
- 与 Q4-V2 入选线路重合数：`{overlap_count}/18`

## 论文表述建议

Q4-V2 可继续作为“候选线路生成 + 容量仿真 + 策略评估”的主模型；Q4-V3 可作为候选产品空间内的精确组合优化层。若采用 MILP 组合，应将结论写成“在63条候选线路、固定产品配额和区域覆盖政策下的精确最优投放组合”，而不是“所有可能旅游线路的全局最优”。
"""
    write_text(report, Q4_EXACT / "reports" / "q4_v3_exact_solver_report.md")
    write_text(
        "# Q4-V3 精确求解器验证\n\n"
        "本文件夹在 Q4-V2 的63条候选线路产品上，补充18线路投放组合的 MILP 精确优化。"
        "输出位于 `outputs/`，报告位于 `reports/`。\n",
        Q4_EXACT / "README.md",
    )
    return comparison.iloc[0].to_dict()


def main() -> None:
    summary = {
        "solver": "HiGHS-MILP via scipy.optimize.milp",
        "global_note": "Exact claims are restricted to explicitly defined subproblems/candidate spaces, not the full stochastic route-generation problems.",
        "q1": q1_exact_order(),
        "q2": q2_exact_two_year_mip(),
        "q4": q4_exact_portfolio(),
    }
    out_path = ROOT / "outputs" / "q124_exact_verification_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
