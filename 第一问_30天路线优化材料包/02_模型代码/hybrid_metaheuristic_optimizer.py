from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
SCRIPT_DIR = ROOT / "scripts"
HYBRID_DIR = ROOT / "hybrid_metaheuristic_outputs"
ADVANCED_DIR = ROOT / "advanced_model_outputs"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(SCRIPT_DIR))
import stochastic_integrated_joint_optimization as base  # noqa: E402


RNG_SEED = 20260610 + 91
ACO_GENERATIONS = 24
ACO_ANTS = 42
HYBRID_ITERATIONS = 900
MONTE_CARLO_TRIALS = 260
MIN_ROUTE_SPOTS = 24
MAX_ROUTE_SPOTS = 38
MAX_SCHEDULED_DAYS_SOFT = 40


def clean(value: Any) -> str:
    return base.clean(value)


def num(value: Any, default: float = 0.0) -> float:
    return base.num(value, default)


def truthy(value: Any) -> bool:
    return base.truthy(value)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def candidate_spots(ctx: dict[str, Any]) -> list[str]:
    out = []
    for _, row in ctx["spots"].iterrows():
        sid = clean(row["spot_id"])
        if sid in ctx["depot_map"] and not truthy(row.get("ordinary_tourist_restricted")):
            out.append(sid)
    return out


def node_key(current: str) -> str:
    return "DEPOT" if current == "DEPOT" else current


class Evaluator:
    def __init__(self, ctx: dict[str, Any]):
        self.ctx = ctx
        self.cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def score(self, route_ids: list[str]) -> dict[str, Any]:
        route_ids = list(dict.fromkeys(route_ids))
        key = tuple(route_ids)
        if key not in self.cache:
            self.cache[key] = base.fast_score_route(route_ids, self.ctx)
        return self.cache[key]

    def objective(self, route_ids: list[str]) -> float:
        return float(self.score(route_ids)["objective"])


def heuristic_attraction(current: str, target: str, ctx: dict[str, Any]) -> float:
    value = num(ctx["value_map"].get(target), 1.0)
    service_h = num(ctx["service_map"].get(target), 1.0)
    ticket = num(ctx["ticket_map"].get(target), 0.0)
    travel = base.travel_time_between(current, target, ctx)
    risk = 0.0
    if current != "DEPOT":
        risk = num(ctx["od_map"].get((current, target), {}).get("risk"), 0.0)
    spot = ctx["spot_map"].get(target, {})
    hotel_bonus = 0.7 if clean(spot.get("hub_name")) in ctx["hotel_hub_set"] else -0.4
    return max(0.05, (value + hotel_bonus) / (0.8 + travel + 0.20 * service_h + ticket / 900.0 + 3.0 * risk))


def roulette(items: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += max(0.0, weight)
        if acc >= r:
            return item
    return items[-1]


def local_order_improve(route_ids: list[str], ctx: dict[str, Any], evaluator: Evaluator, passes: int = 3) -> list[str]:
    original = list(dict.fromkeys(route_ids))
    best = original[:]
    best_obj = evaluator.objective(best)
    route = base.two_opt_order(original, ctx, max_passes=passes)
    route_obj = evaluator.objective(route)
    if route_obj > best_obj:
        best = route[:]
        best_obj = route_obj
    for _ in range(2):
        improved = False
        for i in range(max(0, len(best) - 1)):
            for j in range(i + 1, min(len(best), i + 8)):
                cand = best[:]
                cand[i], cand[j] = cand[j], cand[i]
                obj = evaluator.objective(cand)
                if obj > best_obj:
                    best, best_obj, improved = cand, obj, True
        if not improved:
            break
    return best


def repair_insert(
    route_ids: list[str],
    removed: list[str],
    candidates: list[str],
    ctx: dict[str, Any],
    evaluator: Evaluator,
    rng: random.Random,
    max_insertions: int = 12,
) -> list[str]:
    route = list(dict.fromkeys(route_ids))
    pool = []
    seen = set(route)
    for sid in removed + sorted(candidates, key=lambda x: num(ctx["value_map"].get(x)), reverse=True):
        if sid not in seen and sid not in pool:
            pool.append(sid)
    pool = sorted(pool, key=lambda x: num(ctx["value_map"].get(x)), reverse=True)

    insertions = 0
    for sid in pool[: max(20, max_insertions * 3)]:
        if len(route) >= MAX_ROUTE_SPOTS or insertions >= max_insertions:
            break
        base_obj = evaluator.objective(route)
        positions = list(range(len(route) + 1))
        if len(positions) > 14:
            ranked = []
            for pos in positions:
                prev_id = "DEPOT" if pos == 0 else route[pos - 1]
                next_cost = base.travel_time_between(prev_id, sid, ctx)
                if pos < len(route):
                    next_cost += base.travel_time_between(sid, route[pos], ctx)
                ranked.append((next_cost, pos))
            positions = [p for _, p in sorted(ranked)[:10]]
            positions += rng.sample(list(range(len(route) + 1)), min(4, len(route) + 1))
            positions = list(dict.fromkeys(positions))
        best_pos = None
        best_obj = -math.inf
        for pos in positions:
            cand = route[:pos] + [sid] + route[pos:]
            ev = evaluator.score(cand)
            summary = ev["summary"]
            if num(summary.get("time_window_violations")) > 1:
                continue
            if num(summary.get("scheduled_days")) > MAX_SCHEDULED_DAYS_SOFT + 4:
                continue
            obj = float(ev["objective"])
            if obj > best_obj:
                best_obj, best_pos = obj, pos
        if best_pos is None:
            continue
        forced_coverage = len(route) < MIN_ROUTE_SPOTS and best_obj > base_obj - 6000
        if best_obj > base_obj or forced_coverage:
            route = route[:best_pos] + [sid] + route[best_pos:]
            insertions += 1
    return route


def destroy_route(
    route_ids: list[str],
    ctx: dict[str, Any],
    evaluator: Evaluator,
    rng: random.Random,
    operator: str,
) -> tuple[list[str], list[str]]:
    route = route_ids[:]
    if not route:
        return route, []
    count = max(1, int(len(route) * rng.choice([0.10, 0.16, 0.24, 0.32])))
    removed: list[str]

    if operator == "random_removal":
        removed = rng.sample(route, min(count, len(route)))
    elif operator == "worst_removal":
        base_obj = evaluator.objective(route)
        impacts = []
        sample = route if len(route) <= 28 else rng.sample(route, 28)
        for sid in sample:
            cand = [x for x in route if x != sid]
            impacts.append((evaluator.objective(cand) - base_obj, sid))
        removed = [sid for _, sid in sorted(impacts, reverse=True)[:count]]
    elif operator == "long_arc_removal":
        arcs = []
        for a, b in zip(route[:-1], route[1:]):
            arcs.append((base.travel_time_between(a, b, ctx), b))
        if not arcs:
            removed = rng.sample(route, min(count, len(route)))
        else:
            removed = [sid for _, sid in sorted(arcs, reverse=True)[:count]]
    elif operator == "region_removal":
        regions = defaultdict(list)
        for sid in route:
            regions[clean(ctx["spot_map"].get(sid, {}).get("region_cluster"))].append(sid)
        region = rng.choice([r for r, vals in regions.items() if vals])
        region_spots = regions[region]
        removed = rng.sample(region_spots, min(len(region_spots), count))
    elif operator == "lodging_pressure_removal":
        non_hotel = [sid for sid in route if clean(ctx["spot_map"].get(sid, {}).get("hub_name")) not in ctx["hotel_hub_set"]]
        removed = rng.sample(non_hotel, min(len(non_hotel), count)) if non_hotel else rng.sample(route, min(count, len(route)))
    else:
        removed = rng.sample(route, min(count, len(route)))

    removed_set = set(removed)
    return [sid for sid in route if sid not in removed_set], removed


def aco_initialize(ctx: dict[str, Any], evaluator: Evaluator, candidates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]]]:
    rng = random.Random(RNG_SEED)
    node_list = ["DEPOT"] + candidates
    node_idx = {sid: i for i, sid in enumerate(node_list)}
    pheromone = np.ones((len(node_list), len(node_list)), dtype=float)
    alpha = 1.1
    beta = 3.0
    evaporation = 0.22
    elite_routes: list[list[str]] = []
    generation_rows = []
    solution_rows = []

    for generation in range(1, ACO_GENERATIONS + 1):
        generation_solutions = []
        for ant in range(1, ACO_ANTS + 1):
            route: list[str] = []
            remaining = set(candidates)
            current = "DEPOT"
            failed_extensions = 0
            while remaining and len(route) < MAX_ROUTE_SPOTS and failed_extensions < 10:
                ranked = sorted(
                    remaining,
                    key=lambda sid: heuristic_attraction(current, sid, ctx) * pheromone[node_idx[node_key(current)], node_idx[sid]],
                    reverse=True,
                )
                choice_pool = ranked[: min(16, len(ranked))]
                weights = [
                    (pheromone[node_idx[node_key(current)], node_idx[sid]] ** alpha)
                    * (heuristic_attraction(current, sid, ctx) ** beta)
                    for sid in choice_pool
                ]
                sid = roulette(choice_pool, weights, rng)
                trial = route + [sid]
                ev = evaluator.score(trial)
                summary = ev["summary"]
                acceptable = (
                    num(summary.get("scheduled_days")) <= MAX_SCHEDULED_DAYS_SOFT
                    and num(summary.get("time_window_violations")) <= 1
                    and len(trial) <= MAX_ROUTE_SPOTS
                )
                if acceptable:
                    route = trial
                    remaining.remove(sid)
                    current = sid
                    failed_extensions = 0
                else:
                    remaining.remove(sid)
                    failed_extensions += 1
                if len(route) >= MIN_ROUTE_SPOTS and rng.random() < 0.04 + 0.02 * max(0, len(route) - 30):
                    break
            route = local_order_improve(route, ctx, evaluator, passes=2)
            ev = evaluator.score(route)
            generation_solutions.append((float(ev["objective"]), route, ev))
            solution_rows.append(
                {
                    "generation": generation,
                    "ant": ant,
                    "objective": round(float(ev["objective"]), 3),
                    "spots_count": len(route),
                    **ev["summary"],
                    "route_sequence": base.route_sequence_from_ids(route, ctx["spots"]),
                }
            )

        generation_solutions.sort(key=lambda x: x[0], reverse=True)
        elite_routes.extend([route for _, route, _ in generation_solutions[:5]])
        elite_routes = sorted(elite_routes, key=lambda r: evaluator.objective(r), reverse=True)[:18]
        pheromone *= 1.0 - evaporation
        for rank, (obj, route, _) in enumerate(generation_solutions[:8], start=1):
            deposit = max(0.1, (obj - generation_solutions[-1][0] + 1.0) / max(1.0, abs(generation_solutions[0][0]))) / rank
            prev = "DEPOT"
            for sid in route:
                pheromone[node_idx[node_key(prev)], node_idx[sid]] += deposit
                prev = sid
        best_obj, best_route, best_ev = generation_solutions[0]
        generation_rows.append(
            {
                "generation": generation,
                "best_objective": round(best_obj, 3),
                "mean_objective": round(float(np.mean([x[0] for x in generation_solutions])), 3),
                "best_spots": len(best_route),
                "best_days": int(best_ev["summary"]["scheduled_days"]),
                "best_over_budget_days": int(best_ev["summary"]["over_budget_days"]),
                "best_time_window_violations": int(best_ev["summary"]["time_window_violations"]),
                "best_limited_lodging_nights": int(best_ev["summary"]["limited_lodging_nights"]),
            }
        )
        print(f"ACO generation {generation}/{ACO_GENERATIONS}: best={best_obj:.1f}, spots={len(best_route)}", flush=True)

    return pd.DataFrame(generation_rows), pd.DataFrame(solution_rows), elite_routes


def hybrid_alns_sa(
    ctx: dict[str, Any],
    evaluator: Evaluator,
    candidates: list[str],
    seeds: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = random.Random(RNG_SEED + 7)
    operators = ["random_removal", "worst_removal", "long_arc_removal", "region_removal", "lodging_pressure_removal"]
    op_stats = {op: {"used": 0, "accepted": 0, "improved": 0} for op in operators}
    seed_routes = [list(dict.fromkeys(s)) for s in seeds if s]
    if not seed_routes:
        seed_routes = [base.greedy_nearest_order(candidates[:MIN_ROUTE_SPOTS], ctx)]

    seed_routes = [local_order_improve(route, ctx, evaluator, passes=2) for route in seed_routes]
    current = max(seed_routes, key=evaluator.objective)
    best = current[:]
    current_obj = evaluator.objective(current)
    best_obj = current_obj
    history = []
    temperature = max(2500.0, abs(best_obj) * 0.025)
    cooling = 0.994

    for iteration in range(1, HYBRID_ITERATIONS + 1):
        op = rng.choice(operators)
        op_stats[op]["used"] += 1
        base_route, removed = destroy_route(current, ctx, evaluator, rng, op)
        candidate = repair_insert(base_route, removed, candidates, ctx, evaluator, rng, max_insertions=rng.choice([5, 8, 12]))

        move = rng.choice(["none", "swap", "two_opt", "reorder"])
        if move == "swap" and len(candidate) >= 2:
            i, j = sorted(rng.sample(range(len(candidate)), 2))
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move == "two_opt" and len(candidate) >= 4:
            i, j = sorted(rng.sample(range(len(candidate)), 2))
            if j - i >= 2:
                candidate = candidate[:i] + list(reversed(candidate[i:j])) + candidate[j:]
        elif move == "reorder":
            candidate = base.two_opt_order(candidate, ctx, max_passes=1)

        candidate = list(dict.fromkeys(candidate))
        if len(candidate) < MIN_ROUTE_SPOTS:
            candidate = repair_insert(candidate, [], candidates, ctx, evaluator, rng, max_insertions=MIN_ROUTE_SPOTS - len(candidate) + 4)
        cand_ev = evaluator.score(candidate)
        cand_obj = float(cand_ev["objective"])
        delta = cand_obj - current_obj
        accept = delta >= 0 or rng.random() < math.exp(delta / max(1e-6, temperature))
        if accept:
            current = candidate
            current_obj = cand_obj
            op_stats[op]["accepted"] += 1
        if cand_obj > best_obj:
            best = candidate[:]
            best_obj = cand_obj
            op_stats[op]["improved"] += 1
        temperature *= cooling
        if iteration % 90 == 0:
            improved_best = local_order_improve(best, ctx, evaluator, passes=2)
            improved_obj = evaluator.objective(improved_best)
            if improved_obj > best_obj:
                best = improved_best
                best_obj = improved_obj
            current = best[:]
            current_obj = best_obj
        if iteration % 25 == 0 or iteration == 1:
            best_summary = evaluator.score(best)["summary"]
            print(
                f"Hybrid iteration {iteration}/{HYBRID_ITERATIONS}: best={best_obj:.1f}, "
                f"spots={len(best)}, days={best_summary['scheduled_days']}",
                flush=True,
            )
        history.append(
            {
                "iteration": iteration,
                "operator": op,
                "move": move,
                "temperature": round(temperature, 4),
                "accepted": bool(accept),
                "candidate_objective": round(cand_obj, 3),
                "current_objective": round(current_obj, 3),
                "best_objective": round(best_obj, 3),
                "candidate_spots": len(candidate),
                "best_spots": len(best),
                "candidate_days": int(cand_ev["summary"]["scheduled_days"]),
                "candidate_over_budget_days": int(cand_ev["summary"]["over_budget_days"]),
                "candidate_time_window_violations": int(cand_ev["summary"]["time_window_violations"]),
                "candidate_limited_lodging_nights": int(cand_ev["summary"]["limited_lodging_nights"]),
            }
        )

    op_rows = []
    for op, stats in op_stats.items():
        used = stats["used"]
        op_rows.append(
            {
                "operator": op,
                "used": used,
                "accepted": stats["accepted"],
                "improved": stats["improved"],
                "acceptance_rate": round(stats["accepted"] / used, 4) if used else 0,
                "improvement_rate": round(stats["improved"] / used, 4) if used else 0,
            }
        )
    seed_rows = []
    for idx, route in enumerate(seed_routes, start=1):
        ev = evaluator.score(route)
        seed_rows.append(
            {
                "seed_id": f"SEED_{idx:03d}",
                "objective": round(float(ev["objective"]), 3),
                "spots_count": len(route),
                **ev["summary"],
                "route_sequence": base.route_sequence_from_ids(route, ctx["spots"]),
            }
        )
    return pd.DataFrame(history), pd.DataFrame(op_rows), pd.DataFrame(seed_rows), best


def route_from_existing_sequence(path: Path, sequence_column: str, ctx: dict[str, Any]) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or sequence_column not in df.columns:
        return []
    return base.route_ids_from_sequence(df.iloc[0][sequence_column], ctx["spots"])


def build_baseline_seeds(ctx: dict[str, Any]) -> list[list[str]]:
    seeds = []
    repaired = ctx["repaired"]
    for _, row in repaired.iterrows():
        seeds.append(base.route_ids_from_sequence(row["repaired_sequence"], ctx["spots"]))
    seeds.append(route_from_existing_sequence(ADVANCED_DIR / "integrated_alns_summary.csv", "route_sequence", ctx))
    seeds.append(route_from_existing_sequence(ADVANCED_DIR / "joint_optimization_summary.csv", "route_sequence", ctx))
    return [s for s in seeds if s]


def simulate_hybrid_route(route_ids: list[str], ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 101)
    trials, drivers = base.simulate_single_route("HYBRID_ACO_ALNS_SA", route_ids, ctx, MONTE_CARLO_TRIALS, rng)
    summary = {
        "route_id": "HYBRID_ACO_ALNS_SA",
        "trials": len(trials),
        "spots_count": len(route_ids),
        "feasibility_rate": round(float(trials["route_feasible"].mean()), 4),
        "mean_days": round(float(trials["scheduled_days"].mean()), 2),
        "p95_days": round(float(trials["scheduled_days"].quantile(0.95)), 2),
        "mean_cost_yuan": round(float(trials["total_proxy_cost_yuan"].mean()), 2),
        "p95_cost_yuan": round(float(trials["total_proxy_cost_yuan"].quantile(0.95)), 2),
        "mean_over_budget_days": round(float(trials["over_budget_days"].mean()), 2),
        "prob_any_reservation_failure": round(float((trials["reservation_failures"] > 0).mean()), 4),
        "prob_any_hotel_full": round(float((trials["hotel_full_nights"] > 0).mean()), 4),
        "prob_severe_transport": round(float((trials["severe_transport_events"] > 0).mean()), 4),
        "mean_crowd_penalty": round(float(trials["crowd_penalty"].mean()), 4),
    }
    if not drivers.empty:
        risk = drivers.groupby(["route_id", "risk_driver"], as_index=False)["driver_count"].sum()
    else:
        risk = pd.DataFrame(columns=["route_id", "risk_driver", "driver_count"])
    return {
        "hybrid_monte_carlo_trials": trials,
        "hybrid_monte_carlo_summary": pd.DataFrame([summary]),
        "hybrid_monte_carlo_risk_drivers": risk.sort_values("driver_count", ascending=False) if not risk.empty else risk,
    }


def build_model_comparison(hybrid_summary: pd.DataFrame, hybrid_mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in [
        ADVANCED_DIR / "advanced_model_comparison.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            rows.append(df)
    hybrid = hybrid_summary.copy()
    hybrid.insert(0, "model", "HYBRID_ACO_ALNS_SA")
    hybrid.insert(1, "method", "hybrid_metaheuristic")
    if not hybrid_mc.empty:
        for col in hybrid_mc.columns:
            if col != "route_id":
                hybrid[col] = hybrid_mc.iloc[0][col]
    rows.append(hybrid)
    return pd.concat(rows, ignore_index=True, sort=False)


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    summary = tables["hybrid_route_summary"]
    mc = tables["hybrid_monte_carlo_summary"]
    op = tables["hybrid_operator_stats"]
    comp = tables["hybrid_model_comparison"]
    keep_summary = [
        "route_id",
        "solver_status",
        "objective",
        "optimized_spots_count",
        "scheduled_days",
        "over_budget_days",
        "time_window_violations",
        "limited_lodging_nights",
        "itinerary_proxy_cost_yuan_excluding_meals",
        "route_sequence",
    ]
    keep_comp = [
        "model",
        "method",
        "objective",
        "optimized_spots_count",
        "scheduled_days",
        "over_budget_days",
        "time_window_violations",
        "limited_lodging_nights",
        "itinerary_proxy_cost_yuan_excluding_meals",
        "feasibility_rate",
        "p95_cost_yuan",
    ]
    lines = [
        "# 新疆旅游混合元启发式全量求解报告",
        "",
        "本轮实现 ACO 初始化 + ALNS 破坏修复 + SA 接受准则的混合元启发式主求解器。算法不依赖严格 MILP 在全量尺度下证明最优，而是把严格联合模型作为目标与约束框架，用元启发式在全 40 个候选景点上搜索可执行高质量路线。",
        "",
        "## 1. 最优候选路线",
        "",
        summary[[c for c in keep_summary if c in summary.columns]].to_markdown(index=False),
        "",
        "## 2. Monte Carlo 稳健性",
        "",
        mc.to_markdown(index=False),
        "",
        "## 3. 算子表现",
        "",
        op.to_markdown(index=False),
        "",
        "## 4. 模型对比",
        "",
        comp[[c for c in keep_comp if c in comp.columns]].tail(6).to_markdown(index=False),
        "",
        "## 5. 汇报口径",
        "",
        "建议将该算法作为全量主求解器：严格 MILP 负责定义数学模型，小规模 MILP 提供精确基准，混合元启发式负责全量求解，逐日排程器负责可行性复核，Monte Carlo 负责现实扰动下的稳健性排序。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    HYBRID_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ctx = base.load_context()
    evaluator = Evaluator(ctx)
    candidates = candidate_spots(ctx)

    print("running_aco_initialization", flush=True)
    aco_history, aco_solutions, elite_routes = aco_initialize(ctx, evaluator, candidates)

    print("running_hybrid_alns_sa", flush=True)
    seeds = elite_routes + build_baseline_seeds(ctx)
    alns_history, operator_stats, seed_summary, best_route = hybrid_alns_sa(ctx, evaluator, candidates, seeds)

    print("final_schedule_review", flush=True)
    best_eval = base.score_scheduled_route(best_route, ctx, "HYBRID_ACO_ALNS_SA")
    hybrid_summary = pd.DataFrame(
        [
            {
                "route_id": "HYBRID_ACO_ALNS_SA",
                "solver_status": "aco_initialized_alns_sa_completed",
                "objective": round(float(best_eval["objective"]), 3),
                "total_value_score": round(float(best_eval["total_value"]), 3),
                **best_eval["summary"],
                "route_sequence": base.route_sequence_from_ids(best_route, ctx["spots"]),
            }
        ]
    )
    hybrid_segments = best_eval["scheduled"]["segments"].copy()
    hybrid_segments["route_id"] = "HYBRID_ACO_ALNS_SA"
    hybrid_days = best_eval["scheduled"]["days"].copy()
    hybrid_days["route_id"] = "HYBRID_ACO_ALNS_SA"
    hybrid_time_windows = best_eval["scheduled"]["time_windows"].copy()
    hybrid_time_windows["route_id"] = "HYBRID_ACO_ALNS_SA"
    hybrid_lodging = best_eval["scheduled"]["lodging"].copy()
    hybrid_lodging["route_id"] = "HYBRID_ACO_ALNS_SA"

    print("running_hybrid_monte_carlo", flush=True)
    mc_tables = simulate_hybrid_route(best_route, ctx)
    comparison = build_model_comparison(hybrid_summary, mc_tables["hybrid_monte_carlo_summary"])

    tables: dict[str, pd.DataFrame] = {
        "hybrid_route_summary": hybrid_summary,
        "hybrid_route_segments": hybrid_segments,
        "hybrid_route_days": hybrid_days,
        "hybrid_time_windows": hybrid_time_windows,
        "hybrid_lodging": hybrid_lodging,
        "hybrid_aco_history": aco_history,
        "hybrid_aco_solutions": aco_solutions,
        "hybrid_seed_summary": seed_summary,
        "hybrid_alns_sa_history": alns_history,
        "hybrid_operator_stats": operator_stats,
        **mc_tables,
        "hybrid_model_comparison": comparison,
    }

    for name, df in tables.items():
        write_csv(df, HYBRID_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游混合元启发式全量求解结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游混合元启发式全量求解报告.md")

    summary = {
        "candidate_spots": len(candidates),
        "aco_generations": ACO_GENERATIONS,
        "aco_ants": ACO_ANTS,
        "hybrid_iterations": HYBRID_ITERATIONS,
        "best_route_spots": int(hybrid_summary.iloc[0]["optimized_spots_count"]),
        "best_route_days": int(hybrid_summary.iloc[0]["scheduled_days"]),
        "best_over_budget_days": int(hybrid_summary.iloc[0]["over_budget_days"]),
        "best_time_window_violations": int(hybrid_summary.iloc[0]["time_window_violations"]),
        "best_limited_lodging_nights": int(hybrid_summary.iloc[0]["limited_lodging_nights"]),
        "hybrid_monte_carlo_trials": int(len(mc_tables["hybrid_monte_carlo_trials"])),
        "hybrid_feasibility_rate": float(mc_tables["hybrid_monte_carlo_summary"].iloc[0]["feasibility_rate"]),
    }
    (HYBRID_DIR / "hybrid_solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
