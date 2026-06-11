from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hybrid_metaheuristic_optimizer as hybrid
import stochastic_integrated_joint_optimization as base


ROOT = Path(".")
HARD_DIR = ROOT / "hybrid_30day_outputs"
OUTPUT_DIR = ROOT / "outputs"

ROUTE_ID = "HYBRID_30D_ACO_ALNS_SA"
HARD_MAX_DAYS = 30
HARD_MAX_TIME_WINDOW_VIOLATIONS = 0
MONTE_CARLO_TRIALS = 260
RNG_SEED = hybrid.RNG_SEED + 303


def num(value: Any, default: float = 0.0) -> float:
    return base.num(value, default)


def hard30_penalty(summary: dict[str, Any]) -> float:
    days_excess = max(0.0, num(summary.get("scheduled_days")) - HARD_MAX_DAYS)
    tw_excess = max(0.0, num(summary.get("time_window_violations")) - HARD_MAX_TIME_WINDOW_VIOLATIONS)
    return (
        days_excess * 120000.0
        + tw_excess * 160000.0
        + max(0.0, num(summary.get("limited_lodging_nights")) - 2.0) * 22000.0
    )


def is_hard30_feasible(summary: dict[str, Any]) -> bool:
    return (
        num(summary.get("scheduled_days")) <= HARD_MAX_DAYS
        and num(summary.get("time_window_violations")) <= HARD_MAX_TIME_WINDOW_VIOLATIONS
    )


class Hard30Evaluator(hybrid.Evaluator):
    def score(self, route_ids: list[str]) -> dict[str, Any]:
        route_ids = list(dict.fromkeys(route_ids))
        key = tuple(route_ids)
        if key not in self.cache:
            ev = dict(base.fast_score_route(route_ids, self.ctx))
            summary = dict(ev["summary"])
            base_objective = float(ev["objective"])
            ev["base_objective"] = base_objective
            ev["hard30_objective"] = base_objective - hard30_penalty(summary)
            ev["objective"] = ev["hard30_objective"]
            ev["hard30_feasible"] = is_hard30_feasible(summary)
            self.cache[key] = ev
        return self.cache[key]

    def objective(self, route_ids: list[str]) -> float:
        return float(self.score(route_ids)["hard30_objective"])


def route_from_repaired_milp(ctx: dict[str, Any]) -> list[str]:
    repaired = ctx["repaired"]
    row = repaired[repaired["route_id"] == "PCOP_MILP"]
    if row.empty:
        row = repaired.sort_values(["repaired_time_window_violations", "repaired_days"]).head(1)
    return base.route_ids_from_sequence(row.iloc[0]["repaired_sequence"], ctx["spots"])


def feasible_route_candidates(
    ctx: dict[str, Any],
    evaluator: Hard30Evaluator,
    elite_routes: list[list[str]],
    seed_routes: list[list[str]],
) -> list[list[str]]:
    routes = [route_from_repaired_milp(ctx)]
    routes.extend(seed_routes)
    routes.extend(elite_routes)
    out = []
    seen: set[tuple[str, ...]] = set()
    for route in routes:
        clean_route = list(dict.fromkeys(route))
        key = tuple(clean_route)
        if clean_route and key not in seen:
            seen.add(key)
            if is_hard30_feasible(evaluator.score(clean_route)["summary"]):
                out.append(clean_route)
    return out


def force_best_feasible(
    candidate: list[str],
    ctx: dict[str, Any],
    evaluator: Hard30Evaluator,
    elite_routes: list[list[str]],
    seed_routes: list[list[str]],
) -> list[str]:
    if is_hard30_feasible(evaluator.score(candidate)["summary"]):
        return candidate
    feasible = feasible_route_candidates(ctx, evaluator, elite_routes, seed_routes)
    if not feasible:
        return route_from_repaired_milp(ctx)
    return max(feasible, key=evaluator.objective)


def simulate_hard30_route(route_ids: list[str], ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 101)
    trials, drivers = base.simulate_single_route(ROUTE_ID, route_ids, ctx, MONTE_CARLO_TRIALS, rng)
    summary = {
        "route_id": ROUTE_ID,
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
        risk = risk.sort_values("driver_count", ascending=False)
    else:
        risk = pd.DataFrame(columns=["route_id", "risk_driver", "driver_count"])
    return {
        "hard30_monte_carlo_trials": trials,
        "hard30_monte_carlo_summary": pd.DataFrame([summary]),
        "hard30_monte_carlo_risk_drivers": risk,
    }


def build_model_comparison(hard30_summary: pd.DataFrame, hard30_mc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous = ROOT / "hybrid_metaheuristic_outputs" / "hybrid_model_comparison.csv"
    if previous.exists():
        rows.append(pd.read_csv(previous, encoding="utf-8-sig"))
    hard = hard30_summary.copy()
    hard.insert(0, "model", ROUTE_ID)
    hard.insert(1, "method", "hybrid_metaheuristic_hard_30_days")
    if not hard30_mc.empty:
        for col in hard30_mc.columns:
            if col != "route_id":
                hard[col] = hard30_mc.iloc[0][col]
    rows.append(hard)
    return pd.concat(rows, ignore_index=True, sort=False)


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    summary = tables["hard30_route_summary"]
    mc = tables["hard30_monte_carlo_summary"]
    comparison = tables["hard30_model_comparison"]
    keep_summary = [
        "route_id",
        "solver_status",
        "hard30_feasible",
        "hard30_objective",
        "base_objective",
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
        "hard30_objective",
        "base_objective",
        "optimized_spots_count",
        "scheduled_days",
        "over_budget_days",
        "time_window_violations",
        "limited_lodging_nights",
        "feasibility_rate",
        "p95_cost_yuan",
    ]
    lines = [
        "# 新疆旅游30天硬约束混合元启发式求解报告",
        "",
        "本轮实验将原题第一问的30天期限作为硬约束处理：候选路线只有在正式逐日排程后满足 scheduled_days <= 30 且时间窗违规为0，才可作为可接受解。开放时间版本仍保留为拓展实验，本报告用于和 PCOP_MILP 严格基准进行公平比较。",
        "",
        "## 1. 30天硬约束最优候选路线",
        "",
        summary[[c for c in keep_summary if c in summary.columns]].to_markdown(index=False),
        "",
        "## 2. Monte Carlo 扰动检验",
        "",
        mc.to_markdown(index=False),
        "",
        "## 3. 模型对比",
        "",
        comparison[[c for c in keep_comp if c in comparison.columns]].tail(6).to_markdown(index=False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    HARD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    hybrid.MAX_SCHEDULED_DAYS_SOFT = HARD_MAX_DAYS
    ctx = base.load_context()
    evaluator = Hard30Evaluator(ctx)
    candidates = hybrid.candidate_spots(ctx)

    print("running_hard30_aco_initialization", flush=True)
    aco_history, aco_solutions, elite_routes = hybrid.aco_initialize(ctx, evaluator, candidates)

    print("running_hard30_hybrid_alns_sa", flush=True)
    seed_routes = elite_routes + hybrid.build_baseline_seeds(ctx)
    alns_history, operator_stats, seed_summary, best_route = hybrid.hybrid_alns_sa(ctx, evaluator, candidates, seed_routes)
    best_route = force_best_feasible(best_route, ctx, evaluator, elite_routes, seed_routes)

    print("final_hard30_schedule_review", flush=True)
    best_eval = base.score_scheduled_route(best_route, ctx, ROUTE_ID)
    hard_eval = evaluator.score(best_route)
    hard30_summary = pd.DataFrame(
        [
            {
                "route_id": ROUTE_ID,
                "solver_status": "aco_initialized_alns_sa_completed_hard_30_days",
                "hard30_feasible": bool(hard_eval["hard30_feasible"]),
                "hard30_objective": round(float(hard_eval["hard30_objective"]), 3),
                "base_objective": round(float(best_eval["objective"]), 3),
                "total_value_score": round(float(best_eval["total_value"]), 3),
                **best_eval["summary"],
                "route_sequence": base.route_sequence_from_ids(best_route, ctx["spots"]),
            }
        ]
    )
    hard30_segments = best_eval["scheduled"]["segments"].copy()
    hard30_segments["route_id"] = ROUTE_ID
    hard30_days = best_eval["scheduled"]["days"].copy()
    hard30_days["route_id"] = ROUTE_ID
    hard30_time_windows = best_eval["scheduled"]["time_windows"].copy()
    hard30_time_windows["route_id"] = ROUTE_ID
    hard30_lodging = best_eval["scheduled"]["lodging"].copy()
    hard30_lodging["route_id"] = ROUTE_ID

    print("running_hard30_monte_carlo", flush=True)
    mc_tables = simulate_hard30_route(best_route, ctx)
    comparison = build_model_comparison(hard30_summary, mc_tables["hard30_monte_carlo_summary"])

    tables: dict[str, pd.DataFrame] = {
        "hard30_route_summary": hard30_summary,
        "hard30_route_segments": hard30_segments,
        "hard30_route_days": hard30_days,
        "hard30_time_windows": hard30_time_windows,
        "hard30_lodging": hard30_lodging,
        "hard30_aco_history": aco_history,
        "hard30_aco_solutions": aco_solutions,
        "hard30_seed_summary": seed_summary,
        "hard30_alns_sa_history": alns_history,
        "hard30_operator_stats": operator_stats,
        **mc_tables,
        "hard30_model_comparison": comparison,
    }

    for name, df in tables.items():
        hybrid.write_csv(df, HARD_DIR / f"{name}.csv")
    write_workbook(tables, OUTPUT_DIR / "新疆旅游30天硬约束混合元启发式求解结果.xlsx")
    write_report(tables, OUTPUT_DIR / "新疆旅游30天硬约束混合元启发式求解报告.md")

    solve_summary = {
        "candidate_spots": len(candidates),
        "aco_generations": hybrid.ACO_GENERATIONS,
        "aco_ants": hybrid.ACO_ANTS,
        "hybrid_iterations": hybrid.HYBRID_ITERATIONS,
        "hard_max_days": HARD_MAX_DAYS,
        "best_route_spots": int(hard30_summary.iloc[0]["optimized_spots_count"]),
        "best_route_days": int(hard30_summary.iloc[0]["scheduled_days"]),
        "best_over_budget_days": int(hard30_summary.iloc[0]["over_budget_days"]),
        "best_time_window_violations": int(hard30_summary.iloc[0]["time_window_violations"]),
        "best_limited_lodging_nights": int(hard30_summary.iloc[0]["limited_lodging_nights"]),
        "hard30_feasible": bool(hard30_summary.iloc[0]["hard30_feasible"]),
        "monte_carlo_trials": int(len(mc_tables["hard30_monte_carlo_trials"])),
        "monte_carlo_feasibility_rate": float(mc_tables["hard30_monte_carlo_summary"].iloc[0]["feasibility_rate"]),
    }
    (HARD_DIR / "hard30_solve_summary.json").write_text(
        json.dumps(solve_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(solve_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
