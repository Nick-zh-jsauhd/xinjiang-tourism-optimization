# -*- coding: utf-8 -*-
from q1_v3_build_all import Q1V3Builder, read_csv


if __name__ == "__main__":
    builder = Q1V3Builder()
    builder.candidate_routes = read_csv(builder.outputs / "q1_v3_candidate_routes.csv")
    builder.simulation_summary = read_csv(builder.outputs / "q1_v3_simulation_summary.csv")
    builder.select_robust_pareto_front()
    builder.build_small_exact_check()
    builder.build_figures()
    builder.build_report()
    builder.build_solve_summary()
