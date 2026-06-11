# -*- coding: utf-8 -*-
from q1_v3_build_all import Q1V3Builder, read_csv


if __name__ == "__main__":
    builder = Q1V3Builder()
    builder.candidate_routes = read_csv(builder.outputs / "q1_v3_candidate_routes.csv")
    builder.run_route_simulator()
