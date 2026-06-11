# -*- coding: utf-8 -*-
from q1_v3_build_all import Q1V3Builder


if __name__ == "__main__":
    builder = Q1V3Builder()
    builder.build_transport_labels()
    builder.build_preference_and_diversity_tables()
