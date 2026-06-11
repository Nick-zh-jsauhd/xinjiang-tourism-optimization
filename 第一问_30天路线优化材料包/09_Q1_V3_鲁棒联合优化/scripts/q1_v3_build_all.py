# -*- coding: utf-8 -*-
"""Q1-V3 robust joint optimizer for the Xinjiang 30-day route problem.

This script intentionally implements a matheuristic pipeline instead of claiming
global optimality for the full stochastic, multimodal, scheduled orienteering
problem. It turns the Q1-V3 design document into reproducible artifacts:

1. multimodal non-dominated OD labels from the edge graph;
2. preference/diversity/Loulan substitution rule tables;
3. route-specific scenario samples;
4. coverage-grid route re-optimization with direct label choice;
5. hourly itinerary construction and infeasibility audit;
6. route-specific Monte Carlo simulation;
7. robust Pareto filtering and representative route selection;
8. figures, Excel workbook, Markdown report, and model audit table.
"""

from __future__ import annotations

import heapq
import json
import math
import random
import re
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


try:
    import yaml
except Exception:  # pragma: no cover - fallback only
    yaml = None


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - figures are best effort
    HAS_MPL = False


SCRIPT_PATH = Path(__file__).resolve()
V3_ROOT = SCRIPT_PATH.parents[1]
PKG_ROOT = V3_ROOT.parent
MODEL_DATA = PKG_ROOT / "01_输入数据" / "model_data"
ENHANCED_DATA = PKG_ROOT / "01_输入数据" / "enhanced_data"
V2_OUTPUTS = PKG_ROOT / "08_Q1_V2强化" / "outputs"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "是", "可", "allowed"}


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def parse_time_to_hour(value: str, default: float) -> float:
    text = str(value).strip()
    if not text:
        return default
    m = re.search(r"(\d{1,2})[:：](\d{1,2})", text)
    if not m:
        return default
    return int(m.group(1)) + int(m.group(2)) / 60.0


def hour_to_str(hour: float) -> str:
    hour = max(0.0, hour)
    h = int(math.floor(hour))
    m = int(round((hour - h) * 60))
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}"


def compact_modes(modes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for mode in modes:
        m = str(mode).strip()
        if m == "bus":
            m = "coach"
        if m == "transfer":
            m = "taxi_transfer"
        if not m:
            continue
        if not out or out[-1] != m:
            out.append(m)
    return out or ["same_spot"]


def percentile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.percentile(np.asarray(values, dtype=float), q))


def cvar(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    threshold = np.quantile(arr, alpha)
    tail = arr[arr >= threshold]
    return float(tail.mean()) if len(tail) else float(threshold)


def dominates(a: Sequence[float], b: Sequence[float], eps: float = 1e-9) -> bool:
    """True if vector a weakly improves b in all dimensions and strictly in one."""
    return all(x <= y + eps for x, y in zip(a, b)) and any(x < y - eps for x, y in zip(a, b))


@dataclass
class GraphLabel:
    node: str
    time: float
    cost: float
    risk: float
    fatigue: float
    nodes: Tuple[str, ...]
    modes: Tuple[str, ...]
    edge_ids: Tuple[str, ...]
    quality_notes: Tuple[str, ...]

    def vector(self) -> Tuple[float, float, float, float]:
        return (self.time, self.cost, self.risk, self.fatigue)

    def score(self) -> float:
        return self.time + self.cost / 180.0 + self.risk * 12.0 + self.fatigue * 0.45


class Q1V3Builder:
    def __init__(self) -> None:
        self.root = V3_ROOT
        self.outputs = self.root / "outputs"
        self.figures = self.root / "figures"
        self.reports = self.root / "reports"
        self.config_dir = self.root / "config"
        for path in [self.outputs, self.figures, self.reports, self.config_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config()
        self.rng = np.random.default_rng(int(self.config["random_seed"]))
        self.py_rng = random.Random(int(self.config["random_seed"]))
        self.load_inputs()
        self.transport_labels = pd.DataFrame()
        self.labels_by_pair: Dict[Tuple[str, str], List[dict]] = {}
        self.candidate_routes = pd.DataFrame()
        self.hourly_itinerary = pd.DataFrame()
        self.schedule_summary = pd.DataFrame()
        self.simulation_trials = pd.DataFrame()
        self.simulation_summary = pd.DataFrame()
        self.robust_front = pd.DataFrame()
        self.selected_routes = pd.DataFrame()

    def load_config(self) -> dict:
        defaults = {
            "coverage_grid": [20, 22, 24, 26, 28, 30, 32],
            "scenario_samples": 500,
            "cvar_alpha": 0.75,
            "cvar_alpha_high": 0.90,
            "success_threshold": 0.8,
            "main_route_min_spots": 30,
            "main_route_min_buffer_days": 2,
            "max_red_days_standard": 1,
            "max_red_days_family": 0,
            "max_red_days_senior": 0,
            "transport_label_limit_per_od": 5,
            "max_labels_per_node": 14,
            "max_path_edges": 7,
            "candidate_routes_per_q": 18,
            "random_seed": 20260611,
            "min_region_count": 4,
            "min_theme_count": 5,
            "daily_active_limit_standard": 8.5,
            "daily_active_limit_family": 7.0,
            "daily_active_limit_senior": 6.5,
            "lunch_start": "12:00",
            "lunch_end": "13:00",
            "day_start": "08:30",
            "soft_day_end": "20:30",
            "hard_day_end": "22:00",
            "long_transfer_hours": 6.0,
            "buffer_absorb_hours_per_day": 6.0,
        }
        path = self.config_dir / "q1_v3_config.yaml"
        if path.exists() and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update(loaded)
        return defaults

    def load_inputs(self) -> None:
        self.spots = read_csv(MODEL_DATA / "spot_clean.csv")
        numeric_cols = [
            "visit_hours_min",
            "visit_hours_max",
            "visit_hours_mid",
            "ticket_high_total_yuan_per_person",
            "priority_score_for_op",
            "local_access_hours_from_text",
        ]
        for col in numeric_cols:
            if col in self.spots.columns:
                self.spots[col] = self.spots[col].map(as_float)
        for col in [
            "is_cultural",
            "is_natural",
            "is_topic_preference",
            "requires_approval",
            "requires_border_permit",
            "requires_reservation",
            "high_altitude_or_remote",
            "ordinary_tourist_restricted",
        ]:
            if col in self.spots.columns:
                self.spots[col] = self.spots[col].map(as_bool)
        self.spots_by_id = {str(r["spot_id"]): r for _, r in self.spots.iterrows()}
        self.name_to_id = {str(r["spot_name"]): str(r["spot_id"]) for _, r in self.spots.iterrows()}

        self.edges = read_csv(ENHANCED_DATA / "multimodal_edges.csv")
        for col in [
            "base_time_hours",
            "base_cost_yuan_per_two",
            "base_risk",
            "distance_km",
            "schedule_wait_hours",
        ]:
            if col in self.edges.columns:
                self.edges[col] = self.edges[col].map(as_float)

        self.nodes = read_csv(ENHANCED_DATA / "multimodal_nodes.csv")
        self.time_windows = read_csv(ENHANCED_DATA / "spot_time_windows.csv")
        self.time_window_by_spot = {
            str(r["spot_id"]): {
                "open": parse_time_to_hour(r.get("open_time", ""), 8.5),
                "close": parse_time_to_hour(r.get("close_time", ""), 19.0),
                "quality": str(r.get("time_window_quality", "")),
            }
            for _, r in self.time_windows.iterrows()
        }

        self.capacity = read_csv(ENHANCED_DATA / "capacity_by_spot.csv")
        for col in ["daily_capacity_persons", "instant_capacity_persons", "effective_capacity_beta_085"]:
            if col in self.capacity.columns:
                self.capacity[col] = self.capacity[col].map(as_float)
        self.capacity_by_spot = {str(r["spot_id"]): r for _, r in self.capacity.iterrows()}

        self.hotels = read_csv(ENHANCED_DATA / "hotel_hub_constraints.csv")
        for col in ["default_room_price_yuan_per_night", "hotel_capacity_rooms_simulated"]:
            if col in self.hotels.columns:
                self.hotels[col] = self.hotels[col].map(as_float)
        if "is_hotel_hub" in self.hotels.columns:
            self.hotels["is_hotel_hub"] = self.hotels["is_hotel_hub"].map(as_bool)
        self.hotel_by_hub = {str(r["hub_name"]): r for _, r in self.hotels.iterrows()}

        self.special = read_csv(ENHANCED_DATA / "special_access_constraints.csv")
        self.special_by_spot = {str(r["spot_id"]): r for _, r in self.special.iterrows()}

        self.cultural = read_csv(ENHANCED_DATA / "cultural_tags.csv")
        for col in [
            "tag_silk_road",
            "tag_religion",
            "tag_ethnic_folk",
            "tag_archaeology",
            "tag_world_or_key_heritage",
            "tag_natural",
        ]:
            if col in self.cultural.columns:
                self.cultural[col] = self.cultural[col].map(as_bool)
        if "culture_value_score" in self.cultural.columns:
            self.cultural["culture_value_score"] = self.cultural["culture_value_score"].map(as_float)
        self.cultural_by_spot = {str(r["spot_id"]): r for _, r in self.cultural.iterrows()}

        self.depot = read_csv(ENHANCED_DATA / "depot_access_matrix.csv")
        for col in [
            "depot_to_spot_time",
            "spot_to_depot_time",
            "depot_to_spot_cost",
            "spot_to_depot_cost",
            "depot_to_spot_risk",
            "spot_to_depot_risk",
        ]:
            if col in self.depot.columns:
                self.depot[col] = self.depot[col].map(as_float)
        self.depot_by_spot = {str(r["spot_id"]): r for _, r in self.depot.iterrows()}

        self.rules = read_csv(ENHANCED_DATA / "time_dependent_rules.csv")
        for col in ["probability", "time_factor", "cost_factor", "risk_add"]:
            if col in self.rules.columns:
                self.rules[col] = self.rules[col].map(as_float)

        self.v2_family = read_csv(V2_OUTPUTS / "q1_v2_route_family.csv")
        self.enhanced_od = read_csv(ENHANCED_DATA / "enhanced_od_matrix.csv")
        for col in ["shortest_time_hours", "shortest_cost_yuan_per_two", "path_risk"]:
            if col in self.enhanced_od.columns:
                self.enhanced_od[col] = self.enhanced_od[col].map(as_float)
        self.enhanced_lookup = {}
        for _, r in self.enhanced_od.iterrows():
            if str(r.get("scenario_id", "")) != "base_summer":
                continue
            self.enhanced_lookup[(str(r["from_spot_id"]), str(r["to_spot_id"]))] = r

        self.ordinary_spots = [
            sid
            for sid, row in self.spots_by_id.items()
            if not as_bool(row.get("ordinary_tourist_restricted", False))
            and not as_bool(row.get("requires_approval", False))
        ]

    # ------------------------------------------------------------------
    # P1: Multimodal labels
    # ------------------------------------------------------------------

    def normalize_edge_mode(self, row: pd.Series) -> str:
        mode = str(row.get("mode", "")).strip()
        if mode == "bus":
            return "coach"
        if mode == "self_drive":
            hours = as_float(row.get("base_time_hours", 0.0))
            if hours <= 1.5:
                return "taxi_transfer"
            if hours <= 6.0:
                return "rental_car"
            return "charter_car"
        return mode or "unknown"

    def edge_fatigue(self, mode: str, time_hours: float) -> float:
        params = {
            "rail": (0.55, 8.0),
            "air": (0.65, 5.0),
            "coach": (1.05, 4.0),
            "rental_car": (1.15, 4.0),
            "charter_car": (1.00, 6.0),
            "taxi_transfer": (0.85, 2.0),
            "scenic_shuttle": (0.90, 3.0),
            "transfer": (0.80, 1.2),
        }
        theta, threshold = params.get(mode, (1.0, 4.0))
        return theta * time_hours + 0.08 * max(0.0, time_hours - threshold) ** 2

    def build_graph(self) -> Dict[str, List[dict]]:
        adj: Dict[str, List[dict]] = defaultdict(list)
        for _, row in self.edges.iterrows():
            mode = self.normalize_edge_mode(row)
            wait = as_float(row.get("schedule_wait_hours", 0.0))
            if mode in {"rail", "air", "coach"}:
                time = as_float(row.get("base_time_hours", 0.0)) + max(0.0, wait)
            else:
                time = as_float(row.get("base_time_hours", 0.0))
            cost = as_float(row.get("base_cost_yuan_per_two", 0.0))
            risk = max(0.0, as_float(row.get("base_risk", 0.0)))
            quality = str(row.get("quality_flag", "")).strip()
            if str(row.get("cost_floor_applied", "")).strip():
                quality = (quality + ";local_cost_floor").strip(";")
            edge = {
                "edge_id": str(row.get("edge_id", "")),
                "from": str(row.get("from_node", "")),
                "to": str(row.get("to_node", "")),
                "mode": mode,
                "raw_mode": str(row.get("mode", "")),
                "time": time,
                "cost": cost,
                "risk": risk,
                "fatigue": self.edge_fatigue(mode, time),
                "quality": quality,
            }
            if edge["from"] and edge["to"] and time >= 0:
                adj[edge["from"]].append(edge)
        return adj

    def label_dominated(self, candidate: GraphLabel, labels: Sequence[GraphLabel]) -> bool:
        return any(dominates(label.vector(), candidate.vector()) for label in labels)

    def prune_labels(self, labels: List[GraphLabel], limit: int) -> List[GraphLabel]:
        kept: List[GraphLabel] = []
        for label in sorted(labels, key=lambda x: x.score()):
            if self.label_dominated(label, kept):
                continue
            kept = [old for old in kept if not dominates(label.vector(), old.vector())]
            kept.append(label)
            kept = sorted(kept, key=lambda x: x.score())[:limit]
        return kept

    def multi_label_path(self, src: str, dst: str, adj: Dict[str, List[dict]]) -> List[GraphLabel]:
        if src == dst:
            return [
                GraphLabel(
                    node=dst,
                    time=0.0,
                    cost=0.0,
                    risk=0.0,
                    fatigue=0.0,
                    nodes=(src,),
                    modes=("same_spot",),
                    edge_ids=(),
                    quality_notes=("same_spot",),
                )
            ]
        max_node_labels = int(self.config["max_labels_per_node"])
        max_path_edges = int(self.config["max_path_edges"])
        start = GraphLabel(src, 0.0, 0.0, 0.0, 0.0, (src,), tuple(), tuple(), tuple())
        heap: List[Tuple[float, int, GraphLabel]] = [(0.0, 0, start)]
        counter = 0
        labels_at_node: Dict[str, List[GraphLabel]] = defaultdict(list)
        labels_at_node[src].append(start)
        target_labels: List[GraphLabel] = []
        expansions = 0
        # The raw graph contains a dense self-drive closure between scenic spots.
        # Allowing SPOT -> SPOT -> SPOT intermediate hops creates an artificial
        # combinatorial explosion and unrealistic "visit a spot as a transfer"
        # path. We therefore only allow scenic-spot nodes as source or final
        # destination inside the transport-label subproblem.
        while heap and expansions < 2500:
            _, _, label = heapq.heappop(heap)
            expansions += 1
            if label.node == dst:
                target_labels.append(label)
                target_labels = self.prune_labels(target_labels, int(self.config["transport_label_limit_per_od"]) * 2)
                if len(target_labels) >= int(self.config["transport_label_limit_per_od"]) * 2:
                    continue
            if len(label.edge_ids) >= max_path_edges:
                continue
            for edge in adj.get(label.node, []):
                to_node = edge["to"]
                if to_node.startswith("SPOT::") and to_node != dst:
                    continue
                if to_node in label.nodes:
                    continue
                new_label = GraphLabel(
                    node=to_node,
                    time=label.time + edge["time"],
                    cost=label.cost + edge["cost"],
                    risk=label.risk + edge["risk"],
                    fatigue=label.fatigue + edge["fatigue"],
                    nodes=label.nodes + (to_node,),
                    modes=label.modes + (edge["mode"],),
                    edge_ids=label.edge_ids + (edge["edge_id"],),
                    quality_notes=label.quality_notes + ((edge["quality"],) if edge["quality"] else tuple()),
                )
                if self.label_dominated(new_label, labels_at_node[to_node]):
                    continue
                labels_at_node[to_node] = self.prune_labels(labels_at_node[to_node] + [new_label], max_node_labels)
                counter += 1
                heapq.heappush(heap, (new_label.score(), counter, new_label))
        return self.prune_labels(target_labels, int(self.config["transport_label_limit_per_od"]))

    def fallback_label(self, from_sid: str, to_sid: str) -> dict:
        src_name = self.spots_by_id[from_sid]["spot_name"]
        dst_name = self.spots_by_id[to_sid]["spot_name"]
        if from_sid == to_sid:
            return {
                "from_spot_id": from_sid,
                "from_spot_name": src_name,
                "to_spot_id": to_sid,
                "to_spot_name": dst_name,
                "label_id": f"L_{from_sid}_{to_sid}_01",
                "mode_sequence": "same_spot",
                "node_path": f"SPOT::{from_sid}",
                "edge_id_path": "",
                "time_hours": 0.0,
                "cost_yuan_for_two": 0.0,
                "risk_score": 0.0,
                "fatigue_score": 0.0,
                "transfer_count": 0,
                "is_night_transport": False,
                "hotel_saving_yuan": 0.0,
                "ordinary_tourist_feasible": True,
                "schedule_confirmation_required": False,
                "dominance_rank": 1,
                "label_quality_note": "same_spot",
            }
        od = self.enhanced_lookup.get((from_sid, to_sid))
        time = as_float(od.get("shortest_time_hours", 0.0), 0.0) if od is not None else 8.0
        cost = as_float(od.get("shortest_cost_yuan_per_two", 0.0), 0.0) if od is not None else 800.0
        risk = as_float(od.get("path_risk", 0.1), 0.1) if od is not None else 0.2
        raw_modes = str(od.get("path_modes", "self_drive")) if od is not None else "self_drive"
        if "rail" in raw_modes:
            mode = "rail"
        elif "air" in raw_modes:
            mode = "air"
        elif "bus" in raw_modes:
            mode = "coach"
        elif time <= 1.5:
            mode = "taxi_transfer"
        elif time <= 6:
            mode = "rental_car"
        else:
            mode = "charter_car"
        fatigue = self.edge_fatigue(mode, time)
        return {
            "from_spot_id": from_sid,
            "from_spot_name": src_name,
            "to_spot_id": to_sid,
            "to_spot_name": dst_name,
            "label_id": f"L_{from_sid}_{to_sid}_01",
            "mode_sequence": mode,
            "node_path": f"SPOT::{from_sid}>SPOT::{to_sid}",
            "edge_id_path": "fallback_enhanced_od",
            "time_hours": round(time, 3),
            "cost_yuan_for_two": round(cost, 2),
            "risk_score": round(risk, 4),
            "fatigue_score": round(fatigue, 3),
            "transfer_count": 0,
            "is_night_transport": bool(mode in {"rail", "coach"} and time >= 8.0),
            "hotel_saving_yuan": 180.0 if mode == "rail" and time >= 8.0 else 0.0,
            "ordinary_tourist_feasible": self.is_ordinary_allowed(from_sid) and self.is_ordinary_allowed(to_sid),
            "schedule_confirmation_required": mode in {"rail", "air", "coach"},
            "dominance_rank": 1,
            "label_quality_note": "fallback_from_enhanced_od_matrix",
        }

    def is_ordinary_allowed(self, spot_id: str) -> bool:
        row = self.spots_by_id.get(spot_id)
        if row is None:
            return False
        if as_bool(row.get("ordinary_tourist_restricted", False)) or as_bool(row.get("requires_approval", False)):
            return False
        special = self.special_by_spot.get(spot_id)
        if special is not None and not as_bool(special.get("ordinary_tourist_allowed", True)):
            return False
        return True

    def build_transport_labels(self) -> pd.DataFrame:
        # Fast template enumeration. The original unrestricted multi-label
        # search is kept above as a reference implementation, but a dense
        # self-drive closure makes all-pairs search unnecessarily expensive.
        # Here we enumerate realistic patterns:
        #   1) direct scenic-spot to scenic-spot road label;
        #   2) local connector + one public main edge + local connector.
        # The resulting labels are still filtered by the same non-dominance rule.
        normalized_edges: List[dict] = []
        for _, row in self.edges.iterrows():
            mode = self.normalize_edge_mode(row)
            wait = as_float(row.get("schedule_wait_hours", 0.0), 0.0)
            time = as_float(row.get("base_time_hours", 0.0), 0.0)
            if mode in {"rail", "air", "coach"}:
                time += max(0.0, wait)
            quality = str(row.get("quality_flag", "")).strip()
            if str(row.get("cost_floor_applied", "")).strip():
                quality = (quality + ";local_cost_floor").strip(";")
            normalized_edges.append(
                {
                    "edge_id": str(row.get("edge_id", "")),
                    "from": str(row.get("from_node", "")),
                    "to": str(row.get("to_node", "")),
                    "mode": mode,
                    "raw_mode": str(row.get("mode", "")),
                    "time": time,
                    "cost": as_float(row.get("base_cost_yuan_per_two", 0.0), 0.0),
                    "risk": max(0.0, as_float(row.get("base_risk", 0.0), 0.0)),
                    "fatigue": self.edge_fatigue(mode, time),
                    "quality": quality,
                }
            )

        direct_edges: Dict[Tuple[str, str], List[GraphLabel]] = defaultdict(list)
        local_adj: Dict[str, List[dict]] = defaultdict(list)
        local_rev_adj: Dict[str, List[dict]] = defaultdict(list)
        public_edges: List[dict] = []
        for edge in normalized_edges:
            from_node = edge["from"]
            to_node = edge["to"]
            if from_node.startswith("SPOT::") and to_node.startswith("SPOT::"):
                direct_edges[(from_node, to_node)].append(
                    GraphLabel(
                        node=to_node,
                        time=edge["time"],
                        cost=edge["cost"],
                        risk=edge["risk"],
                        fatigue=edge["fatigue"],
                        nodes=(from_node, to_node),
                        modes=(edge["mode"],),
                        edge_ids=(edge["edge_id"],),
                        quality_notes=((edge["quality"],) if edge["quality"] else tuple()),
                    )
                )
            elif edge["mode"] in {"transfer", "taxi_transfer", "scenic_shuttle"}:
                local_adj[from_node].append(edge)
                rev_edge = dict(edge)
                rev_edge["from"], rev_edge["to"] = edge["to"], edge["from"]
                local_rev_adj[edge["to"]].append(rev_edge)
            elif edge["mode"] in {"rail", "air", "coach"}:
                public_edges.append(edge)

        def dijkstra_local(start: str, adj: Dict[str, List[dict]]) -> Dict[str, GraphLabel]:
            start_label = GraphLabel(start, 0.0, 0.0, 0.0, 0.0, (start,), tuple(), tuple(), tuple())
            heap: List[Tuple[float, int, GraphLabel]] = [(0.0, 0, start_label)]
            best: Dict[str, GraphLabel] = {start: start_label}
            counter = 0
            while heap:
                _, _, label = heapq.heappop(heap)
                if best.get(label.node) is not label:
                    continue
                for edge in adj.get(label.node, []):
                    to_node = edge["to"]
                    if to_node in label.nodes:
                        continue
                    new_label = GraphLabel(
                        node=to_node,
                        time=label.time + edge["time"],
                        cost=label.cost + edge["cost"],
                        risk=label.risk + edge["risk"],
                        fatigue=label.fatigue + edge["fatigue"],
                        nodes=label.nodes + (to_node,),
                        modes=label.modes + (edge["mode"],),
                        edge_ids=label.edge_ids + (edge["edge_id"],),
                        quality_notes=label.quality_notes + ((edge["quality"],) if edge["quality"] else tuple()),
                    )
                    if len(new_label.edge_ids) > 4:
                        continue
                    old = best.get(to_node)
                    if old is None or new_label.score() < old.score():
                        best[to_node] = new_label
                        counter += 1
                        heapq.heappush(heap, (new_label.score(), counter, new_label))
            return best

        connector_from_spot = {}
        connector_to_spot = {}
        spot_nodes = [f"SPOT::{sid}" for sid in self.spots_by_id.keys()]
        for spot_node in spot_nodes:
            connector_from_spot[spot_node] = dijkstra_local(spot_node, local_adj)
            rev = dijkstra_local(spot_node, local_rev_adj)
            converted = {}
            for node, label in rev.items():
                converted[node] = GraphLabel(
                    node=spot_node,
                    time=label.time,
                    cost=label.cost,
                    risk=label.risk,
                    fatigue=label.fatigue,
                    nodes=tuple(reversed(label.nodes)),
                    modes=tuple(reversed(label.modes)),
                    edge_ids=tuple(reversed(label.edge_ids)),
                    quality_notes=tuple(reversed(label.quality_notes)),
                )
            connector_to_spot[spot_node] = converted

        records: List[dict] = []
        spot_ids = list(self.spots_by_id.keys())
        for from_sid in spot_ids:
            for to_sid in spot_ids:
                if from_sid == to_sid:
                    records.append(self.fallback_label(from_sid, to_sid))
                    continue
                src = f"SPOT::{from_sid}"
                dst = f"SPOT::{to_sid}"
                candidates: List[GraphLabel] = []
                candidates.extend(direct_edges.get((src, dst), []))
                out_connectors = connector_from_spot.get(src, {})
                in_connectors = connector_to_spot.get(dst, {})
                for pe in public_edges:
                    left = out_connectors.get(pe["from"])
                    right = in_connectors.get(pe["to"])
                    if left is None or right is None:
                        continue
                    candidates.append(
                        GraphLabel(
                            node=dst,
                            time=left.time + pe["time"] + right.time,
                            cost=left.cost + pe["cost"] + right.cost,
                            risk=left.risk + pe["risk"] + right.risk,
                            fatigue=left.fatigue + pe["fatigue"] + right.fatigue,
                            nodes=left.nodes + (pe["to"],) + right.nodes[1:],
                            modes=left.modes + (pe["mode"],) + right.modes,
                            edge_ids=left.edge_ids + (pe["edge_id"],) + right.edge_ids,
                            quality_notes=left.quality_notes
                            + ("template_public_edge",)
                            + ((pe["quality"],) if pe["quality"] else tuple())
                            + right.quality_notes,
                        )
                    )
                labels = self.prune_labels(candidates, int(self.config["transport_label_limit_per_od"]))
                if not labels:
                    records.append(self.fallback_label(from_sid, to_sid))
                    continue
                for rank, label in enumerate(labels, start=1):
                    modes = compact_modes(label.modes)
                    functional_modes = [m for m in modes if m not in {"taxi_transfer", "scenic_shuttle"}]
                    transfer_count = max(0, len(functional_modes) - 1)
                    schedule_required = any(m in {"rail", "air", "coach"} for m in modes)
                    note_values = sorted({n for n in label.quality_notes if str(n).strip()})
                    records.append(
                        {
                            "from_spot_id": from_sid,
                            "from_spot_name": self.spots_by_id[from_sid]["spot_name"],
                            "to_spot_id": to_sid,
                            "to_spot_name": self.spots_by_id[to_sid]["spot_name"],
                            "label_id": f"L_{from_sid}_{to_sid}_{rank:02d}",
                            "mode_sequence": ">".join(modes),
                            "node_path": ">".join(label.nodes),
                            "edge_id_path": ">".join(label.edge_ids),
                            "time_hours": round(label.time, 3),
                            "cost_yuan_for_two": round(label.cost, 2),
                            "risk_score": round(label.risk, 4),
                            "fatigue_score": round(label.fatigue + 0.45 * transfer_count, 3),
                            "transfer_count": transfer_count,
                            "is_night_transport": bool(
                                any(m in {"rail", "coach"} for m in modes) and label.time >= 8.0
                            ),
                            "hotel_saving_yuan": 180.0
                            if any(m == "rail" for m in modes) and label.time >= 8.0
                            else 0.0,
                            "ordinary_tourist_feasible": self.is_ordinary_allowed(from_sid)
                            and self.is_ordinary_allowed(to_sid),
                            "schedule_confirmation_required": schedule_required,
                            "dominance_rank": rank,
                            "label_quality_note": ";".join(note_values) if note_values else "non_dominated_from_multimodal_edges",
                        }
                    )
        labels = pd.DataFrame(records)
        write_csv(labels, self.outputs / "q1_v3_multimodal_labels.csv")
        self.transport_labels = labels
        self.make_label_lookup(labels)
        return labels

    def make_label_lookup(self, labels: Optional[pd.DataFrame] = None) -> None:
        if labels is None:
            labels_path = self.outputs / "q1_v3_multimodal_labels.csv"
            labels = read_csv(labels_path)
        df = labels.copy()
        for col in [
            "time_hours",
            "cost_yuan_for_two",
            "risk_score",
            "fatigue_score",
            "transfer_count",
            "hotel_saving_yuan",
            "dominance_rank",
        ]:
            if col in df.columns:
                df[col] = df[col].map(as_float)
        grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for _, row in df.iterrows():
            grouped[(str(row["from_spot_id"]), str(row["to_spot_id"]))].append(row.to_dict())
        for key in list(grouped.keys()):
            grouped[key] = sorted(grouped[key], key=lambda r: as_float(r.get("dominance_rank", 1)))
        self.labels_by_pair = grouped

    # ------------------------------------------------------------------
    # P2: Preference, diversity, Loulan substitution
    # ------------------------------------------------------------------

    def preference_groups_definition(self) -> Dict[str, dict]:
        return {
            "G_TIANCHI": {
                "name": "天池组",
                "min_required": 1,
                "spot_ids": ["P001"],
                "weight": 12,
            },
            "G_DABANCHENG": {
                "name": "达坂城组",
                "min_required": 1,
                "spot_ids": ["P002"],
                "weight": 10,
            },
            "G_TURPAN": {
                "name": "吐鲁番组",
                "min_required": 2,
                "spot_ids": ["P003", "P004", "P005", "P006", "P011", "P012", "P013"],
                "weight": 14,
            },
            "G_LOULAN_SUBSTITUTE": {
                "name": "楼兰文化替代组",
                "min_required": 2,
                "spot_ids": ["P006", "P011", "P036", "P031", "P024", "P025", "P026", "P030", "P034", "P037"],
                "weight": 16,
            },
            "G_YILI": {
                "name": "伊犁组",
                "min_required": 2,
                "spot_ids": ["P008", "P009", "P010", "P014", "P015", "P016", "P017"],
                "weight": 14,
            },
        }

    def build_preference_and_diversity_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        groups = self.preference_groups_definition()
        group_rows: List[dict] = []
        for gid, spec in groups.items():
            for sid in spec["spot_ids"]:
                if sid not in self.spots_by_id:
                    continue
                row = self.spots_by_id[sid]
                group_rows.append(
                    {
                        "group_id": gid,
                        "group_name": spec["name"],
                        "min_required": spec["min_required"],
                        "group_weight": spec["weight"],
                        "spot_id": sid,
                        "spot_name": row["spot_name"],
                        "ordinary_tourist_allowed": self.is_ordinary_allowed(sid),
                        "note": "楼兰古城本体受限，使用文化替代节点"
                        if gid == "G_LOULAN_SUBSTITUTE"
                        else "题面偏好组约束",
                    }
                )
        pref = pd.DataFrame(group_rows)
        write_csv(pref, self.outputs / "q1_v3_preference_groups.csv")

        region_rows: List[dict] = []
        for region, sub in self.spots.groupby("region_cluster"):
            ordinary_count = sum(self.is_ordinary_allowed(str(sid)) for sid in sub["spot_id"])
            alpha = 10.0 if region in {"乌鲁木齐周边", "东疆", "伊犁", "南疆", "巴州-南疆北线"} else 8.0
            region_rows.append(
                {
                    "region_cluster": region,
                    "ordinary_candidate_spots": int(ordinary_count),
                    "min_cover_soft": 1 if ordinary_count else 0,
                    "diminishing_alpha": alpha,
                    "diminishing_beta": 0.72,
                    "rule_note": "Value_r(n)=A_r*(1-exp(-B_r*n)); 用于抑制同区域刷点",
                }
            )
        region_rules = pd.DataFrame(region_rows).sort_values("region_cluster")
        write_csv(region_rules, self.outputs / "q1_v3_region_diversity_rules.csv")

        tier_rows = []
        tier_specs = [
            (1, "西域古城/遗址替代", ["P006", "P011", "P036", "P031"], "与楼兰同属丝路遗址/古城叙事"),
            (2, "丝路宗教与城市文化", ["P012", "P024", "P025", "P026"], "补足丝路宗教、绿洲城市和民俗文化"),
            (3, "区域文化补偿", ["P030", "P034", "P037"], "以南疆/巴州文化节点补偿楼兰不可达"),
        ]
        for tier, meaning, spot_ids, note in tier_specs:
            for sid in spot_ids:
                if sid in self.spots_by_id:
                    tier_rows.append(
                        {
                            "tier": tier,
                            "tier_meaning": meaning,
                            "spot_id": sid,
                            "spot_name": self.spots_by_id[sid]["spot_name"],
                            "ordinary_tourist_allowed": self.is_ordinary_allowed(sid),
                            "substitution_note": note,
                        }
                    )
        loulan = pd.DataFrame(tier_rows)
        write_csv(loulan, self.outputs / "q1_v3_loulan_substitution_tiers.csv")
        return pref, region_rules, loulan

    # ------------------------------------------------------------------
    # P3: Scenario samples
    # ------------------------------------------------------------------

    def generate_scenarios(self) -> pd.DataFrame:
        scenario_meta = (
            self.rules.groupby("scenario_id", as_index=False)
            .agg({"probability": "max", "description": "first"})
            .sort_values("scenario_id")
        )
        summer_weights = {
            "base_summer": 0.55,
            "summer_peak": 0.25,
            "duku_weather_disruption": 0.10,
            "hotel_full_price_surge": 0.10,
            "mayday_peak": 0.0,
        }
        scenario_meta["probability"] = scenario_meta.apply(
            lambda r: summer_weights.get(str(r["scenario_id"]), as_float(r["probability"], 0.0)),
            axis=1,
        )
        scenario_meta = scenario_meta[scenario_meta["probability"] > 0].copy()
        scenario_ids = scenario_meta["scenario_id"].astype(str).tolist()
        probs = scenario_meta["probability"].astype(float).to_numpy()
        probs = probs / probs.sum()
        mode_factor = {}
        for _, row in self.rules.iterrows():
            mode = str(row["mode"])
            sid = str(row["scenario_id"])
            if mode == "bus":
                mode = "coach"
            mode_factor[(sid, mode)] = {
                "time_factor": as_float(row.get("time_factor", 1.0), 1.0),
                "cost_factor": as_float(row.get("cost_factor", 1.0), 1.0),
                "risk_add": as_float(row.get("risk_add", 0.0), 0.0),
            }
        rows: List[dict] = []
        n = int(self.config["scenario_samples"])
        for sample_id in range(1, n + 1):
            sid = str(self.rng.choice(scenario_ids, p=probs))
            heatwave = bool(self.rng.random() < (0.28 if sid in {"summer_peak", "mayday_peak"} else 0.12))
            def factor(mode: str, key: str, default: float) -> float:
                base = mode_factor.get((sid, mode), {}).get(key, default)
                if key == "time_factor":
                    return float(base * self.rng.lognormal(mean=0.0, sigma=0.04))
                if key == "cost_factor":
                    return float(base * self.rng.lognormal(mean=0.0, sigma=0.03))
                return float(base)

            if sid == "mayday_peak":
                reservation_pressure = 0.22
                hotel_full_probability = 0.18
            elif sid == "summer_peak":
                reservation_pressure = 0.15
                hotel_full_probability = 0.12
            elif sid == "hotel_full_price_surge":
                reservation_pressure = 0.12
                hotel_full_probability = 0.28
            elif sid == "duku_weather_disruption":
                reservation_pressure = 0.08
                hotel_full_probability = 0.08
            else:
                reservation_pressure = 0.05
                hotel_full_probability = 0.05
            reservation_pressure = max(0.01, min(0.35, reservation_pressure + float(self.rng.normal(0, 0.025))))
            hotel_full_probability = max(0.01, min(0.40, hotel_full_probability + float(self.rng.normal(0, 0.025))))
            road_closure_probability = 0.08 if sid == "duku_weather_disruption" else 0.015
            public_disruption = 0.03 if sid in {"summer_peak", "mayday_peak"} else 0.015
            if sid == "duku_weather_disruption":
                public_disruption += 0.01
            row = {
                "sample_id": sample_id,
                "scenario_id": sid,
                "scenario_description": scenario_meta.loc[
                    scenario_meta["scenario_id"] == sid, "description"
                ].iloc[0],
                "road_time_factor": round(factor("road", "time_factor", 1.0), 4),
                "rail_time_factor": round(factor("rail", "time_factor", 1.0), 4),
                "air_time_factor": round(factor("air", "time_factor", 1.0), 4),
                "coach_time_factor": round(factor("coach", "time_factor", 1.0), 4),
                "local_time_factor": round(float(self.rng.lognormal(0, 0.045)), 4),
                "road_cost_factor": round(factor("road", "cost_factor", 1.0), 4),
                "rail_cost_factor": round(factor("rail", "cost_factor", 1.0), 4),
                "air_cost_factor": round(factor("air", "cost_factor", 1.0), 4),
                "coach_cost_factor": round(factor("coach", "cost_factor", 1.0), 4),
                "service_time_factor": round(float(self.rng.lognormal(0, 0.05)), 4),
                "reservation_pressure": round(reservation_pressure, 4),
                "hotel_full_probability": round(hotel_full_probability, 4),
                "road_closure_probability": round(road_closure_probability, 4),
                "public_transport_disruption_probability": round(public_disruption, 4),
                "heatwave_flag": heatwave,
                "heat_penalty_hours": round(float(self.rng.uniform(0.5, 2.2) if heatwave else self.rng.uniform(0, 0.35)), 3),
                "general_cost_multiplier": round(
                    max(
                        factor("road", "cost_factor", 1.0),
                        factor("rail", "cost_factor", 1.0),
                        factor("air", "cost_factor", 1.0),
                    ),
                    4,
                ),
                "risk_add_road": round(factor("road", "risk_add", 0.0), 4),
                "risk_add_public": round(max(factor("rail", "risk_add", 0.0), factor("air", "risk_add", 0.0)), 4),
            }
            rows.append(row)
        scenarios = pd.DataFrame(rows)
        write_csv(scenarios, self.outputs / "q1_v3_scenario_samples.csv")
        return scenarios

    # ------------------------------------------------------------------
    # P4: Robust master optimizer
    # ------------------------------------------------------------------

    def spot_value(self, sid: str, strategy: str = "balanced") -> float:
        row = self.spots_by_id[sid]
        cultural = self.cultural_by_spot.get(sid)
        culture_score = as_float(cultural.get("culture_value_score", 0.0), 0.0) if cultural is not None else 0.0
        value = (
            as_float(row.get("priority_score_for_op", 0.0)) * 5.0
            + (8.0 if as_bool(row.get("is_topic_preference", False)) else 0.0)
            + (3.0 if as_bool(row.get("is_cultural", False)) else 0.0)
            + (2.4 if as_bool(row.get("is_natural", False)) else 0.0)
            + culture_score * 1.35
        )
        if strategy == "culture":
            value += culture_score * 1.8 + (5 if as_bool(row.get("is_cultural", False)) else 0)
        if strategy == "comfort":
            value -= 5.0 if as_bool(row.get("high_altitude_or_remote", False)) else 0.0
            value -= 2.0 if as_bool(row.get("requires_reservation", False)) else 0.0
        if strategy == "north" and str(row.get("region_cluster", "")) == "北疆":
            value += 8.0
        if strategy == "south" and str(row.get("region_cluster", "")) == "南疆":
            value += 6.0
        return float(value)

    def spot_difficulty(self, sid: str) -> float:
        row = self.spots_by_id[sid]
        depot = self.depot_by_spot.get(sid)
        access = as_float(depot.get("depot_to_spot_time", 2.0), 2.0) if depot is not None else 2.0
        remote = 2.5 if as_bool(row.get("high_altitude_or_remote", False)) else 0.0
        reservation = 1.0 if as_bool(row.get("requires_reservation", False)) else 0.0
        return as_float(row.get("visit_hours_mid", 3.0), 3.0) + 0.25 * access + remote + reservation

    def spot_themes(self, sid: str) -> set:
        row = self.spots_by_id[sid]
        c = self.cultural_by_spot.get(sid)
        themes = set()
        if as_bool(row.get("is_natural", False)) or (c is not None and as_bool(c.get("tag_natural", False))):
            themes.add("自然山水")
        if c is not None and as_bool(c.get("tag_silk_road", False)):
            themes.add("丝路遗址")
        if c is not None and as_bool(c.get("tag_religion", False)):
            themes.add("宗教文化")
        if c is not None and as_bool(c.get("tag_ethnic_folk", False)):
            themes.add("民族民俗")
        if c is not None and (as_bool(c.get("tag_archaeology", False)) or as_bool(c.get("tag_world_or_key_heritage", False))):
            themes.add("西域古城")
        name = str(row.get("spot_name", ""))
        if "博物馆" in name or "古城" in name or "王府" in name:
            themes.add("城市博物馆")
        if as_bool(row.get("high_altitude_or_remote", False)) or "帕米尔" in name or "冰川" in name:
            themes.add("高原边境")
        return themes

    def group_satisfaction(self, selected: Iterable[str]) -> Dict[str, int]:
        selected_set = set(selected)
        out = {}
        for gid, spec in self.preference_groups_definition().items():
            out[gid] = len(selected_set.intersection(spec["spot_ids"]))
        return out

    def group_satisfaction_rate(self, selected: Iterable[str]) -> float:
        counts = self.group_satisfaction(selected)
        total = 0.0
        met = 0.0
        for gid, spec in self.preference_groups_definition().items():
            total += 1.0
            met += min(1.0, counts.get(gid, 0) / max(1, spec["min_required"]))
        return met / total if total else 0.0

    def repair_selection(self, selected: Iterable[str], q: int, strategy: str) -> List[str]:
        selected_set = {sid for sid in selected if sid in self.ordinary_spots}
        groups = self.preference_groups_definition()
        # Add missing preference-group nodes first.
        for gid, spec in groups.items():
            while len(selected_set.intersection(spec["spot_ids"])) < spec["min_required"]:
                options = [
                    sid
                    for sid in spec["spot_ids"]
                    if sid in self.ordinary_spots and sid not in selected_set
                ]
                if not options:
                    break
                best = max(options, key=lambda sid: self.spot_value(sid, strategy) - 0.25 * self.spot_difficulty(sid))
                selected_set.add(best)
        # Fill regional diversity.
        def region_count(ids: Iterable[str]) -> int:
            return len({str(self.spots_by_id[sid]["region_cluster"]) for sid in ids})

        while region_count(selected_set) < int(self.config["min_region_count"]):
            current_regions = {str(self.spots_by_id[sid]["region_cluster"]) for sid in selected_set}
            options = [
                sid
                for sid in self.ordinary_spots
                if str(self.spots_by_id[sid]["region_cluster"]) not in current_regions and sid not in selected_set
            ]
            if not options:
                break
            selected_set.add(max(options, key=lambda sid: self.spot_value(sid, strategy) - 0.18 * self.spot_difficulty(sid)))
        # Fill themes.
        def theme_count(ids: Iterable[str]) -> int:
            themes = set()
            for sid in ids:
                themes.update(self.spot_themes(sid))
            return len(themes)

        while theme_count(selected_set) < int(self.config["min_theme_count"]):
            options = [sid for sid in self.ordinary_spots if sid not in selected_set]
            if not options:
                break
            current = set()
            for sid in selected_set:
                current.update(self.spot_themes(sid))
            selected_set.add(
                max(
                    options,
                    key=lambda sid: len(self.spot_themes(sid) - current) * 8
                    + self.spot_value(sid, strategy)
                    - 0.25 * self.spot_difficulty(sid),
                )
            )
        while len(selected_set) < q:
            region_counter = Counter(str(self.spots_by_id[sid]["region_cluster"]) for sid in selected_set)
            options = [sid for sid in self.ordinary_spots if sid not in selected_set]
            if not options:
                break
            selected_set.add(
                max(
                    options,
                    key=lambda sid: self.spot_value(sid, strategy)
                    + 4.0 / (1.0 + region_counter[str(self.spots_by_id[sid]["region_cluster"])])
                    - 0.35 * self.spot_difficulty(sid),
                )
            )
        # Remove overflow while trying to preserve hard soft-constraints.
        while len(selected_set) > q:
            removable = []
            for sid in list(selected_set):
                trial = selected_set - {sid}
                if len(trial) < q:
                    continue
                ok_groups = True
                counts = self.group_satisfaction(trial)
                for gid, spec in groups.items():
                    if counts.get(gid, 0) < spec["min_required"]:
                        ok_groups = False
                        break
                if ok_groups and region_count(trial) >= int(self.config["min_region_count"]):
                    removable.append(sid)
            if not removable:
                removable = list(selected_set)
            worst = min(removable, key=lambda sid: self.spot_value(sid, strategy) - 0.2 * self.spot_difficulty(sid))
            selected_set.remove(worst)
        return sorted(selected_set, key=lambda sid: self.spot_value(sid, strategy), reverse=True)

    def sequence_from_v2(self, q: int, strategy: str) -> List[str]:
        if self.v2_family.empty:
            return []
        row = self.v2_family.iloc[1] if len(self.v2_family) > 1 else self.v2_family.iloc[0]
        names = [x.strip() for x in str(row.get("route_sequence", "")).split("->") if x.strip()]
        ids = [self.name_to_id[name] for name in names if name in self.name_to_id and self.name_to_id[name] in self.ordinary_spots]
        selected = self.repair_selection(ids[:], min(q, len(self.ordinary_spots)), strategy)
        if len(selected) > q:
            selected = sorted(selected, key=lambda sid: self.spot_value(sid, strategy), reverse=True)[:q]
            selected = self.repair_selection(selected, q, strategy)
        return selected

    def ordered_v2_seed(self, q: int) -> List[str]:
        if self.v2_family.empty:
            return []
        row = self.v2_family.iloc[1] if len(self.v2_family) > 1 else self.v2_family.iloc[0]
        names = [x.strip() for x in str(row.get("route_sequence", "")).split("->") if x.strip()]
        ordered_ids = [
            self.name_to_id[name]
            for name in names
            if name in self.name_to_id and self.name_to_id[name] in self.ordinary_spots
        ]
        selected = ordered_ids[:]
        while len(selected) > q:
            selected_set = set(selected)
            removable = []
            for sid in selected:
                trial = selected_set - {sid}
                counts = self.group_satisfaction(trial)
                group_ok = all(
                    counts[gid] >= spec["min_required"]
                    for gid, spec in self.preference_groups_definition().items()
                )
                if group_ok:
                    removable.append(sid)
            if not removable:
                removable = selected[:]
            worst = min(removable, key=lambda sid: self.spot_value(sid, "balanced") - 0.35 * self.spot_difficulty(sid))
            selected.remove(worst)
        repaired = self.repair_selection(selected, q, "balanced")
        repaired_set = set(repaired)
        ordered = [sid for sid in ordered_ids if sid in repaired_set]
        ordered.extend([sid for sid in repaired if sid not in ordered])
        return ordered[:q]

    def greedy_selection(self, q: int, strategy: str) -> List[str]:
        selected: List[str] = []
        for _, spec in self.preference_groups_definition().items():
            options = [sid for sid in spec["spot_ids"] if sid in self.ordinary_spots]
            options = sorted(options, key=lambda sid: self.spot_value(sid, strategy) - 0.22 * self.spot_difficulty(sid), reverse=True)
            selected.extend(options[: spec["min_required"]])
        selected = list(dict.fromkeys(selected))
        while len(selected) < q:
            current_regions = Counter(str(self.spots_by_id[sid]["region_cluster"]) for sid in selected)
            current_themes = set()
            for sid in selected:
                current_themes.update(self.spot_themes(sid))
            options = [sid for sid in self.ordinary_spots if sid not in selected]
            if not options:
                break
            if strategy == "cost_eff":
                key = lambda sid: self.spot_value(sid, strategy) / (1.0 + self.spot_difficulty(sid))
            elif strategy == "region":
                key = lambda sid: self.spot_value(sid, strategy) + 8.0 / (
                    1.0 + current_regions[str(self.spots_by_id[sid]["region_cluster"])]
                )
            elif strategy == "culture":
                key = lambda sid: self.spot_value(sid, strategy) + 5 * len(self.spot_themes(sid) - current_themes)
            elif strategy == "comfort":
                key = lambda sid: self.spot_value(sid, strategy) - 0.7 * self.spot_difficulty(sid)
            else:
                key = lambda sid: self.spot_value(sid, strategy) - 0.25 * self.spot_difficulty(sid)
            selected.append(max(options, key=key))
        return self.repair_selection(selected, q, strategy)

    def random_selection(self, q: int, strategy: str) -> List[str]:
        base = self.greedy_selection(min(max(8, q // 3), q), strategy)
        pool = [sid for sid in self.ordinary_spots if sid not in base]
        weights = np.array([
            max(0.1, self.spot_value(sid, strategy) / (1.0 + 0.35 * self.spot_difficulty(sid)))
            for sid in pool
        ])
        weights = weights / weights.sum()
        need = max(0, q - len(base))
        if need and pool:
            sampled = list(self.rng.choice(pool, size=min(need, len(pool)), replace=False, p=weights))
        else:
            sampled = []
        return self.repair_selection(base + sampled, q, strategy)

    def label_for_pair(self, from_sid: str, to_sid: str, preference: str = "balanced") -> Optional[dict]:
        labels = self.labels_by_pair.get((from_sid, to_sid))
        if not labels:
            labels = [self.fallback_label(from_sid, to_sid)]
        def score(label: dict) -> float:
            mode_seq = str(label.get("mode_sequence", ""))
            public_bonus = 0.0
            if any(m in mode_seq for m in ["rail", "air", "coach"]):
                public_bonus = 1.8 if preference in {"balanced", "comfort"} else 0.8
            if preference == "cost_eff":
                return (
                    as_float(label.get("cost_yuan_for_two", 0)) * 0.018
                    + as_float(label.get("time_hours", 0)) * 0.75
                    + as_float(label.get("risk_score", 0)) * 18
                    + as_float(label.get("fatigue_score", 0)) * 0.35
                    - public_bonus
                )
            if preference == "comfort":
                return (
                    as_float(label.get("fatigue_score", 0)) * 0.8
                    + as_float(label.get("risk_score", 0)) * 24
                    + as_float(label.get("time_hours", 0)) * 0.9
                    + as_float(label.get("cost_yuan_for_two", 0)) * 0.008
                    - public_bonus
                )
            return (
                as_float(label.get("cost_yuan_for_two", 0)) * 0.012
                + as_float(label.get("time_hours", 0)) * 0.95
                + as_float(label.get("risk_score", 0)) * 22
                + as_float(label.get("fatigue_score", 0)) * 0.48
                - public_bonus
            )
        return min(labels, key=score)

    def depot_access_costs(self, first_sid: str, last_sid: str) -> Tuple[float, float, float, float]:
        first = self.depot_by_spot.get(first_sid)
        last = self.depot_by_spot.get(last_sid)
        in_time = as_float(first.get("depot_to_spot_time", 0.0), 0.0) if first is not None else 0.0
        out_time = as_float(last.get("spot_to_depot_time", 0.0), 0.0) if last is not None else 0.0
        in_cost = as_float(first.get("depot_to_spot_cost", 0.0), 0.0) if first is not None else 0.0
        out_cost = as_float(last.get("spot_to_depot_cost", 0.0), 0.0) if last is not None else 0.0
        in_risk = as_float(first.get("depot_to_spot_risk", 0.0), 0.0) if first is not None else 0.0
        out_risk = as_float(last.get("spot_to_depot_risk", 0.0), 0.0) if last is not None else 0.0
        return in_time + out_time, in_cost + out_cost, in_risk + out_risk, 0.85 * (in_time + out_time)

    def edge_metric(self, from_sid: str, to_sid: str, preference: str = "balanced") -> float:
        if from_sid == to_sid:
            return 0.0
        label = self.label_for_pair(from_sid, to_sid, preference)
        if not label:
            return 9999.0
        return (
            as_float(label.get("time_hours", 0)) * 1.0
            + as_float(label.get("cost_yuan_for_two", 0)) / 180.0
            + as_float(label.get("risk_score", 0)) * 15
            + as_float(label.get("fatigue_score", 0)) * 0.4
        )

    def order_sequence(self, selected: Sequence[str], preference: str = "balanced", variant: int = 0) -> List[str]:
        if not selected:
            return []
        region_orders = [
            ["乌鲁木齐周边", "东疆", "伊犁", "巴州-南疆北线", "南疆", "北疆"],
            ["乌鲁木齐周边", "北疆", "伊犁", "巴州-南疆北线", "南疆", "东疆"],
            ["东疆", "乌鲁木齐周边", "伊犁", "巴州-南疆北线", "南疆", "北疆"],
            ["乌鲁木齐周边", "东疆", "巴州-南疆北线", "南疆", "伊犁", "北疆"],
        ]
        order = region_orders[variant % len(region_orders)]
        region_rank = {region: i for i, region in enumerate(order)}
        grouped = sorted(
            selected,
            key=lambda sid: (
                region_rank.get(str(self.spots_by_id[sid]["region_cluster"]), 99),
                str(self.spots_by_id[sid]["hub_name"]),
                -self.spot_value(sid, preference),
            ),
        )
        seq: List[str] = []
        remaining = grouped[:]
        current = None
        while remaining:
            if current is None:
                depot_key = lambda sid: (
                    region_rank.get(str(self.spots_by_id[sid]["region_cluster"]), 99),
                    as_float(self.depot_by_spot.get(sid, {}).get("depot_to_spot_time", 2.0), 2.0)
                    if isinstance(self.depot_by_spot.get(sid), dict)
                    else as_float(self.depot_by_spot.get(sid).get("depot_to_spot_time", 2.0), 2.0)
                    if self.depot_by_spot.get(sid) is not None
                    else 2.0,
                )
                nxt = min(remaining, key=depot_key)
            else:
                # Prefer same/next planned region, but use label metric inside the band.
                current_region_idx = region_rank.get(str(self.spots_by_id[current]["region_cluster"]), 99)
                nxt = min(
                    remaining,
                    key=lambda sid: (
                        max(0, region_rank.get(str(self.spots_by_id[sid]["region_cluster"]), 99) - current_region_idx),
                        self.edge_metric(current, sid, preference),
                    ),
                )
            seq.append(nxt)
            remaining.remove(nxt)
            current = nxt
        return self.two_opt(seq, preference, max_passes=2)

    def route_metric(self, seq: Sequence[str], preference: str = "balanced") -> float:
        if not seq:
            return 0.0
        total = 0.0
        for a, b in zip(seq[:-1], seq[1:]):
            total += self.edge_metric(a, b, preference)
        depot = self.depot_access_costs(seq[0], seq[-1])
        total += depot[0] + depot[1] / 180.0 + depot[2] * 15
        return total

    def two_opt(self, seq: Sequence[str], preference: str, max_passes: int = 2) -> List[str]:
        seq = list(seq)
        n = len(seq)
        if n < 5:
            return seq
        best = self.route_metric(seq, preference)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    if j - i == 1:
                        continue
                    trial = seq[:i] + list(reversed(seq[i:j])) + seq[j:]
                    metric = self.route_metric(trial, preference)
                    if metric + 1e-6 < best:
                        seq = trial
                        best = metric
                        improved = True
        return seq

    def evaluate_route(self, seq: Sequence[str], target_q: int, generation_method: str, variant: int) -> Optional[dict]:
        if len(seq) < 2:
            return None
        label_rows = []
        for a, b in zip(seq[:-1], seq[1:]):
            label = self.label_for_pair(a, b, generation_method)
            if not label:
                return None
            label_rows.append(label)
        travel_time = sum(as_float(l.get("time_hours", 0.0)) for l in label_rows)
        transport_cost = sum(as_float(l.get("cost_yuan_for_two", 0.0)) for l in label_rows)
        risk = sum(as_float(l.get("risk_score", 0.0)) for l in label_rows)
        fatigue = sum(as_float(l.get("fatigue_score", 0.0)) for l in label_rows)
        depot_time, depot_cost, depot_risk, depot_fatigue = self.depot_access_costs(seq[0], seq[-1])
        travel_time += depot_time
        transport_cost += depot_cost
        risk += depot_risk
        fatigue += depot_fatigue
        service_hours = sum(as_float(self.spots_by_id[sid].get("visit_hours_mid", 3.0), 3.0) for sid in seq)
        ticket_cost = 2.0 * sum(
            as_float(self.spots_by_id[sid].get("ticket_high_total_yuan_per_person", 0.0), 0.0)
            for sid in seq
        )
        regions = [str(self.spots_by_id[sid]["region_cluster"]) for sid in seq]
        region_counts = Counter(regions)
        region_value = 0.0
        for region, n in region_counts.items():
            alpha = 10.0 if region in {"乌鲁木齐周边", "东疆", "伊犁", "南疆", "巴州-南疆北线"} else 8.0
            region_value += alpha * (1.0 - math.exp(-0.72 * n))
        groups = self.preference_groups_definition()
        group_counts = self.group_satisfaction(seq)
        pref_value = sum(
            spec["weight"] * min(1.0, group_counts[gid] / max(1, spec["min_required"]))
            for gid, spec in groups.items()
        )
        themes = set()
        for sid in seq:
            themes.update(self.spot_themes(sid))
        theme_value = len(themes) * 4.0
        spot_value = sum(self.spot_value(sid, generation_method) for sid in seq)
        coverage_value = spot_value + region_value + pref_value + theme_value
        long_transfers = sum(as_float(l.get("time_hours", 0.0)) >= as_float(self.config["long_transfer_hours"]) for l in label_rows)
        reservation_spots = sum(as_bool(self.spots_by_id[sid].get("requires_reservation", False)) for sid in seq)
        capacity_simulated = 0
        for sid in seq:
            cap = self.capacity_by_spot.get(sid)
            if cap is not None and "simulated" in str(cap.get("capacity_source_type", "")):
                capacity_simulated += 1
        active_hours = travel_time + service_hours + max(0, len(seq) // 3) * 0.5
        estimated_active_days = int(math.ceil((active_hours + long_transfers * 0.7) / 8.2))
        estimated_buffer_days = max(0, 30 - estimated_active_days)
        avg_hotel_price = float(np.mean([as_float(r.get("default_room_price_yuan_per_night", 220.0), 220.0) for _, r in self.hotels.iterrows()]))
        hotel_cost = max(0, estimated_active_days - 1) * avg_hotel_price
        total_cost = transport_cost + ticket_cost + hotel_cost
        risk_proxy = (
            risk * 80.0
            + reservation_spots * 1.4
            + capacity_simulated * 0.7
            + long_transfers * 2.0
            + max(0, estimated_active_days - 28) * 2.5
            + max(0, 2 - estimated_buffer_days) * 3.0
        )
        cvar_proxy = risk_proxy * 65 + max(0, estimated_active_days - 28) * 520 + max(0, 2 - estimated_buffer_days) * 480
        fast_score = (
            coverage_value
            - 0.0048 * total_cost
            - 0.42 * fatigue
            - 1.15 * risk_proxy
            - 0.0012 * cvar_proxy
        )
        mode_counter = Counter()
        mode_hours = Counter()
        for label in label_rows:
            modes = compact_modes(str(label.get("mode_sequence", "")).split(">"))
            edge_time = as_float(label.get("time_hours", 0.0))
            for mode in modes:
                mode_counter[mode] += 1
                mode_hours[mode] += edge_time / max(1, len(modes))
        public_hours = sum(mode_hours[m] for m in ["rail", "air", "coach"])
        route_id = f"Q1V3_Q{target_q}_{variant:03d}"
        return {
            "route_id": route_id,
            "target_coverage_q": target_q,
            "generation_method": generation_method,
            "spots_count": len(seq),
            "spot_id_sequence": " -> ".join(seq),
            "route_sequence": " -> ".join(str(self.spots_by_id[sid]["spot_name"]) for sid in seq),
            "label_sequence": " | ".join(str(l.get("label_id", "")) for l in label_rows),
            "transport_mode_sequence": " | ".join(str(l.get("mode_sequence", "")) for l in label_rows),
            "mode_mix": ";".join(f"{k}:{v}" for k, v in sorted(mode_counter.items())),
            "mode_hours_mix": ";".join(f"{k}:{round(v, 2)}" for k, v in sorted(mode_hours.items())),
            "public_transport_hours_share": round(public_hours / travel_time, 4) if travel_time else 0.0,
            "region_count": len(region_counts),
            "region_list": "、".join(sorted(region_counts.keys())),
            "theme_count": len(themes),
            "theme_list": "、".join(sorted(themes)),
            "preference_satisfaction_rate": round(self.group_satisfaction_rate(seq), 4),
            "loulan_substitute_spots": group_counts.get("G_LOULAN_SUBSTITUTE", 0),
            "topic_preference_spots": sum(as_bool(self.spots_by_id[sid].get("is_topic_preference", False)) for sid in seq),
            "cultural_spots": sum(as_bool(self.spots_by_id[sid].get("is_cultural", False)) for sid in seq),
            "natural_spots": sum(as_bool(self.spots_by_id[sid].get("is_natural", False)) for sid in seq),
            "remote_or_high_altitude_spots": sum(as_bool(self.spots_by_id[sid].get("high_altitude_or_remote", False)) for sid in seq),
            "reservation_spots": reservation_spots,
            "capacity_simulated_spots": capacity_simulated,
            "transport_cost_yuan_for_two": round(transport_cost, 2),
            "ticket_cost_yuan_for_two": round(ticket_cost, 2),
            "rough_hotel_cost_yuan": round(hotel_cost, 2),
            "total_cost_yuan_excluding_meals": round(total_cost, 2),
            "total_travel_hours": round(travel_time, 3),
            "total_service_hours": round(service_hours, 3),
            "total_fatigue_score": round(fatigue, 3),
            "total_risk_score": round(risk, 4),
            "long_transfer_edges": int(long_transfers),
            "estimated_active_days": estimated_active_days,
            "estimated_buffer_days": estimated_buffer_days,
            "coverage_value_v3": round(coverage_value, 3),
            "risk_proxy_score": round(risk_proxy, 3),
            "cvar_proxy_loss": round(cvar_proxy, 2),
            "fast_master_score": round(fast_score, 4),
            "hard_constraints_pass": bool(
                len(seq) >= target_q
                and self.group_satisfaction_rate(seq) >= 0.999
                and len(region_counts) >= int(self.config["min_region_count"])
                and len(themes) >= int(self.config["min_theme_count"])
                and "P007" not in seq
                and "P038" not in seq
            ),
            "model_note": "coverage-grid重新搜索；交通标签在候选路线评价中直接选择；CVaR为主搜索快速代理，精评见仿真表",
        }

    def run_master_optimizer(self) -> pd.DataFrame:
        if not self.labels_by_pair:
            labels_path = self.outputs / "q1_v3_multimodal_labels.csv"
            self.make_label_lookup(read_csv(labels_path))
        all_routes: List[dict] = []
        variant_counter = 1
        strategies = ["balanced", "cost_eff", "region", "culture", "comfort", "north", "south"]
        for q in self.config["coverage_grid"]:
            candidates: List[dict] = []
            seqs: List[List[str]] = []
            ordered_seed = self.ordered_v2_seed(q)
            if len(ordered_seed) == q:
                for direct_variant in [ordered_seed, self.two_opt(ordered_seed, "balanced", max_passes=1)]:
                    route = self.evaluate_route(direct_variant, q, "balanced", variant_counter)
                    variant_counter += 1
                    if route is not None:
                        route["generation_method"] = "v2_ordered_seed_direct_label_choice"
                        candidates.append(route)
            for strategy in strategies:
                seqs.append(self.repair_selection(self.sequence_from_v2(q, strategy), q, strategy))
                seqs.append(self.greedy_selection(q, strategy))
                for _ in range(3):
                    seqs.append(self.random_selection(q, strategy))
            seen = set()
            for idx, selected in enumerate(seqs):
                if len(selected) != q:
                    selected = self.repair_selection(selected, q, "balanced")
                strategy = strategies[idx % len(strategies)]
                for order_variant in range(2):
                    ordered = self.order_sequence(selected, strategy, variant=order_variant)
                    sig = tuple(ordered)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    route = self.evaluate_route(ordered, q, strategy, variant_counter)
                    variant_counter += 1
                    if route is not None:
                        candidates.append(route)
            candidates = sorted(
                candidates,
                key=lambda r: (bool(r["hard_constraints_pass"]), r["fast_master_score"]),
                reverse=True,
            )
            kept = candidates[: int(self.config["candidate_routes_per_q"])]
            all_routes.extend(kept)
        df = pd.DataFrame(all_routes)
        if not df.empty:
            df = df.sort_values(["target_coverage_q", "fast_master_score"], ascending=[True, False]).reset_index(drop=True)
        write_csv(df, self.outputs / "q1_v3_candidate_routes.csv")
        self.candidate_routes = df
        return df

    # ------------------------------------------------------------------
    # P5: Hourly scheduler
    # ------------------------------------------------------------------

    def is_heat_sensitive_spot(self, sid: str) -> bool:
        row = self.spots_by_id[sid]
        name = str(row["spot_name"])
        region = str(row["region_cluster"])
        return (
            any(key in name for key in ["火焰山", "吐峪沟", "葡萄沟", "罗布", "沙", "冰川"])
            or region in {"东疆", "南疆", "巴州-南疆北线"}
        ) and as_bool(row.get("is_natural", False))

    def hotel_price_for_spot(self, sid: str) -> Tuple[float, bool, str]:
        hub = str(self.spots_by_id[sid].get("hub_name", ""))
        row = self.hotel_by_hub.get(hub)
        if row is None:
            return 260.0, True, hub
        return (
            as_float(row.get("default_room_price_yuan_per_night", 240.0), 240.0),
            not as_bool(row.get("is_hotel_hub", True)),
            hub,
        )

    def get_label_by_id(self, label_id: str) -> Optional[dict]:
        if not hasattr(self, "_label_by_id"):
            self._label_by_id = {}
            for labels in self.labels_by_pair.values():
                for row in labels:
                    self._label_by_id[str(row.get("label_id", ""))] = row
        return self._label_by_id.get(label_id)

    def add_activity_row(
        self,
        rows: List[dict],
        route_id: str,
        day: int,
        seq_no: int,
        activity_type: str,
        start: float,
        end: float,
        from_name: str = "",
        to_name: str = "",
        spot_id: str = "",
        spot_name: str = "",
        label_id: str = "",
        mode_sequence: str = "",
        travel_hours: float = 0.0,
        service_hours: float = 0.0,
        cost_yuan: float = 0.0,
        open_time: float = 0.0,
        close_time: float = 24.0,
        time_window_feasible: bool = True,
        lodging_hub: str = "",
        late_hotel: bool = False,
        heat_avoidance_applied: bool = False,
        recovery_after_long_transfer: bool = False,
        note: str = "",
    ) -> int:
        rows.append(
            {
                "route_id": route_id,
                "day": day,
                "sequence_no": seq_no,
                "activity_type": activity_type,
                "from_name": from_name,
                "to_name": to_name,
                "spot_id": spot_id,
                "spot_name": spot_name,
                "start_time": hour_to_str(start),
                "end_time": hour_to_str(end),
                "duration_hours": round(max(0.0, end - start), 3),
                "open_time": hour_to_str(open_time) if activity_type == "visit" else "",
                "close_time": hour_to_str(close_time) if activity_type == "visit" else "",
                "time_window_feasible": bool(time_window_feasible),
                "travel_hours": round(travel_hours, 3),
                "service_hours": round(service_hours, 3),
                "cost_yuan_for_two_or_room": round(cost_yuan, 2),
                "label_id": label_id,
                "mode_sequence": mode_sequence,
                "lodging_hub": lodging_hub,
                "late_hotel": bool(late_hotel),
                "heat_avoidance_applied": bool(heat_avoidance_applied),
                "recovery_after_long_transfer": bool(recovery_after_long_transfer),
                "activity_note": note,
            }
        )
        return seq_no + 1

    def run_hourly_scheduler(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.candidate_routes.empty:
            self.candidate_routes = read_csv(self.outputs / "q1_v3_candidate_routes.csv")
        if not self.labels_by_pair:
            self.make_label_lookup(read_csv(self.outputs / "q1_v3_multimodal_labels.csv"))
        day_start = parse_time_to_hour(str(self.config["day_start"]), 8.5)
        soft_end = parse_time_to_hour(str(self.config["soft_day_end"]), 20.5)
        hard_end = parse_time_to_hour(str(self.config["hard_day_end"]), 22.0)
        lunch_start = parse_time_to_hour(str(self.config["lunch_start"]), 12.0)
        lunch_end = parse_time_to_hour(str(self.config["lunch_end"]), 13.0)
        standard_limit = as_float(self.config["daily_active_limit_standard"], 8.5)
        rows: List[dict] = []
        infeasible_rows: List[dict] = []
        summary_rows: List[dict] = []

        for _, route in self.candidate_routes.iterrows():
            route_id = str(route["route_id"])
            seq = [sid.strip() for sid in str(route["spot_id_sequence"]).split("->") if sid.strip()]
            label_ids = [x.strip() for x in str(route.get("label_sequence", "")).split("|") if x.strip()]
            day = 1
            time = day_start
            seq_no = 1
            daily_active: Dict[int, float] = defaultdict(float)
            daily_travel: Dict[int, float] = defaultdict(float)
            daily_service: Dict[int, float] = defaultdict(float)
            daily_late: Dict[int, bool] = defaultdict(bool)
            daily_heat: Dict[int, bool] = defaultdict(bool)
            daily_recovery: Dict[int, bool] = defaultdict(bool)
            violations = Counter()
            prev_sid: Optional[str] = None
            recovery_today = False

            def finish_day(force_next: bool = True) -> None:
                nonlocal day, time, recovery_today
                if force_next:
                    day += 1
                    time = day_start
                    recovery_today = False

            for pos, sid in enumerate(seq):
                spot_row = self.spots_by_id[sid]
                spot_name = str(spot_row["spot_name"])
                service = as_float(spot_row.get("visit_hours_mid", 3.0), 3.0)
                tw = self.time_window_by_spot.get(sid, {"open": 8.5, "close": 19.0})
                open_h = as_float(tw.get("open", 8.5), 8.5)
                close_h = as_float(tw.get("close", 19.0), 19.0)
                heat_sensitive = self.is_heat_sensitive_spot(sid)
                if prev_sid is None:
                    depot = self.depot_by_spot.get(sid)
                    travel = as_float(depot.get("depot_to_spot_time", 0.0), 0.0) if depot is not None else 0.0
                    travel_cost = as_float(depot.get("depot_to_spot_cost", 0.0), 0.0) if depot is not None else 0.0
                    label_id = "DEPOT_TO_FIRST"
                    mode_sequence = "taxi_transfer"
                    from_name = "乌鲁木齐起点"
                else:
                    label = self.get_label_by_id(label_ids[pos - 1]) if pos - 1 < len(label_ids) else None
                    if label is None:
                        label = self.label_for_pair(prev_sid, sid, "balanced")
                    travel = as_float(label.get("time_hours", 0.0), 0.0)
                    travel_cost = as_float(label.get("cost_yuan_for_two", 0.0), 0.0)
                    label_id = str(label.get("label_id", ""))
                    mode_sequence = str(label.get("mode_sequence", ""))
                    from_name = str(self.spots_by_id[prev_sid]["spot_name"])

                long_transfer = travel >= as_float(self.config["long_transfer_hours"], 6.0)
                if long_transfer and (time > day_start + 0.05 or daily_active[day] > 0):
                    finish_day()
                if long_transfer:
                    start = time
                    end = time + travel
                    late = end > soft_end
                    seq_no = self.add_activity_row(
                        rows,
                        route_id,
                        day,
                        seq_no,
                        "long_transfer",
                        start,
                        end,
                        from_name=from_name,
                        to_name=spot_name,
                        label_id=label_id,
                        mode_sequence=mode_sequence,
                        travel_hours=travel,
                        cost_yuan=travel_cost,
                        late_hotel=late,
                        note="长转场日，下一天降低活动强度",
                    )
                    daily_active[day] += travel
                    daily_travel[day] += travel
                    daily_late[day] = daily_late[day] or late
                    if end > hard_end:
                        violations["late_arrival_after_hard_end"] += 1
                    can_visit_same_day = (
                        travel <= 7.25
                        and service <= 3.5
                        and end <= 16.25
                        and end + service <= close_h + 0.5
                        and not (heat_sensitive and end < 16.0)
                    )
                    if can_visit_same_day:
                        time = end
                        recovery_today = True
                        daily_recovery[day] = True
                    else:
                        day += 1
                        time = day_start
                        recovery_today = True
                        daily_recovery[day] = True
                    travel = 0.0
                    travel_cost = 0.0
                    from_name = spot_name
                    label_id = "ARRIVED_FROM_PREVIOUS_LONG_TRANSFER"
                    mode_sequence = "recovery"

                # Try placing non-long travel plus visit. Repair by moving to next day.
                placed = False
                repair_attempts = 0
                while not placed and repair_attempts < 4:
                    repair_attempts += 1
                    limit = standard_limit - (1.5 if recovery_today else 0.0)
                    projected_travel_end = time + travel
                    projected_start = projected_travel_end
                    if projected_start < open_h:
                        projected_start = open_h
                    lunch_needed = False
                    if projected_start < lunch_start and projected_start + service > lunch_start:
                        lunch_needed = True
                    if lunch_start <= projected_start < lunch_end:
                        lunch_needed = True
                        projected_start = lunch_end
                    heat_avoid = False
                    if heat_sensitive and 12.0 <= projected_start < 16.0:
                        heat_avoid = True
                        projected_start = max(projected_start, 16.0)
                    projected_end = projected_start + service
                    added_active = travel + service + (lunch_end - lunch_start if lunch_needed else 0.0)
                    if (
                        (projected_end > close_h + 1e-6 or projected_end > hard_end + 1e-6 or daily_active[day] + added_active > limit + 1e-6)
                        and daily_active[day] > 0.01
                    ):
                        finish_day()
                        continue
                    if projected_end > close_h + 1e-6 and daily_active[day] <= 0.01 and travel >= 3.0:
                        t_start = time
                        t_end = time + travel
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "transfer_for_time_window",
                            t_start,
                            t_end,
                            from_name=from_name,
                            to_name=spot_name,
                            label_id=label_id,
                            mode_sequence=mode_sequence,
                            travel_hours=travel,
                            cost_yuan=travel_cost,
                            late_hotel=t_end > soft_end,
                            recovery_after_long_transfer=recovery_today,
                            note="到达后无法满足景区时间窗，拆为转场日并次日入园",
                        )
                        daily_active[day] += travel
                        daily_travel[day] += travel
                        daily_late[day] = daily_late[day] or (t_end > soft_end)
                        finish_day()
                        travel = 0.0
                        travel_cost = 0.0
                        from_name = spot_name
                        label_id = "ARRIVED_FROM_PREVIOUS_TIME_WINDOW_TRANSFER"
                        mode_sequence = "recovery"
                        continue
                    if projected_end > hard_end + 1e-6:
                        violations["after_hard_day_end"] += 1
                    # Add travel row if needed.
                    if travel > 0:
                        t_start = time
                        t_end = time + travel
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "travel",
                            t_start,
                            t_end,
                            from_name=from_name,
                            to_name=spot_name,
                            label_id=label_id,
                            mode_sequence=mode_sequence,
                            travel_hours=travel,
                            cost_yuan=travel_cost,
                            late_hotel=t_end > soft_end,
                            recovery_after_long_transfer=recovery_today,
                        )
                        daily_active[day] += travel
                        daily_travel[day] += travel
                        daily_late[day] = daily_late[day] or (t_end > soft_end)
                        time = t_end
                    if lunch_needed and time <= lunch_start:
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "lunch_break",
                            lunch_start,
                            lunch_end,
                            note="固定午休块",
                        )
                        daily_active[day] += lunch_end - lunch_start
                        time = lunch_end
                    if time < open_h:
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "wait_open",
                            time,
                            open_h,
                            spot_id=sid,
                            spot_name=spot_name,
                            note="等待开园",
                        )
                        time = open_h
                    if lunch_start <= time < lunch_end:
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "lunch_break",
                            time,
                            lunch_end,
                            note="午休后入园",
                        )
                        daily_active[day] += lunch_end - time
                        time = lunch_end
                    if heat_sensitive and 12.0 <= time < 16.0:
                        seq_no = self.add_activity_row(
                            rows,
                            route_id,
                            day,
                            seq_no,
                            "heat_avoidance_wait",
                            time,
                            16.0,
                            spot_id=sid,
                            spot_name=spot_name,
                            heat_avoidance_applied=True,
                            note="高温户外避让",
                        )
                        daily_heat[day] = True
                        time = 16.0
                    v_start = max(time, open_h)
                    v_end = v_start + service
                    tw_feasible = v_end <= close_h + 1e-6
                    if not tw_feasible:
                        violations["time_window_violation"] += 1
                    seq_no = self.add_activity_row(
                        rows,
                        route_id,
                        day,
                        seq_no,
                        "visit",
                        v_start,
                        v_end,
                        spot_id=sid,
                        spot_name=spot_name,
                        service_hours=service,
                        cost_yuan=2.0 * as_float(spot_row.get("ticket_high_total_yuan_per_person", 0.0), 0.0),
                        open_time=open_h,
                        close_time=close_h,
                        time_window_feasible=tw_feasible,
                        heat_avoidance_applied=heat_sensitive and v_start >= 16.0,
                        recovery_after_long_transfer=recovery_today,
                    )
                    daily_active[day] += service
                    daily_service[day] += service
                    daily_heat[day] = daily_heat[day] or (heat_sensitive and 12.0 <= v_start < 16.0)
                    time = v_end
                    placed = True
                if not placed:
                    infeasible_rows.append(
                        {
                            "route_id": route_id,
                            "spot_id": sid,
                            "spot_name": spot_name,
                            "infeasibility_reason": "placement_repair_failed",
                            "detail": "4次换日修复后仍无法放入小时级时间窗",
                        }
                    )
                prev_sid = sid

            # Return to depot.
            if seq:
                last_sid = seq[-1]
                depot = self.depot_by_spot.get(last_sid)
                back_time = as_float(depot.get("spot_to_depot_time", 0.0), 0.0) if depot is not None else 0.0
                back_cost = as_float(depot.get("spot_to_depot_cost", 0.0), 0.0) if depot is not None else 0.0
                if time + back_time > hard_end and daily_active[day] > 0:
                    finish_day()
                seq_no = self.add_activity_row(
                    rows,
                    route_id,
                    day,
                    seq_no,
                    "return_depot",
                    time,
                    time + back_time,
                    from_name=str(self.spots_by_id[last_sid]["spot_name"]),
                    to_name="乌鲁木齐终点",
                    label_id="RETURN_TO_DEPOT",
                    mode_sequence="taxi_transfer",
                    travel_hours=back_time,
                    cost_yuan=back_cost,
                    late_hotel=time + back_time > soft_end,
                )
                daily_active[day] += back_time
                daily_travel[day] += back_time
                daily_late[day] = daily_late[day] or (time + back_time > soft_end)

            active_days = max(daily_active.keys()) if daily_active else 0
            if active_days > 30:
                infeasible_rows.append(
                    {
                        "route_id": route_id,
                        "spot_id": "",
                        "spot_name": "",
                        "infeasibility_reason": "over_30_days",
                        "detail": f"小时级排程需要{active_days}天，超过30天",
                    }
                )
            buffer_days = max(0, 30 - active_days)
            # Add buffer-day rows for interpretability.
            for bd in range(active_days + 1, min(30, active_days + buffer_days) + 1):
                seq_no = self.add_activity_row(
                    rows,
                    route_id,
                    bd,
                    seq_no,
                    "buffer_day",
                    day_start,
                    day_start,
                    note="机动缓冲日，可吸收天气/预约/交通扰动",
                )

            comfort_scores = []
            red = yellow = green = 0
            late_days = 0
            long_days = 0
            for d, active in daily_active.items():
                travel = daily_travel[d]
                late = daily_late[d]
                if travel >= as_float(self.config["long_transfer_hours"], 6.0):
                    long_days += 1
                if late:
                    late_days += 1
                comfort = 100 - max(0.0, active - 5.5) * 6.0 - max(0.0, travel - 4.0) * 4.0
                comfort -= 4.0 if late else 0.0
                comfort -= 3.0 if daily_heat[d] else 0.0
                comfort = max(35.0, min(100.0, comfort))
                comfort_scores.append(comfort)
                if active > 10.0 or travel > 7.5 or late:
                    red += 1
                elif active > standard_limit or travel > 5.5 or daily_heat[d]:
                    yellow += 1
                else:
                    green += 1
            schedule_soft_feasible = active_days <= 30 and violations["time_window_violation"] <= 2
            schedule_strict_feasible = (
                active_days <= 30
                and violations["time_window_violation"] == 0
                and violations["late_arrival_after_hard_end"] + violations["after_hard_day_end"] == 0
            )
            summary_rows.append(
                {
                    "route_id": route_id,
                    "active_or_transfer_days": active_days,
                    "planned_trip_days": max(30, active_days) if active_days > 30 else 30,
                    "buffer_days": buffer_days,
                    "red_days": red,
                    "yellow_days": yellow,
                    "green_days": green + buffer_days,
                    "long_transfer_days": long_days,
                    "late_hotel_days": late_days,
                    "max_day_active_hours": round(max(daily_active.values()) if daily_active else 0.0, 3),
                    "mean_day_active_hours": round(statistics.mean(daily_active.values()) if daily_active else 0.0, 3),
                    "mean_comfort_score": round(statistics.mean(comfort_scores) if comfort_scores else 100.0, 3),
                    "p10_comfort_score": round(percentile(comfort_scores, 10, 100.0), 3),
                    "time_window_violations": int(violations["time_window_violation"]),
                    "late_after_hard_end_violations": int(violations["late_arrival_after_hard_end"] + violations["after_hard_day_end"]),
                    "schedule_soft_feasible": bool(schedule_soft_feasible),
                    "schedule_strict_feasible": bool(schedule_strict_feasible),
                    "schedule_feasible": bool(schedule_soft_feasible),
                    "scheduler_note": "启发式小时级排程：开放时间、午休、高温避让、长转场恢复、晚到酒店均显式记录",
                }
            )

        itinerary = pd.DataFrame(rows)
        infeasible = pd.DataFrame(infeasible_rows)
        schedule_summary = pd.DataFrame(summary_rows)
        write_csv(itinerary, self.outputs / "q1_v3_hourly_itinerary.csv")
        write_csv(infeasible, self.outputs / "q1_v3_schedule_infeasibility.csv")
        write_csv(schedule_summary, self.outputs / "q1_v3_schedule_summary.csv")
        self.hourly_itinerary = itinerary
        self.schedule_summary = schedule_summary
        # Enrich candidate route output with schedule metrics.
        enriched = self.candidate_routes.merge(schedule_summary, on="route_id", how="left")
        write_csv(enriched, self.outputs / "q1_v3_candidate_routes.csv")
        self.candidate_routes = enriched
        return itinerary, schedule_summary

    # ------------------------------------------------------------------
    # P6: Route-specific simulation
    # ------------------------------------------------------------------

    def mode_factor_for_route(self, mode_hours_mix: str, scenario: pd.Series, suffix: str) -> float:
        pairs = {}
        for part in str(mode_hours_mix).split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                pairs[k.strip()] = as_float(v, 0.0)
        total = sum(pairs.values())
        if total <= 0:
            return 1.0
        factor_sum = 0.0
        for mode, hours in pairs.items():
            if mode in {"rental_car", "charter_car", "taxi_transfer", "scenic_shuttle"}:
                col = f"road_{suffix}" if mode in {"rental_car", "charter_car"} else f"local_{suffix}"
            elif mode == "coach":
                col = f"coach_{suffix}"
            elif mode == "rail":
                col = f"rail_{suffix}"
            elif mode == "air":
                col = f"air_{suffix}"
            else:
                col = f"road_{suffix}"
            factor_sum += hours * as_float(scenario.get(col, 1.0), 1.0)
        return factor_sum / total

    def run_route_simulator(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.candidate_routes.empty:
            self.candidate_routes = read_csv(self.outputs / "q1_v3_candidate_routes.csv")
        scenarios_path = self.outputs / "q1_v3_scenario_samples.csv"
        scenarios = read_csv(scenarios_path) if scenarios_path.exists() else self.generate_scenarios()
        for col in scenarios.columns:
            if col not in {"scenario_id", "scenario_description", "heatwave_flag"}:
                try:
                    scenarios[col] = scenarios[col].map(as_float)
                except Exception:
                    pass
        rows: List[dict] = []
        summary_rows: List[dict] = []
        alpha = as_float(self.config["cvar_alpha"], 0.75)
        alpha_high = as_float(self.config["cvar_alpha_high"], 0.90)
        for _, route in self.candidate_routes.iterrows():
            route_id = str(route["route_id"])
            base_active_days = as_int(route.get("active_or_transfer_days", route.get("estimated_active_days", 30)), 30)
            buffer_days = as_int(route.get("buffer_days", max(0, 30 - base_active_days)), 0)
            total_travel = as_float(route.get("total_travel_hours", 0.0), 0.0)
            total_service = as_float(route.get("total_service_hours", 0.0), 0.0)
            base_cost = as_float(route.get("total_cost_yuan_excluding_meals", 0.0), 0.0)
            base_red = as_int(route.get("red_days", 0), 0)
            base_comfort = as_float(route.get("mean_comfort_score", 82.0), 82.0)
            reservation_spots = as_int(route.get("reservation_spots", 0), 0)
            simulated_capacity_spots = as_int(route.get("capacity_simulated_spots", 0), 0)
            remote_spots = as_int(route.get("remote_or_high_altitude_spots", 0), 0)
            nights = max(1, base_active_days - 1)
            long_transfer_days = as_int(route.get("long_transfer_days", route.get("long_transfer_edges", 0)), 0)
            hot_spots = 0
            for sid in str(route.get("spot_id_sequence", "")).split("->"):
                sid = sid.strip()
                if sid and sid in self.spots_by_id and self.is_heat_sensitive_spot(sid):
                    hot_spots += 1
            losses = []
            sim_days_values = []
            sim_cost_values = []
            success_values = []
            red_values = []
            comfort_values = []
            for _, sc in scenarios.iterrows():
                time_factor = self.mode_factor_for_route(str(route.get("mode_hours_mix", "")), sc, "time_factor")
                cost_factor = self.mode_factor_for_route(str(route.get("mode_hours_mix", "")), sc, "cost_factor")
                service_factor = as_float(sc.get("service_time_factor", 1.0), 1.0)
                heat_penalty = as_float(sc.get("heat_penalty_hours", 0.0), 0.0) * min(3, hot_spots) / 3.0
                scenario_id = str(sc.get("scenario_id", ""))
                if scenario_id == "duku_weather_disruption":
                    mountain_exposure = min(0.72, 0.08 * remote_spots + 0.045 * long_transfer_days)
                    time_factor = 1.0 + (time_factor - 1.0) * mountain_exposure
                    cost_factor = 1.0 + (cost_factor - 1.0) * mountain_exposure
                    heat_penalty *= 0.45
                extra_hours = max(0.0, total_travel * (time_factor - 1.0)) + max(
                    0.0, total_service * (service_factor - 1.0)
                ) + heat_penalty
                absorb = buffer_days * as_float(self.config["buffer_absorb_hours_per_day"], 6.0)
                residual_hours = max(0.0, extra_hours - absorb)
                extra_days = int(math.ceil(residual_hours / 8.0)) if residual_hours > 0 else 0
                road_closure_p = as_float(sc.get("road_closure_probability", 0.0), 0.0)
                if scenario_id == "duku_weather_disruption":
                    road_closure_p *= min(0.72, 0.08 * remote_spots + 0.045 * long_transfer_days)
                public_p = as_float(sc.get("public_transport_disruption_probability", 0.0), 0.0)
                severe_delay_p = min(
                    0.18,
                    road_closure_p * 0.35
                    + public_p * as_float(route.get("public_transport_hours_share", 0.0), 0.0) * 0.55,
                )
                severe_delays = int(self.rng.binomial(max(1, long_transfer_days), severe_delay_p)) if long_transfer_days else int(self.rng.binomial(1, severe_delay_p * 0.35))
                simulated_days = base_active_days + extra_days + int(math.ceil(severe_delays * 0.6))
                reservation_base_p = as_float(sc.get("reservation_pressure", 0.05), 0.05)
                capacity_penalty = 0.015 * simulated_capacity_spots / max(1, reservation_spots)
                # The case assumes planned summer travel rather than walk-up
                # same-day purchase, so reservation pressure is discounted by
                # advance booking. Simulated capacity gaps still add stress.
                res_fail_p = min(0.22, reservation_base_p * 0.38 + capacity_penalty)
                reservation_failures = int(self.rng.binomial(reservation_spots, res_fail_p)) if reservation_spots else 0
                hotel_full_p = as_float(sc.get("hotel_full_probability", 0.05), 0.05)
                # Rooms are assumed to be pre-booked around route hubs; hotel
                # full events represent forced rebooking/price surge, not every
                # night being independently searched at arrival.
                hotel_full_events = int(self.rng.binomial(nights, min(0.12, hotel_full_p * 0.22 + 0.008)))
                cost_multiplier = max(cost_factor, as_float(sc.get("general_cost_multiplier", 1.0), 1.0))
                simulated_cost = base_cost * cost_multiplier + hotel_full_events * 180 + severe_delays * 250
                red_days = base_red + int(extra_hours > 4.0) + severe_delays + int(as_bool(sc.get("heatwave_flag", False)) and hot_spots >= 2)
                comfort = base_comfort - max(0.0, time_factor - 1.0) * 35 - heat_penalty * 1.5 - severe_delays * 4 - reservation_failures * 1.5
                comfort = max(30.0, min(100.0, comfort))
                overrun_days = max(0, simulated_days - 30)
                loss = (
                    900.0 * overrun_days
                    + 260.0 * reservation_failures
                    + 220.0 * hotel_full_events
                    + 420.0 * severe_delays
                    + 180.0 * max(0, red_days - 1)
                    + max(0.0, simulated_cost - base_cost) * 0.30
                    + max(0.0, 75.0 - comfort) * 45.0
                )
                success = bool(
                    simulated_days <= 30
                    and comfort >= 75.0
                    and reservation_failures <= 2
                    and hotel_full_events <= 2
                    and red_days <= max(1, base_red + 1)
                )
                rows.append(
                    {
                        "route_id": route_id,
                        "sample_id": as_int(sc.get("sample_id", 0), 0),
                        "scenario_id": str(sc.get("scenario_id", "")),
                        "simulated_days": simulated_days,
                        "simulated_cost_yuan": round(simulated_cost, 2),
                        "extra_hours_before_buffer": round(extra_hours, 3),
                        "buffer_absorbed_hours": round(min(absorb, extra_hours), 3),
                        "reservation_failures": reservation_failures,
                        "hotel_full_events": hotel_full_events,
                        "severe_delay_events": severe_delays,
                        "red_days": red_days,
                        "mean_comfort_score": round(comfort, 3),
                        "loss": round(loss, 3),
                        "success": success,
                    }
                )
                losses.append(loss)
                sim_days_values.append(simulated_days)
                sim_cost_values.append(simulated_cost)
                success_values.append(1.0 if success else 0.0)
                red_values.append(red_days)
                comfort_values.append(comfort)
            summary_rows.append(
                {
                    "route_id": route_id,
                    "simulation_samples": len(scenarios),
                    "success_probability": round(float(np.mean(success_values)), 4),
                    "overrun_probability": round(float(np.mean([d > 30 for d in sim_days_values])), 4),
                    "expected_days": round(float(np.mean(sim_days_values)), 3),
                    "p90_days": round(percentile(sim_days_values, 90), 3),
                    "p95_days": round(percentile(sim_days_values, 95), 3),
                    "expected_cost_yuan": round(float(np.mean(sim_cost_values)), 2),
                    "p95_cost_yuan": round(percentile(sim_cost_values, 95), 2),
                    "expected_loss": round(float(np.mean(losses)), 3),
                    "cvar75_loss": round(cvar(losses, alpha), 3),
                    "cvar90_loss": round(cvar(losses, alpha_high), 3),
                    "expected_red_days": round(float(np.mean(red_values)), 3),
                    "prob_red_days_gt1": round(float(np.mean([r > 1 for r in red_values])), 4),
                    "mean_simulated_comfort": round(float(np.mean(comfort_values)), 3),
                    "p10_simulated_comfort": round(percentile(comfort_values, 10), 3),
                    "chance_constraint_pass": bool(float(np.mean(success_values)) >= as_float(self.config["success_threshold"], 0.8)),
                    "simulation_note": "route-specific Monte Carlo；扰动由路线交通方式、缓冲日、预约点、酒店压力和高温暴露共同驱动",
                }
            )
        trials = pd.DataFrame(rows)
        summary = pd.DataFrame(summary_rows)
        write_csv(trials, self.outputs / "q1_v3_simulation_trials.csv")
        write_csv(summary, self.outputs / "q1_v3_simulation_summary.csv")
        self.simulation_trials = trials
        self.simulation_summary = summary
        return trials, summary

    # ------------------------------------------------------------------
    # P7: Pareto selector
    # ------------------------------------------------------------------

    def is_pareto_front(self, df: pd.DataFrame) -> List[bool]:
        cols = [
            "spots_count",
            "total_cost_yuan_excluding_meals",
            "mean_comfort_score",
            "success_probability",
            "cvar75_loss",
        ]
        rows = []
        for _, r in df.iterrows():
            rows.append(
                (
                    as_float(r.get("spots_count", 0)),
                    as_float(r.get("total_cost_yuan_excluding_meals", 1e9)),
                    as_float(r.get("mean_comfort_score", 0)),
                    as_float(r.get("success_probability", 0)),
                    as_float(r.get("cvar75_loss", 1e9)),
                )
            )
        front = []
        for i, a in enumerate(rows):
            dominated_flag = False
            for j, b in enumerate(rows):
                if i == j:
                    continue
                better_or_equal = (
                    b[0] >= a[0] - 1e-9
                    and b[1] <= a[1] + 1e-9
                    and b[2] >= a[2] - 1e-9
                    and b[3] >= a[3] - 1e-9
                    and b[4] <= a[4] + 1e-9
                )
                strict = (
                    b[0] > a[0] + 1e-9
                    or b[1] < a[1] - 1e-9
                    or b[2] > a[2] + 1e-9
                    or b[3] > a[3] + 1e-9
                    or b[4] < a[4] - 1e-9
                )
                if better_or_equal and strict:
                    dominated_flag = True
                    break
            front.append(not dominated_flag)
        return front

    def robust_utility(self, df: pd.DataFrame) -> pd.Series:
        def norm_high(series):
            s = pd.to_numeric(series, errors="coerce").fillna(0)
            lo, hi = float(s.min()), float(s.max())
            return (s - lo) / (hi - lo + 1e-9)

        def norm_low(series):
            return 1.0 - norm_high(series)

        return (
            0.30 * norm_high(df["spots_count"])
            + 0.16 * norm_high(df["region_count"])
            + 0.18 * norm_high(df["mean_comfort_score"])
            + 0.20 * norm_high(df["success_probability"])
            + 0.10 * norm_low(df["total_cost_yuan_excluding_meals"])
            + 0.06 * norm_low(df["cvar75_loss"])
        )

    def pick_best(self, df: pd.DataFrame, mask: pd.Series, role: str, fallback_note: str) -> dict:
        pool = df[mask].copy()
        if pool.empty:
            pool = df.copy()
            selection_status = f"fallback: {fallback_note}"
        else:
            selection_status = "criteria_pass"
        pool = pool.sort_values(["robust_utility", "success_probability", "spots_count"], ascending=False)
        row = pool.iloc[0].to_dict()
        row["selected_role"] = role
        row["selection_status"] = selection_status
        return row

    def pick_best_with_metric(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        role: str,
        metric_col: str,
        fallback_note: str,
        exclude_route_ids: Optional[Iterable[str]] = None,
        relaxed_mask: Optional[pd.Series] = None,
    ) -> dict:
        pool = df[mask].copy()
        selection_status = "criteria_pass"
        if pool.empty and relaxed_mask is not None:
            pool = df[relaxed_mask].copy()
            selection_status = f"fallback_relaxed: {fallback_note}"
        if pool.empty:
            pool = df.copy()
            selection_status = f"fallback: {fallback_note}"
        excluded = set(exclude_route_ids or [])
        if excluded and len(pool[~pool["route_id"].astype(str).isin(excluded)]) > 0:
            pool = pool[~pool["route_id"].astype(str).isin(excluded)].copy()
        pool = pool.sort_values([metric_col, "success_probability", "spots_count"], ascending=False)
        row = pool.iloc[0].to_dict()
        row["selected_role"] = role
        row["selection_status"] = selection_status
        return row

    def select_robust_pareto_front(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.candidate_routes.empty:
            self.candidate_routes = read_csv(self.outputs / "q1_v3_candidate_routes.csv")
        if self.simulation_summary.empty:
            self.simulation_summary = read_csv(self.outputs / "q1_v3_simulation_summary.csv")
        merged = self.candidate_routes.merge(self.simulation_summary, on="route_id", how="left")
        if merged.empty:
            raise RuntimeError("No candidate routes for Pareto selection.")
        for col in [
            "spots_count",
            "total_cost_yuan_excluding_meals",
            "mean_comfort_score",
            "success_probability",
            "cvar75_loss",
            "buffer_days",
            "red_days",
            "mean_day_active_hours",
            "late_hotel_days",
            "remote_or_high_altitude_spots",
            "long_transfer_days",
            "active_or_transfer_days",
            "max_day_active_hours",
            "time_window_violations",
            "late_after_hard_end_violations",
        ]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
        for col in ["schedule_soft_feasible", "schedule_strict_feasible"]:
            if col not in merged.columns:
                merged[col] = False
            merged[col] = merged[col].map(as_bool)
        merged["robust_utility"] = self.robust_utility(merged).round(5)
        def norm_high(series):
            s = pd.to_numeric(series, errors="coerce").fillna(0)
            lo, hi = float(s.min()), float(s.max())
            return (s - lo) / (hi - lo + 1e-9)

        def norm_low(series):
            return 1.0 - norm_high(series)

        merged["family_utility"] = (
            0.28 * norm_high(merged["success_probability"])
            + 0.22 * norm_low(merged["red_days"])
            + 0.18 * norm_low(merged["max_day_active_hours"])
            + 0.12 * norm_low(merged["long_transfer_days"])
            + 0.08 * norm_low(merged["late_hotel_days"])
            + 0.07 * norm_high(merged["buffer_days"])
            + 0.05 * norm_low(merged["time_window_violations"])
        ).round(5)
        merged["senior_utility"] = (
            0.26 * norm_high(merged["success_probability"])
            + 0.22 * norm_low(merged["red_days"])
            + 0.18 * norm_low(merged["mean_day_active_hours"])
            + 0.12 * norm_low(merged["remote_or_high_altitude_spots"])
            + 0.10 * norm_low(merged["long_transfer_days"])
            + 0.07 * norm_high(merged["buffer_days"])
            + 0.05 * norm_low(merged["time_window_violations"])
        ).round(5)
        merged["is_robust_pareto"] = self.is_pareto_front(merged)
        front = merged[merged["is_robust_pareto"]].copy().sort_values(
            ["spots_count", "success_probability", "cvar75_loss"],
            ascending=[False, False, True],
        )
        write_csv(front, self.outputs / "q1_v3_robust_pareto_front.csv")
        feasible_pool = merged[
            (merged["success_probability"] >= as_float(self.config["success_threshold"], 0.8))
            & (merged["schedule_strict_feasible"])
        ].copy()
        if not feasible_pool.empty:
            feasible_pool["is_feasible_robust_pareto"] = self.is_pareto_front(feasible_pool)
            feasible_front = feasible_pool[feasible_pool["is_feasible_robust_pareto"]].copy().sort_values(
                ["spots_count", "success_probability", "cvar75_loss"],
                ascending=[False, False, True],
            )
        else:
            feasible_front = feasible_pool
        write_csv(feasible_front, self.outputs / "q1_v3_feasible_robust_pareto_front.csv")
        extreme_mask = merged["spots_count"] == merged["spots_count"].max()
        coverage30_mask = (
            (merged["spots_count"] >= int(self.config["main_route_min_spots"]))
            & (merged["buffer_days"] >= int(self.config["main_route_min_buffer_days"]))
            & (merged["active_or_transfer_days"] <= 30)
        )
        robust_main_mask = (
            (merged["spots_count"] >= 24)
            & (merged["buffer_days"] >= 2)
            & (merged["success_probability"] >= as_float(self.config["success_threshold"], 0.8))
            & (merged["red_days"] <= 2)
            & (merged["mean_comfort_score"] >= 85)
            & (merged["schedule_strict_feasible"])
        )
        family_mask = (
            (merged["spots_count"].between(22, 26))
            & (merged["success_probability"] >= 0.80)
            & (merged["mean_day_active_hours"] <= as_float(self.config["daily_active_limit_family"], 7.0))
            & (merged["red_days"] <= int(self.config["max_red_days_family"]))
            & (merged["late_hotel_days"] == 0)
            & (merged["buffer_days"] >= 4)
            & (merged["schedule_strict_feasible"])
        )
        senior_mask = (
            (merged["spots_count"].between(20, 24))
            & (merged["success_probability"] >= 0.82)
            & (merged["mean_day_active_hours"] <= as_float(self.config["daily_active_limit_senior"], 6.5))
            & (merged["red_days"] <= int(self.config["max_red_days_senior"]))
            & (merged["remote_or_high_altitude_spots"] <= 1)
            & (merged["late_hotel_days"] <= 1)
            & (merged["buffer_days"] >= 6)
            & (merged["schedule_strict_feasible"])
        )
        family_relaxed = (
            (merged["spots_count"].between(20, 26))
            & (merged["success_probability"] >= 0.78)
            & (merged["late_hotel_days"] == 0)
            & (merged["buffer_days"] >= 4)
            & (merged["schedule_soft_feasible"])
        )
        senior_relaxed = (
            (merged["spots_count"].between(20, 24))
            & (merged["success_probability"] >= 0.78)
            & (merged["remote_or_high_altitude_spots"] <= 1)
            & (merged["buffer_days"] >= 6)
            & (merged["schedule_soft_feasible"])
        )
        extreme = self.pick_best(merged, extreme_mask, "极限覆盖版", "仅按最大覆盖选择")
        coverage30 = self.pick_best(merged, coverage30_mask, "30景点均衡覆盖候选版", "未找到30景点且2缓冲的可执行路线，取鲁棒效用最高者")
        robust_main = self.pick_best(merged, robust_main_mask, "鲁棒稳健主推版", "未找到24景点以上且成功率>=80%、红日<=2、严格时间窗可行的路线，取鲁棒效用最高者")
        family = self.pick_best_with_metric(
            merged,
            family_mask,
            "亲子舒适版",
            "family_utility",
            "未找到0红日的严格亲子路线，改按亲子效用函数在放宽红日约束后选择折中方案",
            exclude_route_ids=[str(robust_main.get("route_id", ""))],
            relaxed_mask=family_relaxed,
        )
        senior = self.pick_best_with_metric(
            merged,
            senior_mask,
            "长者慢游版",
            "senior_utility",
            "未找到0红日的严格长者路线，改按长者效用函数在放宽红日约束后选择折中方案",
            exclude_route_ids=[str(robust_main.get("route_id", "")), str(family.get("route_id", ""))],
            relaxed_mask=senior_relaxed,
        )
        selections = [extreme, coverage30, robust_main, family, senior]
        for row in selections:
            if row["selected_role"] == "极限覆盖版":
                if (not as_bool(row.get("schedule_strict_feasible", False))) or as_float(row.get("success_probability", 0.0), 0.0) < as_float(self.config["success_threshold"], 0.8):
                    row["selection_status"] = "coverage_upper_bound_only_not_feasible: 小时级排程/仿真不可作为现实执行方案"
            if row["selected_role"] == "30景点均衡覆盖候选版":
                if as_float(row.get("success_probability", 0.0), 0.0) < as_float(self.config["success_threshold"], 0.8) or as_int(row.get("red_days", 0), 0) > int(self.config["max_red_days_standard"]):
                    row["selection_status"] = (
                        "coverage_candidate_not_strict_robust: 满足30景点和2缓冲，"
                        "但未同时满足80%成功率或红日<=1，应作为覆盖候选而非最终稳健主推"
                    )
            if row["selected_role"] == "鲁棒稳健主推版" and row.get("selection_status") == "criteria_pass":
                row["selection_status"] = "criteria_pass_relaxed_coverage: 成功率>=80%、红日<=2、缓冲>=2；覆盖降至24+以换取稳健性"
        selected = pd.DataFrame(selections)
        selected = selected.drop_duplicates(subset=["selected_role"], keep="first")
        write_csv(selected, self.outputs / "q1_v3_selected_routes.csv")
        # Save merged route table with final robust metrics.
        write_csv(merged.sort_values("robust_utility", ascending=False), self.outputs / "q1_v3_candidate_routes_enriched.csv")
        self.robust_front = front
        self.selected_routes = selected
        return front, selected

    # ------------------------------------------------------------------
    # P8: Figures, reports, audit
    # ------------------------------------------------------------------

    def depot_metric_to_spot(self, sid: str, direction: str) -> float:
        row = self.depot_by_spot.get(sid)
        if row is None:
            return 0.0
        if direction == "to":
            time = as_float(row.get("depot_to_spot_time", 0.0), 0.0)
            cost = as_float(row.get("depot_to_spot_cost", 0.0), 0.0)
            risk = as_float(row.get("depot_to_spot_risk", 0.0), 0.0)
        else:
            time = as_float(row.get("spot_to_depot_time", 0.0), 0.0)
            cost = as_float(row.get("spot_to_depot_cost", 0.0), 0.0)
            risk = as_float(row.get("spot_to_depot_risk", 0.0), 0.0)
        return time + cost / 180.0 + risk * 15.0

    def sequence_metric_for_exact_check(self, seq: Sequence[str]) -> float:
        if not seq:
            return 0.0
        total = self.depot_metric_to_spot(seq[0], "to")
        total += sum(self.edge_metric(a, b, "balanced") for a, b in zip(seq[:-1], seq[1:]))
        total += self.depot_metric_to_spot(seq[-1], "from")
        return total

    def build_small_exact_check(self) -> pd.DataFrame:
        if self.selected_routes.empty:
            selected_path = self.outputs / "q1_v3_selected_routes.csv"
            if selected_path.exists():
                self.selected_routes = read_csv(selected_path)
        if not self.labels_by_pair:
            labels_path = self.outputs / "q1_v3_multimodal_labels.csv"
            if labels_path.exists():
                self.make_label_lookup(read_csv(labels_path))
        if self.selected_routes.empty:
            result = pd.DataFrame()
            write_csv(result, self.outputs / "q1_v3_small_exact_check.csv")
            return result
        target = self.selected_routes[self.selected_routes["selected_role"].astype(str).str.contains("鲁棒稳健主推")]
        row = target.iloc[0] if not target.empty else self.selected_routes.iloc[0]
        route_id = str(row["route_id"])
        seq = [sid.strip() for sid in str(row.get("spot_id_sequence", "")).split("->") if sid.strip()]
        if len(seq) < 4:
            result = pd.DataFrame()
            write_csv(result, self.outputs / "q1_v3_small_exact_check.csv")
            return result
        # Keep the verifier small enough for exact Held-Karp while preserving
        # the route's regional order signal.
        sample = seq[: min(15, len(seq))]
        n = len(sample)
        edge = np.zeros((n, n), dtype=float)
        for i, a in enumerate(sample):
            for j, b in enumerate(sample):
                edge[i, j] = 0.0 if i == j else self.edge_metric(a, b, "balanced")
        dp: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for i, sid in enumerate(sample):
            dp[(1 << i, i)] = (self.depot_metric_to_spot(sid, "to"), -1)
        for mask in range(1, 1 << n):
            for last in range(n):
                state = (mask, last)
                if state not in dp:
                    continue
                base_cost = dp[state][0]
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    nm = mask | (1 << nxt)
                    cand = base_cost + edge[last, nxt]
                    old = dp.get((nm, nxt))
                    if old is None or cand < old[0]:
                        dp[(nm, nxt)] = (cand, last)
        full = (1 << n) - 1
        best_last = min(range(n), key=lambda i: dp[(full, i)][0] + self.depot_metric_to_spot(sample[i], "from"))
        exact_metric = dp[(full, best_last)][0] + self.depot_metric_to_spot(sample[best_last], "from")
        order_idx = []
        mask = full
        last = best_last
        while last >= 0:
            order_idx.append(last)
            prev = dp[(mask, last)][1]
            mask ^= 1 << last
            last = prev
        order_idx.reverse()
        exact_seq = [sample[i] for i in order_idx]
        heuristic_metric = self.sequence_metric_for_exact_check(sample)
        gap = (heuristic_metric - exact_metric) / exact_metric if exact_metric > 0 else 0.0
        result = pd.DataFrame(
            [
                {
                    "check_id": "held_karp_15_node_ordering",
                    "selected_role": str(row["selected_role"]),
                    "route_id": route_id,
                    "node_count": n,
                    "heuristic_metric": round(heuristic_metric, 4),
                    "exact_metric": round(exact_metric, 4),
                    "relative_gap": round(gap, 4),
                    "heuristic_order": " -> ".join(sample),
                    "exact_order": " -> ".join(exact_seq),
                    "check_scope": "固定15节点子集的路径排序精确校验；不等同于完整鲁棒定向游全局最优证明",
                }
            ]
        )
        write_csv(result, self.outputs / "q1_v3_small_exact_check.csv")
        return result

    def build_figures(self) -> None:
        if not HAS_MPL:
            return
        enriched_path = self.outputs / "q1_v3_candidate_routes_enriched.csv"
        df = read_csv(enriched_path) if enriched_path.exists() else self.candidate_routes
        for col in [
            "spots_count",
            "total_cost_yuan_excluding_meals",
            "success_probability",
            "mean_comfort_score",
            "cvar75_loss",
            "public_transport_hours_share",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            df["total_cost_yuan_excluding_meals"],
            df["success_probability"],
            c=df["spots_count"],
            s=35 + df["spots_count"] * 1.5,
            cmap="viridis",
            alpha=0.75,
        )
        ax.set_xlabel("总费用（不含餐饮，元）")
        ax.set_ylabel("路线级仿真成功率")
        ax.set_title("Q1-V3 鲁棒 Pareto 候选：费用-成功率-覆盖")
        fig.colorbar(sc, ax=ax, label="景点数")
        fig.tight_layout()
        fig.savefig(self.figures / "fig_q1_v3_robust_pareto_front.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(df["cvar75_loss"], df["mean_comfort_score"], c=df["success_probability"], cmap="plasma", alpha=0.75)
        ax.set_xlabel("CVaR75 损失")
        ax.set_ylabel("平均舒适度")
        ax.set_title("Q1-V3 风险-舒适度权衡")
        fig.colorbar(sc, ax=ax, label="成功率")
        fig.tight_layout()
        fig.savefig(self.figures / "fig_q1_v3_risk_comfort_tradeoff.png", dpi=180)
        plt.close(fig)

        mode_hours = Counter()
        for mix in df.get("mode_hours_mix", pd.Series(dtype=str)).astype(str):
            for part in mix.split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    mode_hours[k] += as_float(v, 0.0)
        if mode_hours:
            fig, ax = plt.subplots(figsize=(8, 5))
            keys = list(mode_hours.keys())
            vals = [mode_hours[k] for k in keys]
            ax.bar(keys, vals, color="#4c78a8")
            ax.set_ylabel("候选路线累计小时")
            ax.set_title("Q1-V3 交通方式小时占比")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(self.figures / "fig_q1_v3_transport_mode_mix.png", dpi=180)
            plt.close(fig)

        if self.selected_routes.empty:
            selected_path = self.outputs / "q1_v3_selected_routes.csv"
            if selected_path.exists():
                self.selected_routes = read_csv(selected_path)
        if not self.selected_routes.empty and (self.outputs / "q1_v3_simulation_trials.csv").exists():
            selected_id = str(self.selected_routes.iloc[min(1, len(self.selected_routes) - 1)]["route_id"])
            trials = read_csv(self.outputs / "q1_v3_simulation_trials.csv")
            sub = trials[trials["route_id"].astype(str) == selected_id]
            if not sub.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(pd.to_numeric(sub["simulated_days"], errors="coerce").dropna(), bins=range(20, 37), color="#59a14f", alpha=0.85)
                ax.axvline(30, color="#e15759", linestyle="--", label="30天上限")
                ax.set_xlabel("仿真完成天数")
                ax.set_ylabel("样本数")
                ax.set_title(f"主推候选 {selected_id} 完成天数分布")
                ax.legend()
                fig.tight_layout()
                fig.savefig(self.figures / "fig_q1_v3_route_simulation_distribution.png", dpi=180)
                plt.close(fig)

    def md_table(self, df: pd.DataFrame, cols: Sequence[str], max_rows: int = 8) -> str:
        if df.empty:
            return "_无数据_"
        sub = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows).copy()
        headers = list(sub.columns)
        lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in sub.iterrows():
            vals = [str(row[c]).replace("\n", " ") for c in headers]
            lines.append("|" + "|".join(vals) + "|")
        return "\n".join(lines)

    def build_model_audit(self) -> pd.DataFrame:
        audit_rows = [
            {
                "audit_id": "V3-A1",
                "module": "multimodal_labels",
                "claim": "从multimodal_edges.csv生成景点间非支配交通标签",
                "status": "implemented",
                "evidence_file": "outputs/q1_v3_multimodal_labels.csv",
                "remaining_limitation": "班次时刻仍来自种子/模型补全，未实时接入12306或航班库存",
                "next_upgrade": "接入铁路/航班真实时刻与票价余量",
            },
            {
                "audit_id": "V3-A2",
                "module": "robust_optimization",
                "claim": "每个覆盖下界重新搜索路线，交通标签进入主评分",
                "status": "implemented_as_matheuristic",
                "evidence_file": "outputs/q1_v3_candidate_routes.csv",
                "remaining_limitation": "ALNS/贪心/随机重启不证明全局最优",
                "next_upgrade": "小规模用MILP/Gurobi校验启发式最优性缺口",
            },
            {
                "audit_id": "V3-A3",
                "module": "chance_constraint",
                "claim": "路线成功率阈值用于主推方案筛选",
                "status": "implemented_as_filter",
                "evidence_file": "outputs/q1_v3_selected_routes.csv",
                "remaining_limitation": "机会约束不是在单一精确随机规划中硬加入",
                "next_upgrade": "用SAA二阶段模型或CP-SAT回调把失败样本反切回搜索",
            },
            {
                "audit_id": "V3-A4",
                "module": "hourly_schedule",
                "claim": "已实现小时级开放时间、午休、高温避让、长转场恢复和晚到检查",
                "status": "implemented_heuristic",
                "evidence_file": "outputs/q1_v3_hourly_itinerary.csv",
                "remaining_limitation": "未安装OR-Tools，当前为启发式排程修复而非CP-SAT证明可行",
                "next_upgrade": "安装OR-Tools后将日内活动块转为optional interval模型",
            },
            {
                "audit_id": "V3-A5",
                "module": "route_simulation",
                "claim": "对每条候选路线进行route-specific Monte Carlo仿真并计算CVaR",
                "status": "implemented",
                "evidence_file": "outputs/q1_v3_simulation_summary.csv",
                "remaining_limitation": "容量/酒店部分字段含模拟补全，仿真结果应解释为压力测试",
                "next_upgrade": "接入景区预约余量和酒店真实房态",
            },
            {
                "audit_id": "V3-A6",
                "module": "global_optimality",
                "claim": "不声称全局最优",
                "status": "explicitly_not_claimed",
                "evidence_file": "reports/新疆旅游第一问Q1_V3鲁棒联合优化报告.md",
                "remaining_limitation": "启发式质量依赖候选生成多样性",
                "next_upgrade": "增加并行多启动和精确下界对照",
            },
            {
                "audit_id": "V3-A7",
                "module": "small_exact_check",
                "claim": "已增加15节点固定子集路径排序精确校验",
                "status": "implemented_scope_limited",
                "evidence_file": "outputs/q1_v3_small_exact_check.csv",
                "remaining_limitation": "仅校验排序子问题，不证明完整景点选择+随机仿真模型全局最优",
                "next_upgrade": "安装OR-Tools/Gurobi后构建小规模MILP/CP-SAT定向游精确模型",
            },
        ]
        audit = pd.DataFrame(audit_rows)
        write_csv(audit, self.outputs / "q1_v3_model_audit.csv")
        return audit

    def build_report(self) -> None:
        selected = self.selected_routes if not self.selected_routes.empty else read_csv(self.outputs / "q1_v3_selected_routes.csv")
        front = self.robust_front if not self.robust_front.empty else read_csv(self.outputs / "q1_v3_robust_pareto_front.csv")
        feasible_front_path = self.outputs / "q1_v3_feasible_robust_pareto_front.csv"
        feasible_front = read_csv(feasible_front_path) if feasible_front_path.exists() else pd.DataFrame()
        exact_check_path = self.outputs / "q1_v3_small_exact_check.csv"
        exact_check = read_csv(exact_check_path) if exact_check_path.exists() else pd.DataFrame()
        labels = self.transport_labels if not self.transport_labels.empty else read_csv(self.outputs / "q1_v3_multimodal_labels.csv")
        sim = self.simulation_summary if not self.simulation_summary.empty else read_csv(self.outputs / "q1_v3_simulation_summary.csv")
        audit = self.build_model_audit()
        label_pairs = labels.groupby(["from_spot_id", "to_spot_id"]).size()
        avg_labels = float(label_pairs.mean()) if len(label_pairs) else 0.0
        max_labels = int(label_pairs.max()) if len(label_pairs) else 0
        feasible_labels = float(labels["ordinary_tourist_feasible"].map(as_bool).mean()) if "ordinary_tourist_feasible" in labels.columns else 0.0
        report = f"""# 新疆旅游第一问 Q1-V3 鲁棒联合优化报告

## 1. 本版定位

Q1-V3 将 V2.1 中仍属于后验解释的部分推进到求解流程内：从多模式边图生成直接道路标签与“本地接驳-公共主边-本地接驳”模板标签，并进行非支配筛选；随后在每个覆盖下界下重新搜索路线，构造小时级排程，并对每条候选路线做 route-specific Monte Carlo 仿真。当前实现是 matheuristic 高质量可行解框架，不声称全局最优。

## 2. 数据与派生结构

- 景点基础表：40 个景点，其中普通游客基准路线排除楼兰古城、尼雅遗址等受限点。
- 多模式边图：{len(self.edges)} 条边，包含 self_drive、rail、air、coach、transfer、scenic_shuttle 等。
- V3 交通标签：{len(labels)} 条标签，平均每个 OD 保留 {avg_labels:.2f} 个，最多 {max_labels} 个；普通游客可行标签占比 {feasible_labels:.2%}。
- 情景样本：{int(self.config["scenario_samples"])} 个，覆盖暑期普通日、高峰、天气扰动、酒店紧张和价格上浮。

## 3. 数学模型口径

主问题可表述为鲁棒多目标多模式定向游问题。给定覆盖下界 q，模型在候选景点集合 V 中选择 y_i，在交通标签集合 L_ij 中选择 x_ijell，并最大化覆盖、区域、偏好与主题多样性价值，同时惩罚费用、疲劳、风险代理和 CVaR 代理。

机会约束在本实现中采用“仿真后筛选”口径：

```text
P(T(R,w)<=30, Comfort(R,w)>=75, ReservationFail<=2) >= 0.8
```

它用于主推路线筛选，不被包装成精确随机 MILP 硬约束。

## 4. 求解流程

1. 模板标签生成：基于多模式边图生成直接道路标签与“本地接驳-公共主边-本地接驳”模板标签，并做非支配筛选。
2. 覆盖重搜索：对 q in {self.config["coverage_grid"]} 分别生成候选，不再只从 32 点路线嵌套删点。
3. 标签选择：路线边直接选择 label_id，费用、时间、疲劳和风险来自交通标签。
4. 小时排程：显式记录出发/到达、开放时间、午休、高温避让、长转场和晚到酒店。
5. 路线仿真：每条候选路线使用 {int(self.config["scenario_samples"])} 个样本计算成功率、P95天数、CVaR75/90。
6. Pareto 筛选：按覆盖、费用、舒适度、成功率、CVaR 做非支配过滤。

## 5. 代表方案

{self.md_table(selected, ["selected_role", "route_id", "spots_count", "buffer_days", "success_probability", "cvar75_loss", "mean_comfort_score", "time_window_violations", "schedule_strict_feasible", "selection_status"], 8)}

## 6. 严格可行鲁棒 Pareto 前沿样例

本表只保留 `success_probability >= 0.8` 且 `schedule_strict_feasible=True` 的候选；完整数学前沿仍输出到 `q1_v3_robust_pareto_front.csv` 作为覆盖-风险权衡审计。

{self.md_table(feasible_front, ["route_id", "spots_count", "buffer_days", "success_probability", "cvar75_loss", "mean_comfort_score", "time_window_violations", "total_cost_yuan_excluding_meals"], 12)}

## 7. 仿真结果摘要

{self.md_table(sim.sort_values("success_probability", ascending=False), ["route_id", "success_probability", "overrun_probability", "p95_days", "expected_loss", "cvar75_loss", "prob_red_days_gt1"], 12)}

## 8. 小规模精确校验

当前环境未安装 OR-Tools/Gurobi，因此 V3 先增加固定 15 节点子集的 Held-Karp 精确排序校验。该校验只用于衡量路线排序子问题的启发式缺口，不等同于完整随机定向游模型的全局最优证明。

{self.md_table(exact_check, ["check_id", "route_id", "node_count", "heuristic_metric", "exact_metric", "relative_gap", "check_scope"], 5)}

## 9. 楼兰特殊准入处理

楼兰古城在特殊准入表中标记为普通游客不可作为基准路线节点，因此 V3 继续执行 `y_楼兰古城=0`。题面中对楼兰文化的偏好通过“楼兰文化替代组”约束表达，至少选择 2 个丝路遗址、宗教城市文化或南疆/巴州文化补偿节点，例如交河故城、高昌故城、北庭故城、克孜尔石窟、喀什古城、艾提尕尔清真寺、库车王府等。

## 10. 模型审计

{self.md_table(audit, ["audit_id", "module", "claim", "status", "remaining_limitation"], 10)}

## 11. 结论

Q1-V3 已经从“候选路线族 + 后验评价”升级为“交通标签选择 + 覆盖重搜索 + 小时级排程 + 路线级仿真 + 鲁棒前沿筛选”。汇报时应强调：该方案给出的是可复现、约束透明、风险可解释的高质量鲁棒可行解；若要进一步追求严格最优性，应在小规模子问题上引入 MILP/Gurobi/CP-SAT 作为下界与校验。
"""
        report_path = self.reports / "新疆旅游第一问Q1_V3鲁棒联合优化报告.md"
        report_path.write_text(report, encoding="utf-8")

        readme = f"""# Q1-V3 鲁棒联合优化

本目录实现第一问 V3 强化版：多模式交通标签、覆盖重搜索、小时级排程、路线级仿真、鲁棒 Pareto 筛选。

## 一键运行

```powershell
python .\\scripts\\q1_v3_build_all.py
```

## 关键输出

- `outputs/q1_v3_multimodal_labels.csv`
- `outputs/q1_v3_candidate_routes.csv`
- `outputs/q1_v3_hourly_itinerary.csv`
- `outputs/q1_v3_simulation_summary.csv`
- `outputs/q1_v3_robust_pareto_front.csv`
- `outputs/q1_v3_feasible_robust_pareto_front.csv`
- `outputs/q1_v3_selected_routes.csv`
- `outputs/q1_v3_small_exact_check.csv`
- `reports/新疆旅游第一问Q1_V3鲁棒联合优化报告.md`

## 建模边界

当前版本是 matheuristic 高质量可行解框架，不声称全局最优；机会约束通过路线级仿真筛选实现，容量与酒店房量中仍包含模拟补全字段。
"""
        (self.root / "README.md").write_text(readme, encoding="utf-8")

        workbook_path = self.reports / "新疆旅游第一问Q1_V3鲁棒联合优化结果.xlsx"
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            selected.to_excel(writer, sheet_name="selected_routes", index=False)
            front.to_excel(writer, sheet_name="robust_pareto", index=False)
            feasible_front.to_excel(writer, sheet_name="feasible_robust_pareto", index=False)
            read_csv(self.outputs / "q1_v3_candidate_routes_enriched.csv").to_excel(writer, sheet_name="candidate_routes", index=False)
            sim.to_excel(writer, sheet_name="simulation_summary", index=False)
            read_csv(self.outputs / "q1_v3_schedule_summary.csv").to_excel(writer, sheet_name="schedule_summary", index=False)
            read_csv(self.outputs / "q1_v3_preference_groups.csv").to_excel(writer, sheet_name="preference_groups", index=False)
            exact_check.to_excel(writer, sheet_name="small_exact_check", index=False)
            audit.to_excel(writer, sheet_name="model_audit", index=False)

    def build_solve_summary(self) -> dict:
        labels = read_csv(self.outputs / "q1_v3_multimodal_labels.csv")
        candidates = read_csv(self.outputs / "q1_v3_candidate_routes.csv")
        selected = read_csv(self.outputs / "q1_v3_selected_routes.csv")
        sim = read_csv(self.outputs / "q1_v3_simulation_summary.csv")
        exact_path = self.outputs / "q1_v3_small_exact_check.csv"
        exact = read_csv(exact_path) if exact_path.exists() else pd.DataFrame()
        summary = {
            "version": "Q1-V3 Robust Multi-objective Multimodal Orienteering with Simulation-based Scheduling",
            "package_root": str(PKG_ROOT),
            "v3_root": str(self.root),
            "coverage_grid": self.config["coverage_grid"],
            "scenario_samples": int(self.config["scenario_samples"]),
            "transport_labels": int(len(labels)),
            "candidate_routes": int(len(candidates)),
            "simulation_routes": int(len(sim)),
            "pareto_front_routes": int(len(read_csv(self.outputs / "q1_v3_robust_pareto_front.csv"))),
            "feasible_robust_pareto_front_routes": int(len(read_csv(self.outputs / "q1_v3_feasible_robust_pareto_front.csv"))),
            "small_exact_checks": int(len(exact)),
            "selected_route_ids": selected[["selected_role", "route_id", "selection_status"]].to_dict(orient="records"),
            "success_threshold": as_float(self.config["success_threshold"], 0.8),
            "global_optimality_claimed": False,
            "solver_type": "matheuristic: multimodal template labels + non-dominated filtering + coverage-grid search + heuristic hourly scheduler + Monte Carlo simulation + Pareto filtering",
        }
        path = self.outputs / "solve_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def update_manifest(self) -> None:
        rows = []
        for path in sorted(PKG_ROOT.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                rel = path.relative_to(PKG_ROOT)
                rows.append(
                    {
                        "relative_path": str(rel).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "modified_time": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
                    }
                )
        write_csv(pd.DataFrame(rows), PKG_ROOT / "Q1_file_manifest.csv")
        summary = pd.DataFrame(
            [
                {"metric": "manifest_count", "value": len(rows)},
                {"metric": "v3_output_files", "value": sum(1 for p in (self.outputs).glob("*") if p.is_file())},
                {"metric": "v3_report_files", "value": sum(1 for p in (self.reports).glob("*") if p.is_file())},
            ]
        )
        write_csv(summary, PKG_ROOT / "Q1_directory_summary.csv")

    def build_all(self) -> None:
        print("[P1] building multimodal labels...")
        self.build_transport_labels()
        print("[P2] building preference/diversity tables...")
        self.build_preference_and_diversity_tables()
        print("[P3] generating scenario samples...")
        self.generate_scenarios()
        print("[P4] running coverage-grid route optimizer...")
        self.run_master_optimizer()
        print("[P5] building hourly itineraries...")
        self.run_hourly_scheduler()
        print("[P6] running route-specific simulation...")
        self.run_route_simulator()
        print("[P7] selecting robust Pareto front...")
        self.select_robust_pareto_front()
        print("[P8] running small exact ordering check...")
        self.build_small_exact_check()
        print("[P9] building figures, report, workbook, audit...")
        self.build_figures()
        self.build_report()
        summary = self.build_solve_summary()
        self.update_manifest()
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    builder = Q1V3Builder()
    builder.build_all()


if __name__ == "__main__":
    main()
