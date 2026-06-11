# -*- coding: utf-8 -*-
"""
Q1-V2 builder for the first-question package.

The script upgrades the existing single hard-30-day route into a reproducible
route-family experiment:
1. transport labels v2;
2. epsilon-constraint candidate grid;
3. humanized route-family summaries and daily schedules;
4. robustness/CVaR selection tables;
5. figures, workbook and markdown report.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
V2_ROOT = SCRIPT_PATH.parents[1]
PKG_ROOT = SCRIPT_PATH.parents[2]
OUT_DIR = V2_ROOT / "outputs"
FIG_DIR = V2_ROOT / "figures"
REPORT_DIR = V2_ROOT / "reports"
for directory in (OUT_DIR, FIG_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_sequence(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    return [part.strip() for part in str(text).split("->") if part.strip()]


def percentile_or_zero(values: Iterable[float], q: float) -> float:
    values = [float(v) for v in values if not pd.isna(v)]
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def cvar75(values: Iterable[float]) -> float:
    values = sorted(float(v) for v in values if not pd.isna(v))
    if not values:
        return 0.0
    threshold = np.percentile(values, 75)
    tail = [v for v in values if v >= threshold]
    return float(np.mean(tail)) if tail else float(threshold)


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    route_type: str
    persona_id: str
    persona_name: str
    sequence: list[str]
    active_days_target: int | None
    day_active_limit: float
    intended_buffer_days: int
    recommendation_role: str
    model_note: str
    policy_persona_id: str | None = None
    policy_id: str | None = None


class Q1V2Builder:
    def __init__(self) -> None:
        self.spots = read_csv(PKG_ROOT / "01_输入数据/model_data/spot_clean.csv")
        self.enhanced_od = read_csv(PKG_ROOT / "01_输入数据/enhanced_data/enhanced_od_matrix.csv")
        self.depot_access = read_csv(PKG_ROOT / "01_输入数据/enhanced_data/depot_access_matrix.csv")
        self.hotel_default = read_csv(PKG_ROOT / "01_输入数据/model_data/hotel_place_default.csv")
        self.hard_summary = read_csv(PKG_ROOT / "03_第一问主结果/hybrid_30day_outputs/hard30_route_summary.csv")
        self.hard_days = read_csv(PKG_ROOT / "03_第一问主结果/hybrid_30day_outputs/hard30_route_days.csv")
        self.hard_trials = read_csv(PKG_ROOT / "03_第一问主结果/hybrid_30day_outputs/hard30_monte_carlo_trials.csv")
        self.hard_mc_summary = read_csv(PKG_ROOT / "03_第一问主结果/hybrid_30day_outputs/hard30_monte_carlo_summary.csv")
        self.variant_summary = read_csv(PKG_ROOT / "05_画像与鲁棒性/adaptive_strategy_outputs/q1_variant_summary.csv")
        self.variant_days = read_csv(PKG_ROOT / "05_画像与鲁棒性/adaptive_strategy_outputs/q1_variant_days.csv")
        self.dropped_spots = read_csv(PKG_ROOT / "05_画像与鲁棒性/adaptive_strategy_outputs/q1_variant_dropped_spots.csv")
        self.robustness = read_csv(PKG_ROOT / "05_画像与鲁棒性/digital_twin_outputs/q1_persona_robustness.csv")
        self.policy_selection = read_csv(PKG_ROOT / "05_画像与鲁棒性/risk_policy_outputs/q1_policy_selection.csv")
        self.policy_candidates = read_csv(PKG_ROOT / "05_画像与鲁棒性/risk_policy_outputs/q1_policy_candidates.csv")

        self.spot_by_name = self.spots.set_index("spot_name").to_dict("index")
        self.spot_id_by_name = self.spots.set_index("spot_name")["spot_id"].to_dict()
        self.spot_name_by_id = self.spots.set_index("spot_id")["spot_name"].to_dict()
        base_od = self.enhanced_od[self.enhanced_od["scenario_id"].eq("base_summer")].copy()
        self.od_by_name = {
            (row.from_spot_name, row.to_spot_name): row
            for row in base_od.itertuples(index=False)
        }
        self.depot_by_name = self.depot_access.set_index("spot_name").to_dict("index")
        hotel_default = self.hotel_default.copy()
        hotel_default["high_season_room_yuan_per_night"] = hotel_default["high_season_room_yuan_per_night"].map(to_float)
        self.hotel_by_hub = (
            hotel_default.groupby("hub_name", dropna=False)["high_season_room_yuan_per_night"].median().to_dict()
        )

        hard_row = self.hard_summary.iloc[0]
        self.base_sequence = parse_sequence(hard_row["route_sequence"])

    def spot_info(self, name: str) -> dict[str, Any]:
        if name not in self.spot_by_name:
            raise KeyError(f"Unknown spot name: {name}")
        return self.spot_by_name[name]

    def service_hours(self, name: str) -> float:
        return to_float(self.spot_info(name).get("visit_hours_mid"), 2.5)

    def ticket_cost_two(self, name: str) -> float:
        return 2.0 * to_float(self.spot_info(name).get("ticket_high_total_yuan_per_person"), 0.0)

    def priority_score(self, name: str) -> float:
        return to_float(self.spot_info(name).get("priority_score_for_op"), 1.0)

    def coverage_value(self, names: list[str]) -> float:
        value = 0.0
        for name in names:
            info = self.spot_info(name)
            base = self.priority_score(name)
            topic_bonus = 1.2 if boolish(info.get("is_topic_preference")) else 0.0
            culture_bonus = 0.35 if boolish(info.get("is_cultural")) else 0.0
            nature_bonus = 0.35 if boolish(info.get("is_natural")) else 0.0
            remote_penalty = 0.25 if boolish(info.get("high_altitude_or_remote")) else 0.0
            value += base + topic_bonus + culture_bonus + nature_bonus - remote_penalty
        return round(value, 2)

    def hub_name(self, name: str) -> str:
        return str(self.spot_info(name).get("hub_name", ""))

    def region_name(self, name: str) -> str:
        return str(self.spot_info(name).get("region_cluster", "未分区"))

    def hotel_price(self, hub_name: str | None) -> float:
        if hub_name and hub_name in self.hotel_by_hub:
            return to_float(self.hotel_by_hub[hub_name], 220.0)
        return float(self.hotel_default["high_season_room_yuan_per_night"].median())

    def od_edge(self, from_name: str | None, to_name: str, final_return: bool = False) -> dict[str, Any]:
        if final_return:
            access = self.depot_by_name.get(from_name or "", {})
            return {
                "time": to_float(access.get("spot_to_depot_time"), 0.0),
                "cost": to_float(access.get("spot_to_depot_cost"), 0.0),
                "risk": to_float(access.get("spot_to_depot_risk"), 0.0),
                "mode": "return_to_urumqi",
            }
        if from_name is None:
            access = self.depot_by_name.get(to_name, {})
            return {
                "time": to_float(access.get("depot_to_spot_time"), 0.0),
                "cost": to_float(access.get("depot_to_spot_cost"), 0.0),
                "risk": to_float(access.get("depot_to_spot_risk"), 0.0),
                "mode": "depot_to_spot",
            }
        row = self.od_by_name.get((from_name, to_name))
        if row is None:
            return {"time": 4.0, "cost": 240.0, "risk": 0.18, "mode": "missing_od_fallback"}
        return {
            "time": to_float(getattr(row, "shortest_time_hours")),
            "cost": to_float(getattr(row, "shortest_cost_yuan_per_two")),
            "risk": to_float(getattr(row, "path_risk")),
            "mode": str(getattr(row, "path_modes", "")),
        }

    def build_transport_labels_v2(self) -> pd.DataFrame:
        def dominant_mode(path_modes: str, time_hours: float, cost: float) -> str:
            text = str(path_modes or "").lower()
            if not text.strip():
                return "same_spot"
            if "air" in text or "flight" in text:
                return "air"
            if "rail" in text or "train" in text:
                return "rail"
            if "scenic" in text or "shuttle" in text:
                return "scenic_shuttle"
            if "coach" in text or "bus" in text:
                return "coach"
            if "self_drive" in text:
                if time_hours >= 7.0:
                    return "charter_car"
                if time_hours >= 3.5:
                    return "rental_car"
                if cost <= 90:
                    return "taxi_transfer"
                return "rental_car"
            if "transfer" in text:
                return "taxi_transfer"
            return text.split("+")[0].split("->")[0]

        fatigue_coef = {
            "same_spot": 0.0,
            "air": 0.65,
            "rail": 0.58,
            "coach": 1.05,
            "tourist_bus": 0.95,
            "rental_car": 1.12,
            "charter_car": 1.02,
            "carpool": 1.12,
            "taxi_transfer": 0.80,
            "scenic_shuttle": 0.90,
        }
        mode_risk_add = {
            "air": 0.015,
            "rail": 0.005,
            "coach": 0.025,
            "rental_car": 0.020,
            "charter_car": 0.015,
            "taxi_transfer": 0.010,
            "scenic_shuttle": 0.015,
        }

        labels = self.enhanced_od.copy()
        labels["time_hours"] = labels["shortest_time_hours"].astype(float)
        labels["cost_yuan_for_two"] = labels["shortest_cost_yuan_per_two"].astype(float)
        labels["dominant_mode"] = [
            dominant_mode(m, t, c)
            for m, t, c in zip(labels["path_modes"], labels["time_hours"], labels["cost_yuan_for_two"])
        ]
        labels["mode_combo"] = labels["path_modes"].fillna("").replace("", "same_spot")
        labels["is_night_transport"] = (
            labels["dominant_mode"].isin(["rail", "coach"]) & labels["time_hours"].ge(8.0)
        )
        labels["requires_transfer"] = labels["mode_combo"].str.contains(r"transfer|\+|->|/", case=False, regex=True)
        labels["fatigue_score"] = [
            round(
                t * fatigue_coef.get(m, 1.0)
                + max(0.0, t - 4.0) ** 2 * 0.18
                + (0.8 if tr else 0.0)
                + (1.2 if night else 0.0),
                3,
            )
            for t, m, tr, night in zip(
                labels["time_hours"], labels["dominant_mode"], labels["requires_transfer"], labels["is_night_transport"]
            )
        ]
        labels["risk_score"] = [
            round(min(0.95, to_float(base) + mode_risk_add.get(mode, 0.02) + max(0.0, t - 6.0) * 0.015), 3)
            for base, mode, t in zip(labels["path_risk"], labels["dominant_mode"], labels["time_hours"])
        ]
        labels["generalized_cost_score"] = (
            labels["cost_yuan_for_two"] + labels["time_hours"] * 35 + labels["fatigue_score"] * 45 + labels["risk_score"] * 600
        ).round(3)
        labels["label_rank"] = labels.groupby(["from_spot_id", "to_spot_id"])["generalized_cost_score"].rank(
            method="first", ascending=True
        ).astype(int)
        labels["label_quality_note"] = np.where(
            labels["dominant_mode"].isin(["rail", "air"]),
            "public_transport_candidate_needs_schedule_confirmation",
            np.where(labels["dominant_mode"].eq("same_spot"), "diagonal_zero_edge", "amap_or_model_road_edge"),
        )
        keep = [
            "from_spot_id",
            "from_spot_name",
            "to_spot_id",
            "to_spot_name",
            "scenario_id",
            "mode_combo",
            "dominant_mode",
            "time_hours",
            "cost_yuan_for_two",
            "fatigue_score",
            "risk_score",
            "is_night_transport",
            "requires_transfer",
            "label_rank",
            "generalized_cost_score",
            "label_quality_note",
        ]
        out = labels[keep].copy()
        out.to_csv(OUT_DIR / "enhanced_od_labels_v2.csv", index=False, encoding="utf-8-sig")
        return out

    def drop_order(self) -> list[str]:
        ordered_ids: list[str] = []
        if not self.dropped_spots.empty:
            df = self.dropped_spots.copy()
            df["drop_order"] = df["drop_order"].astype(int)
            df = df.sort_values(["drop_order", "priority_score"])
            ordered_ids.extend(df["spot_id"].dropna().astype(str).tolist())

        rows = self.spots.copy()
        rows["priority_num"] = rows["priority_score_for_op"].map(to_float)
        rows["topic_bool"] = rows["is_topic_preference"].map(boolish)
        rows["remote_bool"] = rows["high_altitude_or_remote"].map(boolish)
        rows["drop_sort"] = (
            rows["topic_bool"].astype(int) * 100
            + rows["priority_num"] * 10
            - rows["remote_bool"].astype(int) * 2
        )
        for spot_id in rows.sort_values("drop_sort")["spot_id"].astype(str):
            if spot_id not in ordered_ids:
                ordered_ids.append(spot_id)
        return ordered_ids

    def sequence_for_target(self, q: int, protect_topic_until: int = 26) -> tuple[list[str], list[str]]:
        sequence = list(self.base_sequence)
        drop_ids = self.drop_order()
        dropped: list[str] = []
        for spot_id in drop_ids:
            if len(sequence) <= q:
                break
            name = self.spot_name_by_id.get(spot_id)
            if not name or name not in sequence:
                continue
            info = self.spot_info(name)
            if len(sequence) > protect_topic_until and boolish(info.get("is_topic_preference")):
                continue
            sequence.remove(name)
            dropped.append(name)
        if len(sequence) > q:
            for name in list(sequence):
                if len(sequence) <= q:
                    break
                if name in dropped:
                    continue
                sequence.remove(name)
                dropped.append(name)
        return sequence, dropped

    def segment_metrics(self, sequence: list[str], i: int, j: int) -> dict[str, Any]:
        day_names = sequence[i : j + 1]
        edge_sources: list[str] = []
        travel_hours = 0.0
        travel_cost = 0.0
        risk_values: list[float] = []
        prev = sequence[i - 1] if i > 0 else None
        for name in day_names:
            edge = self.od_edge(prev, name)
            travel_hours += edge["time"]
            travel_cost += edge["cost"]
            risk_values.append(edge["risk"])
            edge_sources.append(edge["mode"])
            prev = name
        if j == len(sequence) - 1:
            edge = self.od_edge(sequence[j], "乌鲁木齐市", final_return=True)
            travel_hours += edge["time"]
            travel_cost += edge["cost"]
            risk_values.append(edge["risk"])
            edge_sources.append(edge["mode"])

        service = sum(self.service_hours(name) for name in day_names)
        ticket = sum(self.ticket_cost_two(name) for name in day_names)
        active = travel_hours + service
        hot_exposure = 0.0
        for name in day_names:
            info = self.spot_info(name)
            if boolish(info.get("is_natural")) or name in {"火焰山", "葡萄沟", "吐峪沟麻扎村"}:
                hot_exposure += self.service_hours(name) * 0.45
        remote_count = sum(1 for name in day_names if boolish(self.spot_info(name).get("high_altitude_or_remote")))
        return {
            "visit_spots_count": len(day_names),
            "visit_spots": " -> ".join(day_names),
            "day_travel_hours": round(travel_hours, 3),
            "day_service_hours": round(service, 3),
            "day_active_hours": round(active, 3),
            "transport_cost_yuan_for_two": round(travel_cost, 2),
            "ticket_cost_yuan_for_two": round(ticket, 2),
            "risk_score": round(float(np.mean(risk_values)) if risk_values else 0.0, 3),
            "mode_combo": " + ".join(sorted(set(str(x) for x in edge_sources if str(x).strip()))),
            "high_temp_exposure_hours": round(hot_exposure, 3),
            "remote_or_altitude_spots": remote_count,
            "lodging_hub": self.hub_name(day_names[-1]) if day_names else "乌鲁木齐市",
        }

    def partition_sequence(self, sequence: list[str], k_days: int, day_limit: float) -> list[dict[str, Any]]:
        n = len(sequence)
        k_days = max(1, min(k_days, n))
        segment_cache: dict[tuple[int, int], dict[str, Any]] = {}
        for i in range(n):
            for j in range(i, n):
                segment_cache[(i, j)] = self.segment_metrics(sequence, i, j)

        total_active = sum(self.service_hours(name) for name in sequence)
        for idx, name in enumerate(sequence):
            prev = sequence[idx - 1] if idx > 0 else None
            total_active += self.od_edge(prev, name)["time"]
        total_active += self.od_edge(sequence[-1], "乌鲁木齐市", final_return=True)["time"]
        target_daily = min(day_limit * 0.88, total_active / k_days)

        inf = 10**18
        dp = np.full((n + 1, k_days + 1), inf)
        parent: dict[tuple[int, int], int] = {}
        dp[0, 0] = 0.0
        for used in range(1, k_days + 1):
            for end in range(used, n + 1):
                best_score = inf
                best_start = -1
                for start in range(used - 1, end):
                    seg = segment_cache[(start, end - 1)]
                    active = seg["day_active_hours"]
                    travel = seg["day_travel_hours"]
                    count = seg["visit_spots_count"]
                    overload = max(0.0, active - day_limit)
                    long_transfer = max(0.0, travel - 6.0)
                    count_overload = max(0, count - 3)
                    penalty = (
                        (active - target_daily) ** 2 * 1.3
                        + overload**2 * 120
                        + long_transfer**2 * 55
                        + count_overload**2 * 20
                        + seg["risk_score"] * 12
                    )
                    score = dp[start, used - 1] + penalty
                    if score < best_score:
                        best_score = score
                        best_start = start
                dp[end, used] = best_score
                parent[(end, used)] = best_start

        parts: list[tuple[int, int]] = []
        end = n
        used = k_days
        while used > 0:
            start = parent[(end, used)]
            parts.append((start, end - 1))
            end = start
            used -= 1
        parts.reverse()
        return [dict(segment_cache[pair]) for pair in parts]

    def comfort_and_stress(self, row: dict[str, Any], day_limit: float, persona: str) -> tuple[float, float, str, str]:
        active = to_float(row.get("day_active_hours"))
        travel = to_float(row.get("day_travel_hours"))
        risk = to_float(row.get("risk_score"))
        hot = to_float(row.get("high_temp_exposure_hours"))
        remote = to_float(row.get("remote_or_altitude_spots"))

        persona_weight = {
            "balanced_robust": 1.00,
            "family_comfort": 1.18,
            "senior_slow": 1.35,
            "extreme_coverage": 0.92,
            "epsilon": 1.00,
        }.get(persona, 1.0)
        fatigue = (
            max(0.0, active - 6.0) * 8.5
            + max(0.0, travel - 3.5) * 7.0
            + max(0.0, active - day_limit) * 16.0
            + hot * 2.2
            + remote * 4.5
            + risk * 35.0
        ) * persona_weight
        comfort = max(35.0, min(100.0, 100.0 - fatigue))
        if comfort < 72 or active > day_limit + 0.8 or travel > 6.0:
            stress = "red"
        elif comfort < 84 or active > day_limit * 0.92 or travel > 4.8:
            stress = "yellow"
        else:
            stress = "green"

        notes: list[str] = []
        if travel > 5.0:
            notes.append("长转场后降低游览密度")
        if hot > 2.0:
            notes.append("中午高温时段安排室内/休息")
        if remote > 0:
            notes.append("提前核验边防/高海拔适应")
        if not notes:
            notes.append("常规执行")
        return round(fatigue, 2), round(comfort, 2), stress, "；".join(notes)

    def schedule_from_sequence(self, spec: RouteSpec) -> pd.DataFrame:
        target_days = spec.active_days_target or min(len(spec.sequence), 30 - spec.intended_buffer_days)
        scheduled = self.partition_sequence(spec.sequence, target_days, spec.day_active_limit)
        rows: list[dict[str, Any]] = []
        for day, metrics in enumerate(scheduled, start=1):
            fatigue, comfort, stress, notes = self.comfort_and_stress(metrics, spec.day_active_limit, spec.persona_id)
            metrics.update(
                {
                    "route_id": spec.route_id,
                    "route_type": spec.route_type,
                    "persona_id": spec.persona_id,
                    "day": day,
                    "calendar_role": "active",
                    "fatigue_index": fatigue,
                    "comfort_score": comfort,
                    "stress_level": stress,
                    "humanized_note": notes,
                    "hotel_cost_yuan": self.hotel_price(metrics.get("lodging_hub")) if day < 30 else 0.0,
                }
            )
            rows.append(metrics)

        planned_days = min(30, len(rows) + spec.intended_buffer_days)
        for day in range(len(rows) + 1, planned_days + 1):
            rows.append(
                {
                    "route_id": spec.route_id,
                    "route_type": spec.route_type,
                    "persona_id": spec.persona_id,
                    "day": day,
                    "calendar_role": "buffer",
                    "visit_spots_count": 0,
                    "visit_spots": "机动缓冲/预约失败补救/天气错峰",
                    "day_travel_hours": 0.0,
                    "day_service_hours": 0.0,
                    "day_active_hours": 0.0,
                    "transport_cost_yuan_for_two": 0.0,
                    "ticket_cost_yuan_for_two": 0.0,
                    "risk_score": 0.0,
                    "mode_combo": "buffer_day",
                    "high_temp_exposure_hours": 0.0,
                    "remote_or_altitude_spots": 0,
                    "lodging_hub": "按实际所在城市顺延",
                    "fatigue_index": 0.0,
                    "comfort_score": 100.0,
                    "stress_level": "green",
                    "humanized_note": "用于道路延误、景区预约失败、酒店满房或高温错峰",
                    "hotel_cost_yuan": float(self.hotel_default["high_season_room_yuan_per_night"].median()) if day < planned_days else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def hard_extreme_schedule(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in self.hard_days.to_dict("records"):
            metrics = {
                "route_id": "Q1V2_EXTREME_32_HARD30",
                "route_type": "极限覆盖版",
                "persona_id": "extreme_coverage",
                "day": int(row["day"]),
                "calendar_role": "active",
                "visit_spots_count": int(row["visit_spots_count"]),
                "visit_spots": row["visit_spots"],
                "day_travel_hours": to_float(row["day_travel_hours"]),
                "day_service_hours": to_float(row["day_service_hours"]),
                "day_active_hours": to_float(row["day_active_hours"]),
                "transport_cost_yuan_for_two": to_float(row["transport_cost_yuan_per_two"]),
                "ticket_cost_yuan_for_two": to_float(row["ticket_cost_yuan_per_two"]),
                "risk_score": to_float(row["risk_score"]),
                "mode_combo": "hard30_original_segments",
                "high_temp_exposure_hours": 0.0,
                "remote_or_altitude_spots": 0,
                "lodging_hub": row.get("lodging_hub", ""),
                "hotel_cost_yuan": to_float(row.get("room_price_yuan"), 0.0),
            }
            fatigue, comfort, stress, notes = self.comfort_and_stress(metrics, 8.0, "extreme_coverage")
            if str(row.get("active_budget_status", "")).lower() != "ok":
                stress = "red"
                notes = "硬约束可行但当天强度超标"
            metrics.update({"fatigue_index": fatigue, "comfort_score": comfort, "stress_level": stress, "humanized_note": notes})
            rows.append(metrics)
        return pd.DataFrame(rows)

    def adaptive_schedule(self, spec: RouteSpec, source_persona_id: str) -> pd.DataFrame:
        src = self.variant_days[self.variant_days["persona_id"].eq(source_persona_id)].copy()
        rows: list[dict[str, Any]] = []
        for row in src.to_dict("records"):
            metrics = {
                "route_id": spec.route_id,
                "route_type": spec.route_type,
                "persona_id": spec.persona_id,
                "day": int(row["day"]),
                "calendar_role": "active" if int(row["visit_spots_count"]) > 0 else "transfer",
                "visit_spots_count": int(row["visit_spots_count"]),
                "visit_spots": row["visit_spots"],
                "day_travel_hours": to_float(row["day_travel_hours"]),
                "day_service_hours": to_float(row["day_service_hours"]),
                "day_active_hours": to_float(row["day_active_hours"]),
                "transport_cost_yuan_for_two": to_float(row["transport_cost_yuan_for_two"]),
                "ticket_cost_yuan_for_two": sum(
                    self.ticket_cost_two(name) for name in parse_sequence(row["visit_spots"]) if name in self.spot_by_name
                ),
                "risk_score": to_float(row["risk_proxy"]),
                "mode_combo": "adaptive_real_amap_route",
                "high_temp_exposure_hours": 0.0,
                "remote_or_altitude_spots": 0,
                "lodging_hub": "",
                "hotel_cost_yuan": float(self.hotel_default["high_season_room_yuan_per_night"].median()),
                "fatigue_index": to_float(row["fatigue_index"]),
                "comfort_score": to_float(row["comfort_score"]),
                "stress_level": row["stress_level"],
                "humanized_note": "来自既有自适应画像排程；V2保留为方案族成员",
            }
            rows.append(metrics)
        planned_days = min(30, len(rows) + spec.intended_buffer_days)
        for day in range(len(rows) + 1, planned_days + 1):
            rows.append(
                {
                    "route_id": spec.route_id,
                    "route_type": spec.route_type,
                    "persona_id": spec.persona_id,
                    "day": day,
                    "calendar_role": "buffer",
                    "visit_spots_count": 0,
                    "visit_spots": "机动缓冲/低强度休整",
                    "day_travel_hours": 0.0,
                    "day_service_hours": 0.0,
                    "day_active_hours": 0.0,
                    "transport_cost_yuan_for_two": 0.0,
                    "ticket_cost_yuan_for_two": 0.0,
                    "risk_score": 0.0,
                    "mode_combo": "buffer_day",
                    "high_temp_exposure_hours": 0.0,
                    "remote_or_altitude_spots": 0,
                    "lodging_hub": "按实际所在城市顺延",
                    "hotel_cost_yuan": float(self.hotel_default["high_season_room_yuan_per_night"].median()) if day < planned_days else 0.0,
                    "fatigue_index": 0.0,
                    "comfort_score": 100.0,
                    "stress_level": "green",
                    "humanized_note": "用于天气/预约/身体状态缓冲",
                }
            )
        return pd.DataFrame(rows)

    def balanced_filtered_schedule(self, spec: RouteSpec) -> pd.DataFrame:
        """Use the validated standard-active day structure, then remove low-value spots.

        This preserves transfer-only days instead of forcing long transfers and visits
        onto the same day. The removed spot days become explicit buffer/recovery days.
        """
        src = self.variant_days[self.variant_days["persona_id"].eq("standard_active")].copy()
        keep_set = set(spec.sequence)
        dropped_name_set = set(self.base_sequence) - keep_set
        rows: list[dict[str, Any]] = []
        for row in src.to_dict("records"):
            original_names = [name for name in parse_sequence(row["visit_spots"]) if name in self.spot_by_name]
            kept_names = [name for name in original_names if name in keep_set]
            removed_names = [name for name in original_names if name not in keep_set]
            is_removed_visit_day = bool(original_names) and not kept_names

            if is_removed_visit_day:
                metrics = {
                    "route_id": spec.route_id,
                    "route_type": spec.route_type,
                    "persona_id": spec.persona_id,
                    "day": int(row["day"]),
                    "calendar_role": "buffer",
                    "visit_spots_count": 0,
                    "visit_spots": f"机动缓冲/删减点释放：{'、'.join(removed_names)}",
                    "day_travel_hours": 0.0,
                    "day_service_hours": 0.0,
                    "day_active_hours": 0.0,
                    "transport_cost_yuan_for_two": 0.0,
                    "ticket_cost_yuan_for_two": 0.0,
                    "risk_score": 0.0,
                    "mode_combo": "buffer_day_from_drop_low_value",
                    "high_temp_exposure_hours": 0.0,
                    "remote_or_altitude_spots": 0,
                    "lodging_hub": "按前后城市顺延",
                    "hotel_cost_yuan": float(self.hotel_default["high_season_room_yuan_per_night"].median()),
                    "fatigue_index": 0.0,
                    "comfort_score": 100.0,
                    "stress_level": "green",
                    "humanized_note": "低收益点删除后形成机动日，用于预约失败、道路延误或身体恢复",
                }
                rows.append(metrics)
                continue

            visit_count = len(kept_names)
            if visit_count:
                removed_service = sum(self.service_hours(name) for name in removed_names)
                removed_ticket = sum(self.ticket_cost_two(name) for name in removed_names)
                service_hours = max(0.0, to_float(row["day_service_hours"]) - removed_service)
                active_hours = max(0.0, to_float(row["day_active_hours"]) - removed_service)
                ticket_cost = sum(self.ticket_cost_two(name) for name in kept_names)
                visit_text = " -> ".join(kept_names)
                role = "active"
            else:
                service_hours = to_float(row["day_service_hours"])
                active_hours = to_float(row["day_active_hours"])
                ticket_cost = 0.0
                visit_text = row["visit_spots"]
                if any(name in str(visit_text) for name in dropped_name_set):
                    visit_text = "跨区转场/机动恢复（绕开删减点，向后续核心景区推进）"
                role = "transfer"

            metrics = {
                "route_id": spec.route_id,
                "route_type": spec.route_type,
                "persona_id": spec.persona_id,
                "day": int(row["day"]),
                "calendar_role": role,
                "visit_spots_count": visit_count,
                "visit_spots": visit_text,
                "day_travel_hours": to_float(row["day_travel_hours"]),
                "day_service_hours": round(service_hours, 3),
                "day_active_hours": round(active_hours, 3),
                "transport_cost_yuan_for_two": to_float(row["transport_cost_yuan_for_two"]),
                "ticket_cost_yuan_for_two": round(ticket_cost, 2),
                "risk_score": to_float(row["risk_proxy"]),
                "mode_combo": "standard_active_amap_route_filtered",
                "high_temp_exposure_hours": 0.0,
                "remote_or_altitude_spots": sum(
                    boolish(self.spot_info(name).get("high_altitude_or_remote")) for name in kept_names
                ),
                "lodging_hub": self.hub_name(kept_names[-1]) if kept_names else "",
                "hotel_cost_yuan": float(self.hotel_default["high_season_room_yuan_per_night"].median()),
                "fatigue_index": to_float(row["fatigue_index"]),
                "comfort_score": to_float(row["comfort_score"]),
                "stress_level": row["stress_level"],
                "humanized_note": "沿用普通体力型人性化转场结构；低收益点已删除" if visit_count else "跨区转场日，不叠加核心景点",
            }
            rows.append(metrics)
        return pd.DataFrame(rows)

    def route_spots_from_days(self, days: pd.DataFrame) -> list[str]:
        names: list[str] = []
        for text in days["visit_spots"].fillna(""):
            for name in parse_sequence(text):
                if name in self.spot_by_name and name not in names:
                    names.append(name)
        return names

    def policy_row(self, persona_id: str | None, policy_id: str | None, profile_id: str = "balanced") -> dict[str, Any]:
        if not persona_id or not policy_id:
            return {}
        df = self.policy_selection[
            self.policy_selection["profile_id"].eq(profile_id)
            & self.policy_selection["persona_id"].eq(persona_id)
            & self.policy_selection["policy_id"].eq(policy_id)
        ]
        if df.empty:
            df = self.policy_selection[
                self.policy_selection["persona_id"].eq(persona_id)
                & self.policy_selection["policy_id"].eq(policy_id)
            ]
        return df.iloc[0].to_dict() if not df.empty else {}

    def persona_robust_summary(self, persona_id: str) -> dict[str, float]:
        df = self.robustness[self.robustness["persona_id"].eq(persona_id)]
        if df.empty:
            return {}
        return {
            "min_success_probability": round(float(df["success_probability"].min()), 4),
            "weighted_success_probability": round(float(df["success_probability"].mean()), 4),
            "weighted_red_risk": round(float(df["red_risk_probability"].mean()), 4),
            "p90_simulated_days_max": round(float(df["p90_simulated_days"].max()), 2),
            "expected_reservation_failures": round(float(df["expected_reservation_failures"].mean()), 2),
            "expected_road_delay_days": round(float(df["expected_road_delay_days"].mean()), 2),
        }

    def hard_cvar(self) -> dict[str, float]:
        base_cost = to_float(self.hard_summary.iloc[0]["itinerary_proxy_cost_yuan_excluding_meals"], 0.0)
        losses = []
        for row in self.hard_trials.to_dict("records"):
            loss = (
                max(0.0, to_float(row["scheduled_days"]) - 30.0) * 600
                + to_float(row["reservation_failures"]) * 400
                + to_float(row["hotel_full_nights"]) * 350
                + to_float(row["severe_transport_events"]) * 800
                + to_float(row["over_budget_days"]) * 150
                + to_float(row["time_window_violations"]) * 500
                + max(0.0, to_float(row["total_proxy_cost_yuan"]) - base_cost) * 0.15
            )
            losses.append(loss)
        mc = self.hard_mc_summary.iloc[0].to_dict()
        return {
            "min_success_probability": round(to_float(mc.get("feasibility_rate")), 4),
            "weighted_success_probability": round(to_float(mc.get("feasibility_rate")), 4),
            "weighted_red_risk": round(1 - to_float(mc.get("feasibility_rate")), 4),
            "p90_simulated_days_max": round(percentile_or_zero(self.hard_trials["scheduled_days"], 90), 2),
            "tail_loss_cvar75": round(cvar75(losses), 2),
            "expected_loss": round(float(np.mean(losses)), 2),
        }

    def summarize_route(self, spec: RouteSpec, days: pd.DataFrame) -> dict[str, Any]:
        spot_names = self.route_spots_from_days(days)
        active_days = int(days[~days["calendar_role"].eq("buffer")]["day"].max()) if not days.empty else 0
        buffer_days = int(days["calendar_role"].eq("buffer").sum())
        regions = [self.region_name(name) for name in spot_names]
        topic = sum(boolish(self.spot_info(name).get("is_topic_preference")) for name in spot_names)
        cultural = sum(boolish(self.spot_info(name).get("is_cultural")) for name in spot_names)
        natural = sum(boolish(self.spot_info(name).get("is_natural")) for name in spot_names)
        remote = sum(boolish(self.spot_info(name).get("high_altitude_or_remote")) for name in spot_names)
        red = int(days["stress_level"].eq("red").sum())
        yellow = int(days["stress_level"].eq("yellow").sum())
        green = int(days["stress_level"].eq("green").sum())

        robust = self.hard_cvar() if spec.persona_id == "extreme_coverage" else self.persona_robust_summary(
            spec.policy_persona_id or spec.persona_id
        )
        policy = self.policy_row(spec.policy_persona_id or spec.persona_id, spec.policy_id)
        if policy:
            robust.update(
                {
                    "min_success_probability": round(to_float(policy.get("min_success_probability")), 4),
                    "weighted_success_probability": round(to_float(policy.get("weighted_success_probability")), 4),
                    "weighted_red_risk": round(to_float(policy.get("weighted_red_risk")), 4),
                    "tail_loss_cvar75": round(to_float(policy.get("tail_loss_cvar75")), 2),
                    "expected_loss": round(to_float(policy.get("expected_loss")), 2),
                    "policy_name": policy.get("policy_name", ""),
                    "policy_cost_penalty_yuan": to_float(policy.get("cost_penalty_yuan"), 0.0),
                    "policy_coverage_loss_spots": to_float(policy.get("coverage_loss_spots"), 0.0),
                }
            )
        robust.setdefault("tail_loss_cvar75", 0.0)
        robust.setdefault("expected_loss", 0.0)
        robust.setdefault("policy_name", "按路线族基准执行")
        robust.setdefault("policy_cost_penalty_yuan", 0.0)
        robust.setdefault("policy_coverage_loss_spots", 0.0)

        transport = round(float(days["transport_cost_yuan_for_two"].sum()), 2)
        ticket = round(float(days["ticket_cost_yuan_for_two"].sum()), 2)
        hotel = round(float(days["hotel_cost_yuan"].sum()), 2)
        total_cost = round(transport + ticket + hotel + robust["policy_cost_penalty_yuan"], 2)
        route_sequence = " -> ".join(spot_names)
        chance_pass = robust["weighted_success_probability"] >= 0.8

        return {
            "route_id": spec.route_id,
            "route_type": spec.route_type,
            "persona_id": spec.persona_id,
            "persona_name": spec.persona_name,
            "recommendation_role": spec.recommendation_role,
            "spots_count": len(spot_names),
            "active_or_transfer_days": active_days,
            "planned_trip_days": int(days["day"].max()) if not days.empty else 0,
            "buffer_days": buffer_days,
            "coverage_value_v2": self.coverage_value(spot_names),
            "region_count": len(set(regions)),
            "region_list": "、".join(sorted(set(regions))),
            "topic_preference_spots": topic,
            "cultural_spots": cultural,
            "natural_spots": natural,
            "remote_or_high_altitude_spots": remote,
            "transport_cost_yuan_for_two": transport,
            "ticket_cost_yuan_for_two": ticket,
            "hotel_cost_yuan": hotel,
            "policy_cost_penalty_yuan": robust["policy_cost_penalty_yuan"],
            "total_cost_yuan_excluding_meals": total_cost,
            "total_travel_hours": round(float(days["day_travel_hours"].sum()), 3),
            "total_service_hours": round(float(days["day_service_hours"].sum()), 3),
            "total_active_hours": round(float(days["day_active_hours"].sum()), 3),
            "mean_day_active_hours": round(float(days[~days["calendar_role"].eq("buffer")]["day_active_hours"].mean()), 3),
            "mean_comfort_score": round(float(days["comfort_score"].mean()), 2),
            "p10_comfort_score": round(percentile_or_zero(days["comfort_score"], 10), 2),
            "red_days": red,
            "yellow_days": yellow,
            "green_days": green,
            "weighted_success_probability": robust["weighted_success_probability"],
            "min_success_probability": robust["min_success_probability"],
            "weighted_red_risk": robust["weighted_red_risk"],
            "tail_loss_cvar75": robust["tail_loss_cvar75"],
            "expected_loss": robust["expected_loss"],
            "chance_constraint_weighted_pass": chance_pass,
            "policy_name": robust["policy_name"],
            "model_note": spec.model_note,
            "route_sequence": route_sequence,
        }

    def build_route_specs(self) -> list[RouteSpec]:
        balanced_seq, balanced_dropped = self.sequence_for_target(30)
        family_seq = parse_sequence(
            self.variant_summary[self.variant_summary["persona_id"].eq("family_comfort")].iloc[0]["route_sequence"]
        )
        senior_seq = parse_sequence(
            self.variant_summary[self.variant_summary["persona_id"].eq("senior_slow")].iloc[0]["route_sequence"]
        )
        return [
            RouteSpec(
                route_id="Q1V2_EXTREME_32_HARD30",
                route_type="极限覆盖版",
                persona_id="extreme_coverage",
                persona_name="极限覆盖展示",
                sequence=self.base_sequence,
                active_days_target=30,
                day_active_limit=8.0,
                intended_buffer_days=0,
                recommendation_role="对照基准，不主推",
                model_note="沿用第一版 HYBRID_30D_ACO_ALNS_SA；展示32景点硬30天覆盖上限",
            ),
            RouteSpec(
                route_id="Q1V2_BALANCED_30_BUFFER2",
                route_type="均衡稳健版",
                persona_id="balanced_robust",
                persona_name="普通成年人现实自由行",
                sequence=balanced_seq,
                active_days_target=28,
                day_active_limit=8.5,
                intended_buffer_days=2,
                recommendation_role="主推",
                model_note=f"删除低收益/高扰动边际点：{'、'.join(balanced_dropped)}；保留2天机动缓冲",
                policy_persona_id="standard_active",
                policy_id="drop_low_value",
            ),
            RouteSpec(
                route_id="Q1V2_FAMILY_30_COMFORT",
                route_type="亲子舒适版",
                persona_id="family_comfort",
                persona_name="亲子舒适型",
                sequence=family_seq,
                active_days_target=None,
                day_active_limit=7.0,
                intended_buffer_days=0,
                recommendation_role="扩展方案",
                model_note="直接复用亲子舒适型自适应排程；以低日强度替代缓冲天数",
                policy_persona_id="family_comfort",
                policy_id="split_slow_trip",
            ),
            RouteSpec(
                route_id="Q1V2_SENIOR_20_SLOW",
                route_type="长者慢游版",
                persona_id="senior_slow",
                persona_name="长者慢游型",
                sequence=senior_seq,
                active_days_target=None,
                day_active_limit=6.5,
                intended_buffer_days=7,
                recommendation_role="扩展方案",
                model_note="删减高海拔/长转场/低收益点，保留7天低强度缓冲",
                policy_persona_id="senior_slow",
                policy_id="as_planned",
            ),
        ]

    def build_route_family(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        specs = self.build_route_specs()
        all_days: list[pd.DataFrame] = []
        summaries: list[dict[str, Any]] = []
        for spec in specs:
            if spec.persona_id == "extreme_coverage":
                days = self.hard_extreme_schedule()
            elif spec.persona_id == "balanced_robust":
                days = self.balanced_filtered_schedule(spec)
            elif spec.persona_id in {"family_comfort", "senior_slow"}:
                days = self.adaptive_schedule(spec, spec.persona_id)
            else:
                days = self.schedule_from_sequence(spec)
            all_days.append(days)
            summaries.append(self.summarize_route(spec, days))

        daily = pd.concat(all_days, ignore_index=True)
        summary = pd.DataFrame(summaries)
        robustness = summary[
            [
                "route_id",
                "route_type",
                "recommendation_role",
                "spots_count",
                "planned_trip_days",
                "buffer_days",
                "weighted_success_probability",
                "min_success_probability",
                "weighted_red_risk",
                "tail_loss_cvar75",
                "expected_loss",
                "chance_constraint_weighted_pass",
                "policy_name",
                "red_days",
                "yellow_days",
                "model_note",
            ]
        ].copy()

        daily.to_csv(OUT_DIR / "q1_v2_daily_itinerary.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT_DIR / "q1_v2_pareto_routes.csv", index=False, encoding="utf-8-sig")
        robustness.to_csv(OUT_DIR / "q1_v2_robustness_summary.csv", index=False, encoding="utf-8-sig")
        return summary, daily, robustness

    def build_epsilon_grid(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for q in [20, 22, 24, 26, 28, 30, 32]:
            seq, dropped = self.sequence_for_target(q)
            active_days = min(max(q - 2, 18), 30)
            buffer_days = max(0, 30 - active_days)
            spec = RouteSpec(
                route_id=f"Q1V2_EPS_Q{q}",
                route_type="epsilon-coverage-grid",
                persona_id="epsilon",
                persona_name=f"覆盖下界q={q}",
                sequence=seq,
                active_days_target=min(active_days, len(seq)),
                day_active_limit=8.3,
                intended_buffer_days=buffer_days,
                recommendation_role="候选网格",
                model_note=f"epsilon约束候选：覆盖不少于{q}个景点；删除：{'、'.join(dropped)}",
                policy_persona_id="standard_active",
                policy_id="drop_low_value" if q <= 30 else "as_planned",
            )
            days = self.balanced_filtered_schedule(spec)
            row = self.summarize_route(spec, days)
            row["epsilon_q"] = q
            row["dropped_spots"] = "、".join(dropped)
            rows.append(row)
        grid = pd.DataFrame(rows)
        grid.to_csv(OUT_DIR / "q1_v2_epsilon_grid.csv", index=False, encoding="utf-8-sig")
        return grid

    def build_cards_and_matrix(self, summary: pd.DataFrame, robustness: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cards = summary[
            [
                "route_id",
                "route_type",
                "recommendation_role",
                "spots_count",
                "planned_trip_days",
                "buffer_days",
                "total_cost_yuan_excluding_meals",
                "mean_comfort_score",
                "weighted_success_probability",
                "tail_loss_cvar75",
                "region_list",
                "model_note",
            ]
        ].copy()
        cards["product_card"] = [
            f"{row.route_type}｜{row.spots_count}景点/{row.planned_trip_days}天/缓冲{row.buffer_days}天｜"
            f"费用约{row.total_cost_yuan_excluding_meals:.0f}元｜舒适度{row.mean_comfort_score:.1f}｜"
            f"加权成功率{row.weighted_success_probability:.1%}"
            for row in cards.itertuples(index=False)
        ]

        matrix_rows = [
            {
                "risk_type": "预约失败",
                "trigger": "重点景区余票不足、实名预约失败",
                "model_signal": "expected_reservation_failures / buffer_days",
                "strategy": "优先消耗缓冲日；同城替代点；提前7-14天预约",
                "applies_to": "全部方案，极限覆盖版风险最高",
            },
            {
                "risk_type": "道路/天气延误",
                "trigger": "独库/山区路段封闭、强降雨、大风沙尘",
                "model_signal": "weighted_red_risk / p90_simulated_days",
                "strategy": "长转场后不安排核心景点；铁路/航班优先替代；保留缓冲日",
                "applies_to": "均衡稳健版、长者慢游版",
            },
            {
                "risk_type": "高温暴露",
                "trigger": "吐鲁番、南疆午后高温",
                "model_signal": "high_temp_exposure_hours / red_days",
                "strategy": "上午户外、午间室内或酒店休息、傍晚补游",
                "applies_to": "亲子舒适版、长者慢游版",
            },
            {
                "risk_type": "酒店满房/晚到",
                "trigger": "旺季热门城市房量不足或晚间到达",
                "model_signal": "hotel_cost / calendar_role / long transfer",
                "strategy": "住宿城市提前锁定；长转场日只做交通或低强度景点",
                "applies_to": "全部方案",
            },
        ]
        cards.to_csv(OUT_DIR / "q1_v2_route_product_cards.csv", index=False, encoding="utf-8-sig")
        matrix = pd.DataFrame(matrix_rows)
        matrix.to_csv(OUT_DIR / "q1_v2_risk_strategy_matrix.csv", index=False, encoding="utf-8-sig")
        return cards, matrix

    def build_figures(self, summary: pd.DataFrame, grid: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
        colors = {
            "极限覆盖版": "#7f1d1d",
            "均衡稳健版": "#0f766e",
            "亲子舒适版": "#2563eb",
            "长者慢游版": "#7c3aed",
        }
        for row in summary.itertuples(index=False):
            ax.scatter(
                row.spots_count,
                row.total_cost_yuan_excluding_meals,
                s=max(70, row.weighted_success_probability * 260),
                color=colors.get(row.route_type, "#475569"),
                edgecolor="white",
                linewidth=1.0,
                alpha=0.88,
            )
            ax.text(row.spots_count + 0.12, row.total_cost_yuan_excluding_meals, row.route_type, fontsize=9)
        ax.plot(grid["spots_count"], grid["total_cost_yuan_excluding_meals"], color="#94a3b8", linestyle="--", linewidth=1.2, label="epsilon候选网格")
        ax.set_xlabel("覆盖景点数")
        ax.set_ylabel("除餐饮外费用（元/两人）")
        ax.set_title("Q1-V2 路线族 Pareto 视图：覆盖-费用-成功率")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_q1_v2_pareto_route_family.png")
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=160)
        x = np.arange(len(summary))
        labels = summary["route_type"].tolist()
        ax1.bar(x - 0.18, summary["mean_comfort_score"], width=0.36, color="#38bdf8", label="平均舒适度")
        ax1.set_ylabel("平均舒适度")
        ax1.set_ylim(0, 105)
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, summary["weighted_success_probability"], width=0.36, color="#22c55e", label="加权成功率")
        ax2.set_ylabel("加权成功率")
        ax2.set_ylim(0, 1.05)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=18, ha="right")
        ax1.set_title("Q1-V2 舒适度与稳健性对比")
        ax1.grid(axis="y", alpha=0.22)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_q1_v2_comfort_robustness.png")
        plt.close(fig)

    def build_workbook(self, tables: dict[str, pd.DataFrame]) -> None:
        out_path = REPORT_DIR / "新疆旅游第一问Q1_V2强化结果.xlsx"
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                for sheet_name, table in tables.items():
                    safe_name = sheet_name[:31]
                    table.to_excel(writer, sheet_name=safe_name, index=False)
        except Exception:
            with pd.ExcelWriter(out_path) as writer:
                for sheet_name, table in tables.items():
                    table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    def build_report(
        self,
        summary: pd.DataFrame,
        robustness: pd.DataFrame,
        grid: pd.DataFrame,
        cards: pd.DataFrame,
        matrix: pd.DataFrame,
    ) -> None:
        main = summary[summary["recommendation_role"].eq("主推")].iloc[0]
        extreme = summary[summary["route_type"].eq("极限覆盖版")].iloc[0]

        def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
            return df[cols].to_markdown(index=False)

        report = f"""# 第一问 Q1-V2 强化建模与实验报告

## 1. 本轮改进定位

第一问原模型已经给出 `HYBRID_30D_ACO_ALNS_SA`：32 景点、30 天硬约束、时间窗违规为 0 的极限覆盖路线。V2 的核心不是继续把景点数量推到最大，而是把问题重构为：

> 面向真实游客体验的鲁棒多模式奖励收集定向游模型。

因此，本轮输出从“单条路线”改为“路线方案族”：极限覆盖版、均衡稳健版、亲子舒适版、长者慢游版。32 景点路线保留为算法能力展示和对照基准，主推方案改为有缓冲、有删减、有风险评价的均衡稳健路线。

## 2. V2 数学建模要点

路线 $R$ 的评价目标被拆成四个维度：

- $F_1(R)$：覆盖价值，综合景点优先级、题面偏好、文化/自然属性；
- $F_2(R)$：除餐饮外总代理成本，包含境内交通、门票、住宿和风险策略成本；
- $F_3(R)$：平均舒适度，来自日活动时长、转场时长、高温暴露、高海拔/远程点；
- $F_4(R)$：风险暴露，使用加权成功率、红色风险日、CVaR75 尾部损失描述。

求解上采用 epsilon-constraint 思路，对覆盖下界 $q=20,22,24,26,28,30,32$ 形成候选网格，再从覆盖、费用、舒适度、稳健性中筛选路线族。

## 3. 本轮生成的数据资产

- `outputs/enhanced_od_labels_v2.csv`：把原 OD 拆成 rail/air/rental_car/charter_car/taxi_transfer/scenic_shuttle 等真实交通标签，并补充疲劳分、风险分、夜间交通、换乘标记。
- `outputs/q1_v2_epsilon_grid.csv`：不同覆盖下界的候选解网格。
- `outputs/q1_v2_pareto_routes.csv`：最终路线族摘要。
- `outputs/q1_v2_daily_itinerary.csv`：逐日人性化行程，含红黄绿压力日、缓冲日、舒适度与风险备注。
- `outputs/q1_v2_robustness_summary.csv`：稳健性与 CVaR 汇总。
- `outputs/q1_v2_route_product_cards.csv`：汇报用路线产品卡。
- `outputs/q1_v2_risk_strategy_matrix.csv`：预约、道路、天气、高温、酒店风险应对矩阵。

## 4. 路线族结果

{markdown_table(summary, [
    "route_type", "recommendation_role", "spots_count", "planned_trip_days", "buffer_days",
    "total_cost_yuan_excluding_meals", "mean_comfort_score", "weighted_success_probability",
    "tail_loss_cvar75", "red_days", "policy_name"
])}

## 5. 主推方案解释

主推方案为 **{main.route_type}**：覆盖 {int(main.spots_count)} 个景点，计划 {int(main.planned_trip_days)} 天，其中缓冲 {int(main.buffer_days)} 天；除餐饮外两人费用约 {main.total_cost_yuan_excluding_meals:.0f} 元，平均舒适度 {main.mean_comfort_score:.1f}，加权成功率 {main.weighted_success_probability:.1%}，CVaR75 尾部损失 {main.tail_loss_cvar75:.0f}。

它相对极限覆盖版牺牲了 {int(extreme.spots_count - main.spots_count)} 个低收益/高扰动边际景点，但把路线从“压线可行”改成“留有机动天数”。这更符合暑期新疆真实旅行中的预约失败、道路延误、高温错峰和酒店满房风险。

## 6. epsilon 候选网格

{markdown_table(grid, [
    "epsilon_q", "spots_count", "planned_trip_days", "buffer_days", "total_cost_yuan_excluding_meals",
    "mean_comfort_score", "weighted_success_probability", "tail_loss_cvar75"
])}

## 7. 风险策略矩阵

{markdown_table(matrix, ["risk_type", "trigger", "strategy", "applies_to"])}

## 8. 结论

第一问 V2 的最终表述应调整为：

> 在 30 天暑期周期内，32 景点路线是硬约束下的覆盖上界，不作为现实主推；对普通游客，推荐 30 景点、28 天有效行程、2 天机动缓冲的均衡稳健版；对亲子和长者，分别给出低日强度和慢游删减方案。

这一版本更适合汇报：它承认“游尽可能多的地方”和“真实可执行”之间的冲突，并用 Pareto 方案族、CVaR 尾部风险和人性化日程把冲突显式表达出来。
"""
        (REPORT_DIR / "新疆旅游第一问Q1_V2强化建模与实验报告.md").write_text(report, encoding="utf-8")

    def run(self) -> None:
        transport = self.build_transport_labels_v2()
        summary, daily, robustness = self.build_route_family()
        grid = self.build_epsilon_grid()
        cards, matrix = self.build_cards_and_matrix(summary, robustness)
        self.build_figures(summary, grid)
        self.build_workbook(
            {
                "route_family": summary,
                "daily_itinerary": daily,
                "robustness": robustness,
                "epsilon_grid": grid,
                "transport_labels": transport,
                "product_cards": cards,
                "risk_matrix": matrix,
            }
        )
        self.build_report(summary, robustness, grid, cards, matrix)
        solve_summary = {
            "package_root": str(PKG_ROOT),
            "v2_root": str(V2_ROOT),
            "routes": int(summary.shape[0]),
            "daily_rows": int(daily.shape[0]),
            "epsilon_candidates": int(grid.shape[0]),
            "transport_label_rows": int(transport.shape[0]),
            "main_route_id": "Q1V2_BALANCED_30_BUFFER2",
            "outputs": [
                str(OUT_DIR / "enhanced_od_labels_v2.csv"),
                str(OUT_DIR / "q1_v2_pareto_routes.csv"),
                str(OUT_DIR / "q1_v2_daily_itinerary.csv"),
                str(OUT_DIR / "q1_v2_robustness_summary.csv"),
                str(OUT_DIR / "q1_v2_epsilon_grid.csv"),
                str(REPORT_DIR / "新疆旅游第一问Q1_V2强化建模与实验报告.md"),
                str(REPORT_DIR / "新疆旅游第一问Q1_V2强化结果.xlsx"),
            ],
        }
        (OUT_DIR / "solve_summary.json").write_text(json.dumps(solve_summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    Q1V2Builder().run()
