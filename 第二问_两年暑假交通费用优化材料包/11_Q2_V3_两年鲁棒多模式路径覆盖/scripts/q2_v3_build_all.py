from __future__ import annotations

import json
import math
import random
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransportLabel:
    label_id: str
    from_id: str
    to_id: str
    mode: str
    mode_combo: str
    time_hours: float
    cost_yuan_for_two: float
    risk_score: float
    fatigue_score: float
    path_desc: str
    source: str
    schedule_required: bool = False
    is_gateway_label: bool = False
    gateway_name: str = ""
    direction: str = ""
    raw_mode: str = ""
    raw_cost_yuan_for_two: float = 0.0
    cost_calibration_note: str = ""


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def num(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name)


def configure_plot_fonts() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_plot_fonts()


class Q2V3Builder:
    def __init__(self) -> None:
        self.q2v3_dir = Path(__file__).resolve().parents[1]
        self.package_dir = self.q2v3_dir.parent
        self.input_dir = self.package_dir / "01_输入数据"
        self.model_data = self.input_dir / "model_data"
        self.enhanced_data = self.input_dir / "enhanced_data"
        self.outputs = self.q2v3_dir / "outputs"
        self.figures = self.q2v3_dir / "figures"
        self.reports = self.q2v3_dir / "reports"
        self.config_dir = self.q2v3_dir / "config"
        for path in [self.outputs, self.figures, self.reports, self.config_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.config = {
            "year_day_limit": 30,
            "daily_effective_hours": 8.0,
            "scenario_samples": 500,
            "success_threshold": 0.8,
            "cost_gap_for_robust_choice": 0.08,
            "max_year_day_difference": 8,
            "transport_label_limit_per_od": 5,
            "min_spots_per_year": 14,
            "max_spots_per_year": 24,
            "random_seed": 20260611,
            "scenic_shuttle_floor_multiplier": 1.0,
            "cost_calibration_version": "scenic_floor_base_v1 + road_mode_normalization_q1v3",
            "gateway_candidates": [
                "乌鲁木齐市",
                "吐鲁番市",
                "伊宁市(伊犁)",
                "喀什市",
                "阿勒泰市",
                "库尔勒市",
                "库车市",
                "和田市",
                "那拉提镇",
                "赛里木湖",
            ],
        }
        self.rng = np.random.default_rng(int(self.config["random_seed"]))
        random.seed(int(self.config["random_seed"]))
        self.label_counter = 0
        self.gateway_label_counter = 0

    # ---------- IO ----------

    def read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path, encoding="utf-8-sig")

    def write_csv(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def load_inputs(self) -> None:
        self.spots = self.read_csv(self.model_data / "spot_clean.csv")
        self.hubs = self.read_csv(self.model_data / "hub_clean.csv")
        self.special = self.read_csv(self.enhanced_data / "special_access_constraints.csv")
        self.time_windows = self.read_csv(self.enhanced_data / "spot_time_windows.csv")
        self.od = self.read_csv(self.enhanced_data / "enhanced_od_matrix_with_amap.csv")
        self.depot = self.read_csv(self.enhanced_data / "depot_access_matrix.csv")
        self.amap_depot = self.read_csv(self.enhanced_data / "amap_depot_access_matrix_clean.csv")
        self.rail = self.read_csv(self.enhanced_data / "rail_12306_model_options.csv")
        self.air = self.read_csv(self.enhanced_data / "air_public_schedule_seed.csv")
        self.graph_baseline = self.read_optional_csv(
            self.package_dir / "03_历史图论基线" / "graph_model_outputs" / "problem2_summary.csv"
        )
        self.amap_baseline = self.read_optional_csv(
            self.package_dir / "04_高德OD重实验" / "amap_selfdrive_model_outputs" / "problem2_amap_summary.csv"
        )
        self.spectral_baseline = self.read_optional_csv(
            self.package_dir / "05_多模式与谱聚类结果" / "enhanced_model_outputs" / "problem2_spectral_summary.csv"
        )
        self.current_status = self.read_optional_csv(
            self.package_dir
            / "07_求解效率与算法升级"
            / "solver_efficiency_outputs"
            / "current_problem2_status.csv"
        )

        access = self.special[["spot_id", "ordinary_tourist_allowed"]].copy()
        self.spots = self.spots.merge(access, on="spot_id", how="left")
        self.spots["ordinary_tourist_allowed"] = self.spots["ordinary_tourist_allowed"].fillna(
            ~self.spots["ordinary_tourist_restricted"].map(truthy)
        )
        ordinary_mask = (
            self.spots["ordinary_tourist_allowed"].map(truthy)
            & ~self.spots["requires_approval"].map(truthy)
            & ~self.spots["ordinary_tourist_restricted"].map(truthy)
        )
        self.ordinary_spots = self.spots[ordinary_mask].copy().reset_index(drop=True)
        self.special_spots = self.spots[~ordinary_mask].copy().reset_index(drop=True)
        self.all_spot_ids = self.spots["spot_id"].tolist()
        self.ordinary_ids = self.ordinary_spots["spot_id"].tolist()
        self.special_ids = self.special_spots["spot_id"].tolist()
        self.spot_by_id = self.spots.set_index("spot_id").to_dict("index")
        self.name_to_id = {clean_text(r.spot_name): clean_text(r.spot_id) for r in self.spots.itertuples()}
        self.gateway_names = [g for g in self.config["gateway_candidates"] if g in set(self.hubs["hub_name"])]
        if "乌鲁木齐市" not in self.gateway_names:
            self.gateway_names.insert(0, "乌鲁木齐市")
        self.gateway_anchors = self.build_gateway_anchors()

    def read_optional_csv(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig")
        return pd.DataFrame()

    # ---------- Label generation ----------

    def next_label_id(self) -> str:
        self.label_counter += 1
        return f"Q2L{self.label_counter:06d}"

    def next_gateway_label_id(self) -> str:
        self.gateway_label_counter += 1
        return f"Q2G{self.gateway_label_counter:06d}"

    def mode_from_text(self, text: Any) -> str:
        lower = clean_text(text).lower()
        if not lower or lower == "nan":
            return "same_spot"
        if "air" in lower or "flight" in lower:
            return "air"
        if "rail" in lower or "train" in lower:
            return "rail"
        if "coach" in lower or "bus" in lower:
            return "coach"
        if "self_drive" in lower:
            return "self_drive"
        if "scenic_shuttle" in lower:
            return "scenic_shuttle"
        return lower.split("->")[0].strip() or "unknown"

    def normalize_road_mode_by_time(self, time_hours: float) -> str:
        if time_hours <= 1.5:
            return "taxi_transfer"
        if time_hours <= 6.0:
            return "rental_car"
        return "charter_car"

    def normalize_mode(self, mode: str, mode_combo: str, time_hours: float) -> str:
        m = clean_text(mode).lower()
        combo = clean_text(mode_combo).lower()
        if m == "same_spot":
            return "same_spot"
        if m in {"self_drive", "fallback_road"}:
            return self.normalize_road_mode_by_time(time_hours)
        if "self_drive" in combo or "amap_selfdrive" in combo:
            return self.normalize_road_mode_by_time(time_hours)
        if m == "bus":
            return "coach"
        return m or "unknown"

    def scenic_shuttle_cost_floor(self, time_hours: float, multiplier: float | None = None) -> float:
        if multiplier is None:
            multiplier = float(self.config["scenic_shuttle_floor_multiplier"])
        if time_hours <= 0:
            base = 0.0
        elif time_hours <= 0.5:
            base = 20.0
        elif time_hours <= 1.0:
            base = 40.0
        elif time_hours <= 2.0:
            base = 70.0
        elif time_hours <= 4.0:
            base = 110.0
        else:
            base = max(160.0, 35.0 * time_hours)
        return round(base * float(multiplier), 2)

    def calibrate_cost(self, mode: str, raw_cost: float, time_hours: float) -> tuple[float, str]:
        raw_cost = max(0.0, float(raw_cost))
        if mode == "scenic_shuttle":
            floor = self.scenic_shuttle_cost_floor(time_hours)
            if raw_cost < floor:
                return round(floor, 2), f"scenic_shuttle_cost_floor_applied:{raw_cost:.2f}->{floor:.2f}"
        return round(raw_cost, 2), "raw_cost_kept"

    def fatigue_for(self, mode: str, time_hours: float) -> float:
        factors = {
            "same_spot": 0.0,
            "rail": 0.45,
            "air": 0.60,
            "coach": 0.85,
            "taxi_transfer": 0.80,
            "rental_car": 0.95,
            "charter_car": 0.85,
            "scenic_shuttle": 0.70,
            "gateway_access": 0.75,
        }
        return round(time_hours * factors.get(mode, 0.90), 3)

    def add_label(
        self,
        bucket: dict[tuple[str, str], list[TransportLabel]],
        from_id: str,
        to_id: str,
        mode: str,
        mode_combo: str,
        time_hours: float,
        cost_yuan_for_two: float,
        risk_score: float,
        path_desc: str,
        source: str,
        schedule_required: bool = False,
    ) -> None:
        if from_id == to_id and mode != "same_spot":
            return
        time_hours = max(0.0, round(float(time_hours), 3))
        raw_mode = clean_text(mode)
        raw_cost = max(0.0, round(float(cost_yuan_for_two), 2))
        normalized_mode = self.normalize_mode(mode, mode_combo, time_hours)
        calibrated_cost, cost_note = self.calibrate_cost(normalized_mode, raw_cost, time_hours)
        risk_score = max(0.0, round(float(risk_score), 3))
        label = TransportLabel(
            label_id=self.next_label_id(),
            from_id=from_id,
            to_id=to_id,
            mode=normalized_mode,
            mode_combo=mode_combo,
            time_hours=time_hours,
            cost_yuan_for_two=calibrated_cost,
            risk_score=risk_score,
            fatigue_score=self.fatigue_for(normalized_mode, time_hours),
            path_desc=path_desc,
            source=source,
            schedule_required=schedule_required or normalized_mode in {"rail", "air", "coach"},
            raw_mode=raw_mode,
            raw_cost_yuan_for_two=raw_cost,
            cost_calibration_note=cost_note,
        )
        bucket[(from_id, to_id)].append(label)

    def canonical_city(self, value: Any) -> str:
        text = clean_text(value)
        text = re.sub(r"[()（）].*?[)）]", "", text)
        text = text.replace("北", "")
        text = text.replace("市", "").replace("县", "").replace("镇", "")
        return text

    def city_matches(self, city: str, hub_name: str) -> bool:
        a = self.canonical_city(city)
        b = self.canonical_city(hub_name)
        return bool(a and b and (a in b or b in a))

    def build_multimodal_labels(self) -> pd.DataFrame:
        bucket: dict[tuple[str, str], list[TransportLabel]] = defaultdict(list)
        od = self.od[self.od["scenario_id"].eq("base_summer")].copy()
        for row in od.itertuples(index=False):
            from_id = clean_text(row.from_spot_id)
            to_id = clean_text(row.to_spot_id)
            if from_id == to_id:
                self.add_label(
                    bucket,
                    from_id,
                    to_id,
                    "same_spot",
                    "same_spot",
                    0.0,
                    0.0,
                    0.0,
                    "same spot",
                    "zero_self_loop",
                )
                continue
            mode_combo = clean_text(row.path_modes)
            mode = self.mode_from_text(mode_combo)
            if mode == "same_spot":
                mode = "self_drive"
            self.add_label(
                bucket,
                from_id,
                to_id,
                mode,
                mode_combo or mode,
                num(row.shortest_time_hours),
                num(row.shortest_cost_yuan_per_two),
                num(row.path_risk),
                mode_combo or "enhanced_od",
                "enhanced_od_matrix_with_amap",
                schedule_required=mode in {"rail", "air", "coach"},
            )
            amap_cost = num(getattr(row, "amap_selfdrive_cost_yuan_per_two", np.nan), np.nan)
            amap_time = num(getattr(row, "driving_duration_hours", np.nan), np.nan)
            if not math.isnan(amap_cost) and not math.isnan(amap_time) and amap_time > 0:
                risk = min(0.75, 0.04 + amap_time * 0.035)
                mode2 = "charter_car" if amap_time >= 5.0 else "rental_car"
                self.add_label(
                    bucket,
                    from_id,
                    to_id,
                    mode2,
                    "amap_selfdrive",
                    amap_time,
                    amap_cost,
                    risk,
                    f"amap road {num(getattr(row, 'driving_distance_km', 0.0)):.1f}km",
                    "amap_driving_od_matrix_clean",
                )

        self.add_public_labels(bucket)
        pruned: dict[tuple[str, str], list[TransportLabel]] = {}
        for pair, labels in bucket.items():
            pruned[pair] = self.prune_labels(labels, int(self.config["transport_label_limit_per_od"]))
        self.labels_by_pair = pruned
        rows = [label.__dict__ for labels in pruned.values() for label in labels]
        labels_df = pd.DataFrame(rows)
        self.write_csv(labels_df, self.outputs / "q2_v3_multimodal_labels.csv")
        return labels_df

    def add_public_labels(self, bucket: dict[tuple[str, str], list[TransportLabel]]) -> None:
        for row in self.rail.itertuples(index=False):
            price = num(getattr(row, "min_ticket_price_yuan", 0.0))
            if price <= 0:
                continue
            from_spots = self.spots_matching_city(row.from_city)
            to_spots = self.spots_matching_city(row.to_city)
            for i in from_spots:
                for j in to_spots:
                    if i == j:
                        continue
                    time_hours = num(row.duration_hours) + 1.1
                    cost = price * 2.0 + 50.0
                    self.add_label(
                        bucket,
                        i,
                        j,
                        "rail",
                        "local_transfer -> rail -> local_transfer",
                        time_hours,
                        cost,
                        0.055,
                        f"{clean_text(row.train_code)} {clean_text(row.from_station)}-{clean_text(row.to_station)}",
                        "rail_12306_model_options",
                        schedule_required=True,
                    )
        for row in self.air.itertuples(index=False):
            from_spots = self.spots_matching_city(row.from_city)
            to_spots = self.spots_matching_city(row.to_city)
            if not from_spots or not to_spots:
                continue
            fare = num(getattr(row, "fare_for_two_yuan", np.nan), np.nan)
            if math.isnan(fare) or fare <= 0:
                fare = max(760.0, 540.0 + num(row.duration_hours) * 160.0)
            for i in from_spots:
                for j in to_spots:
                    if i == j:
                        continue
                    time_hours = num(row.model_air_time_hours, num(row.duration_hours) + 2.5)
                    self.add_label(
                        bucket,
                        i,
                        j,
                        "air",
                        "airport_transfer -> air -> airport_transfer",
                        time_hours,
                        fare,
                        0.075,
                        f"{clean_text(row.flight_no)} {clean_text(row.from_city)}-{clean_text(row.to_city)}",
                        "air_public_schedule_seed",
                        schedule_required=True,
                    )

    def spots_matching_city(self, city: Any) -> list[str]:
        ids: list[str] = []
        for row in self.spots.itertuples(index=False):
            if self.city_matches(city, row.hub_name) or self.city_matches(city, row.region_raw):
                ids.append(clean_text(row.spot_id))
        return ids

    def prune_labels(self, labels: list[TransportLabel], limit: int) -> list[TransportLabel]:
        unique: dict[tuple[str, str, float, float], TransportLabel] = {}
        for label in labels:
            key = (label.from_id, label.to_id, round(label.time_hours, 2), round(label.cost_yuan_for_two, 1))
            old = unique.get(key)
            if old is None or label.risk_score < old.risk_score:
                unique[key] = label
        labels = list(unique.values())
        keep: list[TransportLabel] = []
        for label in labels:
            dominated = False
            for other in labels:
                if label is other:
                    continue
                le_all = (
                    other.cost_yuan_for_two <= label.cost_yuan_for_two + 1e-9
                    and other.time_hours <= label.time_hours + 1e-9
                    and other.risk_score <= label.risk_score + 1e-9
                    and other.fatigue_score <= label.fatigue_score + 1e-9
                )
                lt_one = (
                    other.cost_yuan_for_two < label.cost_yuan_for_two - 1e-9
                    or other.time_hours < label.time_hours - 1e-9
                    or other.risk_score < label.risk_score - 1e-9
                    or other.fatigue_score < label.fatigue_score - 1e-9
                )
                if le_all and lt_one:
                    dominated = True
                    break
            if not dominated:
                keep.append(label)
        keep.sort(key=lambda x: (x.cost_yuan_for_two, x.time_hours, x.risk_score))
        modes_seen = set()
        diverse: list[TransportLabel] = []
        for label in keep:
            if label.mode not in modes_seen:
                diverse.append(label)
                modes_seen.add(label.mode)
        for label in keep:
            if label not in diverse:
                diverse.append(label)
        return diverse[:limit]

    def best_label(self, from_id: str, to_id: str, objective: str = "cost") -> TransportLabel:
        labels = self.labels_by_pair.get((from_id, to_id), [])
        if not labels:
            time_hours = 10.0
            raw_cost = 900.0
            mode = self.normalize_mode("fallback_road", "fallback_road", time_hours)
            return TransportLabel(
                label_id="Q2L_FALLBACK",
                from_id=from_id,
                to_id=to_id,
                mode=mode,
                mode_combo="fallback_road",
                time_hours=time_hours,
                cost_yuan_for_two=raw_cost,
                risk_score=0.50,
                fatigue_score=self.fatigue_for(mode, time_hours),
                path_desc="fallback missing OD",
                source="fallback",
                raw_mode="fallback_road",
                raw_cost_yuan_for_two=raw_cost,
                cost_calibration_note="fallback_cost_kept",
            )
        if objective == "generalized":
            return min(labels, key=lambda x: x.cost_yuan_for_two + x.time_hours * 22 + x.risk_score * 260)
        return min(labels, key=lambda x: (x.cost_yuan_for_two, x.time_hours, x.risk_score))

    # ---------- Gateways ----------

    def build_gateway_anchors(self) -> dict[str, str | None]:
        anchors: dict[str, str | None] = {"乌鲁木齐市": None}
        for gateway in self.gateway_names:
            if gateway == "乌鲁木齐市":
                continue
            exact = self.ordinary_spots[self.ordinary_spots["hub_name"].eq(gateway)]
            if not exact.empty:
                anchors[gateway] = clean_text(exact.sort_values("visit_hours_mid").iloc[0]["spot_id"])
                continue
            hub = self.hubs[self.hubs["hub_name"].eq(gateway)]
            cluster = clean_text(hub.iloc[0]["hub_cluster"]) if not hub.empty else ""
            same_cluster = self.ordinary_spots[self.ordinary_spots["region_cluster"].eq(cluster)]
            if not same_cluster.empty:
                anchors[gateway] = clean_text(same_cluster.sort_values("visit_hours_mid").iloc[0]["spot_id"])
        return anchors

    def build_gateway_labels(self) -> pd.DataFrame:
        rows = []
        self.gateway_label_map: dict[tuple[str, str, str], TransportLabel] = {}
        depot_map = self.depot.set_index("spot_id").to_dict("index")
        for gateway in self.gateway_names:
            anchor = self.gateway_anchors.get(gateway)
            for spot_id in self.all_spot_ids:
                if gateway == "乌鲁木齐市" and spot_id in depot_map:
                    r = depot_map[spot_id]
                    entry = self.make_gateway_label(
                        gateway,
                        "gateway_to_spot",
                        gateway,
                        spot_id,
                        num(r["depot_to_spot_time"]),
                        num(r["depot_to_spot_cost"]),
                        num(r["depot_to_spot_risk"]),
                        "urumqi_depot_access",
                    )
                    exit_label = self.make_gateway_label(
                        gateway,
                        "spot_to_gateway",
                        spot_id,
                        gateway,
                        num(r["spot_to_depot_time"]),
                        num(r["spot_to_depot_cost"]),
                        num(r["spot_to_depot_risk"]),
                        "urumqi_depot_access",
                    )
                elif anchor == spot_id:
                    entry = self.make_gateway_label(
                        gateway,
                        "gateway_to_spot",
                        gateway,
                        spot_id,
                        0.55,
                        30.0,
                        0.025,
                        "gateway_local_access",
                    )
                    exit_label = self.make_gateway_label(
                        gateway,
                        "spot_to_gateway",
                        spot_id,
                        gateway,
                        0.55,
                        30.0,
                        0.025,
                        "gateway_local_access",
                    )
                elif anchor:
                    in_label = self.best_label(anchor, spot_id)
                    out_label = self.best_label(spot_id, anchor)
                    entry = self.make_gateway_label(
                        gateway,
                        "gateway_to_spot",
                        gateway,
                        spot_id,
                        in_label.time_hours + 0.45,
                        in_label.cost_yuan_for_two + 25.0,
                        min(0.9, in_label.risk_score + 0.025),
                        f"gateway_anchor::{self.spot_name(anchor)} -> {in_label.path_desc}",
                        inherited_mode=in_label.mode,
                    )
                    exit_label = self.make_gateway_label(
                        gateway,
                        "spot_to_gateway",
                        spot_id,
                        gateway,
                        out_label.time_hours + 0.45,
                        out_label.cost_yuan_for_two + 25.0,
                        min(0.9, out_label.risk_score + 0.025),
                        f"{out_label.path_desc} -> gateway_anchor::{self.spot_name(anchor)}",
                        inherited_mode=out_label.mode,
                    )
                else:
                    entry = self.make_gateway_label(gateway, "gateway_to_spot", gateway, spot_id, 12.0, 1100.0, 0.6, "fallback")
                    exit_label = self.make_gateway_label(gateway, "spot_to_gateway", spot_id, gateway, 12.0, 1100.0, 0.6, "fallback")
                for label in [entry, exit_label]:
                    self.gateway_label_map[(gateway, label.direction, spot_id)] = label
                    rows.append(label.__dict__)
        df = pd.DataFrame(rows)
        self.write_csv(df, self.outputs / "q2_v3_gateway_labels.csv")
        return df

    def make_gateway_label(
        self,
        gateway: str,
        direction: str,
        from_id: str,
        to_id: str,
        time_hours: float,
        cost: float,
        risk: float,
        source: str,
        inherited_mode: str = "gateway_access",
    ) -> TransportLabel:
        time_hours = round(max(0.0, time_hours), 3)
        raw_cost = round(max(0.0, cost), 2)
        mode = inherited_mode if inherited_mode in {"rail", "air", "coach"} else "gateway_access"
        return TransportLabel(
            label_id=self.next_gateway_label_id(),
            from_id=from_id,
            to_id=to_id,
            mode=mode,
            mode_combo=f"{direction}:{gateway}",
            time_hours=time_hours,
            cost_yuan_for_two=raw_cost,
            risk_score=round(max(0.0, risk), 3),
            fatigue_score=self.fatigue_for(mode, time_hours),
            path_desc=source,
            source=source,
            schedule_required=mode in {"rail", "air", "coach"},
            is_gateway_label=True,
            gateway_name=gateway,
            direction=direction,
            raw_mode=clean_text(inherited_mode),
            raw_cost_yuan_for_two=raw_cost,
            cost_calibration_note="gateway_cost_kept",
        )

    def gateway_label(self, gateway: str, direction: str, spot_id: str) -> TransportLabel:
        return self.gateway_label_map[(gateway, direction, spot_id)]

    # ---------- Plans ----------

    def parse_sequence(self, text: Any) -> list[str]:
        ids = []
        for name in clean_text(text).split("->"):
            spot_id = self.name_to_id.get(name.strip())
            if spot_id:
                ids.append(spot_id)
        return ids

    def spot_name(self, spot_id: str) -> str:
        return clean_text(self.spot_by_id.get(spot_id, {}).get("spot_name", spot_id))

    def route_names(self, seq: list[str]) -> str:
        return " -> ".join(self.spot_name(x) for x in seq)

    def generate_seed_plans(self) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []

        def add_seed(seed_id: str, y1: list[str], y2: list[str], method: str, include_special: bool = False) -> None:
            y1 = self.unique_keep_order(y1)
            y2 = [x for x in self.unique_keep_order(y2) if x not in y1]
            target = set(self.all_spot_ids if include_special else self.ordinary_ids)
            missing = [x for x in (self.all_spot_ids if include_special else self.ordinary_ids) if x not in set(y1 + y2)]
            if missing:
                y1, y2 = self.insert_missing_by_cheapest(y1, y2, missing)
            extra = [x for x in y1 + y2 if x not in target]
            if extra and not include_special:
                y1 = [x for x in y1 if x in target]
                y2 = [x for x in y2 if x in target]
            if y1 and y2:
                for policy in ["rooted_urumqi", "open_gateway"]:
                    seeds.append(
                        {
                            "seed_id": f"{seed_id}_{policy}",
                            "gateway_policy": policy,
                            "year1_seq": y1,
                            "year2_seq": y2,
                            "generation_method": method,
                            "include_special": include_special,
                        }
                    )

        if not self.amap_baseline.empty:
            y1 = []
            y2 = []
            approval = []
            for row in self.amap_baseline.itertuples(index=False):
                rid = clean_text(row.route_id)
                seq = self.parse_sequence(row.route_sequence)
                if "Y1" in rid:
                    y1 = seq
                elif "Approval" in rid:
                    approval = seq
                else:
                    y2 = seq
            add_seed("AMAP_BASELINE", y1, y2, "amap_selfdrive_baseline")
            if y1 and approval:
                add_seed("APPROVAL_EXTENSION", y1, approval, "amap_approval_extension", include_special=True)

        if not self.graph_baseline.empty:
            y1 = []
            y2 = []
            approval = []
            for row in self.graph_baseline.itertuples(index=False):
                rid = clean_text(row.route_id)
                seq = self.parse_sequence(row.route_sequence)
                if "Y1" in rid:
                    y1 = seq
                elif "Approval" in rid:
                    approval = seq
                else:
                    y2 = seq
            add_seed("GRAPH_BASELINE", y1, y2, "early_graph_baseline")
            if y1 and approval:
                add_seed("GRAPH_APPROVAL", y1, approval, "early_graph_approval_extension", include_special=True)

        if not self.spectral_baseline.empty:
            rows = list(self.spectral_baseline.itertuples(index=False))
            if len(rows) >= 2:
                add_seed(
                    "SPECTRAL_REFERENCE",
                    self.parse_sequence(rows[0].route_sequence),
                    self.parse_sequence(rows[1].route_sequence),
                    "spectral_reference_not_final",
                )

        north = self.ordinary_spots[
            self.ordinary_spots["region_cluster"].isin(["北疆", "伊犁", "乌鲁木齐周边"])
        ]["spot_id"].tolist()
        south = [x for x in self.ordinary_ids if x not in north]
        add_seed("REGION_NORTH_SOUTH", north, south, "region_aware_cost_seed")

        grand = self.greedy_grand_tour(self.ordinary_ids, start_gateway="乌鲁木齐市")
        for cut in range(int(self.config["min_spots_per_year"]), int(self.config["max_spots_per_year"]) + 1):
            add_seed(f"GRAND_SPLIT_{cut:02d}", grand[:cut], grand[cut:], "grand_tour_split")

        for idx in range(28):
            shuffled = self.random_region_order()
            cut = int(self.rng.integers(int(self.config["min_spots_per_year"]), int(self.config["max_spots_per_year"]) + 1))
            add_seed(f"RANDOM_RESTART_{idx:02d}", shuffled[:cut], shuffled[cut:], "random_region_restart")

        return seeds

    def unique_keep_order(self, seq: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def insert_missing_by_cheapest(self, y1: list[str], y2: list[str], missing: list[str]) -> tuple[list[str], list[str]]:
        for spot_id in missing:
            candidates = []
            for year_idx, route in enumerate([y1, y2]):
                best_route, delta = self.best_insert(route, spot_id, "rooted_urumqi")
                candidates.append((delta + len(route) * 4, year_idx, best_route))
            _, year_idx, route2 = min(candidates, key=lambda x: x[0])
            if year_idx == 0:
                y1 = route2
            else:
                y2 = route2
        return y1, y2

    def greedy_grand_tour(self, ids: list[str], start_gateway: str) -> list[str]:
        remaining = set(ids)
        current_gateway = start_gateway
        current_spot = None
        route: list[str] = []
        while remaining:
            if current_spot is None:
                next_id = min(remaining, key=lambda x: self.gateway_label(current_gateway, "gateway_to_spot", x).cost_yuan_for_two)
            else:
                next_id = min(remaining, key=lambda x: self.best_label(current_spot, x).cost_yuan_for_two)
            route.append(next_id)
            remaining.remove(next_id)
            current_spot = next_id
        return route

    def random_region_order(self) -> list[str]:
        regions = list(self.ordinary_spots["region_cluster"].drop_duplicates())
        self.rng.shuffle(regions)
        seq = []
        for region in regions:
            sub = self.ordinary_spots[self.ordinary_spots["region_cluster"].eq(region)]["spot_id"].tolist()
            self.rng.shuffle(sub)
            seq.extend(sub)
        return seq

    def best_insert(self, route: list[str], spot_id: str, policy: str) -> tuple[list[str], float]:
        base = self.evaluate_route(route, policy, year=1, collect_segments=False)["route_score"]
        best = (route + [spot_id], float("inf"))
        for pos in range(len(route) + 1):
            cand = route[:pos] + [spot_id] + route[pos:]
            score = self.evaluate_route(cand, policy, year=1, collect_segments=False)["route_score"]
            delta = score - base
            if delta < best[1]:
                best = (cand, delta)
        return best

    def optimize_candidate(self, seed: dict[str, Any]) -> dict[str, Any]:
        y1 = list(seed["year1_seq"])
        y2 = list(seed["year2_seq"])
        policy = seed["gateway_policy"]
        include_special = bool(seed.get("include_special", False))
        y1 = self.improve_route_order(y1, policy)
        y2 = self.improve_route_order(y2, policy)
        best_score = self.plan_score(y1, y2, policy, include_special)
        for _ in range(3):
            improved = False
            for source_year in [1, 2]:
                src = y1 if source_year == 1 else y2
                dst = y2 if source_year == 1 else y1
                if len(src) <= int(self.config["min_spots_per_year"]) and not include_special:
                    continue
                for spot_id in list(src):
                    src_removed = [x for x in src if x != spot_id]
                    if not include_special and len(dst) >= int(self.config["max_spots_per_year"]):
                        continue
                    new_dst, _ = self.best_insert(dst, spot_id, policy)
                    cand_y1, cand_y2 = (src_removed, new_dst) if source_year == 1 else (new_dst, src_removed)
                    cand_y1 = self.improve_route_order(cand_y1, policy, max_rounds=1)
                    cand_y2 = self.improve_route_order(cand_y2, policy, max_rounds=1)
                    score = self.plan_score(cand_y1, cand_y2, policy, include_special)
                    if score + 1e-6 < best_score:
                        y1, y2, best_score = cand_y1, cand_y2, score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            for i, a in enumerate(y1):
                for j, b in enumerate(y2):
                    cand_y1 = y1.copy()
                    cand_y2 = y2.copy()
                    cand_y1[i] = b
                    cand_y2[j] = a
                    cand_y1 = self.improve_route_order(cand_y1, policy, max_rounds=1)
                    cand_y2 = self.improve_route_order(cand_y2, policy, max_rounds=1)
                    score = self.plan_score(cand_y1, cand_y2, policy, include_special)
                    if score + 1e-6 < best_score:
                        y1, y2, best_score = cand_y1, cand_y2, score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        seed = seed.copy()
        seed["year1_seq"] = y1
        seed["year2_seq"] = y2
        return seed

    def improve_route_order(self, route: list[str], policy: str, max_rounds: int = 2) -> list[str]:
        best = route[:]
        for _ in range(max_rounds):
            before = self.evaluate_route(best, policy, year=1, collect_segments=False)["route_score"]
            best = self.two_opt(best, policy, max_passes=1)
            best = self.relocate_within_route(best, policy, max_passes=1)
            after = self.evaluate_route(best, policy, year=1, collect_segments=False)["route_score"]
            if after + 1e-6 >= before:
                break
        return best

    def two_opt(self, route: list[str], policy: str, max_passes: int = 2) -> list[str]:
        if len(route) < 4:
            return route
        best = route[:]
        best_cost = self.evaluate_route(best, policy, year=1, collect_segments=False)["route_score"]
        for _ in range(max_passes):
            improved = False
            for i in range(0, len(best) - 1):
                for j in range(i + 2, len(best) + 1):
                    cand = best[:i] + best[i:j][::-1] + best[j:]
                    score = self.evaluate_route(cand, policy, year=1, collect_segments=False)["route_score"]
                    if score + 1e-6 < best_cost:
                        best, best_cost = cand, score
                        improved = True
            if not improved:
                break
        return best

    def relocate_within_route(self, route: list[str], policy: str, max_passes: int = 1) -> list[str]:
        if len(route) < 3:
            return route
        best = route[:]
        best_cost = self.evaluate_route(best, policy, year=1, collect_segments=False)["route_score"]
        for _ in range(max_passes):
            improved = False
            for i, node in enumerate(best):
                reduced = best[:i] + best[i + 1 :]
                for pos in range(len(reduced) + 1):
                    cand = reduced[:pos] + [node] + reduced[pos:]
                    if cand == best:
                        continue
                    score = self.evaluate_route(cand, policy, year=1, collect_segments=False)["route_score"]
                    if score + 1e-6 < best_cost:
                        best, best_cost = cand, score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def plan_score(self, y1: list[str], y2: list[str], policy: str, include_special: bool = False) -> float:
        r1 = self.evaluate_route(y1, policy, 1, collect_segments=False)
        r2 = self.evaluate_route(y2, policy, 2, collect_segments=False)
        target_count = len(self.all_spot_ids if include_special else self.ordinary_ids)
        covered = len(set(y1 + y2))
        days_penalty = 5000 * max(0, r1["estimated_days"] - 30) + 5000 * max(0, r2["estimated_days"] - 30)
        balance_penalty = 60 * abs(r1["estimated_days"] - r2["estimated_days"])
        count_penalty = 1500 * max(0, target_count - covered)
        red_penalty = 120 * (r1["red_days"] + r2["red_days"])
        return r1["transport_cost_yuan_for_two"] + r2["transport_cost_yuan_for_two"] + days_penalty + balance_penalty + count_penalty + red_penalty

    def evaluate_route(
        self, seq: list[str], policy: str, year: int, collect_segments: bool = True
    ) -> dict[str, Any]:
        if not seq:
            return {}
        internal_labels = [self.best_label(a, b) for a, b in zip(seq[:-1], seq[1:])]
        if policy == "rooted_urumqi":
            entry_gateway = exit_gateway = "乌鲁木齐市"
            entry = self.gateway_label(entry_gateway, "gateway_to_spot", seq[0])
            exit_label = self.gateway_label(exit_gateway, "spot_to_gateway", seq[-1])
        else:
            best = None
            internal_cost = sum(l.cost_yuan_for_two for l in internal_labels)
            for g_in in self.gateway_names:
                entry_l = self.gateway_label(g_in, "gateway_to_spot", seq[0])
                for g_out in self.gateway_names:
                    exit_l = self.gateway_label(g_out, "spot_to_gateway", seq[-1])
                    cost = internal_cost + entry_l.cost_yuan_for_two + exit_l.cost_yuan_for_two
                    if best is None or cost < best[0]:
                        best = (cost, g_in, g_out, entry_l, exit_l)
            assert best is not None
            entry_gateway, exit_gateway, entry, exit_label = best[1], best[2], best[3], best[4]
        labels = [entry] + internal_labels + [exit_label]
        service = sum(num(self.spot_by_id[x].get("visit_hours_mid", 3.0), 3.0) for x in seq)
        travel = sum(l.time_hours for l in labels)
        cost = sum(l.cost_yuan_for_two for l in labels)
        risk = sum(l.risk_score for l in labels)
        fatigue = sum(l.fatigue_score for l in labels)
        long_edges = sum(1 for l in labels if l.time_hours >= 5.0)
        very_long_edges = sum(1 for l in labels if l.time_hours >= 7.5)
        active_hours = travel + service + long_edges * 0.35
        estimated_days = int(math.ceil(active_hours / float(self.config["daily_effective_hours"])))
        red_days = int(very_long_edges + max(0, estimated_days - 26))
        yellow_days = int(max(0, long_edges - red_days))
        time_window_violations = int(sum(1 for l in labels if l.time_hours >= 8.5))
        max_day_active = round(max([l.time_hours for l in labels] + [active_hours / max(1, estimated_days)]) + 1.5, 3)
        schedule_soft_feasible = estimated_days <= 30 and time_window_violations <= 2
        schedule_strict_feasible = estimated_days <= 30 and time_window_violations == 0 and red_days <= 3
        mode_hours = defaultdict(float)
        mode_cost = defaultdict(float)
        for label in labels:
            mode_hours[label.mode] += label.time_hours
            mode_cost[label.mode] += label.cost_yuan_for_two
        segments = []
        if collect_segments:
            ordered_labels = labels
            segment_from = [entry_gateway] + seq[:-1] + [seq[-1]]
            segment_to = [seq[0]] + seq[1:] + [exit_gateway]
            for idx, label in enumerate(ordered_labels, start=1):
                segments.append(
                    {
                        "year": year,
                        "segment_order": idx,
                        "from_node": segment_from[idx - 1],
                        "from_name": self.spot_name(segment_from[idx - 1]) if segment_from[idx - 1].startswith("P") else segment_from[idx - 1],
                        "to_node": segment_to[idx - 1],
                        "to_name": self.spot_name(segment_to[idx - 1]) if segment_to[idx - 1].startswith("P") else segment_to[idx - 1],
                        "label_id": label.label_id,
                        "mode": label.mode,
                        "mode_combo": label.mode_combo,
                        "time_hours": round(label.time_hours, 3),
                        "cost_yuan_for_two": round(label.cost_yuan_for_two, 2),
                        "raw_mode": label.raw_mode,
                        "raw_cost_yuan_for_two": round(label.raw_cost_yuan_for_two, 2),
                        "cost_calibration_note": label.cost_calibration_note,
                        "risk_score": round(label.risk_score, 3),
                        "fatigue_score": round(label.fatigue_score, 3),
                        "path_desc": label.path_desc,
                        "schedule_required": label.schedule_required,
                    }
                )
        return {
            "entry_gateway": entry_gateway,
            "exit_gateway": exit_gateway,
            "spots_count": len(seq),
            "spot_id_sequence": " -> ".join(seq),
            "route_sequence": self.route_names(seq),
            "transport_cost_yuan_for_two": round(cost, 2),
            "travel_hours": round(travel, 3),
            "service_hours": round(service, 3),
            "active_hours": round(active_hours, 3),
            "estimated_days": estimated_days,
            "long_transfer_edges": long_edges,
            "red_days": red_days,
            "yellow_days": yellow_days,
            "time_window_violations": time_window_violations,
            "max_day_active_hours": max_day_active,
            "mean_day_active_hours": round(active_hours / max(1, estimated_days), 3),
            "risk_score": round(risk, 3),
            "fatigue_score": round(fatigue, 3),
            "mode_hours_mix": json.dumps({k: round(v, 3) for k, v in sorted(mode_hours.items())}, ensure_ascii=False),
            "mode_cost_mix": json.dumps({k: round(v, 2) for k, v in sorted(mode_cost.items())}, ensure_ascii=False),
            "schedule_soft_feasible": schedule_soft_feasible,
            "schedule_strict_feasible": schedule_strict_feasible,
            "route_score": cost + max(0, estimated_days - 30) * 5000 + red_days * 120 + time_window_violations * 700,
            "segments": segments,
        }

    def evaluate_plan(self, seed: dict[str, Any], plan_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        y1 = seed["year1_seq"]
        y2 = seed["year2_seq"]
        policy = seed["gateway_policy"]
        r1 = self.evaluate_route(y1, policy, 1)
        r2 = self.evaluate_route(y2, policy, 2)
        covered = set(y1 + y2)
        target = set(self.all_spot_ids if seed.get("include_special") else self.ordinary_ids)
        missing = target - covered
        duplicate_count = len(y1 + y2) - len(covered)
        total_cost = r1["transport_cost_yuan_for_two"] + r2["transport_cost_yuan_for_two"]
        max_days = max(r1["estimated_days"], r2["estimated_days"])
        row = {
            "plan_id": plan_id,
            "gateway_policy": policy,
            "generation_method": seed["generation_method"],
            "include_special": bool(seed.get("include_special", False)),
            "covered_spots": len(covered & target),
            "target_spots": len(target),
            "coverage_complete": len(missing) == 0,
            "missing_spots": "、".join(self.spot_name(x) for x in sorted(missing)),
            "duplicate_spots_count": duplicate_count,
            "year1_entry_gateway": r1["entry_gateway"],
            "year1_exit_gateway": r1["exit_gateway"],
            "year2_entry_gateway": r2["entry_gateway"],
            "year2_exit_gateway": r2["exit_gateway"],
            "year1_spots": r1["spots_count"],
            "year2_spots": r2["spots_count"],
            "year1_days": r1["estimated_days"],
            "year2_days": r2["estimated_days"],
            "max_year_days": max_days,
            "year_day_difference": abs(r1["estimated_days"] - r2["estimated_days"]),
            "year1_transport_cost": r1["transport_cost_yuan_for_two"],
            "year2_transport_cost": r2["transport_cost_yuan_for_two"],
            "total_intra_transport_cost": round(total_cost, 2),
            "total_travel_hours": round(r1["travel_hours"] + r2["travel_hours"], 3),
            "total_service_hours": round(r1["service_hours"] + r2["service_hours"], 3),
            "red_days": r1["red_days"] + r2["red_days"],
            "long_transfer_edges": r1["long_transfer_edges"] + r2["long_transfer_edges"],
            "time_window_violations": r1["time_window_violations"] + r2["time_window_violations"],
            "schedule_soft_feasible": bool(r1["schedule_soft_feasible"] and r2["schedule_soft_feasible"]),
            "schedule_strict_feasible": bool(r1["schedule_strict_feasible"] and r2["schedule_strict_feasible"]),
            "year1_route_sequence": r1["route_sequence"],
            "year2_route_sequence": r2["route_sequence"],
            "year1_spot_id_sequence": r1["spot_id_sequence"],
            "year2_spot_id_sequence": r2["spot_id_sequence"],
            "year1_mode_hours_mix": r1["mode_hours_mix"],
            "year2_mode_hours_mix": r2["mode_hours_mix"],
            "proved_optimal": False,
            "solver_status": "matheuristic_seed_2opt_relocate_swap",
            "cost_calibration_version": self.config["cost_calibration_version"],
        }
        year_rows = []
        for year, r in [(1, r1), (2, r2)]:
            year_rows.append(
                {
                    "plan_id": plan_id,
                    "year": year,
                    "gateway_policy": policy,
                    "entry_gateway": r["entry_gateway"],
                    "exit_gateway": r["exit_gateway"],
                    "spots_count": r["spots_count"],
                    "estimated_days": r["estimated_days"],
                    "transport_cost_yuan_for_two": r["transport_cost_yuan_for_two"],
                    "travel_hours": r["travel_hours"],
                    "service_hours": r["service_hours"],
                    "active_hours": r["active_hours"],
                    "red_days": r["red_days"],
                    "yellow_days": r["yellow_days"],
                    "long_transfer_edges": r["long_transfer_edges"],
                    "time_window_violations": r["time_window_violations"],
                    "max_day_active_hours": r["max_day_active_hours"],
                    "mean_day_active_hours": r["mean_day_active_hours"],
                    "risk_score": r["risk_score"],
                    "fatigue_score": r["fatigue_score"],
                    "schedule_soft_feasible": r["schedule_soft_feasible"],
                    "schedule_strict_feasible": r["schedule_strict_feasible"],
                    "spot_id_sequence": r["spot_id_sequence"],
                    "route_sequence": r["route_sequence"],
                    "mode_hours_mix": r["mode_hours_mix"],
                    "mode_cost_mix": r["mode_cost_mix"],
                }
            )
        segments = []
        for r in [r1, r2]:
            for seg in r["segments"]:
                seg["plan_id"] = plan_id
                segments.append(seg)
        return row, year_rows, segments

    def build_candidate_plans(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seeds = self.generate_seed_plans()
        rows = []
        year_rows = []
        segment_rows = []
        seen = set()
        idx = 0
        for seed in seeds:
            optimized = self.optimize_candidate(seed)
            key = (
                optimized["gateway_policy"],
                tuple(optimized["year1_seq"]),
                tuple(optimized["year2_seq"]),
                bool(optimized.get("include_special")),
            )
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            plan_id = f"Q2V3_{idx:03d}_{optimized['gateway_policy'].upper()}"
            row, yrs, segs = self.evaluate_plan(optimized, plan_id)
            rows.append(row)
            year_rows.extend(yrs)
            segment_rows.extend(segs)
        candidates = pd.DataFrame(rows)
        years = pd.DataFrame(year_rows)
        segments = pd.DataFrame(segment_rows)
        candidates = candidates.sort_values(
            ["include_special", "gateway_policy", "total_intra_transport_cost", "max_year_days"]
        ).reset_index(drop=True)
        self.write_csv(candidates, self.outputs / "q2_v3_candidate_plans.csv")
        self.write_csv(years, self.outputs / "q2_v3_year_route_summary.csv")
        self.write_csv(segments, self.outputs / "q2_v3_route_segments.csv")
        self.candidate_plans = candidates
        self.year_routes = years
        self.route_segments = segments
        return candidates, years, segments

    # ---------- Schedule and simulation ----------

    def build_schedule_summary(self) -> pd.DataFrame:
        rows = []
        for row in self.candidate_plans.itertuples(index=False):
            rows.append(
                {
                    "plan_id": row.plan_id,
                    "gateway_policy": row.gateway_policy,
                    "year1_days": row.year1_days,
                    "year2_days": row.year2_days,
                    "max_year_days": row.max_year_days,
                    "year_day_difference": row.year_day_difference,
                    "red_days": row.red_days,
                    "long_transfer_edges": row.long_transfer_edges,
                    "time_window_violations": row.time_window_violations,
                    "schedule_soft_feasible": row.schedule_soft_feasible,
                    "schedule_strict_feasible": row.schedule_strict_feasible,
                    "schedule_note": self.schedule_note(row),
                }
            )
        df = pd.DataFrame(rows)
        self.write_csv(df, self.outputs / "q2_v3_schedule_summary.csv")
        self.schedule_summary = df
        return df

    def schedule_note(self, row: Any) -> str:
        notes = []
        if row.max_year_days > 30:
            notes.append("年度天数超过30天")
        if row.time_window_violations > 0:
            notes.append("存在长转场导致的时间窗压力")
        if row.red_days > 0:
            notes.append("存在红色压力转场日")
        if row.year_day_difference > self.config["max_year_day_difference"]:
            notes.append("两年负担差异偏大")
        return "；".join(notes) if notes else "年度排程软硬约束均通过"

    def simulate_plans(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        trials = []
        summary = []
        samples = int(self.config["scenario_samples"])
        rooted_min = self.rooted_baseline_cost()
        scenario_probs = [
            ("normal_summer", 0.45),
            ("peak_summer", 0.25),
            ("rain_flood_delay", 0.12),
            ("heatwave", 0.10),
            ("hotel_tight", 0.05),
            ("compound_extreme", 0.03),
        ]
        names = [x[0] for x in scenario_probs]
        probs = np.array([x[1] for x in scenario_probs], dtype=float)
        probs = probs / probs.sum()
        for plan in self.candidate_plans.itertuples(index=False):
            losses = []
            op_success = []
            strict_success = []
            costs = []
            max_days_list = []
            open_cheaper = []
            threshold = rooted_min - plan.total_intra_transport_cost if plan.gateway_policy == "open_gateway" else 0.0
            for trial_id in range(1, samples + 1):
                scenario = str(self.rng.choice(names, p=probs))
                factors = self.scenario_factors(scenario)
                cost_factor = factors["cost_factor"]
                time_factor = factors["time_factor"]
                red_add = int(self.rng.poisson(factors["red_lambda"]))
                delay_events = int(self.rng.binomial(3, factors["major_delay_prob"]))
                y1_days = int(math.ceil(plan.year1_days * time_factor + delay_events * 0.35))
                y2_days = int(math.ceil(plan.year2_days * time_factor + delay_events * 0.25))
                sim_cost = round(plan.total_intra_transport_cost * cost_factor, 2)
                ext_premium = self.external_premium_sample(scenario) if plan.gateway_policy == "open_gateway" else 0.0
                red_days = int(plan.red_days + red_add)
                max_days = max(y1_days, y2_days)
                operational = (
                    y1_days <= 30
                    and y2_days <= 30
                    and red_days <= 5
                    and delay_events <= 2
                    and plan.coverage_complete
                )
                strict = operational and red_days <= 2 and plan.time_window_violations == 0
                loss = (
                    900 * max(0, y1_days - 30)
                    + 900 * max(0, y2_days - 30)
                    + 0.35 * max(0, sim_cost - plan.total_intra_transport_cost)
                    + 160 * red_days
                    + 260 * delay_events
                    + max(0, ext_premium - max(0, threshold)) * 0.4
                )
                trials.append(
                    {
                        "plan_id": plan.plan_id,
                        "trial_id": trial_id,
                        "scenario": scenario,
                        "sim_year1_days": y1_days,
                        "sim_year2_days": y2_days,
                        "sim_max_year_days": max_days,
                        "sim_intra_transport_cost": sim_cost,
                        "external_gateway_premium": round(ext_premium, 2),
                        "red_days": red_days,
                        "major_delay_events": delay_events,
                        "operational_success": operational,
                        "strict_comfort_success": strict,
                        "open_gateway_cheaper": bool(ext_premium < threshold) if plan.gateway_policy == "open_gateway" else False,
                        "loss": round(loss, 3),
                    }
                )
                losses.append(loss)
                op_success.append(operational)
                strict_success.append(strict)
                costs.append(sim_cost)
                max_days_list.append(max_days)
                if plan.gateway_policy == "open_gateway":
                    open_cheaper.append(ext_premium < threshold)
            losses_arr = np.array(losses)
            q75 = float(np.quantile(losses_arr, 0.75))
            q90 = float(np.quantile(losses_arr, 0.90))
            cvar75 = float(losses_arr[losses_arr >= q75].mean()) if len(losses_arr) else 0.0
            cvar90 = float(losses_arr[losses_arr >= q90].mean()) if len(losses_arr) else 0.0
            summary.append(
                {
                    "plan_id": plan.plan_id,
                    "simulation_samples": samples,
                    "operational_success_probability": round(float(np.mean(op_success)), 4),
                    "strict_comfort_success_probability": round(float(np.mean(strict_success)), 4),
                    "expected_intra_transport_cost": round(float(np.mean(costs)), 2),
                    "p95_intra_transport_cost": round(float(np.quantile(costs, 0.95)), 2),
                    "expected_max_year_days": round(float(np.mean(max_days_list)), 3),
                    "p95_max_year_days": round(float(np.quantile(max_days_list, 0.95)), 3),
                    "expected_loss": round(float(np.mean(losses_arr)), 3),
                    "cvar75_loss": round(cvar75, 3),
                    "cvar90_loss": round(cvar90, 3),
                    "break_even_external_premium": round(max(0.0, threshold), 2),
                    "open_gateway_cheaper_probability": round(float(np.mean(open_cheaper)), 4) if open_cheaper else 0.0,
                    "simulation_note": "route-specific two-year Monte Carlo; external premium only applies to open gateway plans",
                }
            )
        trials_df = pd.DataFrame(trials)
        summary_df = pd.DataFrame(summary)
        self.write_csv(trials_df, self.outputs / "q2_v3_simulation_trials.csv")
        self.write_csv(summary_df, self.outputs / "q2_v3_simulation_summary.csv")
        self.simulation_trials = trials_df
        self.simulation_summary = summary_df
        return summary_df, trials_df

    def scenario_factors(self, scenario: str) -> dict[str, float]:
        table = {
            "normal_summer": (1.00, 1.00, 0.15, 0.03),
            "peak_summer": (1.08, 1.16, 0.50, 0.06),
            "rain_flood_delay": (1.16, 1.05, 0.65, 0.12),
            "heatwave": (1.07, 1.04, 0.60, 0.05),
            "hotel_tight": (1.03, 1.03, 0.25, 0.04),
            "compound_extreme": (1.24, 1.22, 1.10, 0.16),
        }
        time_factor, cost_factor, red_lambda, delay_prob = table[scenario]
        return {
            "time_factor": float(self.rng.normal(time_factor, 0.035)),
            "cost_factor": float(self.rng.normal(cost_factor, 0.04)),
            "red_lambda": red_lambda,
            "major_delay_prob": delay_prob,
        }

    def external_premium_sample(self, scenario: str) -> float:
        if scenario == "normal_summer":
            return float(self.rng.triangular(0, 800, 2200))
        if scenario in {"peak_summer", "compound_extreme"}:
            return float(self.rng.triangular(300, 1500, 3800))
        return float(self.rng.triangular(100, 1000, 2800))

    def rooted_baseline_cost(self) -> float:
        rooted = self.candidate_plans[
            (~self.candidate_plans["include_special"]) & self.candidate_plans["gateway_policy"].eq("rooted_urumqi")
        ]
        if rooted.empty:
            return float(self.candidate_plans["total_intra_transport_cost"].min())
        return float(rooted["total_intra_transport_cost"].min())

    # ---------- Selection and checks ----------

    def build_gateway_thresholds(self) -> pd.DataFrame:
        regular = self.candidate_plans[~self.candidate_plans["include_special"]]
        rooted = regular[regular["gateway_policy"].eq("rooted_urumqi")].sort_values("total_intra_transport_cost").head(1)
        opened = regular[regular["gateway_policy"].eq("open_gateway")].sort_values("total_intra_transport_cost").head(1)
        rows = []
        if not rooted.empty and not opened.empty:
            r = rooted.iloc[0]
            o = opened.iloc[0]
            threshold = float(r["total_intra_transport_cost"]) - float(o["total_intra_transport_cost"])
            rows.append(
                {
                    "comparison": "open_gateway_vs_rooted_urumqi",
                    "rooted_plan_id": r["plan_id"],
                    "open_gateway_plan_id": o["plan_id"],
                    "rooted_cost_yuan_for_two": round(float(r["total_intra_transport_cost"]), 2),
                    "open_gateway_cost_yuan_for_two": round(float(o["total_intra_transport_cost"]), 2),
                    "break_even_external_premium_yuan_for_two": round(threshold, 2),
                    "decision_rule": "若两人多口岸外部大交通额外差价低于该阈值，则开放式多口岸总费用更优"
                    if threshold > 0
                    else "当前数据下开放式境内费用未低于乌鲁木齐起讫，优先采用保守方案",
                }
            )
        df = pd.DataFrame(rows)
        self.write_csv(df, self.outputs / "q2_v3_gateway_thresholds.csv")
        self.gateway_thresholds = df
        return df

    def select_representative_plans(self) -> pd.DataFrame:
        df = self.candidate_plans.merge(self.simulation_summary, on="plan_id", how="left")
        regular = df[~df["include_special"]].copy()
        selected_rows = []

        def add(role: str, row: pd.Series, status: str) -> None:
            if row is None or row.empty:
                return
            r = row.to_dict()
            r["selected_role"] = role
            r["selection_status"] = status
            r["external_premium_break_even"] = r.get("break_even_external_premium", 0.0)
            selected_rows.append(r)

        rooted = regular[regular["gateway_policy"].eq("rooted_urumqi")].sort_values(
            ["schedule_strict_feasible", "total_intra_transport_cost"], ascending=[False, True]
        )
        if not rooted.empty:
            add("ROOTED_URUMQI_MAIN", rooted.iloc[0], "criteria_pass" if rooted.iloc[0]["schedule_strict_feasible"] else "soft_candidate")

        opened = regular[regular["gateway_policy"].eq("open_gateway")].sort_values("total_intra_transport_cost")
        if not opened.empty:
            add("OPEN_GATEWAY_MIN_INTRA_COST", opened.iloc[0], "intra_xj_cost_lower_bound")

        if not opened.empty:
            policy = opened.sort_values(
                ["open_gateway_cheaper_probability", "operational_success_probability", "total_intra_transport_cost"],
                ascending=[False, False, True],
            ).iloc[0]
            add("OPEN_GATEWAY_THRESHOLD_POLICY", policy, "conditional_on_external_premium")

        min_cost = float(regular["total_intra_transport_cost"].min()) if not regular.empty else 0.0
        balanced_regular = regular[
            regular["schedule_strict_feasible"].astype(bool)
            & regular["year_day_difference"].le(int(self.config["max_year_day_difference"]))
            & regular["operational_success_probability"].ge(float(self.config["success_threshold"]))
        ]
        robust_pool = balanced_regular[
            balanced_regular["total_intra_transport_cost"].le(min_cost * (1 + float(self.config["cost_gap_for_robust_choice"])))
        ]
        if robust_pool.empty:
            robust_pool = balanced_regular
        if robust_pool.empty:
            robust_pool = regular[regular["schedule_soft_feasible"].astype(bool)]
        if robust_pool.empty:
            robust_pool = regular
        robust = robust_pool.sort_values(
            [
                "operational_success_probability",
                "strict_comfort_success_probability",
                "cvar75_loss",
                "total_intra_transport_cost",
            ],
            ascending=[False, False, True, True],
        )
        if not robust.empty:
            add("ROBUST_TWO_YEAR_MAIN", robust.iloc[0], "operational_robust_balanced_choice")

        comfort = regular[regular["schedule_soft_feasible"].astype(bool)].sort_values(
            ["red_days", "year_day_difference", "max_year_days", "total_intra_transport_cost"]
        )
        if not comfort.empty:
            add("COMFORT_BALANCED_BACKUP", comfort.iloc[0], "comfort_balance_backup")

        approval = df[df["include_special"]].sort_values(["max_year_days", "total_intra_transport_cost"])
        if not approval.empty:
            add("APPROVAL_EXTENSION_OPTIONAL", approval.iloc[0], "approval_required_not_main_route")

        selected = pd.DataFrame(selected_rows)
        cols = [
            "selected_role",
            "plan_id",
            "gateway_policy",
            "year1_entry_gateway",
            "year1_exit_gateway",
            "year2_entry_gateway",
            "year2_exit_gateway",
            "covered_spots",
            "year1_spots",
            "year2_spots",
            "year1_days",
            "year2_days",
            "max_year_days",
            "year_day_difference",
            "year1_transport_cost",
            "year2_transport_cost",
            "total_intra_transport_cost",
            "external_premium_break_even",
            "operational_success_probability",
            "strict_comfort_success_probability",
            "cvar75_loss",
            "red_days",
            "time_window_violations",
            "schedule_soft_feasible",
            "schedule_strict_feasible",
            "proved_optimal",
            "solver_status",
            "selection_status",
            "cost_calibration_version",
            "year1_mode_hours_mix",
            "year2_mode_hours_mix",
            "year1_route_sequence",
            "year2_route_sequence",
        ]
        selected = selected[[c for c in cols if c in selected.columns]]
        self.write_csv(selected, self.outputs / "q2_v3_selected_plans.csv")
        self.selected_plans = selected
        return selected

    def build_pareto_front(self) -> pd.DataFrame:
        df = self.candidate_plans.merge(self.simulation_summary, on="plan_id", how="left")
        df = df[~df["include_special"]].copy()
        front = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                better_or_equal = (
                    other["total_intra_transport_cost"] <= row["total_intra_transport_cost"]
                    and other["max_year_days"] <= row["max_year_days"]
                    and other["cvar75_loss"] <= row["cvar75_loss"]
                    and other["operational_success_probability"] >= row["operational_success_probability"]
                )
                strictly = (
                    other["total_intra_transport_cost"] < row["total_intra_transport_cost"]
                    or other["max_year_days"] < row["max_year_days"]
                    or other["cvar75_loss"] < row["cvar75_loss"]
                    or other["operational_success_probability"] > row["operational_success_probability"]
                )
                if better_or_equal and strictly:
                    dominated = True
                    break
            if not dominated:
                front.append(row.to_dict())
        out = pd.DataFrame(front).sort_values(["total_intra_transport_cost", "max_year_days"])
        self.write_csv(out, self.outputs / "q2_v3_feasible_pareto_front.csv")
        self.pareto_front = out
        return out

    def held_karp_path(
        self, nodes: list[str], start_gateway: str, end_gateway: str, end_spot: str | None = None
    ) -> float:
        n = len(nodes)
        if n == 0:
            return 0.0
        start_cost = [self.gateway_label(start_gateway, "gateway_to_spot", x).cost_yuan_for_two for x in nodes]
        if end_spot:
            end_cost = [self.best_label(x, end_spot).cost_yuan_for_two for x in nodes]
        else:
            end_cost = [self.gateway_label(end_gateway, "spot_to_gateway", x).cost_yuan_for_two for x in nodes]
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = self.best_label(nodes[i], nodes[j]).cost_yuan_for_two
        dp: dict[tuple[int, int], float] = {}
        for i in range(n):
            dp[(1 << i, i)] = start_cost[i]
        full = (1 << n) - 1
        for mask in range(1, full + 1):
            for j in range(n):
                if not (mask & (1 << j)) or (mask, j) not in dp:
                    continue
                val = dp[(mask, j)]
                remain = full ^ mask
                kbits = remain
                while kbits:
                    lsb = kbits & -kbits
                    k = lsb.bit_length() - 1
                    new = mask | lsb
                    key = (new, k)
                    cand = val + dist[j][k]
                    if cand < dp.get(key, float("inf")):
                        dp[key] = cand
                    kbits -= lsb
        return min(dp[(full, j)] + end_cost[j] for j in range(n))

    def route_cost_for_subset(
        self, seq: list[str], start_gateway: str, end_gateway: str, end_spot: str | None = None
    ) -> float:
        if not seq:
            return 0.0
        cost = self.gateway_label(start_gateway, "gateway_to_spot", seq[0]).cost_yuan_for_two
        cost += sum(self.best_label(a, b).cost_yuan_for_two for a, b in zip(seq[:-1], seq[1:]))
        if end_spot:
            cost += self.best_label(seq[-1], end_spot).cost_yuan_for_two
        else:
            cost += self.gateway_label(end_gateway, "spot_to_gateway", seq[-1]).cost_yuan_for_two
        return cost

    def build_exact_checks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        checks = []
        selected = self.selected_plans.head(4)
        for row in selected.itertuples(index=False):
            for year in [1, 2]:
                seq_text = getattr(row, f"year{year}_route_sequence")
                seq = self.parse_sequence(seq_text)
                if len(seq) < 6:
                    continue
                subset = seq[: min(15, len(seq))]
                end_spot = seq[len(subset)] if len(seq) > len(subset) else None
                start_gateway = getattr(row, f"year{year}_entry_gateway")
                end_gateway = getattr(row, f"year{year}_exit_gateway")
                heuristic = self.route_cost_for_subset(subset, start_gateway, end_gateway, end_spot=end_spot)
                exact = self.held_karp_path(subset, start_gateway, end_gateway, end_spot=end_spot)
                gap = (heuristic - exact) / exact if exact > 0 else 0.0
                checks.append(
                    {
                        "check_id": f"{row.plan_id}_Y{year}_prefix{len(subset)}",
                        "plan_id": row.plan_id,
                        "selected_role": row.selected_role,
                        "year": year,
                        "node_count": len(subset),
                        "gateway_policy": row.gateway_policy,
                        "heuristic_cost": round(heuristic, 2),
                        "exact_held_karp_cost": round(exact, 2),
                        "relative_gap": round(gap, 4),
                        "status": "completed_fixed_subset_order_check",
                        "end_anchor_type": "next_spot" if end_spot else "exit_gateway",
                        "end_anchor": self.spot_name(end_spot) if end_spot else end_gateway,
                    }
                )
        exact_df = pd.DataFrame(checks)
        self.write_csv(exact_df, self.outputs / "q2_v3_exact_order_check.csv")
        mip_df = pd.DataFrame(
            [
                {
                    "check_id": "Q2V3_20_NODE_TWO_PATH_MIP",
                    "node_count": 20,
                    "status": "not_run_no_mip_solver",
                    "model_scope": "two-path coverage MILP with subtour elimination",
                    "reason": "当前环境未配置OR-Tools/Gurobi/CPLEX；保留接口，后续可接入MIP start与lazy subtour cuts",
                }
            ]
        )
        self.write_csv(mip_df, self.outputs / "q2_v3_small_mip_check.csv")
        combined = pd.concat([exact_df, mip_df], ignore_index=True, sort=False)
        self.write_csv(combined, self.outputs / "q2_v3_small_exact_check.csv")
        self.exact_checks = exact_df
        self.small_mip_check = mip_df
        return exact_df, mip_df

    # ---------- Cost calibration audit ----------

    def build_transport_cost_calibration_audit(self) -> pd.DataFrame:
        labels = pd.concat([self.multimodal_labels, self.gateway_labels], ignore_index=True, sort=False)
        rows = []
        for r in labels.to_dict("records"):
            raw_cost = num(r.get("raw_cost_yuan_for_two", r.get("cost_yuan_for_two", 0.0)))
            calibrated = num(r.get("cost_yuan_for_two", 0.0))
            rows.append(
                {
                    "label_id": r.get("label_id", ""),
                    "from_id": r.get("from_id", ""),
                    "to_id": r.get("to_id", ""),
                    "raw_mode": r.get("raw_mode", r.get("mode", "")),
                    "normalized_mode": r.get("mode", ""),
                    "mode_combo": r.get("mode_combo", ""),
                    "raw_cost_yuan_for_two": round(raw_cost, 2),
                    "calibrated_cost_yuan_for_two": round(calibrated, 2),
                    "cost_delta": round(calibrated - raw_cost, 2),
                    "time_hours": r.get("time_hours", ""),
                    "cost_calibration_note": r.get("cost_calibration_note", ""),
                    "source": r.get("source", ""),
                    "is_gateway_label": r.get("is_gateway_label", False),
                }
            )
        audit = pd.DataFrame(rows)
        self.write_csv(audit, self.outputs / "q2_v3_transport_cost_calibration_audit.csv")
        self.transport_cost_calibration_audit = audit
        return audit

    def validate_no_zero_cost_scenic_shuttle(self) -> None:
        labels = self.multimodal_labels
        segments = self.route_segments
        bad_labels = labels[(labels["mode"].astype(str).eq("scenic_shuttle")) & (labels["cost_yuan_for_two"] <= 0)]
        bad_segments = segments[
            (segments["mode"].astype(str).eq("scenic_shuttle")) & (segments["cost_yuan_for_two"] <= 0)
        ]
        if not bad_labels.empty:
            raise ValueError(f"仍存在0成本scenic_shuttle标签: {len(bad_labels)}")
        if not bad_segments.empty:
            raise ValueError(f"路线中仍使用0成本scenic_shuttle: {len(bad_segments)}")

    def validate_no_self_drive_mode(self) -> None:
        labels = self.multimodal_labels
        segments = self.route_segments
        selected = self.selected_plans
        if "self_drive" in set(labels["mode"].astype(str)):
            raise ValueError("labels中仍存在self_drive规范模式")
        if "self_drive" in set(segments["mode"].astype(str)):
            raise ValueError("segments中仍存在self_drive规范模式")
        for col in ["year1_mode_hours_mix", "year2_mode_hours_mix"]:
            if col in selected.columns and selected[col].astype(str).str.contains("self_drive", regex=False).any():
                raise ValueError(f"selected plans 的 {col} 中仍含 self_drive")

    def build_cost_calibration_sensitivity(self) -> pd.DataFrame:
        scenarios = [
            ("low_floor", 0.75, "低估接驳费用"),
            ("base_floor", 1.00, "主模型"),
            ("high_floor", 1.25, "保守高估接驳费用"),
        ]
        base_roles = self.selected_plans.set_index("selected_role")["plan_id"].to_dict()
        segment_base = self.route_segments.copy()
        rows = []
        plan_df = self.candidate_plans.merge(self.simulation_summary, on="plan_id", how="left")
        regular = plan_df[~plan_df["include_special"]].copy()
        for scenario, multiplier, meaning in scenarios:
            seg = segment_base.copy()
            scenic = seg["mode"].astype(str).eq("scenic_shuttle")
            adjusted = seg["cost_yuan_for_two"].astype(float).copy()
            if scenic.any():
                floors = seg.loc[scenic, "time_hours"].astype(float).map(
                    lambda t: self.scenic_shuttle_cost_floor(float(t), multiplier)
                )
                raw = seg.loc[scenic, "raw_cost_yuan_for_two"].astype(float)
                adjusted.loc[scenic] = np.maximum(raw.to_numpy(), floors.to_numpy())
            seg["sensitivity_cost"] = adjusted
            costs = (
                seg.groupby("plan_id")["sensitivity_cost"]
                .sum()
                .round(2)
                .rename("sensitivity_total_cost")
                .reset_index()
            )
            df = regular.merge(costs, on="plan_id", how="left")
            df["sensitivity_total_cost"] = df["sensitivity_total_cost"].fillna(df["total_intra_transport_cost"])

            rooted = df[df["gateway_policy"].eq("rooted_urumqi")].sort_values("sensitivity_total_cost").iloc[0]
            opened = df[df["gateway_policy"].eq("open_gateway")].sort_values("sensitivity_total_cost").iloc[0]

            min_cost = float(df["sensitivity_total_cost"].min())
            balanced = df[
                df["schedule_strict_feasible"].astype(bool)
                & df["year_day_difference"].le(int(self.config["max_year_day_difference"]))
                & df["operational_success_probability"].ge(float(self.config["success_threshold"]))
            ]
            robust_pool = balanced[balanced["sensitivity_total_cost"].le(min_cost * (1 + float(self.config["cost_gap_for_robust_choice"])))]
            if robust_pool.empty:
                robust_pool = balanced
            if robust_pool.empty:
                robust_pool = df[df["schedule_soft_feasible"].astype(bool)]
            if robust_pool.empty:
                robust_pool = df
            robust = robust_pool.sort_values(
                [
                    "operational_success_probability",
                    "strict_comfort_success_probability",
                    "cvar75_loss",
                    "sensitivity_total_cost",
                ],
                ascending=[False, False, True, True],
            ).iloc[0]
            changed = any(
                [
                    base_roles.get("ROOTED_URUMQI_MAIN") != rooted["plan_id"],
                    base_roles.get("OPEN_GATEWAY_MIN_INTRA_COST") != opened["plan_id"],
                    base_roles.get("ROBUST_TWO_YEAR_MAIN") != robust["plan_id"],
                ]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "multiplier": multiplier,
                    "meaning": meaning,
                    "rooted_plan_id": rooted["plan_id"],
                    "rooted_cost": round(float(rooted["sensitivity_total_cost"]), 2),
                    "open_min_plan_id": opened["plan_id"],
                    "open_min_cost": round(float(opened["sensitivity_total_cost"]), 2),
                    "robust_plan_id": robust["plan_id"],
                    "robust_cost": round(float(robust["sensitivity_total_cost"]), 2),
                    "break_even_external_premium": round(
                        float(rooted["sensitivity_total_cost"]) - float(opened["sensitivity_total_cost"]), 2
                    ),
                    "selected_plan_changed": changed,
                    "notes": "fixed_candidate_repricing_no_route_reoptimization",
                }
            )
        out = pd.DataFrame(rows)
        self.write_csv(out, self.outputs / "q2_v3_cost_calibration_sensitivity.csv")
        self.cost_calibration_sensitivity = out
        return out

    # ---------- Audit, figures, reports ----------

    def build_model_audit(self) -> pd.DataFrame:
        rows = [
            {
                "audit_id": "Q2V3-A1",
                "module": "problem_definition",
                "claim": "第二问主目标为新疆境内交通费用最小化，覆盖是约束不是收益目标",
                "status": "implemented",
                "evidence_file": "reports/新疆旅游第二问Q2_V3两年交通费用优化报告.md",
                "remaining_limitation": "当前仍是matheuristic高质量可行解，不声称完整全局最优",
                "next_upgrade": "接入Gurobi/CPLEX lazy DFJ或OR-Tools RoutingModel做更强求解",
            },
            {
                "audit_id": "Q2V3-A2",
                "module": "ordinary_access",
                "claim": "普通游客主模型排除楼兰古城、尼雅遗址等特殊审批点",
                "status": "implemented",
                "evidence_file": "outputs/q2_v3_candidate_plans.csv",
                "remaining_limitation": "审批扩展仅作为可选方案，不作为普通游客主线",
                "next_upgrade": "为审批扩展增加审批成功率与向导/越野车成本",
            },
            {
                "audit_id": "Q2V3-A3",
                "module": "multimodal_labels",
                "claim": "生成直接道路、增强OD公共交通与铁路/航班种子模板标签并做非支配筛选",
                "status": "implemented_template_labels",
                "evidence_file": "outputs/q2_v3_multimodal_labels.csv",
                "remaining_limitation": "不是对全图每对OD运行无限制多标签最短路",
                "next_upgrade": "接入班次级铁路/航班票价与真实余票",
            },
            {
                "audit_id": "Q2V3-A4",
                "module": "gateway_policy",
                "claim": "同时输出固定乌鲁木齐起讫与开放式多口岸两种口径",
                "status": "implemented",
                "evidence_file": "outputs/q2_v3_gateway_thresholds.csv",
                "remaining_limitation": "开放式多口岸外部大交通价格仍以情景仿真表达",
                "next_upgrade": "接入实时机票/铁路票价后把外部差价变成实时报价策略",
            },
            {
                "audit_id": "Q2V3-A5",
                "module": "route_specific_simulation",
                "claim": "对每条候选两年方案进行route-specific Monte Carlo仿真",
                "status": "implemented",
                "evidence_file": "outputs/q2_v3_simulation_summary.csv",
                "remaining_limitation": "扰动分布为校准仿真，不是真实实时天气/库存",
                "next_upgrade": "接入天气、道路封闭、票价和酒店房态API",
            },
            {
                "audit_id": "Q2V3-A6",
                "module": "exact_check",
                "claim": "增加小规模Held-Karp排序精确校验，20节点MIP接口保留",
                "status": "implemented_scope_limited",
                "evidence_file": "outputs/q2_v3_small_exact_check.csv",
                "remaining_limitation": "仅校验固定子集排序，不证明完整38点两路径全局最优",
                "next_upgrade": "安装OR-Tools/Gurobi后运行20节点两路径MILP",
            },
            {
                "audit_id": "Q2V3-A7",
                "module": "scenic_shuttle_cost_calibration",
                "claim": "scenic_shuttle不再允许0元参与费用最小化",
                "status": "implemented",
                "evidence_file": "outputs/q2_v3_transport_cost_calibration_audit.csv",
                "remaining_limitation": "接驳费用为代理估计，非实时票价",
                "next_upgrade": "接入景区区间车/当地包车/出租价格数据",
            },
            {
                "audit_id": "Q2V3-A8",
                "module": "road_mode_normalization",
                "claim": "self_drive统一映射为taxi_transfer/rental_car/charter_car",
                "status": "implemented",
                "evidence_file": "outputs/q2_v3_multimodal_labels.csv",
                "remaining_limitation": "当前道路成本仍为代理成本，未拆分完整租车日租/包车日费",
                "next_upgrade": "建立真实租车/包车/司机费用模型",
            },
        ]
        df = pd.DataFrame(rows)
        self.write_csv(df, self.outputs / "q2_v3_model_audit.csv")
        self.model_audit = df
        return df

    def build_figures(self) -> None:
        df = self.candidate_plans.merge(self.simulation_summary, on="plan_id", how="left")
        plt.figure(figsize=(9, 6))
        colors = df["gateway_policy"].map({"rooted_urumqi": "#2563eb", "open_gateway": "#059669"}).fillna("#6b7280")
        plt.scatter(df["total_intra_transport_cost"], df["max_year_days"], c=colors, s=52, alpha=0.78)
        plt.xlabel("Intra-XJ transport cost for two (yuan)")
        plt.ylabel("Max year days")
        plt.title("Q2-V3 Cost-Days Frontier")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.figures / "fig_q2_v3_cost_days_frontier.png", dpi=180)
        plt.close()

        if not self.gateway_thresholds.empty:
            row = self.gateway_thresholds.iloc[0]
            plt.figure(figsize=(8, 5))
            plt.bar(["Rooted Urumqi", "Open gateway"], [row["rooted_cost_yuan_for_two"], row["open_gateway_cost_yuan_for_two"]], color=["#2563eb", "#059669"])
            plt.title(f"Gateway break-even premium: {row['break_even_external_premium_yuan_for_two']:.2f} yuan")
            plt.ylabel("Intra-XJ cost for two")
            plt.tight_layout()
            plt.savefig(self.figures / "fig_q2_v3_gateway_threshold.png", dpi=180)
            plt.close()

        role_order = [
            "ROOTED_URUMQI_MAIN",
            "OPEN_GATEWAY_MIN_INTRA_COST",
            "ROBUST_TWO_YEAR_MAIN",
            "COMFORT_BALANCED_BACKUP",
        ]
        selected = self.selected_plans.copy()
        if not selected.empty and "selected_role" in selected.columns:
            selected = selected.set_index("selected_role").reindex(role_order).dropna(how="all").reset_index()
        if not selected.empty:
            mode_rows = []
            for row in selected.itertuples(index=False):
                for year in [1, 2]:
                    mix = json.loads(getattr(row, f"year{year}_mode_hours_mix", "{}")) if hasattr(row, f"year{year}_mode_hours_mix") else {}
                    for mode, hours in mix.items():
                        mode_rows.append({"role": row.selected_role, "mode": mode, "hours": hours})
            if mode_rows:
                mode_df = pd.DataFrame(mode_rows).pivot_table(index="role", columns="mode", values="hours", aggfunc="sum", fill_value=0)
                mode_df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
                plt.ylabel("Transport hours")
                plt.title("Q2-V3 Mode Mix")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                plt.savefig(self.figures / "fig_q2_v3_mode_mix.png", dpi=180)
                plt.close()

        if not selected.empty:
            rows_to_plot = []
            for row in selected.itertuples(index=False):
                for year in [1, 2]:
                    route = getattr(row, f"year{year}_route_sequence", "")
                    label = f"{row.selected_role} Y{year}"
                    wrapped = textwrap.wrap(label + ": " + route, width=104)
                    rows_to_plot.append("\n".join(wrapped[:3]))
            fig_height = max(8, 1.05 * len(rows_to_plot) + 1.4)
            plt.figure(figsize=(14, fig_height))
            y = 0.94
            for block in rows_to_plot:
                line_count = block.count("\n") + 1
                plt.text(0.01, y, block, fontsize=8, va="top", ha="left", wrap=True)
                y -= 0.055 * line_count + 0.045
            plt.axis("off")
            plt.title("Q2-V3 Two-Year Route Sketch")
            plt.tight_layout()
            plt.savefig(self.figures / "fig_q2_v3_two_year_map.png", dpi=180)
            plt.close()

    def md_table(self, df: pd.DataFrame, cols: list[str], limit: int = 10) -> str:
        if df is None or df.empty:
            return "_无数据_"
        use = df[[c for c in cols if c in df.columns]].head(limit).copy()
        return use.to_markdown(index=False)

    def build_reports(self) -> None:
        selected = self.selected_plans.copy()
        thresholds = self.gateway_thresholds.copy()
        front = self.pareto_front.copy()
        audit = self.model_audit.copy()
        exact = self.exact_checks.copy()
        cal_audit = self.transport_cost_calibration_audit.copy()
        sensitivity = self.cost_calibration_sensitivity.copy()
        floor_count = int(cal_audit["cost_calibration_note"].astype(str).str.contains("scenic_shuttle_cost_floor_applied").sum())
        self_drive_norm_count = int(
            (
                cal_audit["raw_mode"].astype(str).str.contains("self_drive", regex=False)
                | cal_audit["mode_combo"].astype(str).str.contains("selfdrive", regex=False)
                | cal_audit["mode_combo"].astype(str).str.contains("self_drive", regex=False)
            ).sum()
        )
        total_cost_delta = round(float(cal_audit["cost_delta"].sum()), 2)
        report = f"""# 新疆旅游第二问 Q2-V3 两年境内交通费用最小化报告

## 1. 问题重定义

第二问不是简单聚类，也不是第一问覆盖价值最大化的延伸。Q2-V3 将问题定义为：

> 在今、明两年暑假内覆盖普通游客可达景点，每年形成一条新疆境内可执行旅游路径，并最小化两人新疆境内交通费用。

本版本主模型排除需审批或普通游客受限的特殊点；楼兰古城、尼雅遗址仅进入审批扩展方案。

## 2. 数据结构

- 普通游客可达景点：{len(self.ordinary_ids)} 个；
- 特殊准入/审批点：{len(self.special_ids)} 个；
- 入离疆口岸候选：{len(self.gateway_names)} 个；
- 多模式交通标签：{len(self.multimodal_labels)} 条；
- 口岸接驳标签：{len(self.gateway_labels)} 条。

交通标签由增强 OD、高德道路 OD、铁路 12306 种子和航班种子共同构成，并经过费用、时间、风险、疲劳维度的非支配筛选。

为避免低估新疆境内交通费用，Q2-V3 对交通标签进行了费用校准。原始数据中 `scenic_shuttle` 若费用缺失或为 0，不再被视作免费交通，而是按转场时间设置最低接驳费用；`self_drive` 不作为最终交通方式输出，而是按转场时间统一归类为 `taxi_transfer`、`rental_car` 或 `charter_car`。

成本校准版本：`{self.config["cost_calibration_version"]}`。本轮共有 {floor_count} 条标签应用 scenic shuttle 成本下限，{self_drive_norm_count} 条原始 self-drive 标签被规范化，标签层总校准增量为 {total_cost_delta} 元。

## 3. 数学模型

固定乌鲁木齐起讫模型：

```text
min sum_k sum_(i,j,l) c_ijl z_ijlk
s.t. 每个普通游客景点恰好分配给一个年份
     每年从乌鲁木齐进入并返回乌鲁木齐
     每年形成一条连续路径
     年度天数不超过30天
```

开放式多口岸模型：

```text
min 新疆境内交通费用
s.t. 每年可从候选口岸中选择入疆口岸和离疆口岸
     外部大交通费用不进入主目标，只通过阈值策略解释
```

## 4. 算法

本版本采用 matheuristic：

```text
多模式标签生成
+ 历史/高德/谱聚类/区域/Grand-tour/random seed
+ 2-opt、relocate、swap 联合改进
+ 年度排程可行性检查
+ route-specific 两年 Monte Carlo 仿真
+ Pareto 筛选和代表方案选择
```

## 5. 代表方案

{self.md_table(selected, [
    "selected_role", "plan_id", "gateway_policy", "year1_entry_gateway", "year1_exit_gateway",
    "year2_entry_gateway", "year2_exit_gateway", "covered_spots", "year1_days", "year2_days",
    "year_day_difference", "red_days", "time_window_violations",
    "total_intra_transport_cost", "external_premium_break_even", "operational_success_probability",
    "strict_comfort_success_probability", "cvar75_loss", "schedule_strict_feasible", "selection_status",
    "cost_calibration_version"
], 8)}

说明：`OPEN_GATEWAY_MIN_INTRA_COST` 是新疆境内交通费用下界方案；`ROBUST_TWO_YEAR_MAIN` 额外要求硬排程可行、两年天数差不超过 {self.config["max_year_day_difference"]} 天且运营成功率达标，更适合作为可执行主推。

## 6. 多口岸阈值

{self.md_table(thresholds, [
    "comparison", "rooted_cost_yuan_for_two", "open_gateway_cost_yuan_for_two",
    "break_even_external_premium_yuan_for_two", "decision_rule"
], 5)}

解释：开放式多口岸模型只优化新疆境内交通费用，是否在现实总费用上更优，取决于华东到不同新疆口岸的大交通差价是否低于阈值。

## 7. 费用-天数-鲁棒前沿

{self.md_table(front, [
    "plan_id", "gateway_policy", "total_intra_transport_cost", "max_year_days",
    "year_day_difference", "red_days", "time_window_violations",
    "operational_success_probability", "strict_comfort_success_probability", "cvar75_loss",
    "open_gateway_cheaper_probability", "schedule_strict_feasible"
], 12)}

## 8. 小规模精确校验

{self.md_table(exact, [
    "check_id", "selected_role", "year", "node_count", "heuristic_cost",
    "exact_held_karp_cost", "relative_gap", "end_anchor_type", "end_anchor", "status"
], 12)}

前缀校验若该年路线仍有后续景点，则使用下一景点作为续接锚点；只有前缀覆盖该年全部景点时才直接返回口岸。20节点两路径 MILP 当前保留接口，但因未配置 OR-Tools/Gurobi/CPLEX，输出为 `not_run_no_mip_solver`。因此本报告不声称完整38点两路径全局最优。

## 9. 成本校准敏感性

{self.md_table(sensitivity, [
    "scenario", "multiplier", "rooted_plan_id", "rooted_cost", "open_min_plan_id",
    "open_min_cost", "robust_plan_id", "robust_cost", "break_even_external_premium",
    "selected_plan_changed", "notes"
], 10)}

说明：敏感性实验采用固定候选集重计价，不重新搜索路线，用于观察接驳费用下限对代表方案和多口岸阈值的局部影响。

## 10. 模型边界审计

{self.md_table(audit, ["audit_id", "module", "claim", "status", "remaining_limitation"], 10)}

## 11. 结论

Q2-V3 已将第二问从“区域聚类/道路基线”升级为“两年鲁棒多模式路径覆盖”闭环。推荐叙事应分三层：

1. 保守执行：固定乌鲁木齐起讫方案，实施难度最低；
2. 省境内交通：开放式多口岸方案，给出新疆境内费用下界；
3. 现实总费用最优：比较多口岸外部大交通额外差价是否低于阈值。

修正后，所有 scenic_shuttle 标签均计入最低接驳费用，所有 self_drive 标签均按转场时长映射为 taxi_transfer、rental_car 或 charter_car。由此得到的交通费用更适合作为“新疆境内交通费用最小化”的目标函数输入。

当前版本为高质量可行解与局部精确校验，不是商业求解器证明的全局最优解。
"""
        report_path = self.reports / "新疆旅游第二问Q2_V3两年交通费用优化报告.md"
        report_path.write_text(report, encoding="utf-8")

        workbook_path = self.reports / "新疆旅游第二问Q2_V3两年交通费用优化结果.xlsx"
        tables = {
            "selected_plans": selected,
            "candidate_plans": self.candidate_plans,
            "year_route_summary": self.year_routes,
            "route_segments": self.route_segments,
            "schedule_summary": self.schedule_summary,
            "simulation_summary": self.simulation_summary,
            "gateway_thresholds": self.gateway_thresholds,
            "pareto_front": self.pareto_front,
            "cost_calibration_audit": self.transport_cost_calibration_audit,
            "cost_sensitivity": self.cost_calibration_sensitivity,
            "model_audit": self.model_audit,
            "exact_order_check": self.exact_checks,
            "small_mip_check": self.small_mip_check,
        }
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for name, df in tables.items():
                df.to_excel(writer, index=False, sheet_name=name[:31])
        self.report_path = report_path
        self.workbook_path = workbook_path

    def update_package_manifest(self) -> None:
        files = sorted([p for p in self.package_dir.rglob("*") if p.is_file()])
        manifest = []
        for p in files:
            rel = p.relative_to(self.package_dir)
            manifest.append(
                {
                    "relative_path": rel.as_posix(),
                    "directory": rel.parent.as_posix(),
                    "file_name": p.name,
                    "extension": p.suffix,
                    "size_bytes": p.stat().st_size,
                    "last_write_time": pd.Timestamp.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        self.write_csv(pd.DataFrame(manifest), self.package_dir / "Q2_file_manifest.csv")
        summary_rows = []
        for directory, group in pd.DataFrame(manifest).groupby("directory"):
            summary_rows.append(
                {
                    "directory": directory,
                    "file_count": len(group),
                    "total_size_bytes": int(group["size_bytes"].sum()),
                }
            )
        self.write_csv(pd.DataFrame(summary_rows).sort_values("directory"), self.package_dir / "Q2_directory_summary.csv")

    def build_readme(self) -> None:
        text = """# Q2-V3 两年鲁棒多模式路径覆盖

本目录是第二问的主模型升级版，目标是把“今、明两年暑假完成新疆旅游，并使新疆境内交通费用尽量节省”落实为可复现的模型、代码、输出、审计和报告。

## 复现入口

```bash
python -X utf8 "第二问_两年暑假交通费用优化材料包/11_Q2_V3_两年鲁棒多模式路径覆盖/scripts/q2_v3_build_all.py"
```

## 核心输出

```text
outputs/q2_v3_multimodal_labels.csv
outputs/q2_v3_gateway_labels.csv
outputs/q2_v3_candidate_plans.csv
outputs/q2_v3_selected_plans.csv
outputs/q2_v3_year_route_summary.csv
outputs/q2_v3_route_segments.csv
outputs/q2_v3_schedule_summary.csv
outputs/q2_v3_simulation_summary.csv
outputs/q2_v3_gateway_thresholds.csv
outputs/q2_v3_feasible_pareto_front.csv
outputs/q2_v3_transport_cost_calibration_audit.csv
outputs/q2_v3_cost_calibration_sensitivity.csv
outputs/q2_v3_model_audit.csv
outputs/q2_v3_small_exact_check.csv
reports/新疆旅游第二问Q2_V3两年交通费用优化报告.md
reports/新疆旅游第二问Q2_V3两年交通费用优化结果.xlsx
```

## 口径

- 覆盖对象：普通游客可达景点；
- 主目标：两人新疆境内交通费用最小；
- 固定口岸：乌鲁木齐起讫作为保守主方案；
- 开放口岸：多口岸方案作为境内费用下界和阈值策略；
- 成本校准：scenic_shuttle 施加最低接驳费用，self_drive 拆分为 taxi_transfer/rental_car/charter_car；
- 特殊点：楼兰古城、尼雅遗址进入审批扩展，不进入普通游客主线；
- 最优性：matheuristic 高质量可行解 + 小规模 Held-Karp 校验，不声称完整全局最优。
"""
        (self.q2v3_dir / "README.md").write_text(text, encoding="utf-8")

    def run(self) -> None:
        self.load_inputs()
        self.multimodal_labels = self.build_multimodal_labels()
        self.gateway_labels = self.build_gateway_labels()
        self.build_transport_cost_calibration_audit()
        self.build_candidate_plans()
        self.validate_no_zero_cost_scenic_shuttle()
        self.build_schedule_summary()
        self.simulate_plans()
        self.build_gateway_thresholds()
        self.select_representative_plans()
        self.build_cost_calibration_sensitivity()
        self.validate_no_self_drive_mode()
        self.build_pareto_front()
        self.build_exact_checks()
        self.build_model_audit()
        self.build_figures()
        self.build_reports()
        self.build_readme()
        floor_count = int(
            self.transport_cost_calibration_audit["cost_calibration_note"]
            .astype(str)
            .str.contains("scenic_shuttle_cost_floor_applied")
            .sum()
        )
        self_drive_norm_count = int(
            (
                self.transport_cost_calibration_audit["raw_mode"].astype(str).str.contains("self_drive", regex=False)
                | self.transport_cost_calibration_audit["mode_combo"].astype(str).str.contains("selfdrive", regex=False)
                | self.transport_cost_calibration_audit["mode_combo"].astype(str).str.contains("self_drive", regex=False)
            ).sum()
        )
        total_cost_delta = round(float(self.transport_cost_calibration_audit["cost_delta"].sum()), 2)
        solve_summary = {
            "ordinary_spots": len(self.ordinary_ids),
            "special_spots": len(self.special_ids),
            "multimodal_labels": int(len(self.multimodal_labels)),
            "gateway_labels": int(len(self.gateway_labels)),
            "candidate_plans": int(len(self.candidate_plans)),
            "selected_plans": int(len(self.selected_plans)),
            "pareto_front_size": int(len(self.pareto_front)),
            "cost_calibration_version": self.config["cost_calibration_version"],
            "scenic_shuttle_cost_floor_applied_count": floor_count,
            "self_drive_normalized_count": self_drive_norm_count,
            "total_cost_delta_from_calibration": total_cost_delta,
            "report": str(self.report_path.relative_to(self.q2v3_dir)),
            "workbook": str(self.workbook_path.relative_to(self.q2v3_dir)),
        }
        (self.outputs / "solve_summary.json").write_text(json.dumps(solve_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_package_manifest()
        print(json.dumps(solve_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    Q2V3Builder().run()
