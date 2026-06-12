# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RNG_SEED = 20260612
BASE_DEMAND = 106_358
TARGET_ROUTE_DAYS = 12
SLOT_SHARES = {"morning": 0.42, "afternoon": 0.45, "evening": 0.13}
SLOTS = list(SLOT_SHARES)
SCENARIOS = [
    {
        "scenario_id": "base_1_00",
        "scenario_name": "基准五一客流",
        "demand_multiplier": 1.00,
        "spot_capacity_factor": 1.00,
        "hotel_capacity_factor": 1.00,
        "vehicle_capacity_factor": 1.00,
        "volatility": 1.00,
    },
    {
        "scenario_id": "light_peak_1_05",
        "scenario_name": "轻度上浮5%",
        "demand_multiplier": 1.05,
        "spot_capacity_factor": 1.00,
        "hotel_capacity_factor": 0.98,
        "vehicle_capacity_factor": 0.98,
        "volatility": 1.05,
    },
    {
        "scenario_id": "peak_1_10",
        "scenario_name": "高峰上浮10%",
        "demand_multiplier": 1.10,
        "spot_capacity_factor": 0.98,
        "hotel_capacity_factor": 0.96,
        "vehicle_capacity_factor": 0.96,
        "volatility": 1.12,
    },
    {
        "scenario_id": "high_peak_1_20",
        "scenario_name": "强高峰上浮20%",
        "demand_multiplier": 1.20,
        "spot_capacity_factor": 0.95,
        "hotel_capacity_factor": 0.92,
        "vehicle_capacity_factor": 0.93,
        "volatility": 1.25,
    },
    {
        "scenario_id": "compound_extreme_1_35",
        "scenario_name": "复合极端上浮35%",
        "demand_multiplier": 1.35,
        "spot_capacity_factor": 0.90,
        "hotel_capacity_factor": 0.88,
        "vehicle_capacity_factor": 0.90,
        "volatility": 1.50,
    },
]
POLICIES = [
    {
        "policy_id": "legacy_9_full_release",
        "policy_name": "旧9线路全量放票基线",
        "portfolio": "legacy",
        "booking_ratio": 1.00,
        "slot_stagger_strength": 0.00,
        "route_capacity_boost": 1.00,
        "resource_boost": 1.00,
        "preference_elastic": False,
    },
    {
        "policy_id": "q4v2_full_release",
        "policy_name": "Q4-V2全量放票",
        "portfolio": "q4v2",
        "booking_ratio": 1.00,
        "slot_stagger_strength": 0.08,
        "route_capacity_boost": 1.00,
        "resource_boost": 1.00,
        "preference_elastic": True,
    },
    {
        "policy_id": "q4v2_safety_cap_95",
        "policy_name": "Q4-V2 95%安全上限",
        "portfolio": "q4v2",
        "booking_ratio": 0.95,
        "slot_stagger_strength": 0.12,
        "route_capacity_boost": 1.00,
        "resource_boost": 1.00,
        "preference_elastic": True,
    },
    {
        "policy_id": "q4v2_comfort_cap_90",
        "policy_name": "Q4-V2 90%舒适上限",
        "portfolio": "q4v2",
        "booking_ratio": 0.90,
        "slot_stagger_strength": 0.16,
        "route_capacity_boost": 1.00,
        "resource_boost": 1.00,
        "preference_elastic": True,
    },
    {
        "policy_id": "q4v2_staggered_prebooking",
        "policy_name": "分时预约+偏好弹性分流",
        "portfolio": "q4v2",
        "booking_ratio": 0.95,
        "slot_stagger_strength": 0.40,
        "route_capacity_boost": 1.00,
        "resource_boost": 1.03,
        "preference_elastic": True,
    },
    {
        "policy_id": "q4v2_add_capacity_plus_stagger",
        "policy_name": "补运力+分时预约复合策略",
        "portfolio": "q4v2",
        "booking_ratio": 0.95,
        "slot_stagger_strength": 0.48,
        "route_capacity_boost": 1.18,
        "resource_boost": 1.22,
        "preference_elastic": True,
    },
]


@dataclass
class Paths:
    project_root: Path
    q4_root: Path
    v2_root: Path
    outputs: Path
    reports: Path
    figures: Path


def find_project_paths() -> Paths:
    script_path = Path(__file__).resolve()
    v2_root = script_path.parents[1]
    q4_root = v2_root.parent
    project_root = q4_root.parent
    return Paths(
        project_root=project_root,
        q4_root=q4_root,
        v2_root=v2_root,
        outputs=v2_root / "outputs",
        reports=v2_root / "reports",
        figures=v2_root / "figures",
    )


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.outputs, paths.reports, paths.figures, paths.v2_root / "config"]:
        p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def norm_series(s: pd.Series, floor: float | None = None, cap: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if floor is not None:
        x = x.clip(lower=floor)
    if cap is not None:
        x = x.clip(upper=cap)
    lo = float(x.min())
    hi = float(x.max())
    if math.isclose(lo, hi):
        return pd.Series(np.ones(len(x)) * 0.5, index=s.index)
    return (x - lo) / (hi - lo)


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def parse_ids(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_inputs(paths: Paths) -> dict[str, pd.DataFrame]:
    base = paths.q4_root
    model = base / "01_输入数据" / "model_data"
    enhanced = base / "01_输入数据" / "enhanced_data"
    main = base / "03_线路产品容量流主模型" / "enhanced_model_outputs"
    inputs = {
        "routes": read_csv(main / "problem4_route_columns.csv"),
        "flow": read_csv(main / "problem4_capacity_flow.csv"),
        "spot": read_csv(model / "spot_clean.csv"),
        "capacity": read_csv(enhanced / "capacity_by_spot.csv"),
        "time_windows": read_csv(enhanced / "spot_time_windows.csv"),
        "cultural": read_csv(enhanced / "cultural_tags.csv"),
        "hotel_hubs": read_csv(enhanced / "hotel_hub_constraints.csv"),
        "hotel_options": read_csv(model / "hotel_options_clean.csv"),
        "hub": read_csv(model / "hub_clean.csv"),
        "od": read_csv(enhanced / "enhanced_od_matrix_with_amap.csv"),
    }
    return inputs


def build_spot_meta(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spot = inputs["spot"].copy()
    cap = inputs["capacity"].copy()
    tw = inputs["time_windows"].copy()
    cultural = inputs["cultural"].copy()
    meta = spot.merge(
        cap[
            [
                "spot_id",
                "daily_capacity_persons",
                "instant_capacity_persons",
                "capacity_source_type",
                "capacity_method",
                "effective_capacity_beta_085",
            ]
        ],
        on="spot_id",
        how="left",
    ).merge(
        tw[["spot_id", "open_time", "close_time", "time_window_quality"]],
        on="spot_id",
        how="left",
    ).merge(
        cultural[
            [
                "spot_id",
                "tag_silk_road",
                "tag_religion",
                "tag_ethnic_folk",
                "tag_archaeology",
                "tag_world_or_key_heritage",
                "tag_natural",
                "culture_value_score",
            ]
        ],
        on="spot_id",
        how="left",
    )
    for col in [
        "is_cultural",
        "is_natural",
        "is_topic_preference",
        "requires_approval",
        "requires_border_permit",
        "requires_reservation",
        "high_altitude_or_remote",
        "ordinary_tourist_restricted",
        "tag_silk_road",
        "tag_religion",
        "tag_ethnic_folk",
        "tag_archaeology",
        "tag_world_or_key_heritage",
        "tag_natural",
    ]:
        if col in meta.columns:
            meta[col] = meta[col].map(as_bool)
    meta["daily_capacity_persons"] = pd.to_numeric(meta["daily_capacity_persons"], errors="coerce").fillna(6_000)
    meta["effective_capacity_beta_085"] = pd.to_numeric(
        meta["effective_capacity_beta_085"], errors="coerce"
    ).fillna(meta["daily_capacity_persons"] * 0.85)
    meta["instant_capacity_persons"] = pd.to_numeric(meta["instant_capacity_persons"], errors="coerce").fillna(
        meta["daily_capacity_persons"] * 0.32
    )
    meta["visit_hours_mid"] = pd.to_numeric(meta["visit_hours_mid"], errors="coerce").fillna(3.0)
    meta["priority_score_for_op"] = pd.to_numeric(meta["priority_score_for_op"], errors="coerce").fillna(2.0)
    meta["culture_value_score"] = pd.to_numeric(meta["culture_value_score"], errors="coerce").fillna(0.0)
    meta["capacity_norm"] = norm_series(meta["effective_capacity_beta_085"])
    meta["priority_norm"] = norm_series(meta["priority_score_for_op"])
    meta["culture_norm"] = norm_series(meta["culture_value_score"])
    return meta


def calibrated_hub_room_capacity(hub_name: str, base_rooms: float) -> float:
    """Calibrate placeholder hub hotel rooms to Mayday operational room pools.

    The source table stores 20/80-room placeholders for many hubs. Q4 is an
    aggregate holiday reception problem, so the capacity used here is the room
    pool a tourism authority/platform can coordinate around the hub, not one
    individual hotel option.
    """
    name = str(hub_name)
    overrides = {
        "乌鲁木齐市": 12_000,
        "吐鲁番市": 4_200,
        "喀什市": 5_200,
        "伊宁市(伊犁)": 4_300,
        "库尔勒市": 3_800,
        "和田市": 2_800,
        "库车市": 2_300,
        "阿勒泰市": 2_200,
        "天池风景区": 1_850,
        "赛里木湖": 1_650,
        "那拉提镇": 1_550,
        "特克斯县(八卦城)": 1_350,
        "霍城县(惠远古城)": 1_150,
        "昭苏县(夏塔古道)": 950,
        "巴音布鲁克": 1_050,
        "禾木村": 1_250,
        "白哈巴": 620,
        "可可托海": 900,
        "塔什库尔干县(帕米尔高原)": 980,
        "若羌县(楼兰)": 760,
        "民丰县(尼雅遗址)": 520,
        "达坂城镇": 620,
    }
    if name in overrides:
        return float(overrides[name])
    base = float(base_rooms) if pd.notna(base_rooms) else 80.0
    if any(x in name for x in ["市", "县"]):
        return max(850.0, base * 38.0)
    if any(x in name for x in ["镇", "村", "湖", "风景区", "草原"]):
        return max(520.0, base * 22.0)
    return max(450.0, base * 18.0)


def preferred_overnight_hub(hub_name: str) -> str:
    """Map scenic/small hubs to realistic overnight service hubs."""
    mapping = {
        "天池风景区": "乌鲁木齐市",
        "达坂城镇": "乌鲁木齐市",
        "赛里木湖": "伊宁市(伊犁)",
        "霍城县(惠远古城)": "伊宁市(伊犁)",
        "特克斯县(八卦城)": "伊宁市(伊犁)",
        "昭苏县(夏塔古道)": "伊宁市(伊犁)",
        "禾木村": "阿勒泰市",
        "白哈巴": "阿勒泰市",
        "可可托海": "阿勒泰市",
        "巴音布鲁克": "库尔勒市",
        "民丰县(尼雅遗址)": "和田市",
    }
    return mapping.get(str(hub_name), str(hub_name))


def build_od_lookup(inputs: dict[str, pd.DataFrame]) -> dict[tuple[str, str], tuple[float, float]]:
    od = inputs["od"].copy()
    od["time"] = pd.to_numeric(od["driving_duration_hours"], errors="coerce").fillna(
        pd.to_numeric(od["shortest_time_hours"], errors="coerce").fillna(3.0)
    )
    od["cost"] = pd.to_numeric(od["amap_selfdrive_cost_yuan_per_two"], errors="coerce").fillna(
        pd.to_numeric(od["shortest_cost_yuan_per_two"], errors="coerce").fillna(250.0)
    )
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in od.itertuples(index=False):
        lookup[(row.from_spot_id, row.to_spot_id)] = (float(row.time), float(row.cost))
    return lookup


def route_theme_tags(theme: str) -> set[str]:
    lower = theme.lower()
    tags = set()
    if any(x in lower for x in ["heritage", "culture", "silk", "hotan", "south"]):
        tags.add("culture")
    if any(x in lower for x in ["lake", "grass", "kanas", "pamir", "desert", "north"]):
        tags.add("nature")
    if any(x in lower for x in ["east", "heritage", "silk"]):
        tags.add("east_culture")
    if any(x in lower for x in ["ili", "lake", "grass"]):
        tags.add("grass_lake")
    if any(x in lower for x in ["south", "hotan", "pamir", "border"]):
        tags.add("south_pamir")
    if any(x in lower for x in ["bazhou", "desert"]):
        tags.add("bazhou_desert")
    return tags or {"balanced"}


VARIANTS = [
    {
        "variant_code": "balanced_mass",
        "product_type": "capacity_balanced",
        "target_segment": "大众均衡",
        "target_spots": 10,
        "capacity_factor": 0.80,
        "comfort_factor": 0.96,
        "focus": "balanced",
        "description": "保留原线路主题，扩展为12天中等强度产品",
    },
    {
        "variant_code": "family_comfort",
        "product_type": "family_comfort",
        "target_segment": "亲子舒适",
        "target_spots": 8,
        "capacity_factor": 0.56,
        "comfort_factor": 1.08,
        "focus": "comfort",
        "description": "减少长转场与高强度景点，增加缓冲日",
    },
    {
        "variant_code": "senior_slow",
        "product_type": "senior_slow",
        "target_segment": "长者慢游",
        "target_spots": 7,
        "capacity_factor": 0.45,
        "comfort_factor": 1.15,
        "focus": "slow",
        "description": "低海拔、低转场、低红日的慢节奏产品",
    },
    {
        "variant_code": "low_pressure",
        "product_type": "decongested_route",
        "target_segment": "错峰分流",
        "target_spots": 9,
        "capacity_factor": 0.62,
        "comfort_factor": 1.00,
        "focus": "low_pressure",
        "description": "优先选高承载、低热点替代点，承担拥挤分流",
    },
    {
        "variant_code": "culture_deep",
        "product_type": "culture_deep",
        "target_segment": "文化深游",
        "target_spots": 9,
        "capacity_factor": 0.53,
        "comfort_factor": 0.98,
        "focus": "culture",
        "description": "增强丝路、民俗、考古文化主题",
    },
    {
        "variant_code": "nature_flagship",
        "product_type": "nature_flagship",
        "target_segment": "自然风景",
        "target_spots": 9,
        "capacity_factor": 0.58,
        "comfort_factor": 0.96,
        "focus": "nature",
        "description": "增强湖泊、草原、峡谷、冰川等自然体验",
    },
    {
        "variant_code": "premium_reserve",
        "product_type": "premium_reserve",
        "target_segment": "精品预约",
        "target_spots": 8,
        "capacity_factor": 0.48,
        "comfort_factor": 1.05,
        "focus": "quality",
        "description": "小团预约制，承载低但满意度和可控性高",
    },
]

PRODUCT_TYPE_ZH = {
    "capacity_balanced": "大众均衡",
    "family_comfort": "亲子舒适",
    "senior_slow": "长者慢游",
    "decongested_route": "错峰分流",
    "culture_deep": "文化深游",
    "nature_flagship": "自然风景",
    "premium_reserve": "精品预约",
}

SEED_THEME_ZH = {
    "North_Kanas": "北疆喀纳斯线",
    "Ili_Lake_Grass": "伊犁湖泊草原线",
    "East_Heritage": "东疆文化遗产线",
    "South_Culture": "南疆文化线",
    "Bazhou_Desert": "巴州沙漠湖泊线",
    "Hotan_Plus": "和田南疆扩展线",
    "Short_Urumqi_East": "乌鲁木齐东疆短线",
    "SilkRoad_Deep": "丝路深游线",
    "Border_Pamir": "帕米尔边境线",
}

SPOT_TOKEN_PRIORITY = {
    "P001": "乌鲁木齐",
    "P002": "乌鲁木齐",
    "P003": "东疆",
    "P004": "东疆",
    "P005": "东疆",
    "P006": "东疆",
    "P011": "东疆",
    "P012": "东疆",
    "P013": "东疆",
    "P008": "赛湖",
    "P017": "赛湖",
    "P009": "伊犁",
    "P010": "伊犁",
    "P014": "伊犁",
    "P015": "伊犁",
    "P016": "伊犁",
    "P018": "北疆",
    "P019": "北疆",
    "P020": "北疆",
    "P021": "北疆",
    "P022": "北疆",
    "P023": "北疆",
    "P024": "喀什",
    "P025": "喀什",
    "P026": "喀什",
    "P027": "帕米尔",
    "P028": "帕米尔",
    "P040": "帕米尔",
    "P029": "巴州",
    "P030": "巴州",
    "P031": "巴州",
    "P032": "巴州",
    "P033": "巴州",
    "P034": "巴州",
    "P037": "和田",
    "P039": "和田",
}


def compressed_route_tokens(sequence: list[str]) -> list[str]:
    tokens = ordered_unique([SPOT_TOKEN_PRIORITY.get(sid, "") for sid in sequence if SPOT_TOKEN_PRIORITY.get(sid, "")])
    token_set = set(tokens)
    if {"东疆", "赛湖", "喀什"}.issubset(token_set):
        return ["东疆", "赛湖", "喀什"]
    if {"北疆", "伊犁", "东疆"}.issubset(token_set):
        return ["北疆", "伊犁", "东疆"]
    if {"巴州", "东疆", "乌鲁木齐"}.issubset(token_set) and len(tokens) > 3:
        return ["巴州", "东疆", "乌鲁木齐"]
    if len(tokens) <= 3:
        return tokens
    if "乌鲁木齐" in tokens and len(tokens) > 3:
        tokens = [t for t in tokens if t != "乌鲁木齐"]
    if len(tokens) <= 3:
        return tokens
    return [tokens[0], tokens[len(tokens) // 2], tokens[-1]]


def build_route_product_name(
    sequence: list[str],
    seed_theme: str,
    product_type: str,
) -> tuple[str, str]:
    tokens = compressed_route_tokens(sequence)
    if not tokens:
        tokens = [SEED_THEME_ZH.get(seed_theme, seed_theme)]
    product_type_zh = PRODUCT_TYPE_ZH.get(product_type, product_type)
    name = f"{'—'.join(tokens)}{product_type_zh}12日线"
    seed_zh = SEED_THEME_ZH.get(seed_theme, seed_theme)
    note = f"由{seed_zh}种子生成的{product_type_zh}12天产品；seed仅表示初始线路来源，不等同于最终空间范围。"
    if seed_theme == "Short_Urumqi_East" and len(set(tokens)) >= 3:
        note = f"由乌鲁木齐东疆短线种子扩展为跨区{product_type_zh}12天产品，不再按短线产品解释。"
    return name, note


def candidate_pool_for_seed(
    seed_ids: list[str],
    seed_theme: str,
    focus: str,
    spot_meta: pd.DataFrame,
) -> pd.DataFrame:
    seed = spot_meta[spot_meta["spot_id"].isin(seed_ids)]
    seed_regions = set(seed["region_cluster"].dropna())
    seed_tags = route_theme_tags(seed_theme)
    pool = spot_meta[
        (~spot_meta["spot_id"].isin(seed_ids))
        & (~spot_meta["ordinary_tourist_restricted"].map(as_bool))
        & (~spot_meta["requires_approval"].map(as_bool))
    ].copy()
    if pool.empty:
        return pool

    same_region = pool["region_cluster"].isin(seed_regions).astype(float)
    nearby_region = pool["hub_name"].isin(set(seed["hub_name"].dropna())).astype(float)
    culture = pool["is_cultural"].astype(float) + pool["culture_norm"]
    nature = pool["is_natural"].astype(float) + pool["tag_natural"].astype(float)
    capacity = pool["capacity_norm"]
    priority = pool["priority_norm"]
    remote_penalty = pool["high_altitude_or_remote"].astype(float)

    if focus == "culture":
        score = 0.34 * same_region + 0.36 * culture + 0.16 * priority + 0.14 * capacity - 0.18 * remote_penalty
    elif focus == "nature":
        score = 0.32 * same_region + 0.36 * nature + 0.14 * priority + 0.18 * capacity - 0.12 * remote_penalty
    elif focus == "low_pressure":
        score = 0.28 * capacity + 0.25 * (1 - same_region) + 0.22 * priority + 0.15 * nature + 0.10 * culture
    elif focus == "comfort":
        score = 0.34 * same_region + 0.18 * nearby_region + 0.20 * capacity + 0.18 * priority - 0.28 * remote_penalty
    elif focus == "slow":
        score = 0.40 * same_region + 0.22 * capacity + 0.18 * priority - 0.40 * remote_penalty
    elif focus == "quality":
        score = 0.25 * same_region + 0.30 * priority + 0.22 * culture + 0.16 * nature + 0.07 * capacity
    else:
        score = 0.30 * same_region + 0.22 * priority + 0.20 * capacity + 0.14 * culture + 0.14 * nature

    if "culture" in seed_tags:
        score += 0.10 * culture
    if "nature" in seed_tags:
        score += 0.10 * nature

    pool["extension_score"] = score
    return pool.sort_values(["extension_score", "effective_capacity_beta_085"], ascending=False)


def order_sequence_by_region(sequence: list[str], spot_meta: pd.DataFrame, od_lookup: dict[tuple[str, str], tuple[float, float]]) -> list[str]:
    if len(sequence) <= 2:
        return sequence
    meta = spot_meta.set_index("spot_id")
    remaining = sequence[:]
    ordered = [remaining.pop(0)]
    while remaining:
        prev = ordered[-1]
        prev_region = meta.at[prev, "region_cluster"] if prev in meta.index else None
        best_idx = 0
        best_score = float("inf")
        for idx, sid in enumerate(remaining):
            travel = od_lookup.get((prev, sid), (4.5, 500.0))[0]
            region_penalty = 0 if sid in meta.index and meta.at[sid, "region_cluster"] == prev_region else 1.1
            score = travel + region_penalty
            if score < best_score:
                best_score = score
                best_idx = idx
        ordered.append(remaining.pop(best_idx))
    return ordered


def choose_primary_slot(row: pd.Series, day_index: int, focus: str) -> str:
    region = str(row.get("region_cluster", ""))
    visit_hours = float(row.get("visit_hours_mid", 3.0))
    heat_sensitive = any(x in region for x in ["东疆", "南疆", "巴州"])
    if focus in {"family_comfort", "slow", "comfort"} and day_index % 4 == 0:
        return "afternoon"
    if heat_sensitive or visit_hours >= 4.5:
        return "morning"
    if day_index % 5 == 0:
        return "evening"
    return "afternoon" if day_index % 2 == 0 else "morning"


def build_daily_itinerary_for_candidate(
    route_id: str,
    sequence: list[str],
    candidate: dict[str, Any],
    spot_meta: pd.DataFrame,
    od_lookup: dict[tuple[str, str], tuple[float, float]],
) -> pd.DataFrame:
    meta = spot_meta.set_index("spot_id")
    target_spots = len(sequence)
    buffer_days_needed = max(0, TARGET_ROUTE_DAYS - target_spots)
    buffer_after = set()
    if buffer_days_needed:
        step = max(2, math.floor(target_spots / buffer_days_needed))
        pos = step
        while len(buffer_after) < buffer_days_needed and pos <= target_spots:
            buffer_after.add(pos)
            pos += step
        pos = target_spots
        while len(buffer_after) < buffer_days_needed and pos >= 1:
            buffer_after.add(pos)
            pos -= 1

    rows = []
    day = 1
    previous_spot: str | None = None
    active_index = 0
    for sid in sequence:
        active_index += 1
        if day > TARGET_ROUTE_DAYS:
            break
        row = meta.loc[sid]
        travel_hours = od_lookup.get((previous_spot, sid), (1.2, 160.0))[0] if previous_spot else 1.3
        travel_hours = min(float(travel_hours), 8.5)
        raw_visit_hours = float(row["visit_hours_mid"])
        visit_hours = raw_visit_hours
        long_stay_note = ""
        if raw_visit_hours > 8.0:
            visit_hours = 6.8
            long_stay_note = "；原始建议为1-2天深度停留，运营排程按半日主活动+住宿体验处理"
        slot = choose_primary_slot(row, day, str(candidate["focus"]))
        activity_hours = visit_hours + min(travel_hours, 4.5) * 0.72
        if activity_hours > 9.5:
            activity_hours = 9.5
            long_stay_note += "；当日强度按产品质量红线封顶，需现场拆分团队或增加小交通"
        rows.append(
            {
                "route_id": route_id,
                "day_index": day,
                "day_type": "active",
                "spot_id": sid,
                "spot_name": row["spot_name"],
                "region_cluster": row["region_cluster"],
                "hub_id": row["hub_id"],
                "hub_name": row["hub_name"],
                "primary_slot": slot,
                "visit_hours": round(visit_hours, 2),
                "inbound_travel_hours_proxy": round(travel_hours, 2),
                "day_active_hours_proxy": round(activity_hours, 2),
                "overnight_hub_name": preferred_overnight_hub(row["hub_name"]),
                "time_window_note": f'{row.get("open_time", "")}-{row.get("close_time", "")}',
                "arrangement_note": f"主景点预约入园；按分时策略拆分团队{long_stay_note}",
            }
        )
        day += 1
        previous_spot = sid
        if active_index in buffer_after and day <= TARGET_ROUTE_DAYS:
            rows.append(
                {
                    "route_id": route_id,
                    "day_index": day,
                    "day_type": "buffer_transfer",
                    "spot_id": "",
                    "spot_name": "机动缓冲/城市补给",
                    "region_cluster": row["region_cluster"],
                    "hub_id": row["hub_id"],
                    "hub_name": row["hub_name"],
                    "primary_slot": "evening",
                    "visit_hours": 0.0,
                    "inbound_travel_hours_proxy": 0.8,
                    "day_active_hours_proxy": 2.2,
                    "overnight_hub_name": preferred_overnight_hub(row["hub_name"]),
                    "time_window_note": "not_applicable",
                    "arrangement_note": "用于错峰、补给、天气/交通扰动吸收",
                }
            )
            day += 1

    while day <= TARGET_ROUTE_DAYS:
        last = rows[-1] if rows else {}
        rows.append(
            {
                "route_id": route_id,
                "day_index": day,
                "day_type": "buffer_transfer",
                "spot_id": "",
                "spot_name": "机动缓冲/城市补给",
                "region_cluster": last.get("region_cluster", "乌鲁木齐周边"),
                "hub_id": last.get("hub_id", "H001"),
                "hub_name": last.get("hub_name", "乌鲁木齐市"),
                "primary_slot": "evening",
                "visit_hours": 0.0,
                "inbound_travel_hours_proxy": 0.7,
                "day_active_hours_proxy": 2.0,
                "overnight_hub_name": last.get("overnight_hub_name", "乌鲁木齐市"),
                "time_window_note": "not_applicable",
                "arrangement_note": "末端缓冲，便于航铁返程和预约修复",
            }
        )
        day += 1
    return pd.DataFrame(rows)


def generate_route_candidates(inputs: dict[str, pd.DataFrame], spot_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = inputs["routes"].copy()
    od_lookup = build_od_lookup(inputs)
    meta_by_id = spot_meta.set_index("spot_id")
    candidates: list[dict[str, Any]] = []
    itinerary_frames: list[pd.DataFrame] = []

    for seed_idx, seed in routes.iterrows():
        seed_ids = [sid for sid in parse_ids(seed["route_spot_ids"]) if sid in meta_by_id.index]
        if not seed_ids:
            continue
        seed_cap = float(seed["route_capacity_persons_12day"])
        seed_meta = spot_meta[spot_meta["spot_id"].isin(seed_ids)]
        seed_q25 = float(seed_meta["effective_capacity_beta_085"].quantile(0.25))
        seed_theme = str(seed["route_theme"])
        for variant_idx, variant in enumerate(VARIANTS, start=1):
            target_spots = int(variant["target_spots"])
            sequence = seed_ids[:]
            pool = candidate_pool_for_seed(seed_ids, seed_theme, str(variant["focus"]), spot_meta)
            for sid in pool["spot_id"].tolist():
                if len(sequence) >= target_spots:
                    break
                sequence.append(sid)
            sequence = ordered_unique(sequence)
            sequence = order_sequence_by_region(sequence, spot_meta, od_lookup)
            active_meta = spot_meta[spot_meta["spot_id"].isin(sequence)].copy()
            active_meta = active_meta.set_index("spot_id").loc[sequence].reset_index()
            regions = ordered_unique(active_meta["region_cluster"].astype(str).tolist())
            hubs = ordered_unique(active_meta["hub_name"].astype(str).tolist())
            culture_count = int(active_meta["is_cultural"].sum())
            nature_count = int(active_meta["is_natural"].sum())
            remote_count = int(active_meta["high_altitude_or_remote"].sum())
            reservation_count = int(active_meta["requires_reservation"].sum())
            q25 = float(active_meta["effective_capacity_beta_085"].quantile(0.25))
            capacity_ratio = np.clip(q25 / max(seed_q25, 1), 0.72, 1.18)
            region_diversity = len(regions)
            theme_balance = 1 - abs(culture_count - nature_count) / max(1, len(sequence))
            travel_times = []
            for a, b in zip(sequence[:-1], sequence[1:]):
                travel_times.append(od_lookup.get((a, b), (4.5, 500.0))[0])
            avg_travel = float(np.mean(travel_times)) if travel_times else 1.5
            long_transfer_days = int(sum(t > 5.5 for t in travel_times))
            buffer_days = TARGET_ROUTE_DAYS - len(sequence)
            route_capacity = seed_cap * float(variant["capacity_factor"]) * capacity_ratio
            if variant["focus"] == "low_pressure":
                route_capacity *= 1.08
            route_capacity *= max(0.84, 1.0 - 0.025 * remote_count)
            route_capacity = int(round(np.clip(route_capacity, 2_000, 16_500) / 10) * 10)

            attraction_score = (
                44 * float(active_meta["priority_norm"].mean())
                + 18 * float(active_meta["culture_norm"].mean())
                + 14 * float(active_meta["capacity_norm"].mean())
                + 12 * theme_balance
                + 12 * min(region_diversity / 4, 1)
            )
            comfort_score = 100 - 4.8 * avg_travel - 4.2 * long_transfer_days - 4.5 * remote_count + 3.4 * buffer_days
            comfort_score *= float(variant["comfort_factor"])
            comfort_score = float(np.clip(comfort_score, 45, 98))
            low_pressure_score = (
                52 * float(active_meta["capacity_norm"].mean())
                + 26 * min(region_diversity / 5, 1)
                + 12 * buffer_days / max(1, TARGET_ROUTE_DAYS)
                + 10 * (1 - reservation_count / max(1, len(sequence)))
            )
            diversity_score = 45 * min(region_diversity / 5, 1) + 25 * theme_balance + 18 * min(
                (culture_count + nature_count) / max(1, len(sequence)), 1
            ) + 12 * min(buffer_days / 4, 1)
            resource_intensity = 0.55 * len(sequence) + 0.35 * avg_travel + 0.45 * remote_count
            quality_score = (
                0.34 * attraction_score
                + 0.24 * comfort_score
                + 0.20 * low_pressure_score
                + 0.16 * diversity_score
                + 0.00055 * route_capacity
                - 1.8 * resource_intensity
            )
            route_id = f"Q4V2_R{len(candidates) + 1:03d}"
            route_theme_code = f"{seed_theme}_{variant['variant_code']}"
            route_product_name_zh, route_name_note = build_route_product_name(
                sequence, seed_theme, str(variant["product_type"])
            )
            candidate = {
                "route_id": route_id,
                "seed_column_id": seed["column_id"],
                "seed_route_theme": seed_theme,
                "route_theme_code": route_theme_code,
                "route_theme": route_product_name_zh,
                "route_name_note": route_name_note,
                "product_type": variant["product_type"],
                "target_segment": variant["target_segment"],
                "variant_code": variant["variant_code"],
                "days": TARGET_ROUTE_DAYS,
                "active_spot_days": len(sequence),
                "buffer_days": buffer_days,
                "spots_count": len(sequence),
                "route_spot_ids": ";".join(sequence),
                "route_sequence": " -> ".join(active_meta["spot_name"].tolist()),
                "region_set": ";".join(regions),
                "hub_set": ";".join(hubs),
                "culture_spots": culture_count,
                "nature_spots": nature_count,
                "remote_or_high_altitude_spots": remote_count,
                "reservation_spots": reservation_count,
                "route_capacity_persons_12day": route_capacity,
                "seed_capacity_persons_12day": int(seed_cap),
                "capacity_factor_vs_seed": round(route_capacity / seed_cap, 3),
                "attraction_score": round(attraction_score, 3),
                "diversity_score": round(diversity_score, 3),
                "comfort_score": round(comfort_score, 3),
                "low_pressure_score": round(low_pressure_score, 3),
                "avg_interspot_travel_hours_proxy": round(avg_travel, 3),
                "long_transfer_days_proxy": long_transfer_days,
                "resource_intensity_score": round(resource_intensity, 3),
                "portfolio_quality_score": round(quality_score, 3),
                "capacity_source_note": "由旧9线路容量锚定，并按景点安全承载量、线路长度、分流属性校准",
                "generation_note": variant["description"],
            }
            candidates.append(candidate)
            itinerary_frames.append(build_daily_itinerary_for_candidate(route_id, sequence, {**candidate, **variant}, spot_meta, od_lookup))

    candidates_df = pd.DataFrame(candidates)
    itinerary_df = pd.concat(itinerary_frames, ignore_index=True)
    return candidates_df, itinerary_df


def overlap_ratio(ids_a: str, ids_b: str) -> float:
    a = set(parse_ids(ids_a))
    b = set(parse_ids(ids_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_portfolio(candidates: pd.DataFrame, target_capacity: int = 157_000) -> pd.DataFrame:
    df = candidates.copy()
    selected_ids: list[str] = []
    selected_capacity = 0
    theme_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    # First ensure every original theme has at least one representative.
    for seed_theme, group in df.groupby("seed_route_theme"):
        row = group.sort_values("portfolio_quality_score", ascending=False).iloc[0]
        selected_ids.append(row["route_id"])
        selected_capacity += int(row["route_capacity_persons_12day"])
        theme_counts[str(row["seed_route_theme"])] += 1
        type_counts[str(row["product_type"])] += 1

    while selected_capacity < target_capacity and len(selected_ids) < 24:
        best_id = None
        best_score = -1e18
        for row in df[~df["route_id"].isin(selected_ids)].itertuples(index=False):
            max_overlap = 0.0
            for selected in selected_ids:
                selected_row = df[df["route_id"] == selected].iloc[0]
                max_overlap = max(max_overlap, overlap_ratio(row.route_spot_ids, selected_row["route_spot_ids"]))
            same_theme_penalty = max(0, theme_counts[str(row.seed_route_theme)] - 1) * 5.8
            type_penalty = max(0, type_counts[str(row.product_type)] - 3) * 4.5
            east_penalty = 5.0 if str(row.seed_route_theme) == "East_Heritage" and theme_counts[str(row.seed_route_theme)] >= 2 else 0.0
            score = (
                float(row.portfolio_quality_score)
                + 0.0011 * float(row.route_capacity_persons_12day)
                + 0.16 * float(row.low_pressure_score)
                - 22.0 * max_overlap
                - same_theme_penalty
                - type_penalty
                - east_penalty
            )
            if score > best_score:
                best_score = score
                best_id = row.route_id
        if best_id is None:
            break
        best_row = df[df["route_id"] == best_id].iloc[0]
        selected_ids.append(best_id)
        selected_capacity += int(best_row["route_capacity_persons_12day"])
        theme_counts[str(best_row["seed_route_theme"])] += 1
        type_counts[str(best_row["product_type"])] += 1

    df["selected_in_portfolio"] = df["route_id"].isin(selected_ids)
    df["selection_status"] = np.where(df["selected_in_portfolio"], "selected_q4v2_portfolio", "candidate_not_selected")
    order_map = {rid: idx + 1 for idx, rid in enumerate(selected_ids)}
    df["selection_order"] = df["route_id"].map(order_map)
    return df


def capped_allocate(demand: float, capacities: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    capacities = np.array(capacities, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = np.where(weights > 0, weights, 1.0)
    q = np.zeros_like(capacities)
    active = capacities > 1e-9
    for _ in range(len(capacities) + 5):
        remaining = demand - float(q.sum())
        if remaining <= 1e-6 or not active.any():
            break
        w = weights * active
        if w.sum() <= 0:
            w = active.astype(float)
        add = remaining * w / w.sum()
        room = np.maximum(capacities - q, 0)
        actual_add = np.minimum(add, room)
        q += actual_add
        active = (capacities - q) > 1e-6
        if actual_add.sum() <= 1e-6:
            break
    overflow = max(0.0, demand - float(q.sum()))
    return q, overflow


def allocation_weights(portfolio: pd.DataFrame, mode: str) -> np.ndarray:
    cap = np.array(portfolio["route_capacity_persons_12day"], dtype=float)
    cap_share = cap / max(cap.sum(), 1.0)
    quality = np.array(portfolio["portfolio_quality_score"], dtype=float)
    attraction = np.array(portfolio["attraction_score"], dtype=float)
    comfort = np.array(portfolio["comfort_score"], dtype=float)
    lowp = np.array(portfolio["low_pressure_score"], dtype=float)
    def scaled(arr: np.ndarray) -> np.ndarray:
        if np.max(arr) - np.min(arr) < 1e-9:
            return np.ones_like(arr)
        return 0.35 + 0.65 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    if mode == "capacity_ratio":
        return cap_share
    if mode == "preference_elastic":
        utility = 0.58 * np.log1p(cap) + 1.40 * scaled(attraction) + 0.75 * scaled(comfort) + 0.55 * scaled(lowp)
        utility = utility - utility.max()
        return np.exp(utility)
    if mode == "balanced_optimized":
        return 0.50 * cap_share + 0.21 * scaled(quality) + 0.17 * scaled(lowp) + 0.12 * scaled(comfort)
    raise ValueError(mode)


def build_capacity_ratio_allocation(candidates: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    portfolio = candidates[candidates["selected_in_portfolio"]].sort_values("selection_order").copy()
    caps = np.array(portfolio["route_capacity_persons_12day"], dtype=float)
    rows = []
    allocations: dict[str, np.ndarray] = {}
    overflows: dict[str, float] = {}
    for mode in ["capacity_ratio", "preference_elastic", "balanced_optimized"]:
        q, overflow = capped_allocate(BASE_DEMAND, caps, allocation_weights(portfolio, mode))
        allocations[mode] = q
        overflows[mode] = overflow

    for idx, route in enumerate(portfolio.itertuples(index=False)):
        old_match = inputs["flow"][inputs["flow"]["route_theme"] == getattr(route, "seed_route_theme")]
        old_alloc = int(old_match["allocated_visitors"].iloc[0]) if not old_match.empty else np.nan
        rows.append(
            {
                "route_id": route.route_id,
                "route_theme": route.route_theme,
                "route_theme_code": route.route_theme_code,
                "route_name_note": route.route_name_note,
                "seed_column_id": route.seed_column_id,
                "seed_route_theme": route.seed_route_theme,
                "product_type": route.product_type,
                "target_segment": route.target_segment,
                "route_capacity_persons_12day": int(route.route_capacity_persons_12day),
                "old_9route_allocation_same_theme": old_alloc,
                "strict_capacity_ratio_visitors": int(round(allocations["capacity_ratio"][idx])),
                "preference_elastic_visitors": int(round(allocations["preference_elastic"][idx])),
                "balanced_optimized_visitors": int(round(allocations["balanced_optimized"][idx])),
                "balanced_utilization": round(float(allocations["balanced_optimized"][idx] / max(route.route_capacity_persons_12day, 1)), 4),
                "attraction_score": route.attraction_score,
                "comfort_score": route.comfort_score,
                "low_pressure_score": route.low_pressure_score,
                "allocation_rule_note": "题设核心口径：游客名额按线路投放容量成比例；V2同时给出偏好弹性与平衡优化对照。",
            }
        )
    out = pd.DataFrame(rows)
    for mode, overflow in overflows.items():
        out[f"{mode}_unserved_total"] = int(round(overflow))
    return out


def selected_portfolio_table(candidates: pd.DataFrame, allocation: pd.DataFrame) -> pd.DataFrame:
    selected = candidates[candidates["selected_in_portfolio"]].sort_values("selection_order").copy()
    out = selected.merge(
        allocation[
            [
                "route_id",
                "strict_capacity_ratio_visitors",
                "preference_elastic_visitors",
                "balanced_optimized_visitors",
                "balanced_utilization",
            ]
        ],
        on="route_id",
        how="left",
    )
    cols = [
        "selection_order",
        "route_id",
        "route_theme",
        "route_theme_code",
        "seed_column_id",
        "seed_route_theme",
        "product_type",
        "target_segment",
        "days",
        "active_spot_days",
        "buffer_days",
        "spots_count",
        "region_set",
        "route_capacity_persons_12day",
        "balanced_optimized_visitors",
        "balanced_utilization",
        "attraction_score",
        "comfort_score",
        "low_pressure_score",
        "diversity_score",
        "avg_interspot_travel_hours_proxy",
        "route_sequence",
        "route_name_note",
        "generation_note",
    ]
    return out[cols]


def build_route_quality_audit(selected: pd.DataFrame, itinerary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_lookup = selected.set_index("route_id")
    for route_id, group in itinerary[itinerary["route_id"].isin(set(selected["route_id"]))].groupby("route_id"):
        active = group[group["day_type"] == "active"].copy()
        route = selected_lookup.loc[route_id]
        max_day = float(active["day_active_hours_proxy"].max()) if not active.empty else 0.0
        red_days = int((active["day_active_hours_proxy"] > 9.0).sum()) if not active.empty else 0
        orange_days = int(((active["day_active_hours_proxy"] > 8.0) & (active["day_active_hours_proxy"] <= 9.0)).sum()) if not active.empty else 0
        long_transfer_days = int((active["inbound_travel_hours_proxy"] > 6.5).sum()) if not active.empty else 0
        buffer_days = int(route["buffer_days"])
        problem_days = active[
            (active["day_active_hours_proxy"] > 8.0) | (active["inbound_travel_hours_proxy"] > 6.5)
        ][["day_index", "spot_name", "day_active_hours_proxy", "inbound_travel_hours_proxy"]]
        quality_pass = bool(max_day <= 9.0 and red_days == 0 and long_transfer_days <= 3 and buffer_days >= 3)
        notes = []
        if max_day > 9.5:
            notes.append("存在超过9.5小时的单日强度，需拆分景点或增加缓冲")
        elif red_days > 0:
            notes.append("存在9小时以上高强度日，不建议直接作为普通团队产品投放，需分批/小交通/拆段修复")
        if long_transfer_days > 3:
            notes.append("长转场日偏多，建议改为跨区全疆产品或拆成两段销售")
        if buffer_days < 3:
            notes.append("缓冲日不足，抗天气/交通扰动能力弱")
        if not notes:
            notes.append("产品强度处于可投放范围")
        rows.append(
            {
                "route_id": route_id,
                "route_theme": route["route_theme"],
                "route_theme_code": route["route_theme_code"],
                "seed_route_theme": route["seed_route_theme"],
                "product_type": route["product_type"],
                "max_day_active_hours": round(max_day, 3),
                "red_days": red_days,
                "orange_days": orange_days,
                "long_transfer_days": long_transfer_days,
                "buffer_days": buffer_days,
                "quality_pass": quality_pass,
                "highest_pressure_days": "; ".join(
                    f"D{int(r.day_index)}:{r.spot_name}/{float(r.day_active_hours_proxy):.1f}h/{float(r.inbound_travel_hours_proxy):.1f}h_travel"
                    for r in problem_days.itertuples(index=False)
                ),
                "quality_note": "；".join(notes),
            }
        )
    return pd.DataFrame(rows).sort_values(["quality_pass", "max_day_active_hours"], ascending=[True, False])


def load_by_spot_slot(
    itinerary: pd.DataFrame,
    portfolio: pd.DataFrame,
    allocations: pd.DataFrame,
    spot_meta: pd.DataFrame,
    booking_ratio: float = 0.95,
    slot_stagger_strength: float = 0.28,
    spot_capacity_factor: float = 1.0,
) -> pd.DataFrame:
    selected_ids = set(portfolio["route_id"])
    itin = itinerary[itinerary["route_id"].isin(selected_ids) & (itinerary["day_type"] == "active")].copy()
    alloc = dict(zip(allocations["route_id"], allocations["balanced_optimized_visitors"]))
    meta = spot_meta.set_index("spot_id")
    rows = []
    for r in itin.itertuples(index=False):
        # q_r is the total number of visitors entering this route product during the
        # 12-day holiday window. For daily/slot loads we convert it into rolling
        # calendar-day cohorts, otherwise a route's whole 12-day volume would be
        # incorrectly placed on every itinerary day.
        visitors = float(alloc.get(r.route_id, 0.0)) / TARGET_ROUTE_DAYS
        if visitors <= 0 or not r.spot_id:
            continue
        primary = r.primary_slot if r.primary_slot in SLOTS else "afternoon"
        primary_load_share = max(0.52, 1.0 - slot_stagger_strength)
        secondary_slot = "afternoon" if primary == "morning" else "morning"
        tertiary_slot = "evening"
        split = {
            primary: primary_load_share,
            secondary_slot: min(0.34, slot_stagger_strength * 0.78),
            tertiary_slot: max(0.0, 1.0 - primary_load_share - min(0.34, slot_stagger_strength * 0.78)),
        }
        for slot, share in split.items():
            if share <= 0:
                continue
            spot_row = meta.loc[r.spot_id]
            slot_capacity = (
                float(spot_row["daily_capacity_persons"])
                * SLOT_SHARES[slot]
                * booking_ratio
                * spot_capacity_factor
            )
            rows.append(
                {
                    "route_id": r.route_id,
                    "day_index": int(r.day_index),
                    "spot_id": r.spot_id,
                    "spot_name": r.spot_name,
                    "region_cluster": r.region_cluster,
                    "time_slot": slot,
                    "allocated_visitors_share": round(share, 4),
                    "visitor_load": visitors * share,
                    "slot_capacity_persons": slot_capacity,
                    "calendar_flow_assumption": "route_total_visitors_divided_by_12_day_rolling_departures",
                }
            )
    if not rows:
        return pd.DataFrame()
    load = pd.DataFrame(rows)
    agg = load.groupby(
        ["day_index", "spot_id", "spot_name", "region_cluster", "time_slot"], as_index=False
    ).agg(
        visitor_load=("visitor_load", "sum"),
        slot_capacity_persons=("slot_capacity_persons", "max"),
        contributing_routes=("route_id", lambda x: ";".join(sorted(set(x)))),
    )
    agg["utilization"] = agg["visitor_load"] / agg["slot_capacity_persons"].replace(0, np.nan)
    agg["overload_persons"] = (agg["visitor_load"] - agg["slot_capacity_persons"]).clip(lower=0)
    agg["pressure_level"] = pd.cut(
        agg["utilization"],
        bins=[-np.inf, 0.75, 0.9, 1.0, np.inf],
        labels=["green", "yellow", "orange", "red"],
    ).astype(str)
    for col in ["visitor_load", "slot_capacity_persons", "utilization", "overload_persons"]:
        agg[col] = agg[col].astype(float).round(4)
    return agg.sort_values(["day_index", "region_cluster", "spot_id", "time_slot"])


def build_hotel_load(
    itinerary: pd.DataFrame,
    portfolio: pd.DataFrame,
    allocations: pd.DataFrame,
    hotel_hubs: pd.DataFrame,
    booking_ratio: float = 0.95,
    hotel_capacity_factor: float = 1.0,
    resource_boost: float = 1.0,
) -> pd.DataFrame:
    selected_ids = set(portfolio["route_id"])
    itin = itinerary[itinerary["route_id"].isin(selected_ids)].copy()
    alloc = dict(zip(allocations["route_id"], allocations["balanced_optimized_visitors"]))
    hub_caps = hotel_hubs.copy()
    hub_caps["hotel_capacity_rooms_simulated"] = pd.to_numeric(
        hub_caps["hotel_capacity_rooms_simulated"], errors="coerce"
    ).fillna(40)
    hub_caps["q4_operational_room_capacity"] = hub_caps.apply(
        lambda r: calibrated_hub_room_capacity(r["hub_name"], r["hotel_capacity_rooms_simulated"]), axis=1
    )
    cap_by_hub = dict(zip(hub_caps["hub_name"], hub_caps["q4_operational_room_capacity"]))
    source_by_hub = dict(zip(hub_caps["hub_name"], hub_caps["source_type"]))
    rows = []
    for r in itin.itertuples(index=False):
        visitors = float(alloc.get(r.route_id, 0.0)) / TARGET_ROUTE_DAYS
        if visitors <= 0:
            continue
        rooms = visitors / 2.25
        hub = str(r.overnight_hub_name or r.hub_name)
        base_cap = float(cap_by_hub.get(hub, calibrated_hub_room_capacity(hub, 80)))
        room_capacity = base_cap * booking_ratio * hotel_capacity_factor * resource_boost
        rows.append(
            {
                "day_index": int(r.day_index),
                "hub_name": hub,
                "room_demand": rooms,
                "room_capacity": room_capacity,
                "contributing_route": r.route_id,
                "capacity_source_type": f"{source_by_hub.get(hub, 'derived_from_hub_default')}|q4_operational_room_pool_calibrated",
                "calendar_flow_assumption": "route_total_visitors_divided_by_12_day_rolling_departures",
            }
        )
    load = pd.DataFrame(rows)
    agg = load.groupby(["day_index", "hub_name"], as_index=False).agg(
        room_demand=("room_demand", "sum"),
        room_capacity=("room_capacity", "max"),
        contributing_routes=("contributing_route", lambda x: ";".join(sorted(set(x)))),
        capacity_source_type=("capacity_source_type", "first"),
    )
    agg["utilization"] = agg["room_demand"] / agg["room_capacity"].replace(0, np.nan)
    agg["overloaded_rooms"] = (agg["room_demand"] - agg["room_capacity"]).clip(lower=0)
    agg["resource_status"] = pd.cut(
        agg["utilization"],
        bins=[-np.inf, 0.78, 0.93, 1.0, np.inf],
        labels=["green", "yellow", "orange", "red"],
    ).astype(str)
    for col in ["room_demand", "room_capacity", "utilization", "overloaded_rooms"]:
        agg[col] = agg[col].astype(float).round(4)
    return agg.sort_values(["day_index", "utilization"], ascending=[True, False])


def build_vehicle_guide_load(
    itinerary: pd.DataFrame,
    portfolio: pd.DataFrame,
    allocations: pd.DataFrame,
    hotel_hubs: pd.DataFrame,
    spot_load: pd.DataFrame,
    vehicle_capacity_factor: float = 1.0,
    resource_boost: float = 1.0,
) -> pd.DataFrame:
    selected_ids = set(portfolio["route_id"])
    itin = itinerary[itinerary["route_id"].isin(selected_ids)].copy()
    alloc = dict(zip(allocations["route_id"], allocations["balanced_optimized_visitors"]))
    hub_caps = hotel_hubs.copy()
    hub_caps["hotel_capacity_rooms_simulated"] = pd.to_numeric(
        hub_caps["hotel_capacity_rooms_simulated"], errors="coerce"
    ).fillna(40)
    hub_caps["q4_operational_room_capacity"] = hub_caps.apply(
        lambda r: calibrated_hub_room_capacity(r["hub_name"], r["hotel_capacity_rooms_simulated"]), axis=1
    )
    cap_by_hub = dict(zip(hub_caps["hub_name"], hub_caps["q4_operational_room_capacity"]))
    rows = []
    for r in itin.itertuples(index=False):
        visitors = float(alloc.get(r.route_id, 0.0)) / TARGET_ROUTE_DAYS
        if visitors <= 0:
            continue
        hub = str(r.hub_name)
        rooms = float(cap_by_hub.get(hub, calibrated_hub_room_capacity(hub, 80)))
        available_buses = max(8.0, rooms / 4.5) * vehicle_capacity_factor * resource_boost
        available_guides = max(10.0, rooms / 3.2) * vehicle_capacity_factor * resource_boost
        available_shuttles = max(6.0, rooms / 5.0) * vehicle_capacity_factor * resource_boost
        if r.day_type == "active":
            bus_demand = visitors / 38.0
            guide_demand = visitors / 28.0
            shuttle_demand = visitors / 45.0
        else:
            bus_demand = visitors / 55.0
            guide_demand = visitors / 70.0
            shuttle_demand = visitors / 120.0
        for resource_type, demand, capacity in [
            ("coach_bus", bus_demand, available_buses),
            ("licensed_guide", guide_demand, available_guides),
            ("scenic_shuttle_or_parking", shuttle_demand, available_shuttles),
        ]:
            rows.append(
                {
                    "day_index": int(r.day_index),
                    "hub_name": hub,
                    "resource_type": resource_type,
                    "resource_demand_units": demand,
                    "resource_capacity_units": capacity,
                    "contributing_route": r.route_id,
                    "resource_capacity_note": "由住宿枢纽规模推导的校准运力/导游/摆渡车能力，非实时调度库存",
                    "calendar_flow_assumption": "route_total_visitors_divided_by_12_day_rolling_departures",
                }
            )
    load = pd.DataFrame(rows)
    agg = load.groupby(["day_index", "hub_name", "resource_type"], as_index=False).agg(
        resource_demand_units=("resource_demand_units", "sum"),
        resource_capacity_units=("resource_capacity_units", "max"),
        contributing_routes=("contributing_route", lambda x: ";".join(sorted(set(x)))),
        resource_capacity_note=("resource_capacity_note", "first"),
    )
    agg["utilization"] = agg["resource_demand_units"] / agg["resource_capacity_units"].replace(0, np.nan)
    agg["shortage_units"] = (agg["resource_demand_units"] - agg["resource_capacity_units"]).clip(lower=0)
    agg["resource_status"] = pd.cut(
        agg["utilization"],
        bins=[-np.inf, 0.78, 0.93, 1.0, np.inf],
        labels=["green", "yellow", "orange", "red"],
    ).astype(str)
    for col in ["resource_demand_units", "resource_capacity_units", "utilization", "shortage_units"]:
        agg[col] = agg[col].astype(float).round(4)

    if not spot_load.empty:
        spot_aux = spot_load.groupby(["day_index", "region_cluster"], as_index=False).agg(
            scenic_peak_load=("visitor_load", "max"),
            scenic_overload_persons=("overload_persons", "sum"),
        )
        # Keep the resource output unified; scenic shuttle/parking rows already represent this capacity family.
        agg = agg.merge(
            spot_aux.rename(columns={"region_cluster": "hub_name"}),
            on=["day_index", "hub_name"],
            how="left",
        )
    return agg.sort_values(["day_index", "utilization"], ascending=[True, False])


def build_shadow_prices(
    spot_load: pd.DataFrame,
    hotel_load: pd.DataFrame,
    resource_load: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not spot_load.empty:
        for r in spot_load.itertuples(index=False):
            util = float(r.utilization)
            pressure = max(0.0, util - 0.82)
            price = 30 + 160 * pressure + 0.018 * float(r.overload_persons)
            rows.append(
                {
                    "constraint_family": "spot_timeslot_capacity",
                    "constraint_id": f"D{r.day_index}_{r.spot_id}_{r.time_slot}",
                    "day_index": int(r.day_index),
                    "location": r.spot_name,
                    "resource_or_slot": r.time_slot,
                    "load": float(r.visitor_load),
                    "capacity": float(r.slot_capacity_persons),
                    "utilization": util,
                    "overload": float(r.overload_persons),
                    "shadow_price_index": price,
                    "interpretation": "增加一个分时预约容量单位对拥挤罚项的边际缓解强度",
                }
            )
    for r in hotel_load.itertuples(index=False):
        util = float(r.utilization)
        pressure = max(0.0, util - 0.80)
        price = 45 + 210 * pressure + 0.09 * float(r.overloaded_rooms)
        rows.append(
            {
                "constraint_family": "hotel_room_capacity",
                "constraint_id": f"D{r.day_index}_{r.hub_name}",
                "day_index": int(r.day_index),
                "location": r.hub_name,
                "resource_or_slot": "hotel_rooms",
                "load": float(r.room_demand),
                "capacity": float(r.room_capacity),
                "utilization": util,
                "overload": float(r.overloaded_rooms),
                "shadow_price_index": price,
                "interpretation": "增加一间可售房或跨区住宿协调对拒载/等待罚项的边际缓解强度",
            }
        )
    for r in resource_load.itertuples(index=False):
        util = float(r.utilization)
        pressure = max(0.0, util - 0.80)
        price = 50 + 230 * pressure + 7.5 * float(r.shortage_units)
        rows.append(
            {
                "constraint_family": "vehicle_guide_shuttle_capacity",
                "constraint_id": f"D{r.day_index}_{r.hub_name}_{r.resource_type}",
                "day_index": int(r.day_index),
                "location": r.hub_name,
                "resource_or_slot": r.resource_type,
                "load": float(r.resource_demand_units),
                "capacity": float(r.resource_capacity_units),
                "utilization": util,
                "overload": float(r.shortage_units),
                "shadow_price_index": price,
                "interpretation": "增加车辆/导游/摆渡资源对系统服务损失的边际缓解强度",
            }
        )
    out = pd.DataFrame(rows)
    out["shadow_price_index"] = out["shadow_price_index"].astype(float).round(4)
    return out.sort_values(["shadow_price_index", "utilization"], ascending=False)


def build_reservation_slot_policy(spot_load: pd.DataFrame) -> pd.DataFrame:
    if spot_load.empty:
        return pd.DataFrame()
    rows = []
    for r in spot_load.itertuples(index=False):
        util = float(r.utilization)
        if util >= 1.0:
            ratio = 0.84
            batch = "40%首批+30%二批+20%候补+10%现场应急"
            action = "red: 暂停加量，向低压线路和相邻时段转移，并补摆渡/安检资源"
        elif util >= 0.9:
            ratio = 0.90
            batch = "45%首批+35%二批+20%候补"
            action = "orange: 预约上限降至90%，团队拆成早/午两波"
        elif util >= 0.75:
            ratio = 0.95
            batch = "55%首批+30%二批+15%机动"
            action = "yellow: 保留5%机动名额，实时监测入园速度"
        else:
            ratio = 1.00
            batch = "70%首批+20%二批+10%机动"
            action = "green: 可正常放票，保留少量应急名额"
        rows.append(
            {
                "day_index": int(r.day_index),
                "spot_id": r.spot_id,
                "spot_name": r.spot_name,
                "region_cluster": r.region_cluster,
                "time_slot": r.time_slot,
                "baseline_load": float(r.visitor_load),
                "baseline_capacity": float(r.slot_capacity_persons),
                "baseline_utilization": util,
                "recommended_booking_ratio": ratio,
                "release_batch_plan": batch,
                "operation_action": action,
                "policy_note": "分时预约策略按时段压力动态调整，不再只做12天总容量上限。",
            }
        )
    return pd.DataFrame(rows).sort_values(["day_index", "baseline_utilization"], ascending=[True, False])


def evaluate_policy_scenario(
    policy: dict[str, Any],
    scenario: dict[str, Any],
    selected_portfolio: pd.DataFrame,
    allocation_template: pd.DataFrame,
    itinerary: pd.DataFrame,
    spot_meta: pd.DataFrame,
    hotel_hubs: pd.DataFrame,
    legacy_total_capacity: float,
) -> dict[str, Any]:
    demand = BASE_DEMAND * float(scenario["demand_multiplier"])
    if policy["portfolio"] == "legacy":
        capacity = legacy_total_capacity * float(policy["booking_ratio"])
        served = min(demand, capacity)
        rejected = max(0.0, demand - served)
        utilization = served / max(capacity, 1.0)
        expected_loss = rejected * 1.0 + max(0, utilization - 0.92) * 18_000 * float(scenario["volatility"])
        return {
            "scenario_id": scenario["scenario_id"],
            "scenario_name": scenario["scenario_name"],
            "demand_multiplier": scenario["demand_multiplier"],
            "policy_id": policy["policy_id"],
            "policy_name": policy["policy_name"],
            "portfolio_type": policy["portfolio"],
            "demand_visitors": int(round(demand)),
            "served_visitors": int(round(served)),
            "rejected_or_overflow_visitors": int(round(rejected)),
            "offered_capacity": int(round(capacity)),
            "served_ratio": round(served / demand, 4),
            "system_utilization": round(utilization, 4),
            "max_spot_timeslot_utilization": np.nan,
            "overloaded_spot_timeslots": np.nan,
            "spot_overload_persons": np.nan,
            "max_hotel_utilization": np.nan,
            "hotel_overloaded_rooms": np.nan,
            "max_vehicle_guide_utilization": np.nan,
            "vehicle_guide_shortage_units": np.nan,
            "congestion_wait_index": round(max(0, utilization - 0.86) * 1.7, 4),
            "expected_loss_index": round(expected_loss, 4),
            "cvar90_loss_index": round(expected_loss * 1.18, 4),
            "policy_pass": bool(served / demand >= 0.95 and utilization <= 1.0),
            "soft_policy_pass": bool(served / demand >= 0.95 and utilization <= 1.0),
            "strict_policy_pass": False,
            "resource_evaluable": False,
            "main_risk": "旧9线路只做总容量，无法识别分时/酒店/车辆瓶颈",
        }

    portfolio = selected_portfolio.copy()
    capacities = (
        np.array(portfolio["route_capacity_persons_12day"], dtype=float)
        * float(policy["booking_ratio"])
        * float(policy["route_capacity_boost"])
        * float(scenario["spot_capacity_factor"])
    )
    weights = allocation_weights(portfolio, "preference_elastic" if policy["preference_elastic"] else "capacity_ratio")
    q, overflow = capped_allocate(demand, capacities, weights)
    alloc = pd.DataFrame({"route_id": portfolio["route_id"].tolist(), "balanced_optimized_visitors": q})
    served = float(q.sum())
    rejected = float(overflow)
    spot_load = load_by_spot_slot(
        itinerary,
        portfolio,
        alloc,
        spot_meta,
        booking_ratio=float(policy["booking_ratio"]),
        slot_stagger_strength=float(policy["slot_stagger_strength"]),
        spot_capacity_factor=float(scenario["spot_capacity_factor"]),
    )
    hotel_load = build_hotel_load(
        itinerary,
        portfolio,
        alloc,
        hotel_hubs,
        booking_ratio=float(policy["booking_ratio"]),
        hotel_capacity_factor=float(scenario["hotel_capacity_factor"]),
        resource_boost=float(policy["resource_boost"]),
    )
    resource_load = build_vehicle_guide_load(
        itinerary,
        portfolio,
        alloc,
        hotel_hubs,
        spot_load,
        vehicle_capacity_factor=float(scenario["vehicle_capacity_factor"]),
        resource_boost=float(policy["resource_boost"]),
    )
    max_spot_util = float(spot_load["utilization"].max()) if not spot_load.empty else 0.0
    overloaded_slots = int((spot_load["overload_persons"] > 0).sum()) if not spot_load.empty else 0
    spot_overload = float(spot_load["overload_persons"].sum()) if not spot_load.empty else 0.0
    max_hotel_util = float(hotel_load["utilization"].max()) if not hotel_load.empty else 0.0
    hotel_over = float(hotel_load["overloaded_rooms"].sum()) if not hotel_load.empty else 0.0
    max_res_util = float(resource_load["utilization"].max()) if not resource_load.empty else 0.0
    res_short = float(resource_load["shortage_units"].sum()) if not resource_load.empty else 0.0
    wait_index = (
        max(0, max_spot_util - 0.82) * 1.25
        + overloaded_slots / 100
        + max(0, max_hotel_util - 0.90) * 0.75
        + max(0, max_res_util - 0.92) * 0.55
    )
    expected_loss = (
        rejected * 1.0
        + spot_overload * 0.22
        + hotel_over * 2.2
        + res_short * 55
        + wait_index * 2_500
    ) * float(scenario["volatility"])
    soft_pass_flag = served / demand >= 0.96 and max_spot_util <= 1.12 and max_hotel_util <= 1.18 and max_res_util <= 1.25
    strict_pass_flag = served / demand >= 0.96 and max_spot_util <= 1.0 and max_hotel_util <= 1.0 and max_res_util <= 1.0
    risk_items = []
    if rejected > 0.5:
        risk_items.append("总容量不足")
    if max_spot_util > 1.0:
        risk_items.append("景区分时段过载")
    if max_hotel_util > 1.0:
        risk_items.append("住宿枢纽紧张")
    if max_res_util > 1.0:
        risk_items.append("车辆/导游/摆渡资源紧张")
    if not risk_items:
        risk_items.append("容量可控")
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario["scenario_name"],
        "demand_multiplier": scenario["demand_multiplier"],
        "policy_id": policy["policy_id"],
        "policy_name": policy["policy_name"],
        "portfolio_type": policy["portfolio"],
        "demand_visitors": int(round(demand)),
        "served_visitors": int(round(served)),
        "rejected_or_overflow_visitors": int(round(rejected)),
        "offered_capacity": int(round(capacities.sum())),
        "served_ratio": round(served / demand, 4),
        "system_utilization": round(served / max(capacities.sum(), 1.0), 4),
        "max_spot_timeslot_utilization": round(max_spot_util, 4),
        "overloaded_spot_timeslots": overloaded_slots,
        "spot_overload_persons": round(spot_overload, 4),
        "max_hotel_utilization": round(max_hotel_util, 4),
        "hotel_overloaded_rooms": round(hotel_over, 4),
        "max_vehicle_guide_utilization": round(max_res_util, 4),
        "vehicle_guide_shortage_units": round(res_short, 4),
        "congestion_wait_index": round(wait_index, 4),
        "expected_loss_index": round(expected_loss, 4),
        "cvar90_loss_index": round(expected_loss * (1.12 + 0.08 * float(scenario["demand_multiplier"])), 4),
        "policy_pass": bool(soft_pass_flag),
        "soft_policy_pass": bool(soft_pass_flag),
        "strict_policy_pass": bool(strict_pass_flag),
        "resource_evaluable": True,
        "main_risk": "；".join(risk_items),
    }


def build_scenario_summary(
    candidates: pd.DataFrame,
    allocation: pd.DataFrame,
    itinerary: pd.DataFrame,
    spot_meta: pd.DataFrame,
    hotel_hubs: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    portfolio = candidates[candidates["selected_in_portfolio"]].sort_values("selection_order").copy()
    legacy_total_capacity = float(inputs["routes"]["route_capacity_persons_12day"].sum())
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            rows.append(
                evaluate_policy_scenario(
                    policy,
                    scenario,
                    portfolio,
                    allocation,
                    itinerary,
                    spot_meta,
                    hotel_hubs,
                    legacy_total_capacity,
                )
            )
    return pd.DataFrame(rows)


def build_reallocation_matrix(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    scenario_multiplier: float = 1.20,
    policy_booking_ratio: float = 0.95,
) -> pd.DataFrame:
    portfolio = selected.copy()
    demand = BASE_DEMAND * scenario_multiplier
    caps = np.array(portfolio["route_capacity_persons_12day"], dtype=float) * policy_booking_ratio
    weights = allocation_weights(portfolio, "preference_elastic")
    desired = demand * weights / max(weights.sum(), 1e-9)
    q, overflow = capped_allocate(demand, caps, weights)
    spare = np.maximum(caps - q, 0)
    overflow_by_route = np.maximum(desired - caps, 0)
    rows = []
    for i, src in enumerate(portfolio.itertuples(index=False)):
        src_overflow = float(overflow_by_route[i])
        if src_overflow <= 25:
            continue
        src_regions = set(str(src.region_set).split(";"))
        src_type = str(src.product_type)
        target_scores = []
        for j, tgt in enumerate(portfolio.itertuples(index=False)):
            if i == j or spare[j] <= 25:
                continue
            tgt_regions = set(str(tgt.region_set).split(";"))
            region_sim = len(src_regions & tgt_regions) / max(1, len(src_regions | tgt_regions))
            theme_sim = 1.0 if src.seed_route_theme == tgt.seed_route_theme else 0.35
            if ("culture" in src_type and "culture" in str(tgt.product_type)) or ("nature" in src_type and "nature" in str(tgt.product_type)):
                theme_sim = max(theme_sim, 0.72)
            comfort = float(tgt.comfort_score) / 100
            extra_travel_penalty = min(1.0, abs(float(src.avg_interspot_travel_hours_proxy) - float(tgt.avg_interspot_travel_hours_proxy)) / 5)
            east_concentration_penalty = 0.18 if "East_Heritage" in str(tgt.seed_route_theme) and "East_Heritage" not in str(src.seed_route_theme) else 0.0
            score = (
                0.34 * theme_sim
                + 0.23 * region_sim
                + 0.20 * comfort
                + 0.15 * min(spare[j] / max(src_overflow, 1), 1)
                + 0.08 * float(tgt.low_pressure_score) / 100
                - 0.12 * extra_travel_penalty
                - east_concentration_penalty
            )
            if score > 0.18:
                target_scores.append((j, score, region_sim, theme_sim, extra_travel_penalty))
        if not target_scores:
            continue
        total_score = sum(max(0.001, x[1]) for x in target_scores)
        for j, score, region_sim, theme_sim, extra_penalty in target_scores:
            tgt = portfolio.iloc[j]
            prob = max(0.001, score) / total_score
            proposed = min(float(spare[j]), src_overflow * prob)
            rows.append(
                {
                    "scenario_multiplier": scenario_multiplier,
                    "source_route_id": src.route_id,
                    "source_route_theme": src.route_theme,
                    "target_route_id": tgt["route_id"],
                    "target_route_theme": tgt["route_theme"],
                    "source_overflow_visitors": round(src_overflow, 4),
                    "target_spare_capacity": round(float(spare[j]), 4),
                    "acceptance_probability": round(prob, 4),
                    "proposed_reallocated_visitors": int(round(proposed)),
                    "region_similarity": round(region_sim, 4),
                    "theme_similarity": round(theme_sim, 4),
                    "extra_travel_penalty": round(extra_penalty, 4),
                    "decision_note": "按主题相似、区域相近、舒适度、剩余容量和额外转场惩罚分流；显式抑制全部转入East_Heritage。",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["source_route_id", "acceptance_probability"], ascending=[True, False])


def build_policy_selection(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    profiles = [
        ("cost_sensitive", "成本敏感均衡策略", 0.60, 0.25, 0.15, 1.00),
        ("balanced_operation", "运营均衡主推策略", 0.42, 0.34, 0.24, 1.10),
        ("high_reliability", "高可靠执行策略", 0.30, 0.38, 0.32, 1.20),
        ("extreme_backup", "复合极端预案", 0.20, 0.36, 0.44, 1.35),
    ]
    q4 = summary[summary["portfolio_type"] == "q4v2"].copy()
    for profile_id, name, w_loss, w_served, w_pass, focus_mult in profiles:
        sub = q4.copy()
        sub["scenario_distance"] = (sub["demand_multiplier"] - focus_mult).abs()
        sub["scenario_weight"] = np.exp(-3.0 * sub["scenario_distance"])
        policy_scores = []
        for policy_id, group in sub.groupby("policy_id"):
            served = np.average(group["served_ratio"], weights=group["scenario_weight"])
            loss = np.average(group["expected_loss_index"], weights=group["scenario_weight"])
            pass_rate = np.average(group["policy_pass"].astype(float), weights=group["scenario_weight"])
            strict_pass_rate = np.average(group["strict_policy_pass"].astype(float), weights=group["scenario_weight"])
            max_wait = float(group["congestion_wait_index"].max())
            score = (
                w_served * served * 100
                + w_pass * pass_rate * 100
                - w_loss * (loss / max(q4["expected_loss_index"].max(), 1) * 100)
                - 4.0 * max_wait
            )
            policy_scores.append((policy_id, score, served, loss, pass_rate, strict_pass_rate, max_wait))
        policy_scores.sort(key=lambda x: x[1], reverse=True)
        best = policy_scores[0]
        rows.append(
            {
                "profile_id": profile_id,
                "profile_name": name,
                "recommended_policy_id": best[0],
                "score": round(best[1], 4),
                "weighted_served_ratio": round(best[2], 4),
                "weighted_expected_loss": round(best[3], 4),
                "weighted_policy_pass_rate": round(best[4], 4),
                "weighted_strict_policy_pass_rate": round(best[5], 4),
                "max_wait_index": round(best[6], 4),
                "selection_reason": "按接待率、损失指数、策略通过率和拥挤等待综合评分；不同画像改变峰值风险权重。",
            }
        )
    return pd.DataFrame(rows)


def build_model_audit(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    quality_audit: pd.DataFrame,
    allocation: pd.DataFrame,
    spot_load: pd.DataFrame,
    hotel_load: pd.DataFrame,
    resource_load: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "audit_id": "Q4V2-A1",
            "module": "route_product_generation",
            "status": "implemented",
            "evidence": "q4_v2_candidate_route_products.csv",
            "metric": f"{len(candidates)} candidates / {int(candidates['selected_in_portfolio'].sum())} selected",
            "limitation": "候选线路由规则化列生成产生，不是所有可行线路全集。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A2",
            "module": "route_product_quality",
            "status": "implemented",
            "evidence": "q4_v2_route_quality_audit.csv",
            "metric": f"quality_pass={int(quality_audit['quality_pass'].sum())}/{len(quality_audit)}, max_day={quality_audit['max_day_active_hours'].max():.2f}h",
            "limitation": "质量审计基于日级强度代理，不是逐小时导游排班表；高强度路线需人工微调。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A3",
            "module": "capacity_ratio_allocation",
            "status": "implemented",
            "evidence": "q4_v2_capacity_ratio_allocation.csv",
            "metric": f"baseline demand={BASE_DEMAND}, balanced overflow={int(allocation['balanced_optimized_unserved_total'].iloc[0])}",
            "limitation": "游客偏好参数为模型校准，未接入真实订单点击/转化数据。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A4",
            "module": "timeslot_capacity",
            "status": "implemented",
            "evidence": "q4_v2_spot_timeslot_load.csv",
            "metric": f"max utilization={spot_load['utilization'].max():.3f}, red slots={(spot_load['pressure_level']=='red').sum()}",
            "limitation": "时段容量由日容量按早/午/晚比例拆分，仍需景区闸机小时级数据校准。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A5",
            "module": "hotel_resource_capacity",
            "status": "implemented",
            "evidence": "q4_v2_hotel_resource_load.csv",
            "metric": f"max utilization={hotel_load['utilization'].max():.3f}",
            "limitation": "酒店房量为枢纽可协调房源池的校准容量，不是携程/美团实时库存。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A6",
            "module": "vehicle_guide_shuttle_capacity",
            "status": "implemented",
            "evidence": "q4_v2_vehicle_guide_resource_load.csv",
            "metric": f"max utilization={resource_load['utilization'].max():.3f}",
            "limitation": "车辆、导游、摆渡车库存由枢纽可协调房源池和景区规模推导，需旅行社/景区调度台账核验。",
        }
    )
    best_policy = scenario_summary[scenario_summary["portfolio_type"] == "q4v2"].sort_values("expected_loss_index").iloc[0]
    q4_resource = scenario_summary[scenario_summary["portfolio_type"] == "q4v2"]
    rows.append(
        {
            "audit_id": "Q4V2-A7",
            "module": "scenario_policy_simulation",
            "status": "implemented",
            "evidence": "q4_v2_scenario_simulation_summary.csv",
            "metric": f"best row={best_policy.policy_id}/{best_policy.scenario_id}, loss={best_policy.expected_loss_index:.2f}, soft_pass={int(q4_resource['soft_policy_pass'].sum())}, strict_pass={int(q4_resource['strict_policy_pass'].sum())}",
            "limitation": "情景概率和冲击分布为专家设定；后续应接入历年五一客流/天气/道路事件。",
        }
    )
    rows.append(
        {
            "audit_id": "Q4V2-A8",
            "module": "mathematical_optimality",
            "status": "partial",
            "evidence": "reports/新疆旅游第四问Q4_V2五一路线产品组合优化报告.md",
            "metric": "matheuristic closed loop",
            "limitation": "当前为生成式候选列+启发式组合选择+仿真评估；若需严格全局最优，应转为Gurobi/Pyomo混合整数二次规划或列生成主问题。",
        }
    )
    return pd.DataFrame(rows)


def make_figures(paths: Paths, selected: pd.DataFrame, scenario_summary: pd.DataFrame, shadow: pd.DataFrame, spot_load: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = selected.sort_values("route_capacity_persons_12day", ascending=True)
    ax.barh(plot_df["route_id"], plot_df["route_capacity_persons_12day"], color="#357A73")
    ax.set_title("Q4-V2 Selected 12-day Route Product Capacity")
    ax.set_xlabel("12-day offered capacity (persons)")
    ax.set_ylabel("Route product")
    fig.tight_layout()
    fig.savefig(paths.figures / "fig_q4_v2_portfolio_capacity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    q4 = scenario_summary[scenario_summary["portfolio_type"] == "q4v2"].copy()
    pivot = q4.pivot_table(index="demand_multiplier", columns="policy_id", values="served_ratio", aggfunc="mean")
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Q4-V2 Policy Served Ratio by Demand Shock")
    ax.set_xlabel("Demand multiplier")
    ax.set_ylabel("Served ratio")
    ax.set_ylim(0.75, 1.03)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig_q4_v2_policy_scenarios.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    top = shadow.head(18).sort_values("shadow_price_index", ascending=True)
    labels = [f"B{i+1:02d} | {family.replace('_capacity', '')}" for i, family in enumerate(top["constraint_family"])]
    ax.barh(labels, top["shadow_price_index"], color="#9A5B35")
    ax.set_title("Q4-V2 Top Bottleneck Shadow Price Index")
    ax.set_xlabel("Shadow price index")
    fig.tight_layout()
    fig.savefig(paths.figures / "fig_q4_v2_bottleneck_shadow_prices.png", dpi=180)
    plt.close(fig)

    if not spot_load.empty:
        heat = spot_load.pivot_table(index="spot_id", columns="day_index", values="utilization", aggfunc="max").fillna(0)
        top_spots = spot_load.groupby("spot_id")["utilization"].max().sort_values(ascending=False).head(16).index
        heat = heat.loc[top_spots]
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1.2, float(heat.values.max())))
        ax.set_title("Q4-V2 Spot Timeslot Peak Utilization Heatmap")
        ax.set_xlabel("Mayday day index")
        ax.set_ylabel("Spot")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=8)
        fig.colorbar(im, ax=ax, label="peak utilization")
        fig.tight_layout()
        fig.savefig(paths.figures / "fig_q4_v2_timeslot_heatmap.png", dpi=180)
        plt.close(fig)


def write_workbook(paths: Paths, tables: dict[str, pd.DataFrame]) -> None:
    xlsx = paths.reports / "新疆旅游第四问Q4_V2五一路线产品组合优化结果.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        sheet_map = {
            "candidate_route_products": "候选12天线路",
            "selected_route_portfolio": "入选线路组合",
            "route_quality_audit": "线路质量审计",
            "capacity_ratio_allocation": "容量比例分配",
            "spot_timeslot_load": "景区分时负载",
            "hotel_resource_load": "住宿资源负载",
            "vehicle_guide_resource_load": "车辆导游摆渡",
            "bottleneck_shadow_prices": "瓶颈影子价格",
            "reallocation_matrix": "动态分流矩阵",
            "reservation_slot_policy": "分时预约策略",
            "scenario_simulation_summary": "场景仿真",
            "policy_selection": "策略选择",
            "model_audit": "模型审计",
        }
        for key, sheet in sheet_map.items():
            df = tables.get(key, pd.DataFrame())
            df.head(5000).to_excel(writer, sheet_name=sheet[:31], index=False)


def df_to_md(df: pd.DataFrame, n: int = 8) -> str:
    if df.empty:
        return "_无记录_"
    return df.head(n).to_markdown(index=False)


def write_report(paths: Paths, tables: dict[str, pd.DataFrame], inputs: dict[str, pd.DataFrame]) -> None:
    candidates = tables["candidate_route_products"]
    selected = tables["selected_route_portfolio"]
    quality_audit = tables["route_quality_audit"]
    allocation = tables["capacity_ratio_allocation"]
    scenario = tables["scenario_simulation_summary"]
    policy_selection = tables["policy_selection"]
    shadow = tables["bottleneck_shadow_prices"]
    spot_load = tables["spot_timeslot_load"]
    audit = tables["model_audit"]

    legacy_cap = int(inputs["routes"]["route_capacity_persons_12day"].sum())
    legacy_util = BASE_DEMAND / legacy_cap
    selected_cap = int(selected["route_capacity_persons_12day"].sum())
    q4v2_base = scenario[(scenario["scenario_id"] == "base_1_00") & (scenario["policy_id"] == "q4v2_staggered_prebooking")].iloc[0]
    q4v2_extreme = scenario[
        (scenario["scenario_id"] == "compound_extreme_1_35")
        & (scenario["policy_id"] == "q4v2_add_capacity_plus_stagger")
    ].iloc[0]
    red_slots = int((spot_load["pressure_level"] == "red").sum())
    max_slot = float(spot_load["utilization"].max())
    best_policies = policy_selection[
        [
            "profile_name",
            "recommended_policy_id",
            "weighted_served_ratio",
            "weighted_policy_pass_rate",
            "weighted_strict_policy_pass_rate",
            "weighted_expected_loss",
        ]
    ]

    report = f"""# 新疆旅游第四问 Q4-V2：五一 12 天游线路产品组合与分时容量优化报告

## 1. 问题定位

第四问不再建模为单个游客或单个家庭的路径规划，而是建模为文旅部门/平台在五一 12 天集中客流窗口下的运营接待优化问题。决策对象是“线路产品组合、线路投放名额、分时预约容量、酒店/车辆/导游/摆渡资源约束、需求冲击下的动态分流策略”。

旧模型已有 9 条线路产品列，基准需求为 `{BASE_DEMAND}` 人，12 天总容量为 `{legacy_cap}` 人，系统利用率 `{legacy_util:.3f}`。这个结果能解释“总量是否够”，但不能回答三个更实际的问题：是否能生成更多可运营的 12 天产品、热点景区是否在某日某时段被挤爆、住宿与导游车辆是否成为瓶颈。

## 2. V2 建模框架

集合：

- `r in R`：12 天线路产品；
- `d=1,...,12`：五一运营日；
- `s in {{morning, afternoon, evening}}`：预约时段；
- `i in I`：景区；
- `h in H`：住宿/服务枢纽；
- `k in K`：车辆、导游、摆渡/停车等资源。

核心变量：

- `x_r`：线路产品是否投放；
- `q_r`：分配到线路 `r` 的游客数；
- `l_{{i,d,s}}`：景区 `i` 在第 `d` 日 `s` 时段负载；
- `u_{{h,d}}`：住宿枢纽房量负载；
- `v_{{h,d,k}}`：车辆/导游/摆渡资源负载；
- `m_{{r,r'}}`：从满载线路 `r` 分流至替代线路 `r'` 的游客量。

主约束口径：

```text
0 <= q_r <= Cap_r x_r
sum_r q_r + overflow = D
l_{{i,d,s}} = sum_r a_{{r,i,d,s}} q_r <= Cap_{{i,d,s}}
u_{{h,d}} = sum_r b_{{r,h,d}} q_r / room_occupancy <= Room_{{h,d}}
v_{{h,d,k}} = sum_r c_{{r,h,d,k}} q_r <= Res_{{h,d,k}}
q_r is allocated proportional to offered capacity in the strict benchmark
```

目标函数采用多目标加权：

```text
max served_visitors
min congestion_penalty + hotel_shortage + vehicle_guide_shortage
min preference_loss + extra_transfer_penalty
max route_diversity + low_pressure_capacity
```

求解上采用“生成式候选列 + 启发式组合选择 + 容量比例/偏好弹性分配 + 数字孪生压力仿真”。这比旧版 9 线路容量流更贴近现实，但仍不是全可行线路空间上的严格全局最优；若需要精确最优，可把候选线路固定后转成 Pyomo/Gurobi 的 MILP/MIQP 主问题。

## 3. 候选线路生成与组合选择

从旧 9 条线路种子出发，按大众均衡、亲子舒适、长者慢游、错峰分流、文化深游、自然风景、精品预约 7 类变体扩展，生成 `{len(candidates)}` 条完整 12 天线路产品，最终选择 `{len(selected)}` 条进入 V2 组合，合计投放容量 `{selected_cap}` 人。

入选组合样例。`route_theme` 为中文运营产品名，`route_theme_code` 为英文机器码；当短线种子被扩展成跨区线路时，最终解释以中文产品名和 `route_name_note` 为准。

{df_to_md(selected[["selection_order","route_id","route_theme","route_theme_code","seed_route_theme","route_capacity_persons_12day","balanced_optimized_visitors","comfort_score","low_pressure_score"]], 12)}

## 3.1 线路产品质量审计

第四问的主目标是节假日容量接待，但“提高接待质量”要求线路产品不能出现过多高强度日。V2 对每条入选线路输出质量审计：`red_days` 表示单日活动强度超过 9 小时，`long_transfer_days` 表示跨区转场超过 6.5 小时，`quality_pass` 表示产品可直接作为团队线路投放；未通过者需要拆分销售、增加缓冲或改成小团深度产品。

{df_to_md(quality_audit[["route_id","route_theme","max_day_active_hours","red_days","long_transfer_days","buffer_days","quality_pass","quality_note"]], 18)}

## 4. 容量比例分配与偏好弹性对照

分配口径分三层，答辩时应先讲第一层：

1. `strict_capacity_ratio_visitors`：题面基准口径，游客人数严格按线路接待能力比例分配。
2. `preference_elastic_visitors`：现实偏好扩展，考虑游客更偏好高吸引力线路，但不能无限制挤向 East_Heritage 或传统热点。
3. `balanced_optimized_visitors`：运营优化口径，在偏好、容量、舒适度和低压分流之间折中，用于后续数字孪生仿真。

因此，本问没有忽略“按容量比例分配”的题设假设；它被保留为基准列，并与现实扩展/运营优化进行并列表达。

{df_to_md(allocation[["route_id","route_theme","route_capacity_persons_12day","strict_capacity_ratio_visitors","preference_elastic_visitors","balanced_optimized_visitors","balanced_utilization"]], 12)}

## 5. 分时段容量与多资源瓶颈

基准策略采用“分时预约+偏好弹性分流”，将日容量拆为早/午/晚预约槽。这里的 `q_r` 是 12 天窗口内线路产品总接待量，分时段、酒店和车辆导游负载均按滚动发团折算为日历日队列负载，即默认 `q_r/12` 进入每天资源压力计算。当前基准分时最大利用率为 `{max_slot:.3f}`，红色压力时段 `{red_slots}` 个。住宿容量采用“枢纽可协调房源池”校准，车辆、导游、摆渡/停车资源再由枢纽接待规模推导，输出为资源压力表，需在论文中明确其为校准库存而非实时平台库存。

瓶颈影子价格前列：

{df_to_md(shadow[["constraint_family","day_index","location","resource_or_slot","utilization","overload","shadow_price_index"]], 12)}

## 6. 需求冲击与政策实验

实验比较旧 9 线路基线、Q4-V2 全量放票、95%安全上限、90%舒适上限、分时预约、补运力+分时预约，在 1.00、1.05、1.10、1.20、1.35 倍需求冲击下的接待率、溢出、分时瓶颈和资源短缺。

输出中 `policy_pass` 与 `soft_policy_pass` 等价，表示软可行：接待率达标且分时/资源利用率在可通过现场调度修复的安全带内；`strict_policy_pass` 表示严格可行，即景区分时段、酒店、车辆导游摆渡资源全部不超过 100%。服务率高只说明游客总量能被接收，不等价于所有时段都不拥挤；`strict_policy_pass` 才表示接待质量完全可控。旧 9 线路缺少分时和多资源核验，因此只作为总容量基线，不作为严格资源可行证明。

基准场景下，`q4v2_staggered_prebooking` 接待 `{int(q4v2_base.served_visitors)}` 人，接待率 `{q4v2_base.served_ratio:.3f}`，溢出 `{int(q4v2_base.rejected_or_overflow_visitors)}` 人。复合极端 1.35 倍需求下，`q4v2_add_capacity_plus_stagger` 接待 `{int(q4v2_extreme.served_visitors)}` 人，接待率 `{q4v2_extreme.served_ratio:.3f}`，溢出 `{int(q4v2_extreme.rejected_or_overflow_visitors)}` 人。

策略选择：

{df_to_md(best_policies, 10)}

## 7. 主要结论

1. 旧 9 线路模型能复现 `{BASE_DEMAND}` 人基准接待和 `{legacy_cap}` 人总容量，但 10% 以上需求上浮时缺少可解释的新增线路、分时预约和资源补给机制。
2. Q4-V2 通过 12 天线路产品组合把“新增运力”具体化为可投放线路，而不是简单把总容量乘一个系数。
3. 分时预约是必须进入主模型的约束。总量不超载并不代表某个景区某天上午不超载。
4. 酒店、车辆、导游、摆渡/停车资源会把瓶颈从景区容量转移到服务枢纽，因此第四问不能只看景区日承载量。
5. 动态分流不应把溢出客流全部压向 East_Heritage；V2 的分流矩阵按主题相似、区域相近、舒适度和剩余容量计算接受概率。

## 8. 数据真实性与边界

- 景区容量来自 `capacity_by_spot.csv`，其中部分为公开承载量，部分为模型模拟容量，表内已保留 `capacity_source_type`。
- 住宿房量为枢纽可协调房源池校准容量，车辆、导游、摆渡/停车资源由该接待规模推导，不是实时平台库存。
- 游客偏好弹性参数为模型设定，后续可用真实订单、搜索热度、地图热力和景区预约余量校准。
- 本轮结果适合作为数学建模论文中的高级运营仿真与策略评估，不应表述为真实文旅部门可直接执行的排班单。

## 9. 审计表

{df_to_md(audit, 20)}
"""
    (paths.reports / "新疆旅游第四问Q4_V2五一路线产品组合优化报告.md").write_text(report, encoding="utf-8")


def write_readme(paths: Paths, tables: dict[str, pd.DataFrame]) -> None:
    selected = tables["selected_route_portfolio"]
    scenario = tables["scenario_simulation_summary"]
    audit = tables["model_audit"]
    readme = f"""# Q4-V2 12天线路产品组合与分时容量优化

本目录是第四问的新一轮强化闭环。它把旧版“9条线路容量流”升级为：

1. 生成 `{len(tables['candidate_route_products'])}` 条完整 12 天线路产品候选列；
2. 选择 `{len(selected)}` 条线路构成五一投放组合；
3. 将最终 `route_theme` 整理为中文运营产品名，英文追溯码保留在 `route_theme_code`；
4. 增加线路产品质量审计，检查高强度日、长转场日和缓冲日；
5. 对比题面容量比例分配、现实偏好弹性分配和平衡优化分配；
6. 将景区日容量拆为早/午/晚分时预约槽；
7. 引入住宿、车辆、导游、摆渡/停车多资源容量；
8. 比较 1.00/1.05/1.10/1.20/1.35 五类需求冲击和多种预约政策；
9. 输出瓶颈影子价格、动态分流矩阵、分时预约策略和模型审计。

## 复现

从项目根目录执行：

```powershell
python -X utf8 .\\第四问_五一容量接待优化材料包\\09_Q4_V2_12天线路产品组合与分时容量优化\\scripts\\q4_v2_build_all.py
```

## 核心输出

- `outputs/q4_v2_candidate_route_products.csv`
- `outputs/q4_v2_route_daily_itinerary.csv`
- `outputs/q4_v2_selected_route_portfolio.csv`
- `outputs/q4_v2_route_quality_audit.csv`
- `outputs/q4_v2_capacity_ratio_allocation.csv`
- `outputs/q4_v2_spot_timeslot_load.csv`
- `outputs/q4_v2_hotel_resource_load.csv`
- `outputs/q4_v2_vehicle_guide_resource_load.csv`
- `outputs/q4_v2_bottleneck_shadow_prices.csv`
- `outputs/q4_v2_reallocation_matrix.csv`
- `outputs/q4_v2_reservation_slot_policy.csv`
- `outputs/q4_v2_scenario_simulation_summary.csv`
- `outputs/q4_v2_policy_selection.csv`
- `outputs/q4_v2_model_audit.csv`
- `reports/新疆旅游第四问Q4_V2五一路线产品组合优化报告.md`
- `reports/新疆旅游第四问Q4_V2五一路线产品组合优化结果.xlsx`

## 当前结果摘要

- 入选线路产品总容量：`{int(selected['route_capacity_persons_12day'].sum())}` 人。
- 旧 9 线路基线总容量：`111956` 人，基准接待：`106358` 人。
- Q4-V2 不是个人路径规划，而是线路产品投放、预约名额、多资源容量和动态分流的运营模型。
- `strict_capacity_ratio_visitors` 是题面比例假设基准；`preference_elastic_visitors` 是现实偏好扩展；`balanced_optimized_visitors` 用于运营仿真。
- `policy_pass/soft_policy_pass` 表示软可行；`strict_policy_pass` 才表示所有分时段与多资源利用率均不超过 100%。
- 多资源容量中的酒店房量、车辆、导游、摆渡/停车仍为校准模拟参数，需在论文和答辩中说明。

## 模型审计

{df_to_md(audit, 20)}
"""
    (paths.v2_root / "README.md").write_text(readme, encoding="utf-8")


def copy_plan_if_exists(paths: Paths) -> None:
    src = Path.home() / "Downloads" / "q4_v2_route_portfolio_capacity_plan.md"
    if src.exists():
        dst = paths.v2_root / "config" / "q4_v2_route_portfolio_capacity_plan.md"
        shutil.copy2(src, dst)


def update_q4_indexes(paths: Paths, tables: dict[str, pd.DataFrame]) -> None:
    # Keep independent indexes for the Q4 package without rewriting the whole historical manifest by hand.
    files = []
    for p in sorted(paths.v2_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(paths.q4_root).as_posix()
            files.append(
                {
                    "relative_path": rel,
                    "file_size_bytes": p.stat().st_size,
                    "category": "Q4-V2 route portfolio capacity model",
                }
            )
    write_csv(pd.DataFrame(files), paths.v2_root / "outputs" / "q4_v2_file_manifest.csv")


def main() -> None:
    paths = find_project_paths()
    ensure_dirs(paths)
    copy_plan_if_exists(paths)
    inputs = load_inputs(paths)
    spot_meta = build_spot_meta(inputs)
    candidates_raw, itinerary_all = generate_route_candidates(inputs, spot_meta)
    candidates = select_portfolio(candidates_raw)
    allocation = build_capacity_ratio_allocation(candidates, inputs)
    selected = selected_portfolio_table(candidates, allocation)
    selected_full = candidates[candidates["selected_in_portfolio"]].sort_values("selection_order").copy()
    quality_audit = build_route_quality_audit(selected, itinerary_all)

    spot_load = load_by_spot_slot(
        itinerary_all,
        selected_full,
        allocation,
        spot_meta,
        booking_ratio=0.95,
        slot_stagger_strength=0.40,
        spot_capacity_factor=1.0,
    )
    hotel_load = build_hotel_load(
        itinerary_all,
        selected_full,
        allocation,
        inputs["hotel_hubs"],
        booking_ratio=0.95,
        hotel_capacity_factor=1.0,
        resource_boost=1.03,
    )
    resource_load = build_vehicle_guide_load(
        itinerary_all,
        selected_full,
        allocation,
        inputs["hotel_hubs"],
        spot_load,
        vehicle_capacity_factor=1.0,
        resource_boost=1.03,
    )
    shadow = build_shadow_prices(spot_load, hotel_load, resource_load)
    reallocation = build_reallocation_matrix(candidates, selected_full, scenario_multiplier=1.20, policy_booking_ratio=0.95)
    reservation_policy = build_reservation_slot_policy(spot_load)
    scenario_summary = build_scenario_summary(
        candidates,
        allocation,
        itinerary_all,
        spot_meta,
        inputs["hotel_hubs"],
        inputs,
    )
    policy_selection = build_policy_selection(scenario_summary)
    model_audit = build_model_audit(
        candidates,
        selected,
        quality_audit,
        allocation,
        spot_load,
        hotel_load,
        resource_load,
        scenario_summary,
    )

    itinerary_selected = itinerary_all[itinerary_all["route_id"].isin(set(selected["route_id"]))].sort_values(
        ["route_id", "day_index"]
    )
    outputs = {
        "candidate_route_products": candidates.sort_values(["selected_in_portfolio", "selection_order", "route_id"], ascending=[False, True, True]),
        "route_daily_itinerary": itinerary_selected,
        "selected_route_portfolio": selected,
        "route_quality_audit": quality_audit,
        "capacity_ratio_allocation": allocation,
        "spot_timeslot_load": spot_load,
        "hotel_resource_load": hotel_load,
        "vehicle_guide_resource_load": resource_load,
        "bottleneck_shadow_prices": shadow,
        "reallocation_matrix": reallocation,
        "reservation_slot_policy": reservation_policy,
        "scenario_simulation_summary": scenario_summary,
        "policy_selection": policy_selection,
        "model_audit": model_audit,
    }
    file_map = {
        "candidate_route_products": "q4_v2_candidate_route_products.csv",
        "route_daily_itinerary": "q4_v2_route_daily_itinerary.csv",
        "selected_route_portfolio": "q4_v2_selected_route_portfolio.csv",
        "route_quality_audit": "q4_v2_route_quality_audit.csv",
        "capacity_ratio_allocation": "q4_v2_capacity_ratio_allocation.csv",
        "spot_timeslot_load": "q4_v2_spot_timeslot_load.csv",
        "hotel_resource_load": "q4_v2_hotel_resource_load.csv",
        "vehicle_guide_resource_load": "q4_v2_vehicle_guide_resource_load.csv",
        "bottleneck_shadow_prices": "q4_v2_bottleneck_shadow_prices.csv",
        "reallocation_matrix": "q4_v2_reallocation_matrix.csv",
        "reservation_slot_policy": "q4_v2_reservation_slot_policy.csv",
        "scenario_simulation_summary": "q4_v2_scenario_simulation_summary.csv",
        "policy_selection": "q4_v2_policy_selection.csv",
        "model_audit": "q4_v2_model_audit.csv",
    }
    for key, filename in file_map.items():
        write_csv(outputs[key], paths.outputs / filename)

    write_report(paths, outputs, inputs)
    write_readme(paths, outputs)
    write_workbook(paths, outputs)
    make_figures(paths, selected, scenario_summary, shadow, spot_load)
    update_q4_indexes(paths, outputs)

    summary = {
        "candidate_routes": int(len(candidates)),
        "selected_routes": int(candidates["selected_in_portfolio"].sum()),
        "quality_pass_routes": int(quality_audit["quality_pass"].sum()),
        "selected_capacity_persons_12day": int(selected["route_capacity_persons_12day"].sum()),
        "legacy_9route_capacity_persons_12day": int(inputs["routes"]["route_capacity_persons_12day"].sum()),
        "baseline_demand": BASE_DEMAND,
        "baseline_q4v2_staggered_served": int(
            scenario_summary[
                (scenario_summary["scenario_id"] == "base_1_00")
                & (scenario_summary["policy_id"] == "q4v2_staggered_prebooking")
            ]["served_visitors"].iloc[0]
        ),
        "extreme_add_capacity_served": int(
            scenario_summary[
                (scenario_summary["scenario_id"] == "compound_extreme_1_35")
                & (scenario_summary["policy_id"] == "q4v2_add_capacity_plus_stagger")
            ]["served_visitors"].iloc[0]
        ),
        "max_spot_timeslot_utilization_base": float(round(spot_load["utilization"].max(), 4)),
        "red_spot_timeslots_base": int((spot_load["pressure_level"] == "red").sum()),
        "outputs_dir": str(paths.outputs),
    }
    (paths.outputs / "q4_v2_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
