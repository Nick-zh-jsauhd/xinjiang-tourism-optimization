from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DAY_HOURS = 8.0


def as_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        m = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else None


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_duration_hours(value: Any) -> tuple[float | None, float | None, str]:
    s = clean_text(value)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None, None, s
    if "天" in s:
        if "-" in s and len(nums) >= 2:
            return nums[0] * DAY_HOURS, nums[1] * DAY_HOURS, s
        return nums[0] * DAY_HOURS, nums[0] * DAY_HOURS, s
    if "-" in s and len(nums) >= 2:
        return nums[0], nums[1], s
    return nums[0], nums[0], s


def parse_ticket(value: Any) -> dict[str, Any]:
    s = clean_text(value)
    if not s:
        return {
            "ticket_base_yuan": None,
            "ticket_mandatory_local_yuan": None,
            "ticket_total_yuan": None,
            "ticket_parse_note": "blank",
        }
    if "免费" in s:
        return {
            "ticket_base_yuan": 0.0,
            "ticket_mandatory_local_yuan": 0.0,
            "ticket_total_yuan": 0.0,
            "ticket_parse_note": "free",
        }
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return {
            "ticket_base_yuan": None,
            "ticket_mandatory_local_yuan": None,
            "ticket_total_yuan": None,
            "ticket_parse_note": f"unparsed:{s}",
        }
    if "+" in s and len(nums) >= 2:
        base = nums[0]
        local = sum(nums[1:])
        total = base + local
        note = "base_plus_mandatory_local"
    else:
        base = nums[0]
        local = 0.0
        total = nums[0]
        note = "numeric" if re.fullmatch(r"\d+(?:\.\d+)?", s) else f"numeric_with_note:{s}"
    return {
        "ticket_base_yuan": base,
        "ticket_mandatory_local_yuan": local,
        "ticket_total_yuan": total,
        "ticket_parse_note": note,
    }


def contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def extract_first_distance_time(text: str) -> tuple[float | None, float | None]:
    km = None
    hour = None
    m = re.search(r"全程约?(\d+(?:\.\d+)?)公里", text)
    if m:
        km = float(m.group(1))
    m = re.search(r"车程约?(\d+(?:\.\d+)?)(?:-|－)?(\d+(?:\.\d+)?)?小时", text)
    if m:
        if m.group(2):
            hour = (float(m.group(1)) + float(m.group(2))) / 2
        else:
            hour = float(m.group(1))
    return km, hour


def parse_time_from_note(text: str) -> float | None:
    m = re.search(r"(?:飞行时间|车程)(\d+(?:\.\d+)?)(?:-|－)?(\d+(?:\.\d+)?)?小时", text)
    if not m:
        return None
    if m.group(2):
        return (float(m.group(1)) + float(m.group(2))) / 2
    return float(m.group(1))


def normalize_route_name(name: str) -> str:
    s = clean_text(name)
    s = re.sub(r"\(.+?\)|（.+?）", "", s)
    for token in ["市", "县", "镇", "地区", "风景区", "景区"]:
        s = s.replace(token, "")
    return s.strip()


def cluster_for_hub(name: str) -> str:
    if name == "茫崖市":
        return "外部出口"
    if any(k in name for k in ["乌鲁木齐", "天池", "达坂城"]):
        return "乌鲁木齐周边"
    if any(k in name for k in ["吐鲁番", "鄯善"]):
        return "东疆"
    if any(k in name for k in ["伊宁", "赛里木湖", "那拉提", "霍城", "特克斯", "昭苏"]):
        return "伊犁"
    if any(k in name for k in ["阿勒泰", "禾木", "白哈巴", "可可托海"]):
        return "北疆"
    if any(k in name for k in ["库尔勒", "若羌", "巴音布鲁克", "库车"]):
        return "巴州-南疆北线"
    if any(k in name for k in ["喀什", "塔什库尔干", "莎车", "和田", "民丰"]):
        return "南疆"
    return "未分区"


def spot_cluster(region: str, spot: str) -> str:
    text = f"{region} {spot}"
    if any(k in text for k in ["乌鲁木齐", "昌吉", "天池", "达坂城", "江布拉克", "北庭"]):
        return "乌鲁木齐周边"
    if "吐鲁番" in text or any(k in text for k in ["火焰山", "葡萄沟", "坎儿井", "交河", "高昌", "苏公塔", "吐峪沟"]):
        return "东疆"
    if "伊犁" in text or any(k in text for k in ["赛里木", "那拉提", "喀拉峻", "霍城", "昭苏", "夏塔", "果子沟"]):
        return "伊犁"
    if "阿勒泰" in text or any(k in text for k in ["喀纳斯", "禾木", "白哈巴", "可可托海", "五彩滩", "魔鬼城"]):
        return "北疆"
    if any(k in text for k in ["巴音郭楞", "库车", "巴音布鲁克", "博斯腾", "罗布", "楼兰", "克孜尔"]):
        return "巴州-南疆北线"
    if any(k in text for k in ["喀什", "和田", "克孜勒苏", "尼雅", "帕米尔", "塔什库尔干", "奥依塔克"]):
        return "南疆"
    return "未分区"


SPOT_HUB_OVERRIDES = {
    "天山天池": ("天池风景区", "high", "题面偏好点；边表有天池风景区节点"),
    "达坂城古镇": ("达坂城镇", "high", "题面偏好点；边表有达坂城镇节点"),
    "火焰山": ("吐鲁番市", "high", "吐鲁番片区景点，先映射到吐鲁番市"),
    "葡萄沟": ("吐鲁番市", "high", "吐鲁番市内/近郊景点"),
    "坎儿井民俗园": ("吐鲁番市", "high", "吐鲁番市内景点"),
    "交河故城": ("吐鲁番市", "high", "吐鲁番市内/近郊文化景点"),
    "高昌故城": ("吐鲁番市", "medium", "吐鲁番片区，需本地接驳"),
    "苏公塔": ("吐鲁番市", "high", "吐鲁番市内景点"),
    "吐峪沟麻扎村": ("吐鲁番市", "medium", "吐鲁番-鄯善方向，先映射到吐鲁番市"),
    "楼兰古城": ("若羌县(楼兰)", "high", "题面偏好点；需审批，边表有若羌县(楼兰)"),
    "赛里木湖": ("赛里木湖", "high", "景点名与边表节点一致"),
    "那拉提草原": ("那拉提镇", "high", "边表有那拉提镇节点"),
    "喀拉峻大草原": ("特克斯县(八卦城)", "medium", "喀拉峻位于特克斯方向，边表无喀拉峻专门节点"),
    "霍城薰衣草花田": ("霍城县(惠远古城)", "medium", "霍城片区，边表节点为惠远古城"),
    "昭苏草原": ("昭苏县(夏塔古道)", "medium", "昭苏片区，边表节点为夏塔古道"),
    "夏塔古道": ("昭苏县(夏塔古道)", "high", "边表有昭苏县(夏塔古道)节点"),
    "果子沟大桥": ("赛里木湖", "medium", "近赛里木湖/霍城，先映射到赛里木湖节点"),
    "喀纳斯湖": ("禾木村", "low", "边表缺布尔津/喀纳斯节点，临时挂到禾木村"),
    "禾木村": ("禾木村", "high", "景点名与边表节点一致"),
    "白哈巴村": ("白哈巴", "high", "边表有白哈巴节点"),
    "可可托海": ("可可托海", "high", "景点名与边表节点一致"),
    "五彩滩": ("禾木村", "low", "边表缺布尔津节点，临时挂到禾木村"),
    "世界魔鬼城": ("阿勒泰市", "low", "边表缺克拉玛依/乌尔禾节点，需补边"),
    "喀什古城": ("喀什市", "high", "喀什市内景点"),
    "艾提尕尔清真寺": ("喀什市", "high", "喀什市内景点"),
    "香妃园": ("喀什市", "high", "喀什市内景点"),
    "帕米尔高原白沙湖": ("塔什库尔干县(帕米尔高原)", "high", "帕米尔/塔县方向，需边防证"),
    "石头城遗址": ("塔什库尔干县(帕米尔高原)", "high", "塔县景点，需边防证"),
    "天山神秘大峡谷": ("库车市", "medium", "库车方向景点，需本地接驳"),
    "库车王府": ("库车市", "high", "库车市内景点"),
    "克孜尔石窟": ("库车市", "medium", "拜城县方向，边表缺拜城，先映射库车"),
    "博斯腾湖": ("库尔勒市", "medium", "博湖方向，边表缺博湖，先映射库尔勒"),
    "巴音布鲁克草原": ("巴音布鲁克", "high", "边表有巴音布鲁克节点"),
    "罗布人村寨": ("库尔勒市", "medium", "尉犁方向，边表缺尉犁，先映射库尔勒"),
    "江布拉克草原": ("乌鲁木齐市", "low", "边表缺奇台/江布拉克节点，需补边"),
    "北庭故城遗址": ("乌鲁木齐市", "low", "边表缺吉木萨尔节点，需补边"),
    "和田博物馆": ("和田市", "high", "和田市内景点"),
    "尼雅遗址": ("民丰县(尼雅遗址)", "high", "边表有民丰县(尼雅遗址)节点；需审批/向导"),
    "千里葡萄长廊": ("和田市", "medium", "和田县近郊，先映射和田市"),
    "奥依塔克冰川": ("喀什市", "medium", "阿克陶方向，边表缺阿克陶，先映射喀什"),
}


def first_xlsx(path: Path) -> Path:
    files = sorted(path.glob("*.xlsx"))
    for item in files:
        if "全品类" in item.name:
            return item
    if files:
        return files[0]
    raise FileNotFoundError("No .xlsx source file found")


def read_source(input_path: Path) -> dict[str, pd.DataFrame]:
    return {
        "road": pd.read_excel(input_path, sheet_name=0, header=1).dropna(how="all"),
        "hotel": pd.read_excel(input_path, sheet_name=1, header=1).dropna(how="all"),
        "transport": pd.read_excel(input_path, sheet_name=2, header=1).dropna(how="all"),
        "spot": pd.read_excel(input_path, sheet_name=3, header=0).dropna(how="all").dropna(axis=1, how="all"),
        "sources": pd.read_excel(input_path, sheet_name=4, header=1).dropna(how="all").dropna(axis=1, how="all"),
    }


def build_hubs(road: pd.DataFrame, mapped_hubs: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    order: list[str] = []
    for _, r in road.iterrows():
        for value in [r.iloc[0], r.iloc[1]]:
            name = clean_text(value)
            if name and name not in order:
                order.append(name)
    for name in sorted(mapped_hubs):
        if name not in order:
            order.append(name)
    records = []
    hub_id = {}
    for i, name in enumerate(order, 1):
        hid = f"H{i:03d}"
        hub_id[name] = hid
        records.append(
            {
                "hub_id": hid,
                "hub_name": name,
                "hub_cluster": cluster_for_hub(name),
                "is_road_node": bool(name in set(road.iloc[:, 0]).union(set(road.iloc[:, 1]))),
                "is_external_exit": name == "茫崖市",
                "model_role": "external_exit" if name == "茫崖市" else "transport_hub",
            }
        )
    return pd.DataFrame(records), hub_id


def build_spots(spots: pd.DataFrame, hub_id: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cultural_words = ["故城", "古城", "遗址", "清真寺", "王府", "石窟", "博物馆", "民俗", "麻扎", "苏公塔", "坎儿井", "村寨"]
    natural_words = ["湖", "草原", "冰川", "峡谷", "沙漠", "花田", "大桥", "火焰山", "天池", "可可托海", "五彩滩"]
    border_words = ["边防证", "边境"]
    approval_words = ["审批", "申请", "文物局", "仅限科考", "个人无法进入", "有资质"]
    reservation_words = ["预约", "提前"]
    high_alt_words = ["高原", "海拔", "高原反应"]

    spot_records = []
    map_records = []
    for idx, row in spots.reset_index(drop=True).iterrows():
        spot_id = f"P{idx + 1:03d}"
        name = clean_text(row.iloc[0])
        region = clean_text(row.iloc[1])
        raw_ticket_high = clean_text(row.iloc[2])
        raw_ticket_low = clean_text(row.iloc[3])
        raw_duration = clean_text(row.iloc[4])
        open_time = clean_text(row.iloc[5])
        remark = clean_text(row.iloc[6])
        access = clean_text(row.iloc[7])
        hotel_note = clean_text(row.iloc[8])
        traffic_note = clean_text(row.iloc[9])
        full_text = " ".join([remark, access, hotel_note, traffic_note])
        d_min, d_max, _ = parse_duration_hours(raw_duration)
        high_ticket = parse_ticket(raw_ticket_high)
        low_ticket = parse_ticket(raw_ticket_low)
        local_km, local_hours = extract_first_distance_time(access)

        hub_name, confidence, note = SPOT_HUB_OVERRIDES.get(name, ("", "unmapped", "no override"))
        hid = hub_id.get(hub_name, "")
        is_cultural = contains_any(name + " " + remark, cultural_words)
        is_natural = contains_any(name + " " + remark, natural_words)
        requires_approval = contains_any(full_text, approval_words)
        requires_border = contains_any(full_text, border_words)
        requires_reservation = contains_any(full_text, reservation_words) and contains_any(full_text, ["预约", "提前"])
        high_altitude = contains_any(full_text, high_alt_words)
        ordinary_restricted = contains_any(full_text, ["个人无法进入", "仅限科考", "文物局"])
        topic_pref = int(
            name in ["天山天池", "达坂城古镇", "楼兰古城"]
            or "吐鲁番" in region
            or "伊犁" in region
        )
        priority_score = 1 + 3 * topic_pref + 2 * int(is_cultural) + int(is_natural)
        if ordinary_restricted:
            priority_score = max(priority_score - 2, 1)

        spot_records.append(
            {
                "spot_id": spot_id,
                "spot_name": name,
                "region_raw": region,
                "region_cluster": spot_cluster(region, name),
                "hub_id": hid,
                "hub_name": hub_name,
                "map_confidence": confidence,
                "visit_hours_min": d_min,
                "visit_hours_max": d_max,
                "visit_hours_mid": (d_min + d_max) / 2 if d_min is not None and d_max is not None else None,
                "ticket_high_base_yuan_per_person": high_ticket["ticket_base_yuan"],
                "ticket_high_mandatory_local_yuan_per_person": high_ticket["ticket_mandatory_local_yuan"],
                "ticket_high_total_yuan_per_person": high_ticket["ticket_total_yuan"],
                "ticket_low_total_yuan_per_person": low_ticket["ticket_total_yuan"],
                "ticket_parse_note": high_ticket["ticket_parse_note"],
                "is_cultural": is_cultural,
                "is_natural": is_natural,
                "is_topic_preference": bool(topic_pref),
                "priority_score_for_op": priority_score,
                "requires_approval": requires_approval,
                "requires_border_permit": requires_border,
                "requires_reservation": requires_reservation,
                "high_altitude_or_remote": high_altitude,
                "ordinary_tourist_restricted": ordinary_restricted,
                "open_time_raw": open_time,
                "local_access_km_from_text": local_km,
                "local_access_hours_from_text": local_hours,
                "raw_ticket_high": raw_ticket_high,
                "raw_duration": raw_duration,
                "modeling_note": note,
            }
        )
        map_records.append(
            {
                "spot_id": spot_id,
                "spot_name": name,
                "hub_id": hid,
                "hub_name": hub_name,
                "map_confidence": confidence,
                "local_access_km_from_text": local_km,
                "local_access_hours_from_text": local_hours,
                "mapping_note": note,
            }
        )
    return pd.DataFrame(spot_records), pd.DataFrame(map_records)


def build_edges(road: pd.DataFrame, hub_id: dict[str, str]) -> pd.DataFrame:
    records = []
    eid = 1
    for _, r in road.iterrows():
        a = clean_text(r.iloc[0])
        b = clean_text(r.iloc[1])
        km = as_float(r.iloc[2])
        hours = as_float(r.iloc[3])
        route = clean_text(r.iloc[4])
        for src, dst in [(a, b), (b, a)]:
            records.append(
                {
                    "edge_id": f"E{eid:03d}",
                    "from_hub_id": hub_id.get(src, ""),
                    "from_hub_name": src,
                    "to_hub_id": hub_id.get(dst, ""),
                    "to_hub_name": dst,
                    "mode": "self_drive_road",
                    "distance_km": km,
                    "time_hours": hours,
                    "avg_speed_kmh": km / hours if km is not None and hours else None,
                    "route_note": route,
                    "directionality_assumption": "two_way_from_undirected_source",
                    "is_seasonal_sensitive": bool("独库" in route or "G217" in route),
                    "touches_external_node": bool(src == "茫崖市" or dst == "茫崖市"),
                }
            )
            eid += 1
    return pd.DataFrame(records)


def build_hotels(hotel: pd.DataFrame, hub_id: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grade_rank = {"经济型": 1, "舒适型": 2, "高端型": 3}
    alias = {
        "塔什库尔干县": "塔什库尔干县(帕米尔高原)",
        "巴音布鲁克镇": "巴音布鲁克",
        "若羌县(楼兰)": "若羌县(楼兰)",
        "鄯善县(库木塔格沙漠)": "鄯善县(库木塔格沙漠)",
        "特克斯县(八卦城)": "特克斯县(八卦城)",
        "昭苏县(夏塔古道)": "昭苏县(夏塔古道)",
    }
    records = []
    for idx, r in hotel.reset_index(drop=True).iterrows():
        place = clean_text(r.iloc[0])
        mapped = alias.get(place, place)
        records.append(
            {
                "hotel_option_id": f"L{idx + 1:03d}",
                "place_name": place,
                "hub_id": hub_id.get(mapped, ""),
                "hub_name": mapped if mapped in hub_id else "",
                "high_season_room_yuan_per_night": as_float(r.iloc[1]),
                "low_season_room_yuan_per_night": as_float(r.iloc[2]),
                "hotel_grade": clean_text(r.iloc[3]),
                "hotel_grade_rank": grade_rank.get(clean_text(r.iloc[3]), 9),
                "price_basis": "room_per_night_for_two_people",
                "source_note": clean_text(r.iloc[4]),
            }
        )
    options = pd.DataFrame(records)
    default = (
        options.sort_values(["place_name", "hotel_grade_rank", "high_season_room_yuan_per_night"])
        .groupby("place_name", as_index=False)
        .first()
    )
    default = default[
        [
            "place_name",
            "hub_id",
            "hub_name",
            "high_season_room_yuan_per_night",
            "low_season_room_yuan_per_night",
            "hotel_grade",
            "price_basis",
        ]
    ]
    return options, default


def build_transport(transport: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, r in transport.reset_index(drop=True).iterrows():
        ttype = clean_text(r.iloc[0])
        item = clean_text(r.iloc[1])
        cost_min = as_float(r.iloc[2])
        cost_max = as_float(r.iloc[3])
        note = clean_text(r.iloc[4])
        origin = ""
        dest = ""
        if " - " in item:
            parts = item.split(" - ", 1)
            origin = parts[0].strip()
            dest = re.split(r"\s+", parts[1].strip())[0]
        if ttype in ["民航机票", "城际大巴", "铁路火车", "景区内交通"] or "骑马" in item or "骆驼" in item:
            basis = "per_person"
        elif ttype in ["包车服务", "自驾租车"]:
            basis = "per_vehicle_day"
        elif "打车" in item:
            basis = "per_vehicle_trip"
        else:
            basis = "review_required"
        trip_type = "round_trip" if "往返" in item else "one_way" if "单程" in item else "day_or_use"
        records.append(
            {
                "cost_id": f"C{idx + 1:03d}",
                "transport_type": ttype,
                "item": item,
                "origin_raw": origin,
                "destination_raw": dest,
                "trip_type": trip_type,
                "cost_min_yuan": cost_min,
                "cost_max_yuan": cost_max,
                "cost_mid_yuan": (cost_min + cost_max) / 2 if cost_min is not None and cost_max is not None else None,
                "cost_basis": basis,
                "multiply_by_people_for_two_person_trip": basis == "per_person",
                "time_hours_from_note": parse_time_from_note(note),
                "included_in_ticket_or_duplicate_risk": bool("含门票" in note or "门票已含" in note),
                "source_note": note,
            }
        )
    return pd.DataFrame(records)


def build_scenarios() -> pd.DataFrame:
    rows = [
        ("S1_personal_30day", "travel_days", 30, "days", "第一问：一个月暑假"),
        ("S1_personal_30day", "people", 2, "persons", "王先生夫妇两人"),
        ("S1_personal_30day", "day_hours", DAY_HOURS, "hours/day", "将1天游览折算为8小时"),
        ("S1_personal_30day", "allow_ordinary_restricted_spots", 0, "binary", "普通旅游默认不含楼兰/尼雅等审批点"),
        ("S2_two_summers", "stages", 2, "years", "第二问：今明两年"),
        ("S2_two_summers", "objective_cost_scope", 1, "flag", "仅新疆境内交通费用"),
        ("S3_three_research_groups", "groups", 3, "groups", "第三问：三组并行"),
        ("S3_three_research_groups", "research_time_multiplier", 4, "multiplier", "考察时间为观光时间四倍"),
        ("S4_mayday_distribution", "route_duration_limit", 12, "days", "第四问：五一自治区内游程12天"),
        ("S4_mayday_distribution", "capacity_threshold_beta", 0.85, "ratio", "建议安全负载阈值，可灵敏度分析"),
    ]
    return pd.DataFrame(rows, columns=["scenario_id", "parameter", "value", "unit", "note"])


def build_quality(
    road: pd.DataFrame,
    spots_clean: pd.DataFrame,
    edge_clean: pd.DataFrame,
    transport_clean: pd.DataFrame,
    hotel_options: pd.DataFrame,
) -> pd.DataFrame:
    issues = []

    def add(sev: str, table: str, record: str, issue: str, impact: str, fix: str) -> None:
        issues.append(
            {
                "issue_id": f"Q{len(issues) + 1:03d}",
                "severity": sev,
                "table_name": table,
                "record_ref": record,
                "issue": issue,
                "modeling_impact": impact,
                "recommended_fix": fix,
            }
        )

    road_nodes = set(road.iloc[:, 0]).union(set(road.iloc[:, 1]))
    exact_overlap = set(spots_clean["spot_name"]) & road_nodes
    add(
        "High",
        "spot_clean / edge_road_directed",
        "all",
        f"40个景点中只有{len(exact_overlap)}个与公路节点精确同名，必须使用spot_hub_map。",
        "若直接用景点名建图，大多数景点无法进入路径模型。",
        "先以hub_id建交通主图，再用spot_hub_map连接景点与交通枢纽。",
    )
    add(
        "High",
        "edge_road_directed",
        "all",
        "公路边只有28条原始边，属于稀疏网络，不是完整OD矩阵。",
        "不能直接认为任意景点之间都有边；需要先在枢纽图上求最短路闭包。",
        "用Floyd/Dijkstra补全hub-to-hub最短时间/最短费用矩阵。",
    )
    low_maps = spots_clean[spots_clean["map_confidence"].eq("low")]
    for _, row in low_maps.iterrows():
        add(
            "Medium",
            "spot_hub_map",
            row["spot_id"],
            f"{row['spot_name']}到{row['hub_name']}为低置信映射。",
            "路线计算可行，但距离/时间会偏粗。",
            "补充该景点到最近城市/县镇的独立接驳边。",
        )
    mixed_ticket = spots_clean[spots_clean["ticket_parse_note"].str.contains("base_plus|note|unparsed", na=False)]
    if not mixed_ticket.empty:
        add(
            "Medium",
            "spot_clean",
            ",".join(mixed_ticket["spot_id"].head(8)),
            "部分门票字段包含区间车、说明文字或特殊票种，不是纯数字。",
            "费用模型可能重复计入景区区间车。",
            "使用ticket_high_base_yuan_per_person和ticket_high_mandatory_local_yuan_per_person分项计算。",
        )
    restricted = spots_clean[spots_clean["ordinary_tourist_restricted"]]
    if not restricted.empty:
        add(
            "High",
            "spot_clean",
            ",".join(restricted["spot_name"].tolist()),
            "存在普通游客不可直接进入或需文物审批的景点。",
            "第一问普通旅游路线不能把这些点当作普通必游点。",
            "在模型中设置Permit_i或ordinary_tourist_restricted约束。",
        )
    border = spots_clean[spots_clean["requires_border_permit"]]
    if not border.empty:
        add(
            "Medium",
            "spot_clean",
            ",".join(border["spot_name"].tolist()),
            "部分边境/高原景点需要边防证。",
            "若未加证件约束，路线可能不可执行。",
            "将边防证作为区域准入约束或准备时间参数。",
        )
    seasonal_edges = edge_clean[edge_clean["is_seasonal_sensitive"]]
    if not seasonal_edges.empty:
        add(
            "High",
            "edge_road_directed",
            ",".join(seasonal_edges["edge_id"].head(6).tolist()),
            "独库/G217相关边存在季节性通行风险。",
            "五一和非开放期路线可能不可行。",
            "加入date-dependent a_ij(t)，并设置备选绕行边。",
        )
    if any(edge_clean["touches_external_node"]):
        add(
            "Medium",
            "edge_road_directed",
            "茫崖市",
            "边表含新疆外部节点茫崖市。",
            "第二问若只计算新疆境内交通，应避免把外省段混入目标函数。",
            "给外部节点设置external_exit角色，费用口径单独处理。",
        )
    no_hotel_hub = hotel_options[hotel_options["hub_id"].eq("")]
    if not no_hotel_hub.empty:
        add(
            "Low",
            "hotel_options_clean",
            ",".join(no_hotel_hub["place_name"].unique()[:8]),
            "部分住宿地点没有精确匹配到交通hub。",
            "住宿费用可用，但路径节点连接需人工确认。",
            "补充住宿地点到hub_id的别名映射。",
        )
    dup_local = transport_clean[transport_clean["included_in_ticket_or_duplicate_risk"]]
    if not dup_local.empty:
        add(
            "Medium",
            "transport_cost_clean",
            ",".join(dup_local["cost_id"].tolist()),
            "部分景区交通备注显示可能已含在门票内。",
            "若门票字段也包含区间车，会重复计费。",
            "费用估算时优先使用spot_clean的门票拆分，景区交通只作为校验/备选。",
        )
    for _, r in transport_clean.iterrows():
        if r["transport_type"] == "铁路火车" and r["origin_raw"] and r["destination_raw"] and r["time_hours_from_note"]:
            if r["time_hours_from_note"] <= 4 and any(k in r["item"] for k in ["喀什", "伊宁", "阿勒泰", "库尔勒"]):
                add(
                    "Medium",
                    "transport_cost_clean",
                    r["cost_id"],
                    f"{r['item']}备注时间为{r['time_hours_from_note']}小时，疑似与实际铁路行程不符。",
                    "时间约束会严重偏乐观。",
                    "用12306重新核对班次时间后覆盖该字段。",
                )
    add(
        "High",
        "capacity_inputs",
        "missing",
        "没有景区日容量、酒店房量、停车容量、摆渡容量和五一游客总量。",
        "第四问无法直接计算具体分流人数。",
        "后续需补capacity_by_spot、capacity_by_hotel_area、F_total_mayday。",
    )
    return pd.DataFrame(issues)


def build_dictionary() -> pd.DataFrame:
    rows = [
        ("spot_clean", "景点建模主表", "OP收益、停留时间、门票、特殊准入约束"),
        ("hub_clean", "交通枢纽/城市节点表", "构建主交通图G=(H,E)"),
        ("spot_hub_map", "景点到交通枢纽映射表", "把景点P挂接到枢纽H，处理名称不一致"),
        ("edge_road_directed", "双向公路边表", "路径模型、最短路闭包、时间约束"),
        ("hotel_options_clean", "住宿价格选项", "住宿费用估计；双人房按房间计费"),
        ("hotel_place_default", "默认住宿价", "快速路线估算时默认采用经济型/最低可用价"),
        ("transport_cost_clean", "交通费用目录", "机票/火车/大巴/包车/租车/景区交通费用参数"),
        ("scenario_parameters", "场景参数", "五个小问的时间、人数、容量阈值等默认值"),
        ("data_quality_issues", "数据质量与建模风险", "建模前必须处理或在报告中说明的问题"),
    ]
    return pd.DataFrame(rows, columns=["table_name", "meaning", "optimization_use"])


def write_outputs(tables: dict[str, pd.DataFrame], output_dir: Path, source_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_file": source_file.name,
        "tables": {name: {"rows": int(len(df)), "columns": int(len(df.columns))} for name, df in tables.items()},
    }
    json_tables = {
        name: json.loads(df.where(pd.notna(df), None).to_json(orient="records", force_ascii=False))
        for name, df in tables.items()
    }
    (output_dir / "tables.json").write_text(json.dumps(json_tables, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = [
        "# 新疆旅游运筹优化建模数据底座",
        "",
        f"源文件：{source_file.name}",
        "",
        "## 生成表",
    ]
    for name, meta in summary["tables"].items():
        readme.append(f"- `{name}.csv`：{meta['rows']} 行，{meta['columns']} 列")
    readme.extend(
        [
            "",
            "## 建模使用建议",
            "- 交通主图使用 `hub_clean` 与 `edge_road_directed`，不要直接用景点名建图。",
            "- 景点访问变量使用 `spot_clean.spot_id`，再通过 `spot_hub_map.hub_id` 挂接到交通图。",
            "- 第一问费用估算应区分 `per_person` 与 `per_vehicle/day`，住宿是双人房房费，不按人数乘二。",
            "- 第四问仍缺容量数据，只能先保留参数化模型。",
            "- 建模前先阅读 `data_quality_issues.csv`，其中 High 级问题需要在报告中显式说明或补数。",
        ]
    )
    (output_dir / "README_数据底座.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("model_data"))
    args = parser.parse_args()

    input_path = args.input or first_xlsx(Path("."))
    source = read_source(input_path)

    mapped_hubs = {v[0] for v in SPOT_HUB_OVERRIDES.values()}
    hub_clean, hub_id = build_hubs(source["road"], mapped_hubs)
    spot_clean, spot_hub_map = build_spots(source["spot"], hub_id)
    edge_clean = build_edges(source["road"], hub_id)
    hotel_options, hotel_default = build_hotels(source["hotel"], hub_id)
    transport_clean = build_transport(source["transport"])
    scenario = build_scenarios()
    quality = build_quality(source["road"], spot_clean, edge_clean, transport_clean, hotel_options)
    dictionary = build_dictionary()
    source_sites = source["sources"].copy()
    source_sites.columns = ["source_category", "source_name", "url", "core_use"][: len(source_sites.columns)]

    tables = {
        "spot_clean": spot_clean,
        "hub_clean": hub_clean,
        "spot_hub_map": spot_hub_map,
        "edge_road_directed": edge_clean,
        "hotel_options_clean": hotel_options,
        "hotel_place_default": hotel_default,
        "transport_cost_clean": transport_clean,
        "scenario_parameters": scenario,
        "data_quality_issues": quality,
        "source_sites": source_sites,
        "data_dictionary": dictionary,
    }
    write_outputs(tables, args.output_dir, input_path)
    print(json.dumps({name: list(df.shape) for name, df in tables.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
