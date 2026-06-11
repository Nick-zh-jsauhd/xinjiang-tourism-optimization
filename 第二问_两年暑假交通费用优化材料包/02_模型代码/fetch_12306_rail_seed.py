from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(".")
ENHANCED_DIR = ROOT / "enhanced_data"
QUERY_DATE = "2026-06-20"

STATIONS = {
    "乌鲁木齐": "WAR",
    "乌鲁木齐南": "WMR",
    "吐鲁番北": "TAR",
    "库尔勒": "KLR",
    "伊宁": "YMR",
    "喀什": "KSR",
    "和田": "VTR",
    "阿克苏": "ASR",
    "库车": "KCR",
}

CORE_OD = [
    ("乌鲁木齐", "吐鲁番北"),
    ("乌鲁木齐", "库尔勒"),
    ("乌鲁木齐", "伊宁"),
    ("乌鲁木齐", "喀什"),
    ("喀什", "和田"),
    ("乌鲁木齐", "阿克苏"),
    ("阿克苏", "库车"),
    ("库车", "库尔勒"),
]


def minutes(duration: str) -> int:
    h, m = duration.split(":")
    return int(h) * 60 + int(m)


def price_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace("¥", "").replace("\\xa5", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://kyfw.12306.cn/otn/leftTicket/init",
        }
    )
    session.get("https://kyfw.12306.cn/otn/leftTicket/init", timeout=30)
    return session


def query_left_ticket(session: requests.Session, from_station: str, to_station: str) -> dict[str, Any]:
    params = {
        "leftTicketDTO.train_date": QUERY_DATE,
        "leftTicketDTO.from_station": STATIONS[from_station],
        "leftTicketDTO.to_station": STATIONS[to_station],
        "purpose_codes": "ADULT",
    }
    resp = session.get("https://kyfw.12306.cn/otn/leftTicket/queryG", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") is False:
        raise RuntimeError(f"12306 left ticket failed: {data}")
    return data["data"]


def query_price(session: requests.Session, fields: list[str]) -> dict[str, Any]:
    params = {
        "train_no": fields[2],
        "from_station_no": fields[16],
        "to_station_no": fields[17],
        "seat_types": fields[34],
        "train_date": QUERY_DATE,
    }
    resp = session.get("https://kyfw.12306.cn/otn/leftTicket/queryTicketPrice", params=params, timeout=30)
    resp.raise_for_status()
    if not resp.text.strip():
        return {}
    try:
        data = resp.json()
    except Exception:
        return {}
    if not data.get("status"):
        return {}
    return data.get("data", {})


def parse_record(fields: list[str], station_map: dict[str, str], from_city: str, to_city: str, price: dict[str, Any]) -> dict[str, Any]:
    price_candidates = {
        "second_class": price_to_float(price.get("O")),
        "first_class": price_to_float(price.get("M") or price.get("F")),
        "business_class": price_to_float(price.get("A9") or price.get("P")),
        "hard_seat": price_to_float(price.get("A1")),
        "hard_sleeper": price_to_float(price.get("A3")),
        "soft_sleeper": price_to_float(price.get("A4")),
        "no_seat": price_to_float(price.get("WZ")),
    }
    usable_prices = [v for v in price_candidates.values() if v is not None]
    min_price = min(usable_prices) if usable_prices else None
    return {
        "query_date": QUERY_DATE,
        "from_city": from_city,
        "to_city": to_city,
        "from_station": station_map.get(fields[6], fields[6]),
        "to_station": station_map.get(fields[7], fields[7]),
        "train_no": fields[2],
        "train_code": fields[3],
        "depart_time": fields[8],
        "arrive_time": fields[9],
        "duration": fields[10],
        "duration_hours": round(minutes(fields[10]) / 60, 3),
        "can_book": fields[11],
        "train_date_raw": fields[13],
        "train_location": fields[15],
        "from_station_no": fields[16],
        "to_station_no": fields[17],
        "seat_types": fields[34],
        "availability_business": fields[32] if len(fields) > 32 else "",
        "availability_first_class": fields[31] if len(fields) > 31 else "",
        "availability_second_class": fields[30] if len(fields) > 30 else "",
        "availability_soft_sleeper": fields[23] if len(fields) > 23 else "",
        "availability_hard_sleeper": fields[28] if len(fields) > 28 else "",
        "availability_hard_seat": fields[29] if len(fields) > 29 else "",
        "availability_no_seat": fields[26] if len(fields) > 26 else "",
        "price_second_class_yuan": price_candidates["second_class"],
        "price_first_class_yuan": price_candidates["first_class"],
        "price_business_class_yuan": price_candidates["business_class"],
        "price_hard_seat_yuan": price_candidates["hard_seat"],
        "price_hard_sleeper_yuan": price_candidates["hard_sleeper"],
        "price_soft_sleeper_yuan": price_candidates["soft_sleeper"],
        "price_no_seat_yuan": price_candidates["no_seat"],
        "min_ticket_price_yuan": min_price,
        "source": "12306_queryG_and_queryTicketPrice",
        "source_url": "https://kyfw.12306.cn/otn/leftTicket/init",
        "query_timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }


def select_model_options(raw: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (from_city, to_city), group in raw.groupby(["from_city", "to_city"], sort=False):
        group = group.copy()
        group["min_ticket_price_yuan_fill"] = group["min_ticket_price_yuan"].fillna(1e9)
        picks = []
        picks.append(("fastest", group.sort_values(["duration_hours", "min_ticket_price_yuan_fill"]).iloc[0]))
        picks.append(("cheapest", group.sort_values(["min_ticket_price_yuan_fill", "duration_hours"]).iloc[0]))
        night = group[(group["depart_time"] >= "18:00") | (group["depart_time"] <= "08:00")]
        if not night.empty:
            picks.append(("night_or_early", night.sort_values(["duration_hours", "min_ticket_price_yuan_fill"]).iloc[0]))
        for role, row in picks:
            item = row.to_dict()
            item["model_option_role"] = role
            selected.append(item)
    out = pd.DataFrame(selected).drop_duplicates(
        subset=["from_city", "to_city", "train_code", "depart_time", "arrive_time", "model_option_role"]
    )
    return out.drop(columns=["min_ticket_price_yuan_fill"], errors="ignore")


def main() -> None:
    ENHANCED_DIR.mkdir(exist_ok=True)
    session = build_session()
    rows = []
    for from_city, to_city in CORE_OD:
        data = query_left_ticket(session, from_city, to_city)
        station_map = data.get("map", {})
        for rec in data.get("result", []):
            fields = rec.split("|")
            if len(fields) < 35:
                continue
            price = query_price(session, fields)
            rows.append(parse_record(fields, station_map, from_city, to_city, price))
            time.sleep(0.15)
        time.sleep(0.5)
    raw = pd.DataFrame(rows)
    raw_path = ENHANCED_DIR / "rail_12306_raw_core_od.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    options = select_model_options(raw)
    options_path = ENHANCED_DIR / "rail_12306_model_options.csv"
    options.to_csv(options_path, index=False, encoding="utf-8-sig")

    summary = {
        "query_date": QUERY_DATE,
        "core_od_count": len(CORE_OD),
        "raw_rows": len(raw),
        "model_option_rows": len(options),
        "raw_path": str(raw_path),
        "options_path": str(options_path),
    }
    (ENHANCED_DIR / "rail_12306_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
