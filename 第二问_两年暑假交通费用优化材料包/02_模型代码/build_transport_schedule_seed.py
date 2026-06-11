from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(".")
ENHANCED_DIR = ROOT / "enhanced_data"
OUTPUT_DIR = ROOT / "outputs"


AIR_ROWS = [
    # Urumqi - Kashgar, Ctrip schedule page snapshot.
    ("2026-06-20", "乌鲁木齐", "喀什", "URC", "KHG", "CZ6803", "中国南方航空", "06:55", "09:05", 2.167, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.khg.html", "schedule_high_price_missing"),
    ("2026-06-20", "乌鲁木齐", "喀什", "URC", "KHG", "UQ3509", "乌鲁木齐航空", "07:05", "09:10", 2.083, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.khg.html", "schedule_high_price_missing"),
    ("2026-06-20", "乌鲁木齐", "喀什", "URC", "KHG", "CZ6805", "中国南方航空", "07:55", "09:55", 2.000, "daily", 1220, "Skyscanner: 2026年6月从URC到KHG ¥1220起；平均飞行2小时4分钟", "Ctrip schedule + Skyscanner fare", "https://www.tianxun.com/routes/urc/khg/urumqi-to-kashi.html", "schedule_high_fare_proxy"),
    ("2026-06-20", "乌鲁木齐", "喀什", "URC", "KHG", "CZ6873", "中国南方航空", "08:55", "11:00", 2.083, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.khg.html", "schedule_high_price_missing"),

    # Urumqi - Yining.
    ("2026-06-20", "乌鲁木齐", "伊宁", "URC", "YIN", "CZ*", "中国南方航空", "09:05", "10:25", 1.333, "daily", 559, "Skyscanner: 2026年6月单程¥559起；Wego显示最短直飞1小时15分、最早约07:45", "Wego schedule + Skyscanner fare", "https://www.tianxun.com/routes/urc/yin/urumqi-to-yining.html", "schedule_medium_fare_proxy"),
    ("2026-06-20", "乌鲁木齐", "伊宁", "URC", "YIN", "CZ*", "中国南方航空", "09:15", "10:35", 1.333, "daily", None, "", "Wego schedule snippet", "https://www.wego.com.my/zh-cn/flights/urc/yin/cheapest-flights-from-urumqi-to-yining", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "伊宁", "URC", "YIN", "EU*", "成都航空", "10:05", "11:25", 1.333, "daily", None, "", "Wego schedule snippet", "https://www.wego.com.my/zh-cn/flights/urc/yin/cheapest-flights-from-urumqi-to-yining", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "伊宁", "URC", "YIN", "GS*", "天津航空", "19:30", "20:45", 1.250, "daily", None, "", "Search result schedule snippet", "https://qijiang.ctrip.com/schedule/urc.yin.html", "schedule_medium_price_missing"),

    # Urumqi - Korla.
    ("2026-06-20", "乌鲁木齐", "库尔勒", "URC", "KRL", "GJ8665", "长龙航空", "07:55", "09:00", 1.083, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.krl.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "库尔勒", "URC", "KRL", "CA1781", "中国国际航空", "07:55", "09:00", 1.083, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.krl.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "库尔勒", "URC", "KRL", "CZ6671", "中国南方航空", "08:00", "09:05", 1.083, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.krl.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "库尔勒", "URC", "KRL", "G5*", "华夏航空", "", "", None, "snapshot", 1094, "Skyscanner: 6月20出发、6月24返回往返¥1094起；Trip.com港版称6月20单程HK$556起", "Skyscanner fare snapshot", "https://www.tianxun.com/routes/urc/krl/urumqi-to-korla.html", "fare_proxy_schedule_missing"),

    # Urumqi - Aksu.
    ("2026-06-20", "乌鲁木齐", "阿克苏", "URC", "AKU", "CZ6867", "中国南方航空", "07:55", "09:15", 1.333, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.aku.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "阿克苏", "URC", "AKU", "GJ8581", "长龙航空", "07:55", "09:15", 1.333, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.aku.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "阿克苏", "URC", "AKU", "UQ2607", "乌鲁木齐航空", "08:25", "09:50", 1.417, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.aku.html", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "阿克苏", "URC", "AKU", "OTA_LOW", "OTA报价快照", "14:10", "15:40", 1.500, "snapshot", 880, "Ctrip移动页显示URC-AKU若干经济舱¥880；Trip.com港版显示6月15南航单程HK$718起", "Ctrip mobile fare snapshot", "https://m.ctrip.com/html5/flight/URC-AKU-day-2.html", "fare_proxy"),

    # Urumqi - Hotan.
    ("2026-06-20", "乌鲁木齐", "和田", "URC", "HTN", "UQ3603", "乌鲁木齐航空", "07:45", "10:00", 2.250, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.htn.html", "schedule_high_price_missing"),
    ("2026-06-20", "乌鲁木齐", "和田", "URC", "HTN", "GS7497", "天津航空", "08:10", "10:10", 2.000, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.htn.html", "schedule_high_price_missing"),
    ("2026-06-20", "乌鲁木齐", "和田", "URC", "HTN", "CZ6811", "中国南方航空", "08:40", "10:50", 2.167, "daily", None, "", "Ctrip schedule snippet", "https://qijiang.ctrip.com/schedule/urc.htn.html", "schedule_high_price_missing"),
    ("2026-06-20", "乌鲁木齐", "和田", "URC", "HTN", "GS*", "天津航空", "", "", None, "snapshot", 1126, "Skyscanner: URC-HTN 6月9去、6月10回往返¥1126起；Trip.com称约2小时13分", "Skyscanner fare snapshot", "https://www.tianxun.com/routes/urc/htn/urumqi-to-hotan.html", "fare_proxy_schedule_missing"),

    # Urumqi - Kuqa.
    ("2026-06-20", "乌鲁木齐", "库车", "URC", "KCA", "CZ6871", "中国南方航空", "07:10", "08:35", 1.417, "daily", 530, "Skyscanner: URC-KCA单程¥530起；Ctrip/FlightConnections均显示CZ6871 07:10-08:35", "Ctrip schedule + Skyscanner fare", "https://qijiang.ctrip.com/schedule/urc.kca.html", "schedule_high_fare_proxy"),
    ("2026-06-20", "乌鲁木齐", "库车", "URC", "KCA", "GS7553", "天津航空", "07:35", "08:55", 1.333, "some days", None, "", "FlightConnections schedule snippet", "https://www.flightconnections.com/cn/%E4%BB%8E-urc-%E9%A3%9E%E5%BE%80-kca-%E7%9A%84%E8%88%AA%E7%8F%AD", "schedule_medium_price_missing"),
    ("2026-06-20", "乌鲁木齐", "库车", "URC", "KCA", "CZ6873", "中国南方航空", "13:20", "14:45", 1.417, "some days", None, "", "FlightConnections schedule snippet", "https://www.flightconnections.com/cn/%E4%BB%8E-urc-%E9%A3%9E%E5%BE%80-kca-%E7%9A%84%E8%88%AA%E7%8F%AD", "schedule_medium_price_missing"),

    # Urumqi - Altay / Kanas access.
    ("2026-06-20", "乌鲁木齐", "阿勒泰", "URC", "AAT", "CZ6843", "中国南方航空", "08:20", "09:45", 1.417, "daily", None, "", "China Southern route info", "https://www.csair.com/h5/cn/tourism_strategy/guonei_lvyougonglve/Altay2/1cge14vtq9us3.shtml", "official_schedule_price_missing"),
    ("2026-06-20", "乌鲁木齐", "阿勒泰", "URC", "AAT", "CZ6845", "中国南方航空", "12:00", "13:20", 1.333, ".2345.7", None, "", "China Southern route info", "https://www.csair.com/h5/cn/tourism_strategy/guonei_lvyougonglve/Altay2/1cge14vtq9us3.shtml", "official_schedule_price_missing"),
    ("2026-06-20", "乌鲁木齐", "阿勒泰", "URC", "AAT", "CZ6841", "中国南方航空", "20:55", "22:15", 1.333, "daily", 1145, "Skyscanner: URC-AAT往返¥1145起；南航页面给出每日固定航班", "China Southern schedule + Skyscanner fare", "https://www.tianxun.com/routes/urc/aat/urumqi-to-altay.html", "official_schedule_fare_proxy"),
    ("2026-06-20", "乌鲁木齐", "喀纳斯", "URC", "KJI", "EU/CZ*", "成都航空/南方航空", "17:00", "18:25", 1.417, "tourism season", 830, "Ctrip移动页显示URC-KJI 17:00-18:25，票价¥830/¥960；Trip.com称6月单程HK$821起", "Ctrip mobile + Trip.com fare", "https://m.ctrip.com/html5/flight/urc-kji-day-1.html", "schedule_medium_fare_proxy"),
]


def build_air_table() -> pd.DataFrame:
    cols = [
        "query_date",
        "from_city",
        "to_city",
        "from_airport_code",
        "to_airport_code",
        "flight_no",
        "airline",
        "depart_time",
        "arrive_time",
        "duration_hours",
        "frequency",
        "fare_proxy_yuan",
        "fare_note",
        "source_name",
        "source_url",
        "quality_flag",
    ]
    df = pd.DataFrame(AIR_ROWS, columns=cols)
    df["airport_buffer_hours"] = 2.5
    df["model_air_time_hours"] = df["duration_hours"].fillna(0) + df["airport_buffer_hours"]
    df.loc[df["duration_hours"].isna(), "model_air_time_hours"] = None
    df["fare_for_two_yuan"] = df["fare_proxy_yuan"] * 2
    return df


def build_source_notes() -> pd.DataFrame:
    rows = [
        {
            "source_type": "rail_official",
            "source_name": "中国铁路12306 queryG/queryTicketPrice",
            "source_url": "https://kyfw.12306.cn/otn/leftTicket/init",
            "note": "已通过12306公开余票/票价接口抓取2026-06-20核心铁路OD。",
        },
        {
            "source_type": "air_schedule_public",
            "source_name": "携程/FlightConnections/Wego/南航公开页面",
            "source_url": "多个URL见air_public_schedule_seed.csv",
            "note": "用于航班号、起降时间、班期。Ctrip部分页面反爬，采用可检索公开片段和可打开页面快照。",
        },
        {
            "source_type": "air_fare_proxy",
            "source_name": "Skyscanner/Trip.com/Ctrip/Expedia/Rome2Rio报价片段",
            "source_url": "多个URL见air_public_schedule_seed.csv",
            "note": "票价为OTA快照或月度低价，不等于固定票价；正式模型应在出行日期前重新查价。",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ENHANCED_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    air = build_air_table()
    notes = build_source_notes()
    air_path = ENHANCED_DIR / "air_public_schedule_seed.csv"
    air.to_csv(air_path, index=False, encoding="utf-8-sig")

    rail_raw_path = ENHANCED_DIR / "rail_12306_raw_core_od.csv"
    rail_options_path = ENHANCED_DIR / "rail_12306_model_options.csv"
    rail_raw = pd.read_csv(rail_raw_path, encoding="utf-8-sig") if rail_raw_path.exists() else pd.DataFrame()
    rail_options = pd.read_csv(rail_options_path, encoding="utf-8-sig") if rail_options_path.exists() else pd.DataFrame()

    workbook = OUTPUT_DIR / "新疆旅游铁路航班数据包.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        if not rail_raw.empty:
            rail_raw.to_excel(writer, index=False, sheet_name="rail_12306_raw")
        if not rail_options.empty:
            rail_options.to_excel(writer, index=False, sheet_name="rail_model_options")
        air.to_excel(writer, index=False, sheet_name="air_public_seed")
        notes.to_excel(writer, index=False, sheet_name="source_notes")
    print({"air_rows": len(air), "rail_raw_rows": len(rail_raw), "rail_option_rows": len(rail_options), "workbook": str(workbook)})


if __name__ == "__main__":
    main()
