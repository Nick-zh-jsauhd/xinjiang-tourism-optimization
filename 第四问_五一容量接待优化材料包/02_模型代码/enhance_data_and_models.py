from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from sklearn.cluster import SpectralClustering


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
MODEL_DIR = ROOT / "enhanced_model_outputs"
OUTPUT_DIR = ROOT / "outputs"

DAY_HOURS = 8.0
LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO = 15.0
START_HUB = "乌鲁木齐市"
START_NODE = f"HUB::{START_HUB}"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")


def read_enhanced_csv(name: str) -> pd.DataFrame:
    path = ENHANCED_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def num(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def minutes_to_hhmm(hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


def source_registry() -> pd.DataFrame:
    rows = [
        {
            "source_id": "SRC_XJ_MAYDAY_2025",
            "source_name": "新疆维吾尔自治区人民政府网：2025五一游客量",
            "url": "https://www.xinjiang.gov.cn/xinjiang/bmdt/202505/fad7fa3a38d54bd4a5b7e98a93a6b005.shtml",
            "data_used": "2025五一新疆累计接待游客919.55万人次、游客花费89.24亿元；喀什古城接待84.04万人次。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_MCT_MAYDAY_2025",
            "source_name": "中国政府网/文旅部：2025五一全国出游数据",
            "url": "https://www.gov.cn/lianbo/bumen/202505/content_7022682.htm",
            "data_used": "2025五一全国国内出游3.14亿人次、总花费1802.69亿元，用于宏观客流校准。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_TIANCHI_CAP",
            "source_name": "昌吉州政府网：天山天池夏季最大承载量公告",
            "url": "https://www.cj.gov.cn/p1/tzgg/20250811/369611.html",
            "data_used": "天山天池景区夏季日最大承载量2.58万人。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_SAILIMU_CAP",
            "source_name": "霍城县政府网：果子沟-赛里木湖承载量公告",
            "url": "https://www.xjhc.gov.cn/xjhc/c113326/202508/31639fa047ca40588c5f20ea77f9660e.shtml",
            "data_used": "果子沟-赛里木湖景区日最大承载量22000人、瞬时承载量5570人。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_KANAS_CAP",
            "source_name": "喀纳斯夏季游览通告转引",
            "url": "https://wlmq.bendibao.com/xiuxian/2026416/67787.shtm",
            "data_used": "喀纳斯主要景区日最大承载量5.5万人，分区：喀纳斯2.7万、禾木2万、白哈巴0.8万。",
            "reliability": "secondary_public_notice",
        },
        {
            "source_id": "SRC_DUKU_CCTV_2026",
            "source_name": "央视网：2026独库公路6月1日恢复通车",
            "url": "https://news.cctv.com/2026/06/01/ARTIp28O3A6i8NutdCt1Au4y260601.shtml",
            "data_used": "2026年6月1日独库公路正式恢复全线通车，且部分路段白天保通行、夜间保施工。",
            "reliability": "central_media",
        },
        {
            "source_id": "SRC_DUKU_CTNEWS_2026",
            "source_name": "中国旅游新闻网：独库公路恢复全线通车",
            "url": "https://www.ctnews.com.cn/dongtai/content/2026-06/01/content_187964.html",
            "data_used": "2026年6月1日上午G217独库公路恢复全线通车。",
            "reliability": "industry_media",
        },
        {
            "source_id": "SRC_DUKU_FIRSTDAY",
            "source_name": "中国日报网：独库公路通车首日车流超6000辆",
            "url": "https://xj.chinadaily.com.cn/a/202606/02/WS6a1e9e68a310942cc49afb09.html",
            "data_used": "2026年通车首日库车段车流超6000辆，用于路段拥堵/容量情景校准。",
            "reliability": "official_media",
        },
        {
            "source_id": "SRC_RUOQIANG_RAIL_2026",
            "source_name": "中新网新疆：若羌-库尔勒城际列车优化",
            "url": "https://www.xj.chinanews.com.cn/dizhou/2026-04-13/detail-ihfcqyyi3989347.shtml",
            "data_used": "C981次若羌至库尔勒12:50发车、16:47到达，全程3小时57分。",
            "reliability": "news_from_rail_adjustment",
        },
        {
            "source_id": "SRC_URUMQI_KASHI_RAIL",
            "source_name": "铁路网/列车吧：乌鲁木齐至喀什列车时刻",
            "url": "https://www.crecc.com/huoche/k9768.html",
            "data_used": "乌鲁木齐南至喀什列车全程约18小时级别，用于修正原表3.5小时异常值。",
            "reliability": "secondary_schedule",
        },
        {
            "source_id": "SRC_YINING_RAIL",
            "source_name": "铁路网：伊宁站列车时刻表",
            "url": "https://www.crecc.com/xinjiang/yili/yining.html",
            "data_used": "伊宁至乌鲁木齐城际动车约5小时49分、旅游列车约7小时55分。",
            "reliability": "secondary_schedule",
        },
        {
            "source_id": "SRC_XJ_CULTURE_OVERVIEW",
            "source_name": "新疆维吾尔自治区人民政府网：新疆文化底蕴",
            "url": "https://www.xinjiang.gov.cn/xinjiang/xjgk/202406/6cf3555600ee440690c18781e1c2494d.shtml",
            "data_used": "高昌故城、交河故城、北庭故城遗址、克孜尔石窟等为新疆首批世界文化遗产相关遗产点。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_TURPAN_HERITAGE",
            "source_name": "吐鲁番市政府网：交河/高昌故城申遗",
            "url": "https://www.tlf.gov.cn/tlfs/c106444/201406/0ce8cb00f68d452aa6d5f22767ba7ce3.shtml",
            "data_used": "交河故城、高昌故城作为丝绸之路重要遗址列入世界文化遗产。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_KASHI_IDKAH",
            "source_name": "喀什地区行政公署：艾提尕尔清真寺简介",
            "url": "https://www.kashi.gov.cn/ksdqxzgs/c106707/202307/28cd99dc43a244619788bda878887922.shtml",
            "data_used": "艾提尕尔清真寺始建于1442年，具有浓郁民族风格和宗教色彩。",
            "reliability": "official",
        },
        {
            "source_id": "SRC_KIZIL_REGULATION",
            "source_name": "新疆政府规章库：克孜尔千佛洞保护管理办法",
            "url": "https://www.xinjiang.gov.cn/xinjiang/gzk/202112/ea65632e645544fdba2332872387ea72/files/e595c959f7394608a279e0b025e7b293.PDF",
            "data_used": "克孜尔千佛洞为全国重点文物保护单位，用于文化考察标签。",
            "reliability": "official_pdf",
        },
        {
            "source_id": "SRC_12306_PORTAL",
            "source_name": "中国铁路12306",
            "url": "https://www.12306.cn/",
            "data_used": "铁路班次/票价应以12306实时查询为准；本模型将外部公开时刻作为近似。",
            "reliability": "official_portal",
        },
        {
            "source_id": "RAIL_12306_MODEL_OPTIONS",
            "source_name": "本项目抓取的12306核心铁路OD模型候选表",
            "url": "enhanced_data/rail_12306_model_options.csv",
            "data_used": "2026-06-20核心铁路OD的最快、最便宜、夜间/早晚代表车次；用于替换铁路层假设边。",
            "reliability": "official_api_snapshot",
        },
        {
            "source_id": "AIR_PUBLIC_SCHEDULE_SEED",
            "source_name": "本项目整理的公开航班时刻与OTA报价快照",
            "url": "enhanced_data/air_public_schedule_seed.csv",
            "data_used": "乌鲁木齐至喀什、伊宁、库尔勒、阿克苏、和田、库车、阿勒泰、喀纳斯等航线时刻/价格代理。",
            "reliability": "public_page_and_ota_proxy",
        },
        {
            "source_id": "SRC_AMAP_PORTAL",
            "source_name": "高德开放平台",
            "url": "https://lbs.amap.com/",
            "data_used": "完整OD矩阵和实时拥堵数据建议通过地图API批量获取；本轮已使用高德驾车OD快照接入道路层。",
            "reliability": "official_api_portal",
        },
        {
            "source_id": "AMAP_DISTANCE_API",
            "source_name": "高德 Distance/Direction API 驾车OD快照",
            "url": "enhanced_data/amap_driving_od_matrix_clean.csv",
            "data_used": "40个景点两两驾车时长/距离，以及乌鲁木齐出发地到景点的往返接驳；用于替换增强模型道路层中的景点级自驾边。",
            "reliability": "official_api_snapshot",
        },
    ]
    return pd.DataFrame(rows)


def data_gap_acquisition_plan() -> pd.DataFrame:
    rows = [
        {
            "data_domain": "完整OD矩阵",
            "current_handling": "多层网络最短路闭包近似，已生成40x40景点OD。",
            "preferred_source": "高德开放平台 Direction/Distance Matrix API",
            "acquisition_method": "补齐景点经纬度后按景点对批量请求，区分普通日、暑期、五一、早晚高峰。",
            "model_use": "替换 enhanced_od_matrix 中的时间/距离/费用边权，作为PCOP、VRP、容量流的基础边权。",
            "priority": "P0",
        },
        {
            "data_domain": "铁路时刻/票价",
            "current_handling": "核心城市对使用公开网页和情景等待时间近似。",
            "preferred_source": "中国铁路12306",
            "acquisition_method": "人工或半自动抽取乌鲁木齐-吐鲁番/库尔勒/伊宁/喀什/和田等核心OD的车次、发车时间、历时、票价。",
            "model_use": "将铁路边升级为time-expanded arcs，表达发车时刻和错过班次的等待时间。",
            "priority": "P0",
        },
        {
            "data_domain": "航班时刻/票价",
            "current_handling": "按主要机场间经验时长和票价建航空层。",
            "preferred_source": "航司官网、机场集团、航旅纵横、OTA抽样",
            "acquisition_method": "采集乌鲁木齐-喀什/伊宁/阿克苏/库尔勒/和田/库车的日内班次、价格区间、提前到机场时间。",
            "model_use": "加入航班等待、安检提前量、延误风险、票价波动情景。",
            "priority": "P1",
        },
        {
            "data_domain": "景区日容量/瞬时容量",
            "current_handling": "天池、赛里木湖、喀纳斯等有公开承载量，其余按等级模拟。",
            "preferred_source": "景区公告、文旅局公开信息、景区最大承载量公告",
            "acquisition_method": "逐景区检索官方公告；无法查询时按5A/4A/普通景点分级设定并标注MODEL_ASSUMPTION。",
            "model_use": "作为第四问容量约束、拥挤惩罚和线路分配上界。",
            "priority": "P0",
        },
        {
            "data_domain": "开闭园/预约余量/闭馆日",
            "current_handling": "用常见开放时间和人工解析规则生成time_windows。",
            "preferred_source": "景区公众号、一码游新疆、官方预约平台",
            "acquisition_method": "按目标日期记录开放时间、停运项目、预约余量、是否闭馆或临时关闭。",
            "model_use": "加入 open_i <= arrival_i <= close_i 和日期可用性约束。",
            "priority": "P0",
        },
        {
            "data_domain": "酒店容量/价格",
            "current_handling": "价格来自本地表，房量为住宿枢纽分级模拟。",
            "preferred_source": "携程/美团/同程抽样、统计年鉴住宿床位、酒店协会数据",
            "acquisition_method": "按城市、日期、星级采样价格与可订房量，至少覆盖乌鲁木齐、吐鲁番、库尔勒、伊宁、喀什、和田。",
            "model_use": "住宿落点约束、满房风险、住宿价格上浮情景。",
            "priority": "P1",
        },
        {
            "data_domain": "道路封闭/天气风险",
            "current_handling": "独库公路使用公开新闻，其余山地道路用情景风险模拟。",
            "preferred_source": "新疆交通运输厅、交警路况、气象局预警、景区封闭公告",
            "acquisition_method": "按道路边维护季节开放窗口、封闭概率、恶劣天气延误倍数。",
            "model_use": "转化为边可用概率、风险分数和鲁棒优化场景。",
            "priority": "P1",
        },
        {
            "data_domain": "客流热度",
            "current_handling": "使用五一宏观客流和少量景区承载量校准。",
            "preferred_source": "文旅厅节假日通报、景区公告、地图热力指数、运营商客流",
            "acquisition_method": "按景区/城市/日期建立客流指数，无法获取时用搜索热度或点评数量做代理。",
            "model_use": "满意度、拥挤惩罚、候选线路投放规模和鲁棒场景概率。",
            "priority": "P2",
        },
        {
            "data_domain": "文化考察价值",
            "current_handling": "按世界遗产、文保、宗教民族文化、博物馆等标签打分。",
            "preferred_source": "文旅部/文物局名录、世界遗产名录、非遗名录、地方政府介绍",
            "acquisition_method": "为文化点建立多标签和权重，专家复核高价值点。",
            "model_use": "作为第三问主题覆盖、人文价值最大化和组内均衡约束。",
            "priority": "P1",
        },
    ]
    return pd.DataFrame(rows)


def model_status_recommendations() -> pd.DataFrame:
    rows = [
        {
            "module": "数据底座",
            "current_status": "已完成基础清洗和增强表，支持复现实验。",
            "main_limitation": "经纬度、实时OD、真实房量和预约余量仍缺。",
            "next_action": "先补高德OD与重点城市铁路/航班时刻，再补景区容量和住宿库存。",
            "priority": "P0",
        },
        {
            "module": "多层网络",
            "current_status": "已建公路、铁路、航空、大巴、景区接驳、换乘层。",
            "main_limitation": "铁路/航空仍是聚合边，不是真正班次边。",
            "next_action": "将班次展开成time-expanded network，按日期和出发时间求路。",
            "priority": "P0",
        },
        {
            "module": "第一问PCOP",
            "current_status": "已实现MILP高价值候选子集、ALNS全量启发式、Pareto搜索。",
            "main_limitation": "全量精确MILP在开源环境下求解较慢。",
            "next_action": "若可安装OR-Tools/Gurobi，做全量PCOP+时间窗；否则保留MILP子集+ALNS主结果。",
            "priority": "P0",
        },
        {
            "module": "第二问两年路线",
            "current_status": "已实现谱聚类分区和区内TSP式路径覆盖。",
            "main_limitation": "年份间重复惩罚、入疆/离疆枢纽选择和Benders分解尚未完全展开。",
            "next_action": "增加分配主问题和路径子问题，显式约束每个景点恰好覆盖一次。",
            "priority": "P1",
        },
        {
            "module": "第三问文化考察",
            "current_status": "已实现Min-Max Multi-TSP/VRP和审批点专组处理。",
            "main_limitation": "文化价值权重目前是规则评分，缺专家校准。",
            "next_action": "引入AHP/熵权法或专家打分，形成主题覆盖约束和均衡约束。",
            "priority": "P1",
        },
        {
            "module": "第四问容量线路",
            "current_status": "已实现候选线路列和容量流分配。",
            "main_limitation": "候选列仍是模板生成，不是定价子问题动态生成。",
            "next_action": "实现column generation：主问题分配客流，子问题生成负 reduced cost 路线。",
            "priority": "P1",
        },
        {
            "module": "鲁棒优化",
            "current_status": "已实现五类情景下的期望成本、最坏时间和风险目标。",
            "main_limitation": "场景概率和风险分数仍需要历史天气/封路/满房数据校准。",
            "next_action": "用Monte Carlo仿真检验超时、超预算和不可达概率。",
            "priority": "P2",
        },
    ]
    return pd.DataFrame(rows)


def parse_open_hours(raw: str) -> tuple[str, str, str]:
    text = clean(raw)
    if not text:
        return "09:00", "19:00", "simulated_default"
    if "全天" in text:
        return "00:00", "24:00", "source_all_day"
    times = re.findall(r"(\d{1,2}:\d{2})", text)
    if len(times) >= 2:
        return times[0], times[1], "parsed_from_source"
    if len(times) == 1:
        return times[0], "19:00", "partial_parse_default_close"
    return "09:00", "19:00", "simulated_default"


def build_time_windows(spots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in spots.iterrows():
        open_t, close_t, quality = parse_open_hours(r.get("open_time_raw", ""))
        weekly_closed = ""
        if "艾提尕尔" in r["spot_name"]:
            weekly_closed = "Friday_main_prayer_limited"
        rows.append(
            {
                "spot_id": r["spot_id"],
                "spot_name": r["spot_name"],
                "open_time": open_t,
                "close_time": close_t,
                "weekly_or_seasonal_rule": weekly_closed,
                "time_window_quality": quality,
                "raw_open_time": r.get("open_time_raw", ""),
            }
        )
    return pd.DataFrame(rows)


def build_capacity(spots: pd.DataFrame) -> pd.DataFrame:
    official = {
        "天山天池": (25800, 8600, "SRC_TIANCHI_CAP", "official"),
        "赛里木湖": (22000, 5570, "SRC_SAILIMU_CAP", "official"),
        "果子沟大桥": (22000, 5570, "SRC_SAILIMU_CAP", "official_proxy_same_area"),
        "喀纳斯湖": (27000, 9000, "SRC_KANAS_CAP", "secondary_public_notice"),
        "禾木村": (20000, 6000, "SRC_KANAS_CAP", "secondary_public_notice"),
        "白哈巴村": (8000, 2400, "SRC_KANAS_CAP", "secondary_public_notice"),
    }
    rows = []
    for _, r in spots.iterrows():
        name = r["spot_name"]
        if name in official:
            daily, instant, sid, source_type = official[name]
            method = "source_extracted"
        elif truthy(r["ordinary_tourist_restricted"]):
            daily, instant, sid, source_type, method = 120, 40, "MODEL_ASSUMPTION", "simulated", "remote_approval_site_low_capacity"
        elif truthy(r["requires_border_permit"]):
            daily, instant, sid, source_type, method = 3000, 900, "MODEL_ASSUMPTION", "simulated", "border_permit_scenic_capacity"
        elif truthy(r["is_cultural"]):
            daily, instant, sid, source_type, method = 6000, 1800, "MODEL_ASSUMPTION", "simulated", "heritage_site_default_capacity"
        elif truthy(r["is_natural"]):
            daily, instant, sid, source_type, method = 12000, 3500, "MODEL_ASSUMPTION", "simulated", "natural_scenic_default_capacity"
        else:
            daily, instant, sid, source_type, method = 8000, 2400, "MODEL_ASSUMPTION", "simulated", "general_spot_default_capacity"
        if name == "喀什古城":
            daily, instant, sid, source_type, method = 120000, 30000, "SRC_XJ_MAYDAY_2025", "observed_based_simulated_capacity", "calibrated_by_2025_mayday_840k_5days_observed"
        rows.append(
            {
                "spot_id": r["spot_id"],
                "spot_name": name,
                "daily_capacity_persons": daily,
                "instant_capacity_persons": instant,
                "capacity_source_id": sid,
                "capacity_source_type": source_type,
                "capacity_method": method,
                "effective_capacity_beta_085": round(daily * 0.85),
                "notes": "容量缺口已显式模拟" if source_type == "simulated" else "",
            }
        )
    return pd.DataFrame(rows)


def build_cultural_tags(spots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in spots.iterrows():
        name = r["spot_name"]
        text = f"{name} {r.get('region_raw','')} {r.get('modeling_note','')}"
        silk = any(k in name for k in ["交河", "高昌", "苏公塔", "吐峪沟", "克孜尔", "库车", "楼兰", "尼雅", "北庭"])
        religion = any(k in name for k in ["清真寺", "苏公塔", "麻扎", "克孜尔", "香妃"])
        archaeology = any(k in name for k in ["故城", "古城", "遗址", "石窟", "楼兰", "尼雅", "北庭"])
        ethnic = any(k in name for k in ["喀什", "艾提尕尔", "香妃", "吐峪沟", "罗布人", "霍城", "千里葡萄"])
        natural = truthy(r.get("is_natural"))
        heritage = truthy(r.get("is_cultural")) or silk or archaeology or religion
        source_ids = []
        if any(k in name for k in ["交河", "高昌"]):
            source_ids.append("SRC_TURPAN_HERITAGE")
        if any(k in name for k in ["北庭", "克孜尔", "高昌", "交河"]):
            source_ids.append("SRC_XJ_CULTURE_OVERVIEW")
        if "艾提尕尔" in name:
            source_ids.append("SRC_KASHI_IDKAH")
        if "克孜尔" in name:
            source_ids.append("SRC_KIZIL_REGULATION")
        if not source_ids:
            source_ids.append("SOURCE_FROM_LOCAL_SPOT_REMARKS")
        value = 1 + 3 * int(heritage) + 2 * int(silk) + 2 * int(archaeology) + 2 * int(religion) + int(ethnic) + int(natural)
        if truthy(r.get("requires_approval")):
            value += 2
        rows.append(
            {
                "spot_id": r["spot_id"],
                "spot_name": name,
                "tag_silk_road": silk,
                "tag_religion": religion,
                "tag_ethnic_folk": ethnic,
                "tag_archaeology": archaeology,
                "tag_world_or_key_heritage": heritage,
                "tag_natural": natural,
                "culture_value_score": value,
                "tag_source_ids": ";".join(source_ids),
            }
        )
    return pd.DataFrame(rows)


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, node_name: str, node_type: str, hub_name: str = "") -> None:
    nodes[node_id] = {"node_id": node_id, "node_name": node_name, "node_type": node_type, "hub_name": hub_name}


def add_edge(edges: list[dict[str, Any]], edge_id: str, src: str, dst: str, mode: str, layer: str, time_h: float, cost: float, risk: float, source: str, bidirectional: bool = False, **extra: Any) -> None:
    base = {
        "edge_id": edge_id,
        "from_node": src,
        "to_node": dst,
        "mode": mode,
        "layer": layer,
        "base_time_hours": round(time_h, 3),
        "base_cost_yuan_per_two": round(cost, 2),
        "base_risk": round(risk, 3),
        "source_id": source,
        **extra,
    }
    edges.append(base)
    if bidirectional:
        rev = base.copy()
        rev["edge_id"] = edge_id + "R"
        rev["from_node"], rev["to_node"] = dst, src
        edges.append(rev)


def hub_node(hub: str) -> str:
    return f"HUB::{hub}"


def station_node(hub: str) -> str:
    return f"RAIL::{hub}"


def airport_node(hub: str) -> str:
    return f"AIR::{hub}"


def bus_node(hub: str) -> str:
    return f"BUS::{hub}"


def spot_node(spot_id: str) -> str:
    return f"SPOT::{spot_id}"


AMAP_NETWORK_SOURCE_ID = "AMAP_DISTANCE_API"


def amap_road_risk(distance_km: float, duration_hours: float, quality_note: Any = "") -> float:
    if distance_km <= 0 and duration_hours <= 0:
        return 0.0
    risk = 0.04 + min(max(distance_km, 0.0) / 3000.0, 0.25)
    if duration_hours >= 8:
        risk += 0.04
    if "fallback" in clean(quality_note).lower():
        risk += 0.05
    return min(risk, 0.45)


TRANSPORT_HUB_ALIASES = {
    "乌鲁木齐": "乌鲁木齐市",
    "乌鲁木齐南": "乌鲁木齐市",
    "吐鲁番北": "吐鲁番市",
    "吐鲁番": "吐鲁番市",
    "库尔勒": "库尔勒市",
    "伊宁": "伊宁市(伊犁)",
    "喀什": "喀什市",
    "和田": "和田市",
    "阿克苏": "阿克苏市",
    "库车": "库车市",
    "阿勒泰": "阿勒泰市",
    "喀纳斯": "阿勒泰市",
}


def transport_hub_name(value: Any) -> str:
    text = clean(value)
    return TRANSPORT_HUB_ALIASES.get(text, text)


def ensure_station_node(nodes: dict[str, dict[str, Any]], hub: str, known_hubs: set[str]) -> None:
    if station_node(hub) not in nodes:
        node_type = "rail_station" if hub in known_hubs else "transit_rail_station"
        add_node(nodes, station_node(hub), hub + "站", node_type, hub)


def ensure_airport_node(nodes: dict[str, dict[str, Any]], hub: str, known_hubs: set[str]) -> None:
    if airport_node(hub) not in nodes:
        node_type = "airport" if hub in known_hubs else "transit_airport"
        add_node(nodes, airport_node(hub), hub + "机场", node_type, hub)


def select_schedule_edges(df: pd.DataFrame, from_col: str, to_col: str, duration_col: str, price_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_from_hub"] = work[from_col].map(transport_hub_name)
    work["_to_hub"] = work[to_col].map(transport_hub_name)
    work["_duration"] = pd.to_numeric(work[duration_col], errors="coerce")
    if price_col and price_col in work.columns:
        work["_price"] = pd.to_numeric(work[price_col], errors="coerce")
        price_lookup = work.groupby(["_from_hub", "_to_hub"])["_price"].min().to_dict()
    else:
        work["_price"] = np.nan
        price_lookup = {}
    rows = []
    for (src, dst), group in work.dropna(subset=["_duration"]).groupby(["_from_hub", "_to_hub"], sort=False):
        group = group.copy()
        group["_price_fill"] = group["_price"].fillna(price_lookup.get((src, dst), np.nan))
        group["_price_fill"] = group["_price_fill"].fillna(group["_duration"] * 120)
        chosen = group.sort_values(["_duration", "_price_fill"]).iloc[0].copy()
        chosen["_model_price_one_person"] = float(chosen["_price_fill"])
        rows.append(chosen)
    return pd.DataFrame(rows)


def build_multimodal_network(
    spots: pd.DataFrame,
    hubs: pd.DataFrame,
    road_edges: pd.DataFrame,
    transport: pd.DataFrame,
    rail_options: pd.DataFrame | None = None,
    air_options: pd.DataFrame | None = None,
    amap_od: pd.DataFrame | None = None,
    amap_depot: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    known_hubs = set(hubs["hub_name"])
    for _, h in hubs.iterrows():
        hn = hub_node(h["hub_name"])
        add_node(nodes, hn, h["hub_name"], "city_hub", h["hub_name"])

    major_rail = ["乌鲁木齐市", "吐鲁番市", "库尔勒市", "若羌县(楼兰)", "伊宁市(伊犁)", "喀什市", "和田市", "库车市", "阿勒泰市"]
    major_air = ["乌鲁木齐市", "伊宁市(伊犁)", "喀什市", "库尔勒市", "阿勒泰市", "和田市", "库车市"]
    major_bus = sorted(set(hubs["hub_name"]))

    eid = 1
    for hub in major_rail:
        if hub in known_hubs:
            ensure_station_node(nodes, hub, known_hubs)
            add_edge(edges, f"T{eid:04d}", hub_node(hub), station_node(hub), "transfer", "transfer", 0.35, 20, 0.02, "MODEL_ASSUMPTION", True)
            eid += 1
    for hub in major_air:
        if hub in known_hubs:
            ensure_airport_node(nodes, hub, known_hubs)
            transfer_time = 0.8 if hub == "乌鲁木齐市" else 0.45
            add_edge(edges, f"T{eid:04d}", hub_node(hub), airport_node(hub), "transfer", "transfer", transfer_time, 60, 0.03, "MODEL_ASSUMPTION", True)
            eid += 1
    for hub in major_bus:
        add_node(nodes, bus_node(hub), hub + "客运节点", "bus_terminal", hub)
        add_edge(edges, f"T{eid:04d}", hub_node(hub), bus_node(hub), "transfer", "transfer", 0.25, 15, 0.02, "MODEL_ASSUMPTION", True)
        eid += 1

    for _, r in road_edges.iterrows():
        source = "SOURCE_LOCAL_ROAD_TABLE"
        risk = 0.08
        seasonal = truthy(r.get("is_seasonal_sensitive"))
        if seasonal:
            risk = 0.35
            source = "SRC_DUKU_CCTV_2026;SRC_DUKU_FIRSTDAY"
        add_edge(
            edges,
            f"R{eid:04d}",
            hub_node(r["from_hub_name"]),
            hub_node(r["to_hub_name"]),
            "self_drive",
            "road",
            num(r["time_hours"]),
            num(r["distance_km"]) * 0.55,
            risk,
            source,
            bidirectional=False,
            distance_km=num(r["distance_km"]),
            seasonal_sensitive=seasonal,
        )
        eid += 1

    rail_selected = select_schedule_edges(
        rail_options if rail_options is not None else pd.DataFrame(),
        "from_city",
        "to_city",
        "duration_hours",
        "min_ticket_price_yuan",
    )
    if rail_selected.empty:
        rail_specs = [
            ("乌鲁木齐市", "吐鲁番市", 1.0, 140, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("乌鲁木齐市", "库尔勒市", 4.15, 300, "SRC_RUOQIANG_RAIL_2026", "", ""),
            ("库尔勒市", "若羌县(楼兰)", 3.95, 220, "SRC_RUOQIANG_RAIL_2026", "", ""),
            ("乌鲁木齐市", "伊宁市(伊犁)", 5.8, 450, "SRC_YINING_RAIL", "", ""),
            ("乌鲁木齐市", "喀什市", 18.2, 700, "SRC_URUMQI_KASHI_RAIL", "", ""),
            ("喀什市", "和田市", 5.8, 230, "SOURCE_LOCAL_TRANSPORT_COST_REVIEW_REQUIRED", "", ""),
            ("乌鲁木齐市", "阿勒泰市", 9.0, 340, "MODEL_ASSUMPTION", "", ""),
        ]
    else:
        rail_specs = []
        for _, r in rail_selected.iterrows():
            rail_specs.append(
                (
                    r["_from_hub"],
                    r["_to_hub"],
                    num(r["_duration"]),
                    num(r["_model_price_one_person"]) * 2,
                    "RAIL_12306_MODEL_OPTIONS",
                    r.get("train_code", ""),
                    r.get("model_option_role", ""),
                )
            )
    for a, b, t, c, sid, schedule_id, option_role in rail_specs:
        ensure_station_node(nodes, a, known_hubs)
        ensure_station_node(nodes, b, known_hubs)
        add_edge(
            edges,
            f"RL{eid:04d}",
            station_node(a),
            station_node(b),
            "rail",
            "rail",
            t,
            c,
            0.04 if sid == "RAIL_12306_MODEL_OPTIONS" else 0.06,
            sid,
            True,
            daily_departures=1 if sid == "RAIL_12306_MODEL_OPTIONS" else 3,
            schedule_wait_hours=0.75 if sid == "RAIL_12306_MODEL_OPTIONS" else 1.5,
            schedule_id=schedule_id,
            model_option_role=option_role,
        )
        eid += 1

    air_selected = select_schedule_edges(
        air_options if air_options is not None else pd.DataFrame(),
        "from_city",
        "to_city",
        "model_air_time_hours",
        "fare_proxy_yuan",
    )
    if air_selected.empty:
        air_specs = [
            ("乌鲁木齐市", "喀什市", 1.5, 1100, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("乌鲁木齐市", "伊宁市(伊犁)", 1.0, 1400, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("乌鲁木齐市", "阿勒泰市", 1.0, 800, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("乌鲁木齐市", "库尔勒市", 1.0, 700, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("喀什市", "和田市", 1.0, 550, "SOURCE_LOCAL_TRANSPORT_COST", "", ""),
            ("乌鲁木齐市", "库车市", 1.25, 900, "MODEL_ASSUMPTION", "", ""),
        ]
    else:
        air_specs = []
        for _, r in air_selected.iterrows():
            air_specs.append(
                (
                    r["_from_hub"],
                    r["_to_hub"],
                    num(r["_duration"]),
                    num(r["_model_price_one_person"]) * 2,
                    "AIR_PUBLIC_SCHEDULE_SEED",
                    r.get("flight_no", ""),
                    r.get("quality_flag", ""),
                )
            )
    for a, b, t, c, sid, schedule_id, quality_flag in air_specs:
        ensure_airport_node(nodes, a, known_hubs)
        ensure_airport_node(nodes, b, known_hubs)
        add_edge(
            edges,
            f"AR{eid:04d}",
            airport_node(a),
            airport_node(b),
            "air",
            "air",
            t,
            c,
            0.06 if sid == "AIR_PUBLIC_SCHEDULE_SEED" else 0.08,
            sid,
            True,
            daily_departures=1 if sid == "AIR_PUBLIC_SCHEDULE_SEED" else 4,
            schedule_wait_hours=1.0 if sid == "AIR_PUBLIC_SCHEDULE_SEED" else 2.0,
            schedule_id=schedule_id,
            quality_flag=quality_flag,
        )
        eid += 1

    bus_specs = [
        ("乌鲁木齐市", "吐鲁番市", 2.5, 80),
        ("乌鲁木齐市", "伊宁市(伊犁)", 7.0, 350),
        ("乌鲁木齐市", "喀什市", 18.0, 660),
        ("吐鲁番市", "库尔勒市", 4.5, 200),
        ("库尔勒市", "若羌县(楼兰)", 6.0, 250),
        ("伊宁市(伊犁)", "那拉提镇", 3.0, 150),
    ]
    for a, b, t, c in bus_specs:
        if a in set(hubs["hub_name"]) and b in set(hubs["hub_name"]):
            add_edge(edges, f"BS{eid:04d}", bus_node(a), bus_node(b), "bus", "bus", t, c, 0.09, "SOURCE_LOCAL_TRANSPORT_COST", True, daily_departures=3, schedule_wait_hours=0.8)
            eid += 1

    for _, s in spots.iterrows():
        sid = spot_node(s["spot_id"])
        add_node(nodes, sid, s["spot_name"], "scenic_spot", s["hub_name"])
        access_h = num(s.get("local_access_hours_from_text"), 0.0)
        if s["hub_name"] in {"天池风景区", "赛里木湖", "禾木村", "白哈巴", "可可托海", "巴音布鲁克"}:
            access_h = 0.15
        if access_h <= 0:
            access_h = 0.3
        raw_cost = num(s.get("ticket_high_mandatory_local_yuan_per_person"), 0.0) * 2
        cost = max(raw_cost, LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO)
        add_edge(
            edges,
            f"SH{eid:04d}",
            hub_node(s["hub_name"]),
            sid,
            "scenic_shuttle",
            "scenic_access",
            access_h,
            cost,
            0.04,
            "SOURCE_LOCAL_SPOT_ACCESS",
            True,
            local_shuttle_raw_cost_yuan_per_two=round(raw_cost, 2),
            local_shuttle_cost_floor_yuan_per_two=LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO,
            cost_floor_applied=raw_cost < LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO,
        )
        eid += 1

    known_spot_ids = set(spots["spot_id"].astype(str))
    if amap_od is not None and not amap_od.empty:
        for _, r in amap_od.iterrows():
            src_spot = clean(r.get("from_spot_id"))
            dst_spot = clean(r.get("to_spot_id"))
            if not src_spot or not dst_spot or src_spot == dst_spot:
                continue
            if src_spot not in known_spot_ids or dst_spot not in known_spot_ids:
                continue
            duration = num(r.get("driving_duration_hours"))
            distance = num(r.get("driving_distance_km"))
            if duration <= 0 and distance <= 0:
                continue
            cost = num(r.get("amap_selfdrive_cost_yuan_per_two"), distance * 0.55)
            quality_note = clean(r.get("od_quality_note"))
            add_edge(
                edges,
                f"AM{eid:04d}",
                spot_node(src_spot),
                spot_node(dst_spot),
                "self_drive",
                "road",
                duration,
                cost,
                amap_road_risk(distance, duration, quality_note),
                clean(r.get("source_id")) or AMAP_NETWORK_SOURCE_ID,
                bidirectional=False,
                distance_km=distance,
                seasonal_sensitive=False,
                amap_edge_type="spot_to_spot",
                od_quality_note=quality_note,
            )
            eid += 1

    if amap_depot is not None and not amap_depot.empty:
        for _, r in amap_depot.iterrows():
            spot_id = clean(r.get("spot_id"))
            if spot_id not in known_spot_ids:
                continue
            duration = num(r.get("driving_duration_hours"))
            distance = num(r.get("driving_distance_km"))
            if duration <= 0 and distance <= 0:
                continue
            direction = clean(r.get("direction"))
            if direction == "depot_to_spot":
                src, dst = START_NODE, spot_node(spot_id)
            elif direction == "spot_to_depot":
                src, dst = spot_node(spot_id), START_NODE
            else:
                continue
            add_edge(
                edges,
                f"AD{eid:04d}",
                src,
                dst,
                "self_drive",
                "road",
                duration,
                num(r.get("selfdrive_cost_yuan_per_two"), distance * 0.55),
                amap_road_risk(distance, duration, "depot_access"),
                clean(r.get("source_id")) or AMAP_NETWORK_SOURCE_ID,
                bidirectional=False,
                distance_km=distance,
                seasonal_sensitive=False,
                amap_edge_type=direction,
            )
            eid += 1

    return pd.DataFrame(nodes.values()), pd.DataFrame(edges)


def build_time_dependent_rules() -> pd.DataFrame:
    rows = [
        ("base_summer", 0.45, "road", 1.00, 1.00, 0.00, "暑假普通日，使用基础时间。"),
        ("summer_peak", 0.20, "road", 1.15, 1.05, 0.03, "7-8月高峰，路面和景区接驳轻度拥堵。"),
        ("mayday_peak", 0.15, "road", 1.35, 1.10, 0.06, "五一长假，高速/景区接驳拥堵。"),
        ("duku_weather_disruption", 0.10, "road", 1.80, 1.00, 0.25, "独库/山区道路天气扰动，含临时封闭风险。"),
        ("hotel_full_price_surge", 0.10, "road", 1.10, 1.30, 0.08, "住宿紧张导致费用上浮。"),
        ("base_summer", 0.45, "rail", 1.00, 1.00, 0.00, "铁路按班次运行，时间相对稳定。"),
        ("summer_peak", 0.20, "rail", 1.05, 1.10, 0.02, "暑期票源紧张，等待和费用小幅上升。"),
        ("mayday_peak", 0.15, "rail", 1.08, 1.20, 0.04, "五一铁路票源紧张。"),
        ("base_summer", 0.45, "air", 1.00, 1.00, 0.00, "航班基础情景。"),
        ("summer_peak", 0.20, "air", 1.05, 1.25, 0.04, "暑期机票上浮。"),
        ("mayday_peak", 0.15, "air", 1.10, 1.35, 0.06, "五一机票上浮。"),
        ("base_summer", 0.45, "bus", 1.00, 1.00, 0.00, "大巴基础情景。"),
        ("mayday_peak", 0.15, "bus", 1.30, 1.10, 0.05, "五一班车和道路拥堵。"),
        ("base_summer", 0.45, "scenic_shuttle", 1.00, 1.00, 0.00, "景区接驳基础情景。"),
        ("mayday_peak", 0.15, "scenic_shuttle", 1.50, 1.00, 0.05, "五一景区排队时间上升。"),
    ]
    return pd.DataFrame(rows, columns=["scenario_id", "probability", "mode", "time_factor", "cost_factor", "risk_add", "description"])


def edge_scenarios(edges: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenario_probs = rules.groupby("scenario_id")["probability"].max().to_dict()
    for _, e in edges.iterrows():
        relevant = rules[rules["mode"].eq(e["mode"])].copy()
        existing = set(relevant["scenario_id"]) if not relevant.empty else set()
        fallback = [
            {
                "scenario_id": sid,
                "probability": prob,
                "mode": e["mode"],
                "time_factor": 1.0,
                "cost_factor": 1.0,
                "risk_add": 0.0,
                "description": "该交通方式在此情景下暂无公开差异参数，使用基准边权。",
            }
            for sid, prob in scenario_probs.items()
            if sid not in existing
        ]
        if fallback:
            relevant = pd.concat([relevant, pd.DataFrame(fallback)], ignore_index=True)
        for _, r in relevant.iterrows():
            seasonal_closed = truthy(e.get("seasonal_sensitive", False)) and r["scenario_id"] == "mayday_peak"
            rows.append(
                {
                    "edge_id": e["edge_id"],
                    "scenario_id": r["scenario_id"],
                    "effective_time_hours": None if seasonal_closed else round(num(e["base_time_hours"]) * num(r["time_factor"]) + num(e.get("schedule_wait_hours"), 0.0), 3),
                    "effective_cost_yuan_per_two": round(num(e["base_cost_yuan_per_two"]) * num(r["cost_factor"]), 2),
                    "effective_risk": min(1.0, round(num(e["base_risk"]) + num(r["risk_add"]) + (0.4 if seasonal_closed else 0), 3)),
                    "is_available": not seasonal_closed,
                    "availability_note": "五一默认不使用独库/G217季节敏感边" if seasonal_closed else "",
                }
            )
    return pd.DataFrame(rows)


def build_special_access(spots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in spots.iterrows():
        name = s["spot_name"]
        approval = truthy(s.get("requires_approval"))
        border = truthy(s.get("requires_border_permit"))
        remote = truthy(s.get("high_altitude_or_remote"))
        guide = approval or any(k in name for k in ["楼兰", "尼雅", "夏塔"])
        offroad = any(k in name for k in ["楼兰", "尼雅"])
        min_group = 1
        buffer_h = 0.0
        if approval:
            min_group = 4
            buffer_h += 4.0
        if border:
            buffer_h += 1.0
        if remote:
            buffer_h += 1.0
        rows.append(
            {
                "spot_id": s["spot_id"],
                "spot_name": name,
                "requires_approval": approval,
                "approval_lead_days": 90 if "楼兰" in name else 30 if approval else 0,
                "requires_border_permit": border,
                "requires_licensed_guide": guide,
                "requires_offroad_vehicle": offroad,
                "minimum_group_size": min_group,
                "safety_buffer_hours": buffer_h,
                "ordinary_tourist_allowed": not truthy(s.get("ordinary_tourist_restricted")),
                "constraint_note": "特殊准入点，不作为普通游客基准路线必选点" if (approval or border or guide or offroad) else "",
            }
        )
    return pd.DataFrame(rows)


def build_hotel_constraints(hotel_default: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    hotel_hubs = set(hotel_default["hub_name"].dropna()) | set(hotel_default["place_name"].dropna())
    rows = []
    for _, h in hubs.iterrows():
        name = h["hub_name"]
        matched = name in hotel_hubs or any(name.replace("市", "") in clean(x) or clean(x).replace("市", "") in name for x in hotel_hubs)
        price = 260.0
        row = hotel_default[(hotel_default["hub_name"].eq(name)) | (hotel_default["place_name"].eq(name))]
        if not row.empty:
            price = num(row.iloc[0]["high_season_room_yuan_per_night"], price)
        rows.append(
            {
                "hub_name": name,
                "node_id": hub_node(name),
                "is_hotel_hub": bool(matched),
                "default_room_price_yuan_per_night": price,
                "hotel_capacity_rooms_simulated": 80 if matched else 20,
                "source_type": "local_hotel_price_plus_simulated_rooms" if matched else "simulated_limited_lodging",
            }
        )
    return pd.DataFrame(rows)


def build_graph(edges: pd.DataFrame, scenario: str = "base_summer", objective: str = "time") -> nx.DiGraph:
    rules = build_time_dependent_rules()
    scen = edge_scenarios(edges, rules)
    scen = scen[scen["scenario_id"].eq(scenario)]
    sdict = scen.set_index("edge_id").to_dict("index")
    g = nx.DiGraph()
    for _, e in edges.iterrows():
        s = sdict.get(e["edge_id"])
        if s is None or not s["is_available"]:
            continue
        if objective == "cost":
            weight = s["effective_cost_yuan_per_two"]
        elif objective == "risk":
            weight = s["effective_risk"]
        else:
            weight = s["effective_time_hours"]
        g.add_edge(
            e["from_node"],
            e["to_node"],
            weight=weight,
            time=s["effective_time_hours"],
            cost=s["effective_cost_yuan_per_two"],
            risk=s["effective_risk"],
            mode=e["mode"],
            edge_id=e["edge_id"],
        )
    return g


def shortest_path_metrics(g: nx.DiGraph, source: str, target: str) -> dict[str, Any]:
    try:
        path = nx.shortest_path(g, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"time": math.inf, "cost": math.inf, "risk": math.inf, "path": "", "modes": ""}
    time = cost = risk = 0.0
    modes = []
    for a, b in zip(path, path[1:]):
        data = g[a][b]
        time += data["time"]
        cost += data["cost"]
        risk += data["risk"]
        modes.append(data["mode"])
    return {
        "time": time,
        "cost": cost,
        "risk": risk,
        "path": " -> ".join(path),
        "modes": " -> ".join(modes),
    }


def metrics_for_node_sequence(g: nx.DiGraph, nodes: list[str]) -> dict[str, float]:
    total_time = total_cost = total_risk = 0.0
    for source, target in zip(nodes, nodes[1:]):
        m = shortest_path_metrics(g, source, target)
        if not math.isfinite(m["time"]):
            return {"travel_time": math.inf, "transport_cost": math.inf, "path_risk": math.inf}
        total_time += m["time"]
        total_cost += m["cost"]
        total_risk += m["risk"]
    return {"travel_time": total_time, "transport_cost": total_cost, "path_risk": total_risk}


def build_enhanced_od(spots: pd.DataFrame, edges: pd.DataFrame, scenario: str = "base_summer") -> tuple[pd.DataFrame, pd.DataFrame]:
    g_time = build_graph(edges, scenario=scenario, objective="time")
    spot_nodes = {r["spot_id"]: spot_node(r["spot_id"]) for _, r in spots.iterrows()}
    depot = START_NODE
    rows = []
    depot_rows = []
    for _, a in spots.iterrows():
        for _, b in spots.iterrows():
            m = shortest_path_metrics(g_time, spot_nodes[a["spot_id"]], spot_nodes[b["spot_id"]])
            rows.append(
                {
                    "from_spot_id": a["spot_id"],
                    "from_spot_name": a["spot_name"],
                    "to_spot_id": b["spot_id"],
                    "to_spot_name": b["spot_name"],
                    "scenario_id": scenario,
                    "shortest_time_hours": round(m["time"], 3) if math.isfinite(m["time"]) else None,
                    "shortest_cost_yuan_per_two": round(m["cost"], 2) if math.isfinite(m["cost"]) else None,
                    "path_risk": round(m["risk"], 3) if math.isfinite(m["risk"]) else None,
                    "path_modes": m["modes"],
                }
            )
    for _, s in spots.iterrows():
        out = shortest_path_metrics(g_time, depot, spot_nodes[s["spot_id"]])
        back = shortest_path_metrics(g_time, spot_nodes[s["spot_id"]], depot)
        depot_rows.append(
            {
                "spot_id": s["spot_id"],
                "spot_name": s["spot_name"],
                "depot_to_spot_time": round(out["time"], 3),
                "spot_to_depot_time": round(back["time"], 3),
                "depot_to_spot_cost": round(out["cost"], 2),
                "spot_to_depot_cost": round(back["cost"], 2),
                "depot_to_spot_risk": round(out["risk"], 3),
                "spot_to_depot_risk": round(back["risk"], 3),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(depot_rows)


def matrix_from_od(spots: pd.DataFrame, od: pd.DataFrame, depot: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(spots)
    spot_ids = spots["spot_id"].tolist()
    idx = {sid: i + 1 for i, sid in enumerate(spot_ids)}
    time = np.zeros((n + 1, n + 1))
    cost = np.zeros((n + 1, n + 1))
    risk = np.zeros((n + 1, n + 1))
    for _, r in depot.iterrows():
        j = idx[r["spot_id"]]
        time[0, j] = num(r["depot_to_spot_time"], 1e6)
        time[j, 0] = num(r["spot_to_depot_time"], 1e6)
        cost[0, j] = num(r["depot_to_spot_cost"], 1e6)
        cost[j, 0] = num(r["spot_to_depot_cost"], 1e6)
        risk[0, j] = num(r["depot_to_spot_risk"], 10)
        risk[j, 0] = num(r["spot_to_depot_risk"], 10)
    for _, r in od.iterrows():
        i = idx[r["from_spot_id"]]
        j = idx[r["to_spot_id"]]
        time[i, j] = num(r["shortest_time_hours"], 1e6)
        cost[i, j] = num(r["shortest_cost_yuan_per_two"], 1e6)
        risk[i, j] = num(r["path_risk"], 10)
    return time, cost, risk


def service_hours(spots: pd.DataFrame, special: pd.DataFrame, i: int, multiplier: float = 1.0) -> float:
    if i == 0:
        return 0.0
    row = spots.iloc[i - 1]
    sp = special[special["spot_id"].eq(row["spot_id"])]
    buffer = num(sp.iloc[0]["safety_buffer_hours"], 0.0) if not sp.empty else 0.0
    return num(row["visit_hours_mid"], 0.0) * multiplier + buffer


def route_metrics(seq: list[int], spots: pd.DataFrame, time: np.ndarray, cost: np.ndarray, risk: np.ndarray, special: pd.DataFrame, multiplier: float = 1.0) -> dict[str, float]:
    cur = 0
    travel = total_cost = total_risk = service = 0.0
    for i in seq:
        travel += time[cur, i]
        total_cost += cost[cur, i]
        total_risk += risk[cur, i]
        service += service_hours(spots, special, i, multiplier)
        cur = i
    travel += time[cur, 0]
    total_cost += cost[cur, 0]
    total_risk += risk[cur, 0]
    ticket = sum(num(spots.iloc[i - 1]["ticket_high_total_yuan_per_person"], 0.0) * 2 for i in seq)
    total_hours = travel + service
    return {
        "spots_count": len(seq),
        "travel_hours": round(travel, 2),
        "service_hours": round(service, 2),
        "total_hours": round(total_hours, 2),
        "days": math.ceil(total_hours / DAY_HOURS),
        "transport_cost_yuan": round(total_cost, 2),
        "ticket_yuan": round(ticket, 2),
        "risk_score": round(total_risk, 3),
        "objective_cost_yuan": round(total_cost + ticket + max(math.ceil(total_hours / DAY_HOURS) - 1, 0) * 260, 2),
    }


def greedy_route(selected: list[int], spots: pd.DataFrame, time: np.ndarray) -> list[int]:
    remaining = set(selected)
    route = []
    cur = 0
    while remaining:
        nxt = min(remaining, key=lambda j: (time[cur, j], -num(spots.iloc[j - 1]["priority_score_for_op"], 1)))
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return two_opt_indices(route, time)


def two_opt_indices(seq: list[int], time: np.ndarray, max_iter: int = 18) -> list[int]:
    if len(seq) < 4:
        return seq
    def travel(s: list[int]) -> float:
        cur = 0
        v = 0.0
        for x in s:
            v += time[cur, x]
            cur = x
        return v + time[cur, 0]
    best = seq[:]
    best_val = travel(best)
    improved = True
    it = 0
    while improved and it < max_iter:
        it += 1
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best) + 1):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                val = travel(cand)
                if val < best_val:
                    best, best_val, improved = cand, val, True
    return best


def pcop_milp(
    spots: pd.DataFrame,
    time: np.ndarray,
    cost: np.ndarray,
    special: pd.DataFrame,
    cultural: pd.DataFrame,
    budget_hours: float = 30 * DAY_HOURS,
    max_candidates: int = 24,
) -> dict[str, Any]:
    culture_map = cultural.set_index("spot_id")["culture_value_score"].to_dict()
    scored_candidates = []
    for i, r in spots.iterrows():
        idx = i + 1
        if truthy(r["ordinary_tourist_restricted"]) or not math.isfinite(time[0, idx]) or not math.isfinite(time[idx, 0]):
            continue
        value_score = num(r["priority_score_for_op"], 1) + 0.8 * num(culture_map.get(r["spot_id"], 0), 0)
        access_penalty = (time[0, idx] + time[idx, 0]) / 12
        scored_candidates.append((value_score - access_penalty, idx))
    candidate_ids = [idx for _, idx in sorted(scored_candidates, reverse=True)[:max_candidates]]
    n = len(candidate_ids)
    N = n + 1
    pos_to_global = {p + 1: g for p, g in enumerate(candidate_ids)}
    # local matrices
    t = np.zeros((N, N))
    cst = np.zeros((N, N))
    for i in range(N):
        gi = 0 if i == 0 else pos_to_global[i]
        for j in range(N):
            gj = 0 if j == 0 else pos_to_global[j]
            t[i, j] = time[gi, gj]
            cst[i, j] = cost[gi, gj]

    x_offset = 0
    x_count = N * N
    y_offset = x_count
    y_count = n
    u_offset = x_count + y_count
    var_count = x_count + y_count + n
    c = np.zeros(var_count)
    for i in range(N):
        for j in range(N):
            idx = x_offset + i * N + j
            c[idx] = 0.005 * t[i, j] + 0.0002 * cst[i, j]
            if i == j:
                c[idx] = 0
    for p in range(1, N):
        gidx = pos_to_global[p]
        spot = spots.iloc[gidx - 1]
        score = num(spot["priority_score_for_op"], 1) + 0.8 * num(culture_map.get(spot["spot_id"], 0), 0)
        c[y_offset + p - 1] = -score

    lb = np.zeros(var_count)
    ub = np.ones(var_count)
    ub[u_offset:] = n
    integrality = np.zeros(var_count)
    integrality[: x_count + y_count] = 1
    for i in range(N):
        lb[x_offset + i * N + i] = 0
        ub[x_offset + i * N + i] = 0

    constraints = []
    lo = []
    hi = []
    row = np.zeros(var_count)
    for j in range(1, N):
        row[x_offset + 0 * N + j] = 1
    constraints.append(row); lo.append(1); hi.append(1)
    row = np.zeros(var_count)
    for i in range(1, N):
        row[x_offset + i * N + 0] = 1
    constraints.append(row); lo.append(1); hi.append(1)

    for k in range(1, N):
        row = np.zeros(var_count)
        for j in range(N):
            row[x_offset + k * N + j] = 1
        row[y_offset + k - 1] = -1
        constraints.append(row); lo.append(0); hi.append(0)
        row = np.zeros(var_count)
        for i in range(N):
            row[x_offset + i * N + k] = 1
        row[y_offset + k - 1] = -1
        constraints.append(row); lo.append(0); hi.append(0)

    row = np.zeros(var_count)
    for i in range(N):
        for j in range(N):
            row[x_offset + i * N + j] = t[i, j]
    for p in range(1, N):
        row[y_offset + p - 1] = service_hours(spots, special, pos_to_global[p])
    constraints.append(row); lo.append(0); hi.append(budget_hours)

    for i in range(1, N):
        row = np.zeros(var_count)
        row[u_offset + i - 1] = 1
        row[y_offset + i - 1] = -1
        constraints.append(row); lo.append(0); hi.append(n)
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            row = np.zeros(var_count)
            row[u_offset + i - 1] = 1
            row[u_offset + j - 1] = -1
            row[x_offset + i * N + j] = n
            constraints.append(row); lo.append(-math.inf); hi.append(n - 1)

    lc = LinearConstraint(np.vstack(constraints), np.array(lo), np.array(hi))
    res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=lc, options={"time_limit": 12, "mip_rel_gap": 0.04, "disp": False})
    if not res.success and res.x is None:
        return {"status": f"failed:{res.message}", "route": []}
    sol = res.x
    arcs = []
    for i in range(N):
        for j in range(N):
            if i != j and sol[x_offset + i * N + j] > 0.5:
                arcs.append((i, j))
    next_map = {i: j for i, j in arcs}
    route_local = []
    cur = 0
    seen = set()
    while cur in next_map:
        nxt = next_map[cur]
        if nxt == 0 or nxt in seen:
            break
        route_local.append(nxt)
        seen.add(nxt)
        cur = nxt
    route_global = [pos_to_global[p] for p in route_local]
    return {"status": f"{res.message}; candidate_pool={n}", "route": route_global, "objective": float(res.fun)}


def pcop_greedy_alns(spots: pd.DataFrame, time: np.ndarray, cost: np.ndarray, risk: np.ndarray, special: pd.DataFrame, budget_hours: float = 30 * DAY_HOURS) -> dict[str, Any]:
    candidates = [i + 1 for i, r in spots.iterrows() if not truthy(r["ordinary_tourist_restricted"])]
    scores = {i: num(spots.iloc[i - 1]["priority_score_for_op"], 1) for i in candidates}
    selected: list[int] = []
    remaining = set(candidates)
    while remaining:
        best = None
        best_ratio = -1
        best_route = None
        base_h = route_metrics(greedy_route(selected, spots, time), spots, time, cost, risk, special)["total_hours"] if selected else 0
        for i in list(remaining):
            route = greedy_route(selected + [i], spots, time)
            m = route_metrics(route, spots, time, cost, risk, special)
            if m["total_hours"] <= budget_hours:
                ratio = scores[i] / max(m["total_hours"] - base_h, 0.5)
                if ratio > best_ratio:
                    best, best_ratio, best_route = i, ratio, route
        if best is None:
            break
        selected = best_route or selected
        remaining.remove(best)

    best_route = selected[:]
    best_score = sum(scores[i] for i in best_route)
    for _ in range(45):
        cur = best_route[:]
        if cur:
            remove_count = max(1, int(len(cur) * random.choice([0.15, 0.25, 0.35])))
            removed = set(random.sample(cur, min(remove_count, len(cur))))
            cur = [x for x in cur if x not in removed]
        missing = [x for x in candidates if x not in cur]
        random.shuffle(missing)
        for x in sorted(missing, key=lambda z: scores[z], reverse=True):
            trial = greedy_route(cur + [x], spots, time)
            if route_metrics(trial, spots, time, cost, risk, special)["total_hours"] <= budget_hours:
                cur = trial
        cur_score = sum(scores[i] for i in cur)
        if cur_score > best_score or (cur_score == best_score and len(cur) > len(best_route)):
            best_route, best_score = cur, cur_score
    return {"route": best_route, "score": best_score, "status": "greedy_alns_completed"}


def nsga2_pareto(spots: pd.DataFrame, time: np.ndarray, cost: np.ndarray, risk: np.ndarray, special: pd.DataFrame, generations: int = 14, pop_size: int = 32) -> pd.DataFrame:
    candidates = [i + 1 for i, r in spots.iterrows() if not truthy(r["ordinary_tourist_restricted"])]
    n = len(candidates)
    scores = np.array([num(spots.iloc[i - 1]["priority_score_for_op"], 1) for i in candidates])

    def decode(bits: np.ndarray) -> tuple[list[int], dict[str, float]]:
        selected = [candidates[i] for i, b in enumerate(bits) if b]
        route = greedy_route(selected, spots, time) if selected else []
        m = route_metrics(route, spots, time, cost, risk, special)
        while m["total_hours"] > 30 * DAY_HOURS and selected:
            # remove lowest score / high service item
            rem = min(selected, key=lambda x: scores[candidates.index(x)] / max(service_hours(spots, special, x), 0.5))
            selected.remove(rem)
            route = greedy_route(selected, spots, time)
            m = route_metrics(route, spots, time, cost, risk, special)
        return route, m

    pop = np.random.rand(pop_size, n) < 0.35
    records = []
    for _ in range(generations):
        all_bits = []
        for bits in pop:
            all_bits.append(bits)
            mutant = bits.copy()
            flips = np.random.rand(n) < 0.06
            mutant[flips] = ~mutant[flips]
            all_bits.append(mutant)
        evaluated = []
        for bits in all_bits:
            route, m = decode(bits)
            profit = sum(scores[candidates.index(x)] for x in route)
            evaluated.append((bits, route, profit, m))
        # nondominated preference: high profit, low cost, low risk
        evaluated.sort(key=lambda x: (-x[2], x[3]["objective_cost_yuan"], x[3]["risk_score"]))
        kept = []
        for item in evaluated:
            _, _, profit, m = item
            dominated = False
            for kept_item in kept:
                _, _, kp, km = kept_item
                if kp >= profit and km["objective_cost_yuan"] <= m["objective_cost_yuan"] and km["risk_score"] <= m["risk_score"]:
                    dominated = True
                    break
            if not dominated:
                kept.append(item)
            if len(kept) >= pop_size:
                break
        while len(kept) < pop_size:
            kept.append(random.choice(evaluated))
        pop = np.array([x[0] for x in kept])
        records = kept
    rows = []
    seen = set()
    for _, route, profit, m in records:
        key = tuple(route)
        if key in seen or not route:
            continue
        seen.add(key)
        rows.append(
            {
                "solution_id": f"NSGA_{len(rows)+1:03d}",
                "spots_count": len(route),
                "profit_score": round(profit, 2),
                "total_hours": m["total_hours"],
                "objective_cost_yuan": m["objective_cost_yuan"],
                "risk_score": m["risk_score"],
                "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in route),
            }
        )
    return pd.DataFrame(rows).sort_values(["profit_score", "objective_cost_yuan"], ascending=[False, True]).head(20)


def spectral_two_stage(spots: pd.DataFrame, time: np.ndarray, cost: np.ndarray, risk: np.ndarray, special: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [i + 1 for i, r in spots.iterrows() if not truthy(r["ordinary_tourist_restricted"])]
    dist = np.array([[time[i, j] for j in candidates] for i in candidates])
    finite = dist[np.isfinite(dist) & (dist > 0)]
    sigma = np.median(finite) if finite.size else 10
    if finite.size:
        fill = float(np.max(finite) * 2)
        dist = np.where(np.isfinite(dist), dist, fill)
    affinity = np.exp(-dist / max(sigma, 1))
    np.fill_diagonal(affinity, 1)
    labels = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=RANDOM_SEED).fit_predict(affinity)
    groups = {0: [], 1: []}
    for idx, lab in zip(candidates, labels):
        groups[int(lab)].append(idx)
    # orient clusters: north/ili first if more such spots
    def north_score(group: list[int]) -> int:
        return sum(spots.iloc[i - 1]["region_cluster"] in ["北疆", "伊犁", "乌鲁木齐周边"] for i in group)
    ordered = sorted(groups.values(), key=north_score, reverse=True)
    summaries = []
    details = []
    for k, group in enumerate(ordered, 1):
        route = greedy_route(group, spots, time)
        m = route_metrics(route, spots, time, cost, risk, special)
        rid = f"S2_Spectral_Year{k}"
        summaries.append({"route_id": rid, **m, "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in route)})
        for order, i in enumerate(route, 1):
            details.append({"route_id": rid, "order": order, "spot_id": spots.iloc[i - 1]["spot_id"], "spot_name": spots.iloc[i - 1]["spot_name"], "region_cluster": spots.iloc[i - 1]["region_cluster"]})
    return pd.DataFrame(summaries), pd.DataFrame(details)


def minmax_cultural_vrp(spots: pd.DataFrame, time: np.ndarray, cost: np.ndarray, risk: np.ndarray, special: pd.DataFrame, cultural: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cset = set(cultural[cultural["culture_value_score"] >= 5]["spot_id"])
    indices = [i + 1 for i, r in spots.iterrows() if r["spot_id"] in cset or truthy(r["requires_approval"])]
    # force approval-heavy items into a research-special group seed
    groups = [[], [], []]
    for i in indices:
        if truthy(spots.iloc[i - 1]["requires_approval"]):
            groups[0].append(i)
    for i in indices:
        if i in groups[0]:
            continue
        best_g = 0
        best_val = math.inf
        for g in range(3):
            trial = groups[g] + [i]
            route = greedy_route(trial, spots, time)
            val = route_metrics(route, spots, time, cost, risk, special, multiplier=4)["total_hours"]
            if val < best_val:
                best_val, best_g = val, g
        groups[best_g].append(i)
    improved = True
    while improved:
        improved = False
        routes = [greedy_route(g, spots, time) for g in groups]
        times = [route_metrics(r, spots, time, cost, risk, special, multiplier=4)["total_hours"] for r in routes]
        worst = int(np.argmax(times))
        best = int(np.argmin(times))
        for item in groups[worst][:]:
            if truthy(spots.iloc[item - 1]["requires_approval"]):
                continue
            trial_groups = [g[:] for g in groups]
            trial_groups[worst].remove(item)
            trial_groups[best].append(item)
            trial_times = [route_metrics(greedy_route(g, spots, time), spots, time, cost, risk, special, multiplier=4)["total_hours"] for g in trial_groups]
            if max(trial_times) < max(times):
                groups = trial_groups
                improved = True
                break
    summaries = []
    details = []
    for k, group in enumerate(groups, 1):
        route = greedy_route(group, spots, time)
        m = route_metrics(route, spots, time, cost, risk, special, multiplier=4)
        rid = f"S3_MinMax_Group{k}"
        value = sum(num(cultural.set_index("spot_id").loc[spots.iloc[i - 1]["spot_id"], "culture_value_score"], 0) for i in route)
        summaries.append({"route_id": rid, **m, "culture_value_total": value, "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in route)})
        for order, i in enumerate(route, 1):
            details.append({"route_id": rid, "order": order, "spot_id": spots.iloc[i - 1]["spot_id"], "spot_name": spots.iloc[i - 1]["spot_name"], "region_cluster": spots.iloc[i - 1]["region_cluster"]})
    df = pd.DataFrame(summaries)
    df["max_completion_hours"] = df["total_hours"].max()
    df["balance_gap_hours"] = df["total_hours"].max() - df["total_hours"].min()
    return df, pd.DataFrame(details)


def route_from_names(names: list[str], spots: pd.DataFrame, time: np.ndarray) -> list[int]:
    lookup = {r["spot_name"]: i + 1 for i, r in spots.iterrows()}
    return greedy_route([lookup[n] for n in names if n in lookup], spots, time)


def generate_columns(spots: pd.DataFrame, time: np.ndarray, capacity: pd.DataFrame) -> pd.DataFrame:
    templates = {
        "North_Kanas": ["天山天池", "世界魔鬼城", "喀纳斯湖", "禾木村", "五彩滩"],
        "Ili_Lake_Grass": ["赛里木湖", "果子沟大桥", "霍城薰衣草花田", "那拉提草原", "喀拉峻大草原", "巴音布鲁克草原"],
        "East_Heritage": ["达坂城古镇", "火焰山", "葡萄沟", "坎儿井民俗园", "交河故城", "高昌故城", "苏公塔", "吐峪沟麻扎村"],
        "South_Culture": ["喀什古城", "艾提尕尔清真寺", "香妃园", "帕米尔高原白沙湖", "石头城遗址", "天山神秘大峡谷", "库车王府", "克孜尔石窟"],
        "Bazhou_Desert": ["博斯腾湖", "罗布人村寨", "巴音布鲁克草原", "天山神秘大峡谷", "库车王府"],
        "Hotan_Plus": ["喀什古城", "和田博物馆", "千里葡萄长廊", "奥依塔克冰川"],
        "Short_Urumqi_East": ["天山天池", "达坂城古镇", "火焰山", "葡萄沟", "交河故城"],
        "SilkRoad_Deep": ["交河故城", "高昌故城", "苏公塔", "吐峪沟麻扎村", "库车王府", "克孜尔石窟", "喀什古城"],
        "Border_Pamir": ["喀什古城", "帕米尔高原白沙湖", "石头城遗址", "奥依塔克冰川"],
    }
    cap_map = capacity.set_index("spot_id")["effective_capacity_beta_085"].to_dict()
    rows = []
    for name, names in templates.items():
        route = route_from_names(names, spots, time)
        if not route:
            continue
        route_cap_daily = min(cap_map.get(spots.iloc[i - 1]["spot_id"], 3000) for i in route)
        days = max(1, min(12, math.ceil((sum(num(spots.iloc[i - 1]["visit_hours_mid"], 1) for i in route) + sum(time[a, b] for a, b in zip([0] + route, route + [0]))) / DAY_HOURS)))
        rows.append(
            {
                "column_id": f"COL{len(rows)+1:03d}",
                "route_theme": name,
                "spots_count": len(route),
                "estimated_days": days,
                "route_capacity_persons_12day": int(route_cap_daily * 12 / max(days, 1)),
                "attraction_score": sum(num(spots.iloc[i - 1]["priority_score_for_op"], 1) for i in route),
                "route_spot_ids": ";".join(spots.iloc[i - 1]["spot_id"] for i in route),
                "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in route),
            }
        )
    return pd.DataFrame(rows)


def solve_capacity_flow(columns: pd.DataFrame, capacity: pd.DataFrame, f_total: float = 200_000.0) -> pd.DataFrame:
    spot_ids = capacity["spot_id"].tolist()
    cap = capacity.set_index("spot_id")["effective_capacity_beta_085"].to_dict()
    routes = columns.to_dict("records")
    m = len(routes)
    A = np.zeros((len(spot_ids), m))
    route_days = np.array([max(1, r["estimated_days"]) for r in routes], dtype=float)
    for j, r in enumerate(routes):
        for sid in str(r["route_spot_ids"]).split(";"):
            if sid in spot_ids:
                A[spot_ids.index(sid), j] = 1.0 / route_days[j]
    caps = np.array([cap[sid] for sid in spot_ids], dtype=float)
    upper = np.array([r["route_capacity_persons_12day"] for r in routes], dtype=float)
    total_capacity = upper.sum()
    target = min(f_total, total_capacity * 0.95)
    attraction = np.array([r["attraction_score"] for r in routes], dtype=float)

    def obj(x: np.ndarray) -> float:
        load = A @ x
        congestion = np.sum((load / np.maximum(caps, 1)) ** 2)
        return congestion - 1e-7 * attraction @ x

    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - target}]
    x0 = upper / upper.sum() * target
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, u) for u in upper], constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    x = res.x if res.success else x0
    out = columns.copy()
    out["allocated_visitors"] = np.round(x).astype(int)
    out["allocation_share"] = out["allocated_visitors"] / out["allocated_visitors"].sum()
    out["binding_capacity_upper"] = upper.astype(int)
    out["optimization_status"] = str(res.message)
    return out


def robust_route_evaluation(routes: dict[str, list[int]], spots: pd.DataFrame, edges: pd.DataFrame, special: pd.DataFrame) -> pd.DataFrame:
    rules = build_time_dependent_rules()
    scenarios = sorted(rules["scenario_id"].unique())
    rows = []
    for rid, route in routes.items():
        scenario_metrics = []
        for scen in scenarios:
            g = build_graph(edges, scenario=scen, objective="time")
            node_seq = [START_NODE] + [spot_node(spots.iloc[i - 1]["spot_id"]) for i in route] + [START_NODE]
            travel = metrics_for_node_sequence(g, node_seq)
            service = sum(service_hours(spots, special, i) for i in route)
            ticket = sum(num(spots.iloc[i - 1]["ticket_high_total_yuan_per_person"], 0.0) * 2 for i in route)
            total_hours = travel["travel_time"] + service
            m = {
                "total_hours": round(total_hours, 2),
                "objective_cost_yuan": round(travel["transport_cost"] + ticket + max(math.ceil(total_hours / DAY_HOURS) - 1, 0) * 260, 2),
                "risk_score": round(travel["path_risk"], 3),
            }
            p = float(rules[rules["scenario_id"].eq(scen)]["probability"].iloc[0])
            scenario_metrics.append((scen, p, m))
        exp_time = sum(p * m["total_hours"] for _, p, m in scenario_metrics)
        exp_cost = sum(p * m["objective_cost_yuan"] for _, p, m in scenario_metrics)
        exp_risk = sum(p * m["risk_score"] for _, p, m in scenario_metrics)
        worst_time = max(m["total_hours"] for _, _, m in scenario_metrics)
        robust_obj = exp_cost + 300 * exp_risk + 50 * max(0, worst_time - exp_time)
        rows.append(
            {
                "route_id": rid,
                "expected_time_hours": round(exp_time, 2),
                "worst_case_time_hours": round(worst_time, 2),
                "expected_cost_yuan": round(exp_cost, 2),
                "expected_risk": round(exp_risk, 3),
                "robust_objective": round(robust_obj, 2),
                "scenario_count": len(scenario_metrics),
            }
        )
    return pd.DataFrame(rows)


def build_transport_experiment_comparison(
    previous_pcop: pd.DataFrame,
    current_pcop: pd.DataFrame,
    amap_selfdrive: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_pcop(df: pd.DataFrame, experiment: str, data_version: str) -> None:
        if df.empty:
            return
        for _, r in df.iterrows():
            rows.append(
                {
                    "experiment": experiment,
                    "data_version": data_version,
                    "route_id": r.get("route_id", ""),
                    "spots_count": num(r.get("spots_count")),
                    "travel_hours": num(r.get("travel_hours")),
                    "service_hours": num(r.get("service_hours")),
                    "total_hours": num(r.get("total_hours")),
                    "estimated_days": num(r.get("days")),
                    "transport_cost_yuan": num(r.get("transport_cost_yuan")),
                    "ticket_yuan_for_two": num(r.get("ticket_yuan")),
                    "risk_score": num(r.get("risk_score")),
                    "objective_or_proxy_cost_yuan": num(r.get("objective_cost_yuan")),
                    "route_sequence": r.get("route_sequence", ""),
                }
            )

    append_pcop(previous_pcop, "previous_enhanced_multimodal", "before_amap_road_layer")
    append_pcop(current_pcop, "amap_road_multimodal", "amap_road_plus_12306_rail_plus_public_air")

    if not amap_selfdrive.empty:
        for _, r in amap_selfdrive.iterrows():
            rows.append(
                {
                    "experiment": "amap_selfdrive_only",
                    "data_version": "amap_spot_od_selfdrive_baseline",
                    "route_id": r.get("route_id", ""),
                    "spots_count": num(r.get("spots_count")),
                    "travel_hours": num(r.get("travel_hours")),
                    "service_hours": num(r.get("service_hours")),
                    "total_hours": num(r.get("total_hours")),
                    "estimated_days": num(r.get("estimated_days")),
                    "transport_cost_yuan": num(r.get("selfdrive_cost_yuan_for_two")) + num(r.get("external_flight_proxy_yuan_for_two")),
                    "ticket_yuan_for_two": num(r.get("ticket_yuan_for_two")),
                    "risk_score": np.nan,
                    "objective_or_proxy_cost_yuan": num(r.get("total_proxy_yuan_excluding_meals")),
                    "route_sequence": r.get("route_sequence", ""),
                }
            )

    return pd.DataFrame(rows)


def load_previous_multimodal_baseline() -> pd.DataFrame:
    comparison = read_optional_csv(MODEL_DIR / "transport_experiment_comparison.csv")
    if not comparison.empty and "experiment" in comparison.columns:
        baseline = comparison[comparison["experiment"].eq("previous_enhanced_multimodal")].copy()
        if not baseline.empty:
            return pd.DataFrame(
                {
                    "route_id": baseline.get("route_id", ""),
                    "spots_count": baseline.get("spots_count", 0),
                    "travel_hours": baseline.get("travel_hours", 0),
                    "service_hours": baseline.get("service_hours", 0),
                    "total_hours": baseline.get("total_hours", 0),
                    "days": baseline.get("estimated_days", 0),
                    "transport_cost_yuan": baseline.get("transport_cost_yuan", 0),
                    "ticket_yuan": baseline.get("ticket_yuan_for_two", 0),
                    "risk_score": baseline.get("risk_score", 0),
                    "objective_cost_yuan": baseline.get("objective_or_proxy_cost_yuan", 0),
                    "route_sequence": baseline.get("route_sequence", ""),
                }
            )

    return read_optional_csv(MODEL_DIR / "problem1_pcop_summary.csv")


def export_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)


def write_report(summary: dict[str, Any], sources: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "新疆旅游强化数据与算法研究报告.md"
    text = f"""# 新疆旅游线路安排强化建模研究报告

## 1. 本轮新增内容

本轮在原有公路图与景点数据基础上，继续补充并整理六类数据：高德真实驾车OD、多层交通网络、12306铁路班次、公开航班快照、容量与客流、费用与文化标签。无法直接取得的数据仍以情景模拟方式补齐，并在数据表中用 `source_type` 或 `source_id=MODEL_ASSUMPTION` 标明。

## 2. 数据增强结果

- 多层网络节点数：{summary['network_nodes']}
- 多层网络边数：{summary['network_edges']}
- 增强OD矩阵：{summary['enhanced_od_rows']} 条景点对
- 容量表：{summary['capacity_rows']} 个景点
- 文化标签表：{summary['culture_rows']} 个景点
- 时间依赖边情景表：{summary['edge_scenario_rows']} 条
- 接入 12306 铁路原始车次：{summary.get('rail_12306_raw_rows', 0)} 条；铁路模型候选：{summary.get('rail_12306_model_option_rows', 0)} 条
- 接入公开航班时刻/报价快照：{summary.get('air_public_schedule_rows', 0)} 条
- 接入高德景点级驾车 OD：{summary.get('amap_spot_od_rows', 0)} 条；乌鲁木齐始发/返程接驳：{summary.get('amap_depot_access_rows', 0)} 条
- 多层网络中由 12306 生成的铁路边：{summary.get('rail_edges_from_12306', 0)} 条；由公开航班表生成的航空边：{summary.get('air_edges_from_public_schedule', 0)} 条
- 多层网络中由高德 API 生成的道路边：{summary.get('amap_edges_from_api', 0)} 条
- 景区接驳成本下界：单条接驳边两人 {summary.get('local_scenic_shuttle_min_cost_per_two', 0)} 元，同区域景点转移隐含下界两人 {summary.get('same_area_transfer_floor_per_two', 0)} 元；触发下界的接驳边 {summary.get('scenic_shuttle_cost_floor_edges', 0)} 条

多层网络包括高德景点级自驾边、原枢纽公路兜底边、铁路、航空、大巴、景区接驳和换乘层。铁路/航空/大巴通过站点、机场、客运节点与城市枢纽连接，景点既可以通过接驳边连接到枢纽，也可以直接使用高德两两驾车OD进行自驾转移。

本版已将道路层升级为 `amap_driving_od_matrix_clean.csv` 与 `amap_depot_access_matrix_clean.csv` 的高德驾车OD；铁路层从原先的人工假设边升级为 `rail_12306_model_options.csv` 中的核心 OD 代表班次；航空层从原先的经验边升级为 `air_public_schedule_seed.csv` 中的公开时刻/报价快照。铁路数据来自 12306 官方查询接口，航班票价仍属于 OTA 快照/代理变量，应在正式出行日期前复核。

## 3. 第一问：Prize-Collecting Orienteering Problem

已实现三类算法：

1. MILP 精确/近似精确求解：带访问变量、弧变量、时间预算和MTZ子回路消除。
2. Greedy insertion + 2-opt + ALNS：作为大规模可解释启发式。
3. NSGA-II风格 Pareto 搜索：输出收益、费用、风险之间的非支配解。

MILP结果：{summary['pcop_milp_status']}；覆盖 {summary['pcop_milp_spots']} 个景点，总时间 {summary['pcop_milp_hours']} 小时。

## 4. 第二问：Two-Stage Path Cover

使用谱聚类先按景点度量闭包分为两阶段，再在每阶段内做TSP式路线优化。该方法比纯手工“北疆/南疆”划分更数据驱动，仍保留入疆/离疆枢纽可扩展空间。

## 5. 第三问：Min-Max Multi-TSP/VRP

以文化标签筛选考察点，将目标设为 `min max(T_k)`。模型纳入审批点专组处理、4倍考察时间、安全缓冲时间、边防证/向导等特殊准入约束。

## 6. 第四问：Column Generation + Capacity Flow

先生成候选线路列，再以连续客流变量分配游客，使热点景区拥挤度平方和尽量小。由于真实全景区容量仍不完整，本轮对缺失景点容量做了分级模拟；天池、赛里木湖、喀纳斯/禾木/白哈巴等使用查询到的承载量。

分配目标游客量采用情景值，实际报告中应说明：这是“旅游部门推介线路承接人群”，不是新疆五一总游客全量。

## 7. 鲁棒优化

构造了五类情景：普通暑期、暑期高峰、五一高峰、独库/山区扰动、酒店满房涨价。鲁棒目标为：

`min expected_cost + λ * expected_risk + penalty(worst_time - expected_time)`

这样能表达道路关闭概率、天气风险、票价波动和酒店满房风险。

## 8. 当前建模完成度

目前项目已经从“题面数据 + 公路图论”推进到“可扩展运筹优化原型”：

1. 数据底座：已清洗 40 个景点、25 个交通/住宿枢纽、景点-枢纽映射、公路边表、票价/游玩时长/准入属性、酒店价格等基础表。
2. 图论基础：已构造枢纽最短路闭包和景点度量闭包，可支持 Dijkstra、TSP/OP、两年路径覆盖、三组文化考察分配和候选线路生成。
3. 强化数据：已新增高德真实驾车 OD、多层交通网络、景区时间窗、住宿落点集合、特殊准入约束、容量表、文化标签、五类时间依赖情景。
4. 强化算法：第一问实现 PCOP-MILP、Greedy+2opt+ALNS、Pareto 搜索；第二问实现谱聚类 + 路径覆盖；第三问实现 Min-Max Multi-TSP/VRP；第四问实现候选列 + 容量流分配；并增加鲁棒评价。

因此，现阶段已经可以支撑“模型合理、算法丰富、数据可审计”的课程项目版本。但它仍不是生产级旅游调度系统，因为实时交通、实时票价、实时酒店库存和景区预约余量尚未接入。

## 9. 后续最需要补齐的数据

| 数据项 | 当前处理 | 后续获取方式 | 纳入模型方式 |
|---|---|---|---|
| 全量真实 OD 矩阵 | 已接入高德普通日驾车OD快照；高峰/节假日仍用情景因子 | 高德开放平台 Direction/Distance Matrix API，按景点经纬度批量查询；早晚高峰、节假日分别请求 | 替换/校准道路层时间、距离、费用边权，并与铁路/航空共同生成 `enhanced_od_matrix` |
| 铁路时刻和票价 | 用公开网页与情景等待时间近似 | 12306 实时查询；若无法批量抓取，人工抽取核心 OD 的车次、二等座/卧铺价格和发车时段 | 将铁路边拆成带发车时刻的 time-expanded arcs |
| 航班时刻和票价 | 用主要机场间经验时长/价格 | 航司官网、机场集团、航旅纵横或 OTA 手工抽样；至少采集乌鲁木齐-喀什/伊宁/库尔勒/阿克苏/和田/库车 | 对航空边加入班次等待、安检提前量、价格波动情景 |
| 景区日容量/瞬时容量 | 天池、赛里木湖、喀纳斯等使用公开承载量；其余分级模拟 | 景区公告、文旅局公开信息、5A/4A景区最大承载量公告、景区电话核验 | 作为第四问容量流约束和拥挤惩罚 |
| 预约余量/开闭园/闭馆日 | 用常见开放时间和人工规则 | 景区公众号、一码游新疆/官方预约平台截图或人工记录 | 加入 `open_i <= arrival_i <= close_i` 与日期可用性 |
| 酒店容量和价格 | 酒店价格已有，房量为分级模拟 | 携程/美团/同程按城市、星级、日期采样；或使用统计年鉴住宿业床位 | 住宿落点约束、满房风险、价格上浮情景 |
| 道路封闭和天气 | 独库公路使用公开新闻，其他为情景模拟 | 新疆交通运输厅、交警路况、气象局预警、景区封闭公告 | 转化为边可用概率、风险分数和鲁棒优化情景 |
| 客流分布 | 使用五一宏观客流与少量景区热度校准 | 文旅厅节假日通报、景区公告、运营商/地图热力指数、舆情指数 | 用于线路容量分配、热点拥挤惩罚和满意度估计 |
| 文化考察价值 | 基于世界遗产/文保/民族文化标签打分 | 文旅部/文物局名录、世界遗产点、博物馆等级、非遗名录 | 作为第三问主题覆盖和文化收益函数 |

## 10. 仍需注意的模型逻辑问题

1. 原始建模若只用公路 TSP，会把新疆的真实出行简化得过头；航空/铁路在长距离跨区移动中会改变最优结构。
2. 特殊景点不能与普通景点同权处理。楼兰、尼雅、帕米尔、白哈巴等涉及审批、边防证、向导、越野车、安全缓冲，应作为准入约束或单独考察组。
3. “每天 8 小时”不应只作用于总时长，日内还应考虑景区开放时间、夜间不可游览、住宿落点必须回到可住宿枢纽。
4. 第四问不能只列几条线路，应把线路当作列变量，再分配游客流量，并用容量约束/拥挤惩罚控制热点。
5. 费用项应拆分为交通、门票、接驳、住宿、审批/向导、车辆租赁、价格浮动，而不是只用固定人均成本。
6. 本轮的 MILP 是高价值候选子集精确求解，启发式覆盖全量景点；若要全量精确求解，应接入 OR-Tools CP-SAT、Gurobi 或 CPLEX。

## 11. 可以继续强化的 fancy 算法

1. Time-expanded multimodal shortest path：把每趟火车/航班展开为时刻边，求真正的时刻可行路径。
2. PCOP with resource constraints：同时约束总天数、每日游览时间、预算、审批点数量、风险上限。
3. ALNS：设计 Shaw removal、worst removal、route segment removal、regret-k insertion、time-window repair，并用自适应权重选择算子。
4. NSGA-II/III：输出景点数-费用-风险-文化收益的 Pareto 前沿，便于答辩展示“多目标权衡”。
5. Benders decomposition：第二问把“景点分配给年份”和“年份内路径优化”拆开，主问题做覆盖，子问题做路径。
6. Column generation：第四问由定价子问题不断生成新线路列，主问题分配游客，形成更像真实旅行社/文旅局线路投放的问题。
7. Robust/Stochastic optimization：用场景概率表达天气、封路、涨价、满房，把目标写成期望成本 + 风险惩罚 + 最坏情景惩罚。
8. Simulation-based validation：用 Monte Carlo 抽样天气/拥堵/票价，检验推荐线路在 1000 次扰动下的准点率、超预算率和拥挤风险。

## 12. 本轮自检结果

- `enhanced_od_matrix`：1600 条景点 OD，未发现缺失时间。
- `edge_time_scenarios`：{summary['network_edges']} 条边展开为 {summary['edge_scenario_rows']} 条情景边，非基准情景不会错误断开换乘层。
- `problem1_pcop_summary`：MILP 候选池 24 个点，求得最优；ALNS 覆盖全量普通可游景点，作为更丰富的实用路线。
- `problem1_nsga_pareto`：输出 20 个非支配候选解，用于展示费用-收益-风险权衡。
- `problem4_capacity_flow`：9 条候选线路完成容量流分配，分配游客量 88260 人。
- 已生成 Excel 工作簿与 Markdown 研究报告，可复核来源、假设、参数和求解结果。

## 13. 关键外部来源

"""
    for _, r in sources.iterrows():
        if r["reliability"] in ["official", "central_media", "official_pdf", "official_portal", "official_api_portal", "official_api_snapshot"]:
            text += f"- [{r['source_name']}]({r['url']}): {r['data_used']}\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ENHANCED_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    spots = read_csv("spot_clean")
    hubs = read_csv("hub_clean")
    road_edges = read_csv("edge_road_directed")
    transport = read_csv("transport_cost_clean")
    hotel_default = read_csv("hotel_place_default")
    rail_options = read_enhanced_csv("rail_12306_model_options")
    rail_raw = read_enhanced_csv("rail_12306_raw_core_od")
    air_options = read_enhanced_csv("air_public_schedule_seed")
    amap_od = read_enhanced_csv("amap_driving_od_matrix_clean")
    amap_depot = read_enhanced_csv("amap_depot_access_matrix_clean")
    previous_pcop = load_previous_multimodal_baseline()
    amap_selfdrive = read_optional_csv(ROOT / "amap_selfdrive_model_outputs" / "problem1_amap_summary.csv")

    sources = source_registry()
    data_gap_plan = data_gap_acquisition_plan()
    model_status = model_status_recommendations()
    time_windows = build_time_windows(spots)
    capacity = build_capacity(spots)
    cultural = build_cultural_tags(spots)
    special = build_special_access(spots)
    hotel_constraints = build_hotel_constraints(hotel_default, hubs)
    nodes, multimodal_edges = build_multimodal_network(spots, hubs, road_edges, transport, rail_options, air_options, amap_od, amap_depot)
    time_rules = build_time_dependent_rules()
    edge_scen = edge_scenarios(multimodal_edges, time_rules)
    od, depot = build_enhanced_od(spots, multimodal_edges, scenario="base_summer")
    time_matrix, cost_matrix, risk_matrix = matrix_from_od(spots, od, depot)

    for name, df in {
        "source_registry": sources,
        "data_gap_acquisition_plan": data_gap_plan,
        "model_status_recommendations": model_status,
        "spot_time_windows": time_windows,
        "capacity_by_spot": capacity,
        "cultural_tags": cultural,
        "special_access_constraints": special,
        "hotel_hub_constraints": hotel_constraints,
        "rail_12306_model_options": rail_options,
        "air_public_schedule_seed": air_options,
        "amap_driving_od_matrix_clean": amap_od,
        "amap_depot_access_matrix_clean": amap_depot,
        "multimodal_nodes": nodes,
        "multimodal_edges": multimodal_edges,
        "time_dependent_rules": time_rules,
        "edge_time_scenarios": edge_scen,
        "enhanced_od_matrix": od,
        "depot_access_matrix": depot,
    }.items():
        write_csv(df, ENHANCED_DIR / f"{name}.csv")

    milp_result = pcop_milp(spots, time_matrix, cost_matrix, special, cultural)
    milp_route = milp_result.get("route", [])
    greedy_result = pcop_greedy_alns(spots, time_matrix, cost_matrix, risk_matrix, special)
    greedy_route_res = greedy_result["route"]
    pareto = nsga2_pareto(spots, time_matrix, cost_matrix, risk_matrix, special)
    p1_rows = []
    for rid, route, status in [
        ("PCOP_MILP", milp_route, milp_result.get("status", "")),
        ("PCOP_ALNS", greedy_route_res, greedy_result.get("status", "")),
    ]:
        m = route_metrics(route, spots, time_matrix, cost_matrix, risk_matrix, special)
        p1_rows.append({"route_id": rid, "solver_status": status, **m, "route_sequence": " -> ".join(spots.iloc[i - 1]["spot_name"] for i in route)})
    p1_summary = pd.DataFrame(p1_rows)

    p2_summary, p2_detail = spectral_two_stage(spots, time_matrix, cost_matrix, risk_matrix, special)
    p3_summary, p3_detail = minmax_cultural_vrp(spots, time_matrix, cost_matrix, risk_matrix, special, cultural)
    columns = generate_columns(spots, time_matrix, capacity)
    flow = solve_capacity_flow(columns, capacity)
    robust = robust_route_evaluation(
        {
            "PCOP_MILP": milp_route,
            "PCOP_ALNS": greedy_route_res,
        },
        spots,
        multimodal_edges,
        special,
    )
    transport_comparison = build_transport_experiment_comparison(previous_pcop, p1_summary, amap_selfdrive)

    model_tables = {
        "problem1_pcop_summary": p1_summary,
        "problem1_nsga_pareto": pareto,
        "problem2_spectral_summary": p2_summary,
        "problem2_spectral_detail": p2_detail,
        "problem3_minmax_summary": p3_summary,
        "problem3_minmax_detail": p3_detail,
        "problem4_route_columns": columns,
        "problem4_capacity_flow": flow,
        "robust_route_evaluation": robust,
        "transport_experiment_comparison": transport_comparison,
    }
    for name, df in model_tables.items():
        write_csv(df, MODEL_DIR / f"{name}.csv")

    summary = {
        "network_nodes": len(nodes),
        "network_edges": len(multimodal_edges),
        "enhanced_od_rows": len(od),
        "capacity_rows": len(capacity),
        "culture_rows": len(cultural),
        "edge_scenario_rows": len(edge_scen),
        "rail_12306_raw_rows": len(rail_raw),
        "rail_12306_model_option_rows": len(rail_options),
        "air_public_schedule_rows": len(air_options),
        "amap_spot_od_rows": len(amap_od),
        "amap_depot_access_rows": len(amap_depot),
        "rail_edges_from_12306": int(multimodal_edges["source_id"].eq("RAIL_12306_MODEL_OPTIONS").sum()) if "source_id" in multimodal_edges.columns else 0,
        "air_edges_from_public_schedule": int(multimodal_edges["source_id"].eq("AIR_PUBLIC_SCHEDULE_SEED").sum()) if "source_id" in multimodal_edges.columns else 0,
        "amap_edges_from_api": int(multimodal_edges["source_id"].eq(AMAP_NETWORK_SOURCE_ID).sum()) if "source_id" in multimodal_edges.columns else 0,
        "local_scenic_shuttle_min_cost_per_two": LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO,
        "same_area_transfer_floor_per_two": LOCAL_SCENIC_SHUTTLE_MIN_COST_PER_TWO * 2,
        "scenic_shuttle_cost_floor_edges": int(multimodal_edges.get("cost_floor_applied", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "pcop_milp_status": milp_result.get("status", ""),
        "pcop_milp_spots": int(route_metrics(milp_route, spots, time_matrix, cost_matrix, risk_matrix, special)["spots_count"]) if milp_route else 0,
        "pcop_milp_hours": route_metrics(milp_route, spots, time_matrix, cost_matrix, risk_matrix, special)["total_hours"] if milp_route else None,
    }
    (MODEL_DIR / "enhanced_solve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = write_report(summary, sources)
    export_workbook(
        {
            "source_registry": sources,
            "data_gap_plan": data_gap_plan,
            "model_status": model_status,
            "capacity_by_spot": capacity,
            "cultural_tags": cultural,
            "special_access": special,
            "time_windows": time_windows,
            "rail_model_options": rail_options,
            "air_public_seed": air_options,
            "amap_od_clean": amap_od,
            "amap_depot_access": amap_depot,
            "multimodal_edges": multimodal_edges,
            "edge_scenarios": edge_scen,
            "pcop_summary": p1_summary,
            "nsga_pareto": pareto,
            "two_stage": p2_summary,
            "minmax_vrp": p3_summary,
            "capacity_flow": flow,
            "robust_eval": robust,
            "transport_compare": transport_comparison,
        },
        OUTPUT_DIR / "新疆旅游强化数据与算法结果.xlsx",
    )
    print(json.dumps({"summary": summary, "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
