from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(".")
DATA_DIR = ROOT / "model_data"
ENHANCED_DIR = ROOT / "enhanced_data"
GEOCODE_OUT = ENHANCED_DIR / "spot_coordinates_amap.csv"
OD_OUT = ENHANCED_DIR / "amap_driving_od_matrix.csv"

KEY_NAMES = ("AMAP_API_KEY", "GAODE_API_KEY", "AMAP_KEY", "GAODE_KEY")
SECURITY_KEY_NAMES = ("AMAP_SECURITY_KEY", "GAODE_SECURITY_KEY")


def load_dotenv_value(names: tuple[str, ...]) -> str | None:
    for env_name in names:
        if os.environ.get(env_name):
            return os.environ[env_name].strip()
    for env_file in (ROOT / ".env.local", ROOT / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in names:
                return v.strip().strip('"').strip("'")
    return None


def signed_params(params: dict[str, Any], security_key: str | None) -> dict[str, Any]:
    out = {k: v for k, v in params.items() if v is not None}
    if not security_key:
        return out
    raw = "&".join(f"{k}={out[k]}" for k in sorted(out)) + security_key
    out["sig"] = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return out


def redact_sensitive(text: str) -> str:
    text = str(text)
    for marker in ("key=", "sig="):
        while marker in text:
            start = text.find(marker) + len(marker)
            end_candidates = [idx for idx in (text.find("&", start), text.find(" ", start), text.find("'", start), text.find('"', start)) if idx != -1]
            end = min(end_candidates) if end_candidates else len(text)
            text = text[:start] + "REDACTED" + text[end:]
    return text


def amap_get(endpoint: str, params: dict[str, Any], key: str, security_key: str | None, sleep_seconds: float) -> dict[str, Any]:
    full_params = signed_params({**params, "key": key, "output": "json"}, security_key)
    session = requests.Session()
    # The Codex desktop environment can expose a local proxy that breaks some TLS
    # handshakes. Amap Web Service works with a direct HTTPS request.
    session.trust_env = False
    resp = session.get(endpoint, params=full_params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        info = data.get("info", "unknown")
        code = data.get("infocode", "unknown")
        raise RuntimeError(f"Amap API failed: info={info}, infocode={code}, endpoint={endpoint}")
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return data


def read_spots() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "spot_clean.csv", encoding="utf-8-sig")


def build_geocode_candidates(row: pd.Series) -> list[str]:
    spot = str(row["spot_name"])
    region = str(row.get("region_raw", "") or "")
    hub = str(row.get("hub_name", "") or "")
    candidates = [
        f"新疆维吾尔自治区{region}{spot}",
        f"新疆维吾尔自治区{hub}{spot}",
        f"新疆{spot}",
        spot,
    ]
    seen: set[str] = set()
    return [x for x in candidates if x and not (x in seen or seen.add(x))]


def build_poi_candidates(row: pd.Series) -> list[str]:
    spot = str(row["spot_name"])
    region = str(row.get("region_raw", "") or "")
    hub = str(row.get("hub_name", "") or "")
    candidates = [
        f"新疆 {spot}",
        f"{spot}景区",
        f"{spot}风景区",
        f"{region} {spot}",
        f"{hub} {spot}",
        spot,
    ]
    seen: set[str] = set()
    return [x for x in candidates if x and not (x in seen or seen.add(x))]


def chinese_name_score(spot_name: str, candidate: str) -> float:
    spot = "".join(ch for ch in spot_name if "\u4e00" <= ch <= "\u9fff")
    cand = "".join(ch for ch in candidate if "\u4e00" <= ch <= "\u9fff")
    if not spot or not cand:
        return 0.0
    if spot in cand:
        return 1.0
    trimmed = spot.replace("景区", "").replace("风景区", "").replace("古镇", "").replace("故城", "")
    if trimmed and trimmed in cand:
        return 0.9
    common = sum(1 for ch in set(spot) if ch in cand)
    return common / max(len(set(spot)), 1)


def search_poi(row: pd.Series, key: str, security_key: str | None, sleep_seconds: float) -> tuple[dict[str, Any] | None, str, str]:
    error = ""
    candidates: list[tuple[float, dict[str, Any], str]] = []
    for keyword in build_poi_candidates(row):
        try:
            data = amap_get(
                "https://restapi.amap.com/v3/place/text",
                {
                    "keywords": keyword,
                    "types": "风景名胜|科教文化服务|地名地址信息",
                    "citylimit": "false",
                    "offset": 10,
                    "page": 1,
                    "extensions": "base",
                },
                key,
                security_key,
                sleep_seconds,
            )
            pois = data.get("pois", [])
            for poi in pois:
                haystack = f"{poi.get('name', '')} {poi.get('address', '')}"
                score = chinese_name_score(str(row["spot_name"]), haystack)
                poi = dict(poi)
                poi["_match_score"] = score
                candidates.append((score, poi, keyword))
        except Exception as exc:
            error = redact_sensitive(str(exc))
    if candidates:
        score, poi, keyword = max(candidates, key=lambda x: x[0])
        if score >= 0.35:
            return poi, keyword, ""
        error = f"Low-confidence POI match; best={poi.get('name', '')}; score={score:.3f}"
    return None, "", error


def geocode_address(row: pd.Series, key: str, security_key: str | None, sleep_seconds: float) -> tuple[dict[str, Any] | None, str, str]:
    error = ""
    for address in build_geocode_candidates(row):
        try:
            data = amap_get(
                "https://restapi.amap.com/v3/geocode/geo",
                {"address": address, "city": "新疆"},
                key,
                security_key,
                sleep_seconds,
            )
            geocodes = data.get("geocodes", [])
            if geocodes:
                return geocodes[0], address, ""
        except Exception as exc:
            error = redact_sensitive(str(exc))
    return None, "", error


def geocode_spots(spots: pd.DataFrame, key: str, security_key: str | None, sleep_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in spots.iterrows():
        chosen, query_used, error = search_poi(row, key, security_key, sleep_seconds)
        method = "place_text"
        if chosen is None:
            chosen, query_used, error = geocode_address(row, key, security_key, sleep_seconds)
            method = "geocode_geo"
        location = (chosen or {}).get("location", "")
        lon = lat = None
        if "," in location:
            lon_s, lat_s = location.split(",", 1)
            lon, lat = float(lon_s), float(lat_s)
        match_score = float((chosen or {}).get("_match_score", 0.0) or 0.0)
        formatted = (chosen or {}).get("formatted_address", "") or (chosen or {}).get("address", "")
        if method == "place_text" and match_score >= 0.65:
            coordinate_quality = "high"
        elif method == "place_text" and match_score >= 0.35:
            coordinate_quality = "medium_review"
        elif chosen and chinese_name_score(str(row["spot_name"]), str(formatted)) >= 0.35:
            coordinate_quality = "medium_review"
        elif chosen:
            coordinate_quality = "low_review"
        else:
            coordinate_quality = "failed"
        rows.append(
            {
                "spot_id": row["spot_id"],
                "spot_name": row["spot_name"],
                "region_raw": row.get("region_raw", ""),
                "hub_name": row.get("hub_name", ""),
                "amap_method": method if chosen else "",
                "amap_query": query_used,
                "longitude": lon,
                "latitude": lat,
                "formatted_address": formatted,
                "poi_name": (chosen or {}).get("name", ""),
                "poi_type": (chosen or {}).get("type", ""),
                "match_score": match_score,
                "coordinate_quality": coordinate_quality,
                "province": (chosen or {}).get("province", ""),
                "city": (chosen or {}).get("city", ""),
                "district": (chosen or {}).get("district", ""),
                "adcode": (chosen or {}).get("adcode", ""),
                "level": (chosen or {}).get("level", ""),
                "geocode_status": "ok" if chosen else "failed",
                "error": error,
            }
        )
    out = pd.DataFrame(rows)
    ENHANCED_DIR.mkdir(exist_ok=True)
    out.to_csv(GEOCODE_OUT, index=False, encoding="utf-8-sig")
    return out


def load_or_create_geocodes(spots: pd.DataFrame, key: str, security_key: str | None, sleep_seconds: float, force: bool) -> pd.DataFrame:
    if GEOCODE_OUT.exists() and not force:
        geo = pd.read_csv(GEOCODE_OUT, encoding="utf-8-sig")
        if {"spot_id", "longitude", "latitude"}.issubset(geo.columns) and geo["longitude"].notna().all() and geo["latitude"].notna().all():
            return geo
    return geocode_spots(spots, key, security_key, sleep_seconds)


def query_driving_od(geo: pd.DataFrame, key: str, security_key: str | None, sleep_seconds: float, limit: int | None = None) -> pd.DataFrame:
    geo = geo.copy()
    if limit:
        geo = geo.head(limit)
    if geo[["longitude", "latitude"]].isna().any().any():
        missing = geo[geo[["longitude", "latitude"]].isna().any(axis=1)]["spot_name"].tolist()
        raise RuntimeError(f"Missing coordinates for: {missing}")

    origin_points = [f"{r.longitude:.6f},{r.latitude:.6f}" for r in geo.itertuples()]
    origins_param = "|".join(origin_points)
    rows: list[dict[str, Any]] = []

    for dest_idx, dest in enumerate(geo.itertuples(), start=1):
        destination = f"{dest.longitude:.6f},{dest.latitude:.6f}"
        data = amap_get(
            "https://restapi.amap.com/v3/distance",
            {"origins": origins_param, "destination": destination, "type": 1},
            key,
            security_key,
            sleep_seconds,
        )
        result_by_origin = {}
        for item in data.get("results", []):
            origin_id = int(item.get("origin_id", 0))
            result_by_origin[origin_id] = item
        for origin_idx, origin in enumerate(geo.itertuples(), start=1):
            item = result_by_origin.get(origin_idx, {})
            distance_m = float(item.get("distance", 0) or 0)
            duration_s = float(item.get("duration", 0) or 0)
            rows.append(
                {
                    "from_spot_id": origin.spot_id,
                    "from_spot_name": origin.spot_name,
                    "from_lng": origin.longitude,
                    "from_lat": origin.latitude,
                    "to_spot_id": dest.spot_id,
                    "to_spot_name": dest.spot_name,
                    "to_lng": dest.longitude,
                    "to_lat": dest.latitude,
                    "amap_origin_id": origin_idx,
                    "amap_dest_batch_index": dest_idx,
                    "driving_distance_km": round(distance_m / 1000, 3),
                    "driving_duration_hours": round(duration_s / 3600, 3),
                    "driving_duration_minutes": round(duration_s / 60, 1),
                    "source_id": "AMAP_DISTANCE_API",
                }
            )
    out = pd.DataFrame(rows)
    ENHANCED_DIR.mkdir(exist_ok=True)
    out.to_csv(OD_OUT, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Xinjiang tourism spot coordinates and driving OD matrix from Amap Web Service API.")
    parser.add_argument("--geocode-only", action="store_true", help="Only fetch/cache spot coordinates.")
    parser.add_argument("--od-only", action="store_true", help="Only fetch OD using cached coordinates.")
    parser.add_argument("--force-geocode", action="store_true", help="Refresh geocodes even if cache exists.")
    parser.add_argument("--limit", type=int, default=None, help="Limit spots for API smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API requests.")
    args = parser.parse_args()

    key = load_dotenv_value(KEY_NAMES)
    security_key = load_dotenv_value(SECURITY_KEY_NAMES)
    if not key:
        raise SystemExit(
            "Missing Amap API key. Set AMAP_API_KEY in the current shell or in .env.local. "
            "Do not commit or share the key publicly."
        )

    spots = read_spots()
    if args.limit:
        spots = spots.head(args.limit)

    if args.od_only:
        if not GEOCODE_OUT.exists():
            raise SystemExit(f"Missing {GEOCODE_OUT}; run geocoding first.")
        geo = pd.read_csv(GEOCODE_OUT, encoding="utf-8-sig")
        if args.limit:
            geo = geo.head(args.limit)
    else:
        geo = load_or_create_geocodes(spots, key, security_key, args.sleep, args.force_geocode)

    failed = geo[geo["geocode_status"].ne("ok")] if "geocode_status" in geo.columns else pd.DataFrame()
    if not failed.empty:
        print(f"Geocode completed with {len(failed)} failures. Review {GEOCODE_OUT}.")
    else:
        print(f"Geocode ready: {len(geo)} spots -> {GEOCODE_OUT}")

    if args.geocode_only:
        return

    od = query_driving_od(geo, key, security_key, args.sleep, args.limit)
    missing = int((od["driving_duration_hours"] <= 0).sum())
    print(f"Driving OD ready: {len(od)} rows -> {OD_OUT}; zero-duration rows={missing}")


if __name__ == "__main__":
    main()
