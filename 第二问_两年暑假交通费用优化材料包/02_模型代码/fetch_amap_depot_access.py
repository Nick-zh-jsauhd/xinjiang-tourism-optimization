from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from fetch_amap_od import KEY_NAMES, SECURITY_KEY_NAMES, amap_get, load_dotenv_value, signed_params  # noqa: E402,F401


ROOT = Path(".")
ENHANCED_DIR = ROOT / "enhanced_data"
DEPOT_OUT = ENHANCED_DIR / "amap_depot_access_matrix_clean.csv"


def geocode_depot(key: str, security_key: str | None, sleep_seconds: float) -> tuple[float, float, str]:
    data = amap_get(
        "https://restapi.amap.com/v3/geocode/geo",
        {"address": "新疆维吾尔自治区乌鲁木齐市人民政府", "city": "乌鲁木齐"},
        key,
        security_key,
        sleep_seconds,
    )
    geocodes = data.get("geocodes", [])
    if not geocodes:
        raise RuntimeError("Could not geocode Urumqi depot.")
    loc = geocodes[0]["location"]
    lng_s, lat_s = loc.split(",", 1)
    return float(lng_s), float(lat_s), geocodes[0].get("formatted_address", "乌鲁木齐市人民政府")


def main() -> None:
    key = load_dotenv_value(KEY_NAMES)
    security_key = load_dotenv_value(SECURITY_KEY_NAMES)
    if not key:
        raise SystemExit("Missing Amap API key. Set AMAP_API_KEY in the current shell or .env.local.")

    sleep_seconds = 1.2
    coords_path = ENHANCED_DIR / "spot_coordinates_amap.csv"
    if not coords_path.exists():
        raise SystemExit(f"Missing {coords_path}; run fetch_amap_od.py first.")
    spots = pd.read_csv(coords_path, encoding="utf-8-sig")
    depot_lng, depot_lat, depot_address = geocode_depot(key, security_key, sleep_seconds)
    depot_point = f"{depot_lng:.6f},{depot_lat:.6f}"

    rows = []
    origin_points = "|".join(f"{r.longitude:.6f},{r.latitude:.6f}" for r in spots.itertuples())

    for spot in spots.itertuples():
        dest = f"{spot.longitude:.6f},{spot.latitude:.6f}"
        out_data = amap_get(
            "https://restapi.amap.com/v3/distance",
            {"origins": depot_point, "destination": dest, "type": 1},
            key,
            security_key,
            sleep_seconds,
        )
        out_item = (out_data.get("results") or [{}])[0]
        rows.append(
            {
                "direction": "depot_to_spot",
                "depot_name": "乌鲁木齐市人民政府",
                "depot_address": depot_address,
                "depot_lng": depot_lng,
                "depot_lat": depot_lat,
                "spot_id": spot.spot_id,
                "spot_name": spot.spot_name,
                "spot_lng": spot.longitude,
                "spot_lat": spot.latitude,
                "driving_distance_km": round(float(out_item.get("distance", 0) or 0) / 1000, 3),
                "driving_duration_hours": round(float(out_item.get("duration", 0) or 0) / 3600, 3),
                "driving_duration_minutes": round(float(out_item.get("duration", 0) or 0) / 60, 1),
                "source_id": "AMAP_DISTANCE_API",
            }
        )

    back_data = amap_get(
        "https://restapi.amap.com/v3/distance",
        {"origins": origin_points, "destination": depot_point, "type": 1},
        key,
        security_key,
        sleep_seconds,
    )
    by_origin = {int(item.get("origin_id", 0)): item for item in back_data.get("results", [])}
    for idx, spot in enumerate(spots.itertuples(), start=1):
        item = by_origin.get(idx, {})
        rows.append(
            {
                "direction": "spot_to_depot",
                "depot_name": "乌鲁木齐市人民政府",
                "depot_address": depot_address,
                "depot_lng": depot_lng,
                "depot_lat": depot_lat,
                "spot_id": spot.spot_id,
                "spot_name": spot.spot_name,
                "spot_lng": spot.longitude,
                "spot_lat": spot.latitude,
                "driving_distance_km": round(float(item.get("distance", 0) or 0) / 1000, 3),
                "driving_duration_hours": round(float(item.get("duration", 0) or 0) / 3600, 3),
                "driving_duration_minutes": round(float(item.get("duration", 0) or 0) / 60, 1),
                "source_id": "AMAP_DISTANCE_API",
            }
        )

    out = pd.DataFrame(rows)
    out["selfdrive_cost_yuan_per_two"] = (out["driving_distance_km"] * 0.55).round(2)
    out.to_csv(DEPOT_OUT, index=False, encoding="utf-8-sig")
    print(f"Depot access ready: {len(out)} rows -> {DEPOT_OUT}")


if __name__ == "__main__":
    main()
