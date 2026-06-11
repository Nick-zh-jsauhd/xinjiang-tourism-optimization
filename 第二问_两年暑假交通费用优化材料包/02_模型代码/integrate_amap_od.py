from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(".")
ENHANCED_DIR = ROOT / "enhanced_data"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    raw_path = ENHANCED_DIR / "amap_driving_od_matrix.csv"
    base_path = ENHANCED_DIR / "enhanced_od_matrix.csv"
    geo_path = ENHANCED_DIR / "spot_coordinates_amap.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}; run fetch_amap_od.py first.")
    if not base_path.exists():
        raise SystemExit(f"Missing {base_path}; run enhance_data_and_models.py first.")

    od = pd.read_csv(raw_path, encoding="utf-8-sig")
    geo = pd.read_csv(geo_path, encoding="utf-8-sig") if geo_path.exists() else pd.DataFrame()
    base = pd.read_csv(base_path, encoding="utf-8-sig")

    self_mask = od["from_spot_id"].eq(od["to_spot_id"])
    od.loc[self_mask, ["driving_distance_km", "driving_duration_hours", "driving_duration_minutes"]] = 0.0
    od["amap_selfdrive_cost_yuan_per_two"] = (od["driving_distance_km"] * 0.55).round(2)
    od["od_quality_note"] = "self_loop_forced_zero"
    od.loc[~self_mask, "od_quality_note"] = "amap_distance_api"

    clean_path = ENHANCED_DIR / "amap_driving_od_matrix_clean.csv"
    od.to_csv(clean_path, index=False, encoding="utf-8-sig")

    merged = base.merge(
        od[
            [
                "from_spot_id",
                "to_spot_id",
                "driving_distance_km",
                "driving_duration_hours",
                "driving_duration_minutes",
                "amap_selfdrive_cost_yuan_per_two",
                "od_quality_note",
            ]
        ],
        on=["from_spot_id", "to_spot_id"],
        how="left",
    )
    merged["amap_duration_minus_multimodal_hours"] = (
        merged["driving_duration_hours"] - merged["shortest_time_hours"]
    ).round(3)
    merged["amap_cost_minus_multimodal_yuan"] = (
        merged["amap_selfdrive_cost_yuan_per_two"] - merged["shortest_cost_yuan_per_two"]
    ).round(2)
    merged_path = ENHANCED_DIR / "enhanced_od_matrix_with_amap.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    non_self = od[~self_mask]
    comparison = merged[~merged["from_spot_id"].eq(merged["to_spot_id"])].copy()
    summary_rows = [
        {"metric": "amap_od_rows", "value": len(od), "note": "40x40景点驾车OD"},
        {"metric": "amap_non_self_rows", "value": len(non_self), "note": "剔除自环后的有效景点对"},
        {"metric": "coordinate_high_quality_rows", "value": int((geo.get("coordinate_quality", pd.Series(dtype=str)) == "high").sum()) if not geo.empty else None, "note": "高德POI坐标高置信匹配数"},
        {"metric": "amap_avg_duration_hours_non_self", "value": round(float(non_self["driving_duration_hours"].mean()), 3), "note": "高德驾车平均时长"},
        {"metric": "amap_median_duration_hours_non_self", "value": round(float(non_self["driving_duration_hours"].median()), 3), "note": "高德驾车中位时长"},
        {"metric": "amap_avg_distance_km_non_self", "value": round(float(non_self["driving_distance_km"].mean()), 3), "note": "高德驾车平均距离"},
        {"metric": "avg_amap_minus_multimodal_hours", "value": round(float(comparison["amap_duration_minus_multimodal_hours"].mean()), 3), "note": "高德驾车时长-多层交通最短时长"},
        {"metric": "median_amap_minus_multimodal_hours", "value": round(float(comparison["amap_duration_minus_multimodal_hours"].median()), 3), "note": "高德驾车时长差中位数"},
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = ENHANCED_DIR / "amap_od_integration_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    out_json = {
        "raw_od": str(raw_path),
        "clean_od": str(clean_path),
        "merged_od": str(merged_path),
        "summary": str(summary_path),
        "rows": {
            "raw_od": int(len(od)),
            "merged_od": int(len(merged)),
            "coordinate_rows": int(len(geo)) if not geo.empty else 0,
        },
    }
    (ENHANCED_DIR / "amap_od_integration_summary.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    workbook = next(iter(sorted(OUTPUT_DIR.glob("*强化数据与算法结果.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)), None)
    if workbook:
        with pd.ExcelWriter(workbook, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            if not geo.empty:
                geo.to_excel(writer, index=False, sheet_name="amap_geocode")
            od.to_excel(writer, index=False, sheet_name="amap_od_clean")
            summary.to_excel(writer, index=False, sheet_name="amap_od_summary")
    print(json.dumps(out_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
