# Q1-V3 鲁棒联合优化

本目录实现第一问 V3 强化版：多模式交通标签、覆盖重搜索、小时级排程、路线级仿真、鲁棒 Pareto 筛选。

## 一键运行

```powershell
python .\scripts\q1_v3_build_all.py
```

## 关键输出

- `outputs/q1_v3_multimodal_labels.csv`
- `outputs/q1_v3_candidate_routes.csv`
- `outputs/q1_v3_hourly_itinerary.csv`
- `outputs/q1_v3_simulation_summary.csv`
- `outputs/q1_v3_robust_pareto_front.csv`
- `outputs/q1_v3_selected_routes.csv`
- `reports/新疆旅游第一问Q1_V3鲁棒联合优化报告.md`

## 建模边界

当前版本是 matheuristic 高质量可行解框架，不声称全局最优；机会约束通过路线级仿真筛选实现，容量与酒店房量中仍包含模拟补全字段。
