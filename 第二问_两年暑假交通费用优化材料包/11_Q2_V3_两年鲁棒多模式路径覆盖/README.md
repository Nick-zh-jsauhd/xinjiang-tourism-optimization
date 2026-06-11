# Q2-V3 两年鲁棒多模式路径覆盖

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
