# Q4-V2 12天线路产品组合与分时容量优化

本目录是第四问的新一轮强化闭环。它把旧版“9条线路容量流”升级为：

1. 生成 `63` 条完整 12 天线路产品候选列；
2. 选择 `18` 条线路构成五一投放组合；
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
python -X utf8 .\第四问_五一容量接待优化材料包\09_Q4_V2_12天线路产品组合与分时容量优化\scripts\q4_v2_build_all.py
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

- 入选线路产品总容量：`160190` 人。
- 旧 9 线路基线总容量：`111956` 人，基准接待：`106358` 人。
- Q4-V2 不是个人路径规划，而是线路产品投放、预约名额、多资源容量和动态分流的运营模型。
- `strict_capacity_ratio_visitors` 是题面比例假设基准；`preference_elastic_visitors` 是现实偏好扩展；`balanced_optimized_visitors` 用于运营仿真。
- `policy_pass/soft_policy_pass` 表示软可行；`strict_policy_pass` 才表示所有分时段与多资源利用率均不超过 100%。
- 多资源容量中的酒店房量、车辆、导游、摆渡/停车仍为校准模拟参数，需在论文和答辩中说明。

## 模型审计

| audit_id   | module                         | status      | evidence                                               | metric                                                                                     | limitation                                                                                                       |
|:-----------|:-------------------------------|:------------|:-------------------------------------------------------|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| Q4V2-A1    | route_product_generation       | implemented | q4_v2_candidate_route_products.csv                     | 63 candidates / 18 selected                                                                | 候选线路由规则化列生成产生，不是所有可行线路全集。                                                               |
| Q4V2-A2    | route_product_quality          | implemented | q4_v2_route_quality_audit.csv                          | quality_pass=8/18, max_day=9.24h                                                           | 质量审计基于日级强度代理，不是逐小时导游排班表；高强度路线需人工微调。                                           |
| Q4V2-A3    | capacity_ratio_allocation      | implemented | q4_v2_capacity_ratio_allocation.csv                    | baseline demand=106358, balanced overflow=0                                                | 游客偏好参数为模型校准，未接入真实订单点击/转化数据。                                                            |
| Q4V2-A4    | timeslot_capacity              | implemented | q4_v2_spot_timeslot_load.csv                           | max utilization=1.058, red slots=1                                                         | 时段容量由日容量按早/午/晚比例拆分，仍需景区闸机小时级数据校准。                                                 |
| Q4V2-A5    | hotel_resource_capacity        | implemented | q4_v2_hotel_resource_load.csv                          | max utilization=0.553                                                                      | 酒店房量为枢纽可协调房源池的校准容量，不是携程/美团实时库存。                                                    |
| Q4V2-A6    | vehicle_guide_shuttle_capacity | implemented | q4_v2_vehicle_guide_resource_load.csv                  | max utilization=0.376                                                                      | 车辆、导游、摆渡车库存由枢纽可协调房源池和景区规模推导，需旅行社/景区调度台账核验。                              |
| Q4V2-A7    | scenario_policy_simulation     | implemented | q4_v2_scenario_simulation_summary.csv                  | best row=q4v2_add_capacity_plus_stagger/base_1_00, loss=298.44, soft_pass=4, strict_pass=2 | 情景概率和冲击分布为专家设定；后续应接入历年五一客流/天气/道路事件。                                             |
| Q4V2-A8    | mathematical_optimality        | partial     | reports/新疆旅游第四问Q4_V2五一路线产品组合优化报告.md | matheuristic closed loop                                                                   | 当前为生成式候选列+启发式组合选择+仿真评估；若需严格全局最优，应转为Gurobi/Pyomo混合整数二次规划或列生成主问题。 |
