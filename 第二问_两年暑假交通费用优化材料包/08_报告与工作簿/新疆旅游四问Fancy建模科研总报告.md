# 新疆旅游四问 Fancy 建模科研总报告

## 1. 总体定位

本项目已经不只是求一条路线，而是形成了一个分层旅游决策系统：先用真实与校准数据构造加权图，再用运筹优化给出四问主结果，随后加入游客画像、疲劳、容量、价格、延期和需求扰动，使模型更贴近真实旅游场景。

核心表达可以概括为：

```text
真实数据底座 -> 确定性优化主解 -> 元启发式增强 -> 人性化评价 -> Monte Carlo仿真 -> 条件策略/应急方案
```

## 2. 模型分层

| layer        | content                                      | purpose                                        |
|:-------------|:---------------------------------------------|:-----------------------------------------------|
| 数据层       | 景点属性、高德OD、住宿锚点、容量与费用参数   | 真实交通OD + 题面/Excel清洗数据 + 场景校准参数 |
| 确定性优化层 | PCOP、两年路径覆盖、MinMax Multi-TSP、容量流 | 回答四问的主结果                               |
| 元启发式层   | ACO + ALNS + SA + 2-opt/relocate             | 在复杂约束下生成高质量路线                     |
| 人性化层     | 疲劳凸惩罚、游客画像、长途转场拆日           | 把人的体验转化为可计算约束                     |
| 仿真层       | Monte Carlo价格、延期、需求扰动              | 评估路线和政策在真实波动下是否稳健             |
| 策略层       | 多口岸阈值、缓冲天、预约上限、替代线路分流   | 形成可答辩、可运营的条件方案                   |

## 3. 四问主结论总览

| question   | model_layer         | model_name                  | primary_metric               | key_result                                            | evidence_table                                              |
|:-----------|:--------------------|:----------------------------|:-----------------------------|:------------------------------------------------------|:------------------------------------------------------------|
| Q1         | 确定性主模型        | HYBRID_30D_ACO_ALNS_SA      | 30天内高覆盖低成本路线       | 32景点，30天，总成本14548.53元                        | hybrid_30day_outputs/hard30_route_summary.csv               |
| Q1         | 人性化/策略层       | 普通体力型30天主线          | 画像适配路线                 | 32景点，28天，均值舒适度91.78，状态satisfied          | adaptive_strategy_outputs/q1_variant_summary.csv            |
| Q1         | 人性化/策略层       | 亲子舒适型30天改造          | 画像适配路线                 | 30景点，30天，均值舒适度89.19，状态satisfied          | adaptive_strategy_outputs/q1_variant_summary.csv            |
| Q1         | 人性化/策略层       | 长者慢游型30天改造          | 画像适配路线                 | 20景点，23天，均值舒适度85.51，状态tradeoff_remaining | adaptive_strategy_outputs/q1_variant_summary.csv            |
| Q2         | 路径覆盖主模型/下界 | P2_ROOTED_URUMQI_MINCOST    | 两年覆盖38个普通景点         | 32天，交通费4340.68元，最大单年19天                   | problem2_openpath_outputs/scenario_totals.csv               |
| Q2         | 路径覆盖主模型/下界 | P2_OPEN_GATEWAY_LOWER_BOUND | 两年覆盖38个普通景点         | 28天，交通费2992.69元，最大单年14天                   | problem2_openpath_outputs/scenario_totals.csv               |
| Q2         | 政策仿真层          | optimistic_multicity        | 开放式多口岸更便宜概率       | 胜率98.79%，期望节省701.96元                          | policy_simulation_outputs/q2_gateway_price_summary.csv      |
| Q2         | 政策仿真层          | balanced_multicity          | 开放式多口岸更便宜概率       | 胜率70.01%，期望节省243.33元                          | policy_simulation_outputs/q2_gateway_price_summary.csv      |
| Q2         | 政策仿真层          | peak_summer_multicity       | 开放式多口岸更便宜概率       | 胜率30.38%，期望节省-414.27元                         | policy_simulation_outputs/q2_gateway_price_summary.csv      |
| Q3         | 确定性主模型        | S3_MinMax_MultiTeam         | 三组文化考察最小最大完成时间 | 最大完成98.70小时，时间差12.16小时                    | enhanced_model_outputs/problem3_minmax_summary.csv          |
| Q3         | 风险仿真层          | Fieldwork_Buffer_MC         | 项目级90%可靠性缓冲          | 建议缓冲4天，完成概率91.84%                           | policy_simulation_outputs/q3_project_buffer_curve.csv       |
| Q4         | 容量分配主模型      | RouteColumn_CapacityFlow    | 五一12天接待路线组合         | 9条线路，分配106358人，总容量111956人                 | enhanced_model_outputs/problem4_capacity_flow.csv           |
| Q4         | 自适应分流层        | DemandShock_Reallocation    | 需求上浮10%的分流效果        | 分流新增接待4628人，仍溢出5038人                      | adaptive_strategy_outputs/q4_reallocation_summary.csv       |
| Q4         | 预约管理层          | SafetyCap95                 | 95%预约上限                  | 接待106358人，等待/拒绝10636人，利用率95.0%           | policy_simulation_outputs/q4_reservation_policy_summary.csv |

## 4. 第一问：30天路线

| question   | model_layer   | model_name             | primary_metric         | key_result                                            | evidence_table                                   |
|:-----------|:--------------|:-----------------------|:-----------------------|:------------------------------------------------------|:-------------------------------------------------|
| Q1         | 确定性主模型  | HYBRID_30D_ACO_ALNS_SA | 30天内高覆盖低成本路线 | 32景点，30天，总成本14548.53元                        | hybrid_30day_outputs/hard30_route_summary.csv    |
| Q1         | 人性化/策略层 | 普通体力型30天主线     | 画像适配路线           | 32景点，28天，均值舒适度91.78，状态satisfied          | adaptive_strategy_outputs/q1_variant_summary.csv |
| Q1         | 人性化/策略层 | 亲子舒适型30天改造     | 画像适配路线           | 30景点，30天，均值舒适度89.19，状态satisfied          | adaptive_strategy_outputs/q1_variant_summary.csv |
| Q1         | 人性化/策略层 | 长者慢游型30天改造     | 画像适配路线           | 20景点，23天，均值舒适度85.51，状态tradeoff_remaining | adaptive_strategy_outputs/q1_variant_summary.csv |

第一问正文建议采用 30 天硬约束混合元启发式作为主答案。强化口径是：普通体力型和探索型可以不删点完成；亲子舒适型建议删去少量低优先级非核心点；长者慢游型不建议硬塞30天，应作为慢游拆期方案。

## 5. 第二问：两年暑假路线

| question   | model_layer         | model_name                  | primary_metric         | key_result                          | evidence_table                                         |
|:-----------|:--------------------|:----------------------------|:-----------------------|:------------------------------------|:-------------------------------------------------------|
| Q2         | 路径覆盖主模型/下界 | P2_ROOTED_URUMQI_MINCOST    | 两年覆盖38个普通景点   | 32天，交通费4340.68元，最大单年19天 | problem2_openpath_outputs/scenario_totals.csv          |
| Q2         | 路径覆盖主模型/下界 | P2_OPEN_GATEWAY_LOWER_BOUND | 两年覆盖38个普通景点   | 28天，交通费2992.69元，最大单年14天 | problem2_openpath_outputs/scenario_totals.csv          |
| Q2         | 政策仿真层          | optimistic_multicity        | 开放式多口岸更便宜概率 | 胜率98.79%，期望节省701.96元        | policy_simulation_outputs/q2_gateway_price_summary.csv |
| Q2         | 政策仿真层          | balanced_multicity          | 开放式多口岸更便宜概率 | 胜率70.01%，期望节省243.33元        | policy_simulation_outputs/q2_gateway_price_summary.csv |
| Q2         | 政策仿真层          | peak_summer_multicity       | 开放式多口岸更便宜概率 | 胜率30.38%，期望节省-414.27元       | policy_simulation_outputs/q2_gateway_price_summary.csv |

第二问正文建议保留乌鲁木齐起讫模型作为保守主方案，同时把开放式多口岸作为条件方案。阈值逻辑是：若两人多口岸外部大交通额外差价低于 1347.99 元，则开放式方案在总费用上更优。第二问 DFJ 精确求证仍未生成最终 summary，不能替换当前主答案。

## 6. 第三问：文化考察

| question   | model_layer   | model_name          | primary_metric               | key_result                         | evidence_table                                        |
|:-----------|:--------------|:--------------------|:-----------------------------|:-----------------------------------|:------------------------------------------------------|
| Q3         | 确定性主模型  | S3_MinMax_MultiTeam | 三组文化考察最小最大完成时间 | 最大完成98.70小时，时间差12.16小时 | enhanced_model_outputs/problem3_minmax_summary.csv    |
| Q3         | 风险仿真层    | Fieldwork_Buffer_MC | 项目级90%可靠性缓冲          | 建议缓冲4天，完成概率91.84%        | policy_simulation_outputs/q3_project_buffer_curve.csv |

第三问不应按景点数量平均分配，而应按完成时间、远程交通、审批/风险和文化任务量均衡。仿真表明项目级90%可靠性需要约4天缓冲，因此论文中应把第三问称为文化专项调研调度，而不是普通旅游路线。

## 7. 第四问：五一接待

| question   | model_layer    | model_name               | primary_metric        | key_result                                  | evidence_table                                              |
|:-----------|:---------------|:-------------------------|:----------------------|:--------------------------------------------|:------------------------------------------------------------|
| Q4         | 容量分配主模型 | RouteColumn_CapacityFlow | 五一12天接待路线组合  | 9条线路，分配106358人，总容量111956人       | enhanced_model_outputs/problem4_capacity_flow.csv           |
| Q4         | 自适应分流层   | DemandShock_Reallocation | 需求上浮10%的分流效果 | 分流新增接待4628人，仍溢出5038人            | adaptive_strategy_outputs/q4_reallocation_summary.csv       |
| Q4         | 预约管理层     | SafetyCap95              | 95%预约上限           | 接待106358人，等待/拒绝10636人，利用率95.0% | policy_simulation_outputs/q4_reservation_policy_summary.csv |

第四问应强调路线产品组合和容量管理。基准需求已经接近满负荷；需求上浮10%时，单靠既有替代线路仍有溢出，因此需要预约上限、价格引导、新增运力或分流宣传。

## 8. 论文中建议突出的创新点

1. 统一加权图：把景点、交通、住宿、容量、风险放入同一数据底座。
2. 四问分层建模：PCOP、两年路径覆盖、MinMax Multi-TSP、容量流分别服务不同问题。
3. 混合元启发式：用 ACO 初始化、ALNS 破坏修复、SA 接受准则处理大规模约束。
4. 游客画像：把亲子、长者、探索型游客的疲劳阈值写入模型。
5. 运营策略：多口岸阈值、文化缓冲天、预约上限和需求分流让结果更贴近实际。

## 9. 交付文件索引

| artifact_name   | path                                                | exists   |   size_bytes | last_write_time     | role                               |
|:----------------|:----------------------------------------------------|:---------|-------------:|:--------------------|:-----------------------------------|
| 基础数据底座    | outputs/新疆旅游优化建模数据底座.xlsx               | True     |        35944 | 2026-06-09 22:12:32 | 清洗后的景点、交通、住宿、费用数据 |
| 第一问主结果    | outputs/新疆旅游30天硬约束混合元启发式求解结果.xlsx | True     |       235504 | 2026-06-10 20:26:57 | 30天硬约束混合元启发式路线         |
| 第二问重建结果  | outputs/新疆旅游第二问交通费用最小化重建结果.xlsx   | True     |        16442 | 2026-06-10 22:04:12 | 两年路径覆盖与开放式下界           |
| 四问收束方案    | outputs/新疆旅游线路安排四问最终收束方案.md         | True     |        22895 | 2026-06-10 22:05:04 | 四问正文主口径                     |
| 人性化评价      | outputs/新疆旅游人性化场景强化结果.xlsx             | True     |        23058 | 2026-06-11 00:47:13 | 疲劳、舒适度、画像适配             |
| 自适应策略      | outputs/新疆旅游自适应策略实验结果.xlsx             | True     |        18509 | 2026-06-11 00:53:17 | 画像改造路线与五一分流             |
| 政策仿真        | outputs/新疆旅游政策仿真实验结果.xlsx               | True     |       161110 | 2026-06-11 00:57:36 | 多口岸价格、延期、预约上限仿真     |
| 随机联合优化    | outputs/新疆旅游随机仿真与联合优化结果.xlsx         | True     |        69763 | 2026-06-10 18:12:12 | Monte Carlo与联合调度拓展          |

## 10. 后续还能继续强化的方向

1. 若安装 Gurobi/CPLEX，可把第二问 DFJ lazy constraints 改成真正 branch-and-cut，提升证明效率。
2. 补真实酒店房态、景区预约余量后，第四问可以升级为动态容量控制。
3. 补真实多机场机票/铁路票价后，第二问开放式口岸方案可从下界变成正式主模型候选。
4. 进一步把天气、道路封闭和客流热力接入为时变边权，形成在线重规划模型。