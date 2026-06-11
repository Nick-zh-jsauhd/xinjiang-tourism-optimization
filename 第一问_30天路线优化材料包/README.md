# 第一问 30天路线优化材料包

## 1. 材料包用途

本文件夹专门整理新疆旅游线路安排项目中“第一问：30天个人旅游路线优化”相关的材料、数据、代码和结果。

这不是重新定义模型的地方，而是把第一问从整个四问项目中独立拆出，便于：

- 写论文第一问建模章节；
- 检查第一问使用的数据来源；
- 复核 30 天硬约束结果；
- 追溯逐日排程、人群画像和鲁棒性仿真；
- 后续重构第一问多模式交通模型。

## 2. 目录结构

| 目录 | 内容 |
|---|---|
| `00_原始题目与总纲/` | 原题、原始 Excel、数学建模总纲、最终数据集说明 |
| `01_输入数据/model_data/` | 题面数据清洗后的基础表 |
| `01_输入数据/enhanced_data/` | 第一问实际依赖的增强数据，尤其是多模式 OD、时间窗、住宿、准入约束 |
| `02_模型代码/` | 第一问及其上游数据、排程、画像、仿真相关 Python 脚本 |
| `03_第一问主结果/` | 第一问 PCOP 基准、混合元启发式硬 30 天主结果 |
| `04_逐日排程与修复/` | 路线转逐日行程、时间窗、住宿落点、排程修复结果 |
| `05_画像与鲁棒性/` | 游客画像、自适应删点、数字孪生、风险策略相关数据 |
| `06_图表/` | 第一问相关可视化图表 |
| `07_报告与工作簿/` | 第一问及其相关强化实验报告和 Excel 工作簿 |
| `08_Q1_V2强化/` | 按 `q1_v2_plan.md` 新增的候选路线族、交通标签、稳健性审计和汇报成果 |

## 3. 第一问当前主模型口径

第一问应表述为：

> 在 30 天内，从普通游客可达景点中选择一条高价值、成本可控、时间窗可行、住宿可落地的旅游路线。

数学上更接近：

```text
Time-Windowed Multimodal Prize-Collecting Orienteering Problem
```

而不是简单最短路或 TSP。

当前主结果 `HYBRID_30D_ACO_ALNS_SA` 使用的是 enhanced 多模式 OD 闭包作为输入，但最终硬 30 天路线实际以道路接驳和景区接驳为主，未选中铁路/航班段。因此论文中应写成：

> 第一问读取 enhanced 多模式 OD 闭包；最终解在当前权重和约束下呈现为道路接驳主导路线。后续可通过长距离驾车疲劳惩罚、夜间铁路奖励和班次级公共交通约束继续强化。

## 4. 关键输入数据

| 文件 | 作用 |
|---|---|
| `01_输入数据/model_data/spot_clean.csv` | 40 个景点基础属性、门票、游览时长、普通游客可达性 |
| `01_输入数据/model_data/hub_clean.csv` | 枢纽节点 |
| `01_输入数据/enhanced_data/enhanced_od_matrix.csv` | 景点到景点多模式最短路闭包，是第一问核心 OD 输入 |
| `01_输入数据/enhanced_data/depot_access_matrix.csv` | 乌鲁木齐起终点接驳闭包 |
| `01_输入数据/enhanced_data/spot_time_windows.csv` | 景区开放时间窗 |
| `01_输入数据/enhanced_data/hotel_hub_constraints.csv` | 住宿枢纽和模拟房量 |
| `01_输入数据/enhanced_data/special_access_constraints.csv` | 审批、边防证、向导、普通游客可达性 |
| `01_输入数据/enhanced_data/time_dependent_rules.csv` | 情景扰动参数 |

## 5. 关键代码

| 文件 | 作用 |
|---|---|
| `02_模型代码/stochastic_integrated_joint_optimization.py` | 第一问联合评分、排程、仿真、MILP/ALNS 基础函数 |
| `02_模型代码/hybrid_metaheuristic_optimizer.py` | ACO + ALNS + SA 混合元启发式基础版本 |
| `02_模型代码/hybrid_30day_metaheuristic_optimizer.py` | 第一问硬 30 天约束主求解脚本 |
| `02_模型代码/build_daily_itinerary_layer.py` | 路线转逐日行程与时间窗检查 |
| `02_模型代码/repair_daily_itinerary_layer.py` | 逐日行程修复 |
| `02_模型代码/adaptive_strategy_lab.py` | 游客画像、自适应删点和策略实验 |
| `02_模型代码/digital_twin_robustness_lab.py` | 数字孪生鲁棒性仿真 |
| `02_模型代码/risk_aware_policy_engine.py` | CVaR 风险策略选择 |

注意：这些脚本是从原项目复制出的代码快照。若要直接重跑，建议在原项目根目录运行，因为脚本内部默认读取 `model_data/`、`enhanced_data/`、`hybrid_30day_outputs/` 等根目录相对路径。

## 6. 第一问主结果

核心结果在：

```text
03_第一问主结果/hybrid_30day_outputs/hard30_route_summary.csv
03_第一问主结果/hybrid_30day_outputs/hard30_route_segments.csv
03_第一问主结果/hybrid_30day_outputs/hard30_route_days.csv
```

当前主结果摘要：

- 路线：`HYBRID_30D_ACO_ALNS_SA`
- 覆盖景点：32 个
- 排程天数：30 天
- 时间窗违规：0
- 住宿受限夜：0
- 超强度活动日：2
- 交通费：5516.53 元/两人
- 门票：3402.00 元/两人
- 住宿：5630.00 元
- 总代理成本：14548.53 元

## 7. 鲁棒性与人性化结果

第一问不能只看确定性 30 天路线。真实出行中，天气、预约、酒店满房、道路延误会改变可行性。

相关材料：

```text
05_画像与鲁棒性/adaptive_strategy_outputs/q1_variant_summary.csv
05_画像与鲁棒性/digital_twin_outputs/q1_persona_robustness.csv
05_画像与鲁棒性/risk_policy_outputs/q1_policy_selection.csv
```

核心结论：

- 普通体力型路线可作为正文主线；
- 亲子舒适型在高峰、预约收紧和复合冲击下最脆弱；
- 长者慢游型应减少高海拔和长转场点；
- 第一问最终应区分“确定性主路线”和“风险场景下策略修正”。

## 8. 写论文时的正确顺序

第一问章节应按以下顺序写：

1. 问题分析：30天内高价值路线选择，不是 TSP；
2. 数据结构：景点集合、交通 OD、时间窗、住宿、准入约束；
3. 数学模型：变量、目标函数、约束；
4. 求解方法：PCOP 基准 + ACO/ALNS/SA 硬30天搜索；
5. 结果：32景点30天主路线；
6. 排程验证：逐日时间窗、住宿和强度；
7. 鲁棒性：数字孪生和游客画像策略；
8. 局限：最终路线仍道路接驳主导，公共交通班次级模型需进一步强化。

## 9. Q1-V2 强化版成果

按照 `00_原始题目与总纲/q1_v2_plan.md`，本材料包已经新增 Q1-V2：

```text
08_Q1_V2强化/scripts/q1_v2_build_all.py
```

它将第一问从“单条 32 景点硬 30 天路线”升级为“epsilon-coverage 候选路线族 + 后验稳健性评价”。当前 V2.1 是工程强化版，不应在论文中强称为严格 Pareto 前沿、完整多交通方式标签选择优化或 chance-constrained 主模型。

核心输出：

```text
08_Q1_V2强化/outputs/enhanced_od_labels_v2.csv
08_Q1_V2强化/outputs/q1_v2_route_family.csv
08_Q1_V2强化/outputs/q1_v2_daily_itinerary.csv
08_Q1_V2强化/outputs/q1_v2_robustness_summary.csv
08_Q1_V2强化/outputs/q1_v2_epsilon_grid.csv
08_Q1_V2强化/outputs/q1_v2_model_audit.csv
08_Q1_V2强化/outputs/q1_v2_loulan_substitution.csv
08_Q1_V2强化/reports/新疆旅游第一问Q1_V2强化建模与实验报告.md
08_Q1_V2强化/reports/楼兰古城不可普通访问与文化替代说明.md
08_Q1_V2强化/reports/新疆旅游第一问Q1_V2强化结果.xlsx
```

V2 当前结论：

- `极限覆盖版`：32 景点、30 天，无缓冲，保留为算法能力展示和对照基准；
- `均衡稳健版`：30 景点、28 天，2 天缓冲，作为第一问现实主推方案；
- `亲子舒适版`：30 景点、30 天，通过低日强度换取舒适性；
- `长者慢游版`：20 景点、30 天，保留 7 天缓冲，控制高海拔、长转场和连续疲劳。

论文中建议将第一版 32 景点路线写成“覆盖上界/极限方案”，将 V2 均衡稳健版写成“现实推荐方案”。同时需要说明：楼兰古城虽然是题面偏好点，但因审批和普通游客限制不进入基准自由行主线，模型通过交河故城、高昌故城、北庭故城遗址、克孜尔石窟、罗布人村寨等文化节点进行替代满足。

## 10. Q1-V3 鲁棒联合优化成果

按照 `q1_v3_robust_joint_optimization_plan.md`，本材料包已经新增 Q1-V3：

```text
09_Q1_V3_鲁棒联合优化/scripts/q1_v3_build_all.py
```

它将 V2.1 中仍属于后验解释的部分推进到求解流程内：从 `multimodal_edges.csv` 生成非支配交通标签，对每个覆盖下界重新搜索候选路线，构造小时级排程，并对每条候选路线执行 500 样本 route-specific Monte Carlo 仿真和鲁棒 Pareto 筛选。

核心输出：

```text
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_multimodal_labels.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_candidate_routes_enriched.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_hourly_itinerary.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_simulation_summary.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_robust_pareto_front.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_selected_routes.csv
09_Q1_V3_鲁棒联合优化/outputs/q1_v3_model_audit.csv
09_Q1_V3_鲁棒联合优化/reports/新疆旅游第一问Q1_V3鲁棒联合优化报告.md
09_Q1_V3_鲁棒联合优化/reports/新疆旅游第一问Q1_V3鲁棒联合优化结果.xlsx
```

V3 当前结论：

- `30景点均衡覆盖候选版`：30 景点、28 个活动/转场日、2 天缓冲，路线级成功率约 0.716，红色压力日 3 天；可作为覆盖型展示方案，但不应声称为严格稳健主推；
- `鲁棒稳健主推版`：24 景点、25 个活动/转场日、5 天缓冲，路线级成功率约 0.824，红色压力日 2 天；适合作为严格鲁棒口径下的现实主推；
- `极限覆盖版`：32 景点超过 30 天小时级可执行性边界，仅作为覆盖上界；
- `长者慢游版`：22 景点、7 天缓冲，成功率约 0.820，适合作为低强度扩展方案。

论文中建议将 V2 的 30 景点方案作为“日级稳健候选”，将 V3 的 30 景点方案写成“小时级覆盖候选”，并将 V3 的 24 景点方案作为“严格鲁棒主推”。这比强行把 30 景点路线包装成 80% 成功率、红日不超过 1 天的方案更符合仿真结果。

## 11. 文件索引

完整文件清单见：

```text
Q1_file_manifest.csv
```
