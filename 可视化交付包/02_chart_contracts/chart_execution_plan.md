# 图表执行计划

本计划修订自 `xinjiang_visualization_plan.md`。当前阶段不再追求“所有结果都可视化”，而是围绕论文主结论保留少数高信息密度图。旧版 P0 中的模型演化卡片、漏斗图、独立信息卡、大面积热力图已从 paper-ready 主输出撤下。

## P0：Nature 风格核心证据图

| 编号 | 图名 | 图型 | 数据表 | 关键口径 |
|---|---|---|---|---|
| Fig. 1 | Q1 覆盖、缓冲与两类成功率取舍 | paired dot plot | `q1_visual_route_tiers.csv` | 24景点是运营鲁棒主推，不等于严格舒适主推 |
| Fig. 2 | Q2 多口岸费用与差价阈值 | lollipop + threshold axis | `q2_visual_plan_compare.csv`, `q2_visual_gateway_threshold.csv` | 多口岸是否划算取决于外部大交通差价 |
| Fig. 3 | Q3 三组完成时间与最优性审计 | lollipop + audit table | `q3_visual_group_summary.csv`, `q3_visual_exact_check_card.csv` | exact gap=0 只在固定特殊准入政策空间下成立 |
| Fig. 4 | Q4 容量提升证据图 | lollipop | `q4_visual_capacity_compare.csv` | 18条12天产品把可投放容量提升至160190人 |
| Fig. 5 | Q4 线路质量审计点图 | ranked dot plot | `q4_visual_quality_audit.csv` | 8条可直接投放，10条需要微调或储备 |
| Fig. 6 | Q4 瓶颈影子价格排序 | ranked lollipop | `q4_visual_shadow_top10.csv` | 服务率高不代表分时与资源约束完全无压力 |

## 视觉纪律

1. 白底，无网格线。
2. 轴线使用细黑/深灰线，避免装饰性背景。
3. 单图原则上只使用一个主题色，风险或失败才使用橙/红。
4. 避免大面积柱状色块，优先使用点图、哑铃图、lollipop 和直接标注。
5. 图题必须是结论句或清晰问题句，不能只写“某某对比图”。
6. 密集中文标签用宋体常规，标题和轴标签用宋体加粗。

## P1：后续可补充但不抢主图的图

| 编号 | 图名 | 图型 | 数据表 | 备注 |
|---|---|---|---|---|
| Fig. 1-2 | Q1 主推路线地图 | 路线地图 | 待补路线节点坐标表 | 用于答辩，不作为论文核心证据图 |
| Fig. 1-3 | Q1 每日压力条带 | 条带图 | `q1_visual_daily_pressure.csv` | 若主文需要解释红色压力日，可补 |
| Fig. 2-2 | Q2 年度负担均衡图 | paired dot plot | `q2_visual_year_balance.csv` | 用于解释最低费用方案为何不是主推 |
| Fig. 3-2 | Q3 缓冲策略成功率 | discrete dot plot | `q3_visual_buffer_policy_long.csv` | 离散策略，不画连续趋势线 |
| Fig. 4-2 | Q4 需求冲击策略表现 | compact heatmap | `q4_visual_strategy_matrix.csv` | 需要严格控制色阶和标签密度 |

## 禁用或降级的旧图

- `fig_00_model_evolution`：信息密度不足，适合口头开场，不适合作为论文主图。
- `fig_q1_route_funnel`：漏斗会误导为单调筛选过程，已用取舍点图替代。
- `fig_q3_exact_gap_card`：独立卡片信息量低，已合并到 Q3 审计面板。
- `fig_q4_timeslot_heatmap`：大面积空白且主信息被稀释，已用瓶颈影子价格 Top 排序替代。

## 关键口径

1. Q1 主推方案称为“运营鲁棒主推”，严格舒适成功率另行展示。
2. Q2 所有费用图必须注明只统计新疆境内交通费用。
3. Q3 的 exact gap=0 不能脱离固定特殊准入政策空间表述。
4. Q4 的 `served_ratio=1` 不等于资源严格通过，必须保留分时和多资源瓶颈解释。
