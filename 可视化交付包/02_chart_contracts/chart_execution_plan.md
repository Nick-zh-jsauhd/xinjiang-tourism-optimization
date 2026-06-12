# 图表执行计划

本计划修订自 `xinjiang_visualization_plan.md`，目标是先完成最能支撑论文和答辩主叙事的图，再补充地图、附录和审计图。

## P0：第一批必须完成

| 编号 | 图名 | 图型 | 数据表 | 关键口径 |
|---|---|---|---|---|
| Fig. 0 | 四问决策主体与模型演化总览 | 流程卡片 | `overview_model_evolution.csv` | 四问主模型已冻结 |
| Fig. 1-1 | Q1方案漏斗：覆盖上界到运营主推 | 阶梯/漏斗 | `q1_visual_route_tiers.csv` | 24景点是运营鲁棒，不是严格舒适 |
| Fig. 1-2 | Q1两类成功率对比 | 分组柱状 | `q1_visual_success_comparison.csv` | operational 与 strict comfort 必须并列 |
| Fig. 2-1 | Q2三方案费用与外部差价阈值 | 柱状+阈值标尺 | `q2_visual_plan_compare.csv`, `q2_visual_gateway_threshold.csv` | 多口岸是否划算取决于外部大交通差价 |
| Fig. 3-1 | Q3三组完成时间对比 | 横向条形 | `q3_visual_group_summary.csv` | MinMax按完成时间均衡，不按点数均分 |
| Fig. 3-2 | Q3固定政策空间 exact gap=0 | 信息卡 | `q3_visual_exact_check_card.csv` | 只在固定特殊准入政策空间成立 |
| Fig. 4-1 | Q4旧9线路与Q4-V2容量对比 | 双柱+需求线 | `q4_visual_capacity_compare.csv` | 18条12天产品提升投放容量 |
| Fig. 4-2 | Q4线路质量审计 | 横向条形 | `q4_visual_quality_audit.csv` | 8条可直接投放，10条需微调/储备 |
| Fig. 4-3 | Q4景区分时压力热力图 | 热力图 | `q4_visual_timeslot_pressure.csv` | 总量不超载不代表分时不过载 |

## P1：第二批增强图

| 编号 | 图名 | 图型 | 数据表 | 备注 |
|---|---|---|---|---|
| Fig. 1-3 | Q1主推路线地图 | 地图路线 | `q1_visual_route_map_nodes.csv` 待生成 | 需要坐标与路线节点拆分 |
| Fig. 1-4 | Q1每日压力条带图 | 条带图 | `q1_visual_daily_pressure.csv` | 从小时行程派生，不是原表字段 |
| Fig. 2-2 | Q2年度负担均衡图 | 分组柱状 | `q2_visual_year_balance.csv` | 说明最低费用方案负担不均衡 |
| Fig. 3-3 | Q3缓冲策略成功率点图 | 离散点图 | `q3_visual_buffer_policy_long.csv` | 不使用连续折线 |
| Fig. 4-4 | Q4需求冲击-策略表现热力图 | 热力图 | `q4_visual_strategy_matrix.csv` | 主色用 expected_loss，叠加 strict pass |

## P2：答辩加分图

| 编号 | 图名 | 图型 | 数据表 |
|---|---|---|---|
| Fig. 2-3 | Q2交通方式构成 | 堆叠条形 | `q2_visual_mode_mix.csv` |
| Fig. 3-4 | Q3资源占用日历 | 热力日历 | `q3_visual_resource_calendar.csv` |
| Fig. 4-5 | Q4瓶颈影子价格 Top 10 | 横向条形 | `q4_visual_shadow_top10.csv` |
| Fig. 4-6 | Q4三种分配口径对比 | 分组柱状/小多图 | `q4_visual_allocation_comparison_long.csv` |

## 关键修正

1. Q1 主推方案只能称为“运营鲁棒主推”，严格舒适成功率另行展示。
2. Q2 交通方式构成必须过滤到代表方案，不能混入候选池。
3. Q3 缓冲策略是离散政策，不画连续趋势线。
4. Q4 `served_ratio=1` 不等于资源严格通过，热力图必须保留 `strict_policy_pass`。
5. 18条线路矩阵在PPT中拆成“直接投放”和“储备微调”两块，完整表放附录。

