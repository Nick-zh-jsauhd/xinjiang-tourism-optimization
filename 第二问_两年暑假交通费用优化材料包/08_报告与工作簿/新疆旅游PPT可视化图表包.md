# 新疆旅游PPT可视化图表包

本包将前面模型和仿真实验转成可直接放进PPT/论文的PNG图。所有图由当前CSV结果自动生成，避免手工改数。

| figure_id   | title                | path                                                            | recommended_use       | exists   |   size_bytes |
|:------------|:---------------------|:----------------------------------------------------------------|:----------------------|:---------|-------------:|
| 图1         | 模型分层架构         | visual_assets_outputs/figures/fig01_model_stack.png             | PPT第3页/论文方法框架 | True     |        94707 |
| 图2         | 第一问游客画像舒适度 | visual_assets_outputs/figures/fig02_q1_persona_comfort.png      | 第一问强化结果        | True     |        69324 |
| 图3         | 第二问多口岸胜率     | visual_assets_outputs/figures/fig03_q2_gateway_probability.png  | 第二问政策仿真        | True     |        80402 |
| 图4         | 第三问缓冲可靠性     | visual_assets_outputs/figures/fig04_q3_buffer_curve.png         | 第三问风险仿真        | True     |        71607 |
| 图5         | 第四问需求冲击       | visual_assets_outputs/figures/fig05_q4_demand_reallocation.png  | 第四问容量策略        | True     |        45001 |
| 图6         | 第四问预约上限策略   | visual_assets_outputs/figures/fig06_q4_reservation_policy.png   | 第四问运营策略        | True     |        56837 |
| 图7         | 第二问费用天数对比   | visual_assets_outputs/figures/fig07_q2_cost_days_comparison.png | 第二问主结果对比      | True     |        73517 |

建议使用方式：

1. 答辩第3页放模型分层图。
2. 第一问页放游客画像舒适度图。
3. 第二问页放费用-天数对比图和多口岸胜率图。
4. 第三问页放缓冲可靠性曲线。
5. 第四问页放需求冲击图和预约上限图。