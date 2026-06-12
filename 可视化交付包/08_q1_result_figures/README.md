# 第一问结果分析与可视化图组

本文件夹为 Q1-V3 鲁棒多目标多模式定向游模型的论文图组。图形采用白底、无网格线、低饱和蓝灰主色，并优先表达建模结论而非装饰性总览。

## 图表清单

| 图号 | 文件 | 论文作用 |
|---|---|---|
| fig_q1_01_screening_contraction | `png/fig_q1_01_screening_contraction.png` | 筛选收缩证据：用聚合图和代表方案表解释为何高覆盖路线不能直接作为推荐。 |
| fig_q1_02_route_tiers | `png/fig_q1_02_route_tiers.png` | 代表方案层级：拆分覆盖上界、覆盖候选、运营主推和舒适备选。 |
| fig_q1_03_main_route_map | `png/fig_q1_03_main_route_map.png` | 空间结构：展示24景点运营主推路线的跨区域覆盖。 |
| fig_q1_04_itinerary_pressure | `png/fig_q1_04_itinerary_pressure.png` | 小时级可执行性：展示每日交通、游览、午休与缓冲日。 |
| fig_q1_05_monte_carlo_distribution | `png/fig_q1_05_monte_carlo_distribution.png` | 鲁棒仿真：展示完成天数和红色压力日分布。 |
| fig_q1_06_exact_check | `png/fig_q1_06_exact_check.png` | 算法可信度：展示小规模Held-Karp精确校验结果。 |

## 口径说明

- `operational_success_probability` 表示运营完成成功率，不限制红色压力日。
- `strict_comfort_success_probability` 在运营成功基础上要求红色压力日不超过1天。
- 主推路线 `Q1V3_Q24_120` 是运营鲁棒主推，不是完全舒适主推；其严格舒适成功率仍为0。
- `fig_q1_04_itinerary_pressure` 的红/黄/绿判定复用 V3 小时级排程阈值：活动时长>10小时、交通>7.5小时或晚到为红日；活动>8.5小时、交通>5.5小时或高温避让为黄日。
