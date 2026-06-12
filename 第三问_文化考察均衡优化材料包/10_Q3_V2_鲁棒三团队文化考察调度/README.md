# Q3-V2 鲁棒三团队文化考察调度

本目录是第三问的 V2 独立复现闭环，对应模型：

```text
Robust Multi-Team Cultural Fieldwork Scheduling
```

即“鲁棒三团队文化考察调度模型”。

## 模型口径

第三问不按普通旅游路线建模，也不按景点数量平均分配。Q3-V2 将其定义为：

```text
三组并行文化考察 MinMax 完成时间最小化问题
```

核心规则：

- 文化考察时间 = 普通观光时间的 4 倍；
- 交通时间沿用前两问数据口径，景点间优先使用高德驾车 OD，缺失时回退增强 OD；
- 楼兰古城、尼雅遗址固定为专项审批组，显式计入持证向导、越野车、最小团队规模和安全缓冲；
- 14 个文化点在固定特殊准入资源政策下做精确/半精确 MinMax 校验；
- 日级排程、资源日历和 route-specific Monte Carlo 仿真同步输出。

## 复现

在项目根目录运行：

```bash
python -X utf8 "第三问_文化考察均衡优化材料包/10_Q3_V2_鲁棒三团队文化考察调度/scripts/q3_v2_build_all.py"
```

## 核心结果

当前运行结果：

- 文化候选点：14 个；
- 枚举可行分配：474254 个；
- 固定特殊准入组政策下 exact gap：0；
- 最大完成时间：99.19 小时；
- 组间完成时间差：9.99 小时；
- 基础现场项目天数：13 天；
- 仿真样本：15000 条；
- 均衡稳健主推：5 天项目缓冲；
- 复合极端预案：延后/拆期 + 7 天缓冲。

## 核心输出

```text
outputs/q3_v2_cultural_candidate_set.csv
outputs/q3_v2_multimodal_labels.csv
outputs/q3_v2_assignment_routes.csv
outputs/q3_v2_group_route_segments.csv
outputs/q3_v2_daily_fieldwork_schedule.csv
outputs/q3_v2_resource_usage_calendar.csv
outputs/q3_v2_simulation_trials.csv
outputs/q3_v2_simulation_summary.csv
outputs/q3_v2_buffer_policy.csv
outputs/q3_v2_exact_check.csv
outputs/q3_v2_model_audit.csv
outputs/solve_summary.json
```

报告与工作簿：

```text
reports/新疆旅游第三问Q3_V2鲁棒三团队文化考察报告.md
reports/新疆旅游第三问Q3_V2鲁棒三团队文化考察结果.xlsx
```

## 解释边界

Q3-V2 的 `optimality_gap_under_constraints = 0` 指的是：

> 在“楼兰古城、尼雅遗址固定为同一专项审批组，三组均从乌鲁木齐起讫，组内路径用高德/增强 OD 时间”的政策口径下，枚举分配 + Held-Karp 得到精确最优。

它不代表所有可能资源政策下的全局最优。如果允许楼兰/尼雅拆给不同组并错日共享向导/越野车，需要建立新的资源日历优化模型重新求解。

