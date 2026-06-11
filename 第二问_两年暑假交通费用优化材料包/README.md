# 第二问 两年暑假交通费用优化材料包

## 1. 材料包用途

本文件夹专门整理新疆旅游线路安排项目中“第二问：今、明两年暑假完成新疆旅游，并尽量节省新疆境内交通费用”相关的已有材料、数据、代码和实验结果。

第二问的核心口径应是：

> 将普通游客可达景点分配到两个暑假，每个暑假形成一条新疆境内旅游路径，在覆盖要求满足的前提下最小化两人新疆境内交通费用。

它不是简单平均分景点，也不是以景点价值最大化为主目标。谱聚类、区域分组和最近邻路线只能作为启发式初始解，最终模型必须把“新疆境内交通费用”放进目标函数。

## 2. 目录结构

| 目录 | 内容 |
|---|---|
| `00_原始题目与建模口径/` | 原题、原始 Excel、四问总纲、最终收束方案、论文草案中第二问相关文字 |
| `01_输入数据/model_data/` | 清洗后的景点、枢纽、道路边、场景参数和基础交通费用表 |
| `01_输入数据/enhanced_data/` | 高德 OD、增强 OD、多模式边、铁路种子、航班种子、口岸接驳等增强交通数据 |
| `02_模型代码/` | 第二问及其上游数据采集、图论基线、高德重实验、多口岸仿真、效率研究相关脚本快照 |
| `03_历史图论基线/` | 早期图论最短路闭包与两年路径覆盖基线 |
| `04_高德OD重实验/` | 使用高德驾车 OD 后的第二问道路交通费用基线 |
| `05_多模式与谱聚类结果/` | 增强模型中的第二问谱聚类/路径覆盖参考结果 |
| `06_多口岸阈值与风险仿真/` | 乌鲁木齐起讫与开放式多口岸的阈值、价格不确定性和风险策略结果 |
| `07_求解效率与算法升级/` | HiGHS/DFJ 求解状态、求解器可用性和 OR-Tools/Gurobi 升级建议 |
| `08_报告与工作簿/` | 已有报告、Excel 工作簿和答辩材料中与第二问有关的成果 |
| `09_图表/` | 第二问多口岸概率、费用天数对比、风险前沿等图表 |
| `10_待补与缺口审计/` | 当前缺失或需重跑补齐的第二问主模型成果清单 |

## 3. 当前已有结果的层级

### 3.1 早期图论基线

位置：

```text
03_历史图论基线/graph_model_outputs/problem2_summary.csv
03_历史图论基线/graph_model_outputs/problem2_route_detail.csv
```

该版本基于早期图论闭包和代理费用，得到北疆-伊犁与东疆-南疆两条路径，并附带审批扩展路线。它适合作为历史基线，不应作为最终第二问主答案。

### 3.2 高德道路 OD 基线

位置：

```text
04_高德OD重实验/amap_selfdrive_model_outputs/problem2_amap_summary.csv
04_高德OD重实验/amap_selfdrive_model_outputs/problem2_amap_detail.csv
```

已有结果显示：

| 路线 | 景点数 | 估算天数 | 高德道路费用，两人 | 定位 |
|---|---:|---:|---:|---|
| `P2_AMAP_Y1_North_Ili_loop` | 17 | 17 | 2131.30 | 北疆-伊犁闭环 |
| `P2_AMAP_Y2_East_South_open` | 21 | 14 | 1893.30 | 东疆-南疆开放路径 |
| `P2_AMAP_Y2_Approval_Extension_open` | 23 | 16 | 2101.60 | 加入楼兰、尼雅的审批扩展 |

这一层使用真实高德驾车 OD，适合作为“道路自驾/包车成本基线”，但第二问后续主模型应进一步转向多模式交通费用，而不是只按驾车费用定稿。

### 3.3 报告中的两年路径覆盖主口径

现有报告反复引用两个方案：

| 方案 | 覆盖景点 | 合计天数 | 最大单年天数 | 新疆境内交通费，两人 | 状态 |
|---|---:|---:|---:|---:|---|
| `P2_ROOTED_URUMQI_MINCOST` | 38 | 32 | 19 | 4340.68 | 乌鲁木齐起讫保守主方案 |
| `P2_OPEN_GATEWAY_LOWER_BOUND` | 38 | 28 | 14 | 2992.69 | 开放式多口岸境内费用下界 |

当前工作区有这些结果的状态摘要：

```text
07_求解效率与算法升级/solver_efficiency_outputs/current_problem2_status.csv
```

但报告引用的原始主结果目录 `problem2_openpath_outputs/` 在当前工作区缺失。因此这两个数值现在只能作为“已有报告结论和状态摘要”，不能直接视为可完整复现的最终输出。下一轮第二问重建必须补齐路线明细、年度分配、边级交通标签和求解日志。

### 3.4 多口岸阈值与不确定性

位置：

```text
06_多口岸阈值与风险仿真/humanized_scenario_outputs/q2_gateway_threshold.csv
06_多口岸阈值与风险仿真/policy_simulation_outputs/q2_gateway_price_summary.csv
06_多口岸阈值与风险仿真/digital_twin_outputs/q2_gateway_robustness.csv
```

核心结论：

```text
阈值 = 4340.68 - 2992.69 = 1347.99 元/两人
```

如果两人采用多口岸进出疆带来的外部大交通额外差价低于 1347.99 元，则开放式多口岸方案在总费用上更优。否则，乌鲁木齐起讫方案更稳妥。

## 4. 第二问下一步建模建议

第二问后续应新建一个真正的 Q2 主求解器，而不是继续沿用第一问或旧谱聚类结果。建议模型为：

```text
Two-Summer Multimodal Path Cover / Two-Route VRP
```

关键决策变量：

- `x_ik`：景点 `i` 是否分配给第 `k` 年暑假；
- `y_ijk`：第 `k` 年是否从景点 `i` 转移到景点 `j`；
- `z_ijlk`：第 `k` 年从 `i` 到 `j` 是否选择第 `l` 个交通标签；
- `s_ak, t_ak`：第 `k` 年入疆、离疆口岸选择。

主目标：

```text
min sum_k sum_i sum_j sum_l c_ijl z_ijlk
```

其中 `c_ijl` 只计新疆境内交通费用。外部大交通不放入主目标，但可以进入多口岸阈值策略层。

建议分两版推进：

1. 基准版：固定乌鲁木齐起讫，两年覆盖普通游客可达景点，使用 `enhanced_od_matrix.csv` 或由 `multimodal_edges.csv` 重建的非支配交通标签。
2. 强化版：开放入疆/离疆口岸，加入铁路/航班/道路多模式选择，并输出“境内费用下界 + 外部票价阈值”。

## 5. 复现说明

这些脚本是从原项目根目录复制的快照，内部多数脚本仍默认读取根目录相对路径，例如 `model_data/`、`enhanced_data/`、`problem2_openpath_outputs/`。如需直接重跑，建议回到项目根目录运行，而不是在材料包子目录内运行。

已有基线可重跑入口：

```bash
python -X utf8 scripts/deepen_graph_optimization.py
python -X utf8 scripts/run_amap_selfdrive_experiment.py
python -X utf8 scripts/enhance_data_and_models.py
```

多口岸仿真脚本当前依赖缺失的 `problem2_openpath_outputs/scenario_totals.csv`，需要先补齐第二问主结果后再重跑：

```bash
python -X utf8 scripts/policy_simulation_lab.py
python -X utf8 scripts/humanized_scenario_research.py
python -X utf8 scripts/digital_twin_robustness_lab.py
```

## 6. 当前结论

第二问已经具备较好的数据基础和研究口径，但还没有达到第一问 V3 那种“代码、输入、输出、审计、报告完全闭环”的程度。

汇报上可以先说：

> 当前第二问已明确建模方向为两年路径覆盖与新疆境内交通费用最小化；已有高德道路 OD 基线、多口岸阈值仿真和求解效率研究。下一步需要基于增强多模式 OD 重建主求解器，补齐可复现的年度路线明细和边级交通标签选择结果。
