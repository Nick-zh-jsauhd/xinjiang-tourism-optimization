# Q1-V5 题面优先成本-覆盖校准

本目录在保留 Q1-V3 鲁棒多目标方案的基础上，按题面“花最少的钱游尽可能多的地方”重新校准第一问的求解叙事。

核心输出：

- `outputs/q1_v5_objective_hierarchy.csv`：题面优先目标层级；
- `outputs/q1_v5_cost_coverage_frontier.csv`：成本-覆盖前沿；
- `outputs/q1_v5_selected_comparison.csv`：题面最高覆盖、运营鲁棒、原Q1-V3主推等方案对比；
- `outputs/q1_v5_topic_marginal_cost.csv`：题面前沿边际费用；
- `reports/新疆旅游第一问Q1_V5题面优先成本覆盖校准报告.md`：论文口径报告。

复现：

```powershell
E:\Anaconda\envs\xj-opt\python.exe "E:\Desktop\运筹学项目\第一问_30天路线优化材料包\11_Q1_V5_题面优先成本覆盖校准\scripts\q1_v5_cost_coverage_calibration.py"
```
