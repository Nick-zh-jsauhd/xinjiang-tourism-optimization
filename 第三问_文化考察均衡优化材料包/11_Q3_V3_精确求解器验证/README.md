# Q3-V3 精确求解器验证

本目录用于验证第三问 Q3-V2 三团队文化考察方案的精确最优性。

## 核心口径

- 输入沿用 `10_Q3_V2_鲁棒三团队文化考察调度/outputs` 的文化候选点与多模式 OD 标签。
- 对每个子集使用 Held-Karp 动态规划精确求组内乌鲁木齐起讫闭环。
- 对三组分配使用 HiGHS-MILP 的 0-1 集合划分模型。
- 目标为字典序：最小最大完成时间、最小组间差距、最小风险、最小交通费。
- 最优性只在固定特殊准入组、乌鲁木齐起讫、资源不跨组共享的政策空间内成立。

## 主要结果

- 精确求解器：`HiGHS-MILP`
- 最终状态：`OPTIMAL`
- 最大完成时间：`99.187` 小时
- 组间差距：`9.994` 小时
- 允许列总数：`12262`
- 对应枚举分配空间：`474254`

## 关键输出

- `outputs/q3_v3_exact_solver_assignment_routes.csv`：三组精确分配与路线结果。
- `outputs/q3_v3_exact_solver_segments.csv`：逐段交通标签。
- `outputs/q3_v3_exact_solver_column_summary.csv`：集合划分列规模与选中列。
- `outputs/q3_v3_exact_solver_phase_audit.csv`：四阶段字典序求解审计。
- `outputs/q3_v3_exact_solver_check.csv`：与 Q3-V2 精确校验对照。
- `reports/新疆旅游第三问Q3_V3精确求解器验证报告.md`：论文可用报告。