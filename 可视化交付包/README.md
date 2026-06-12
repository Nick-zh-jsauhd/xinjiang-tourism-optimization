# 新疆旅游线路优化可视化交付包

本目录用于承接四问模型冻结后的论文与答辩可视化制作。当前推荐版本是 **Nature 风格核心证据图组**：少图、高信息密度、白底、无网格线、克制配色，优先服务论文论证而不是把所有结果都画出来。

## 当前状态

已完成：

- 统一视觉风格规范；
- 四问可视化数据底座 `01_visual_data/`；
- Nature 风格 P0 核心图组；
- 论文可用 PNG/SVG/PDF 与 PPT 预览 PNG 导出。

暂不作为主图使用：

- 旧版模型演化卡片；
- 旧版漏斗图；
- 旧版大面积热力图；
- 低信息量的独立结论卡片。

这些图容易造成“装饰多于信息”的问题，当前 paper-ready 输出已不再保留。

## 目录结构

```text
可视化交付包/
├── 00_visual_style_guide/
│   ├── visual_style_guide.md
│   └── color_palette.csv
├── 01_visual_data/
│   ├── README.md
│   └── *.csv
├── 02_chart_contracts/
│   └── chart_execution_plan.md
├── 05_p0_figures/
│   └── *.png
├── 06_paper_ready/
│   ├── png/
│   ├── svg/
│   └── pdf/
├── 07_ppt_ready/
├── scripts/
│   ├── build_visual_data.py
│   └── build_p0_figures.py
└── figure_index.csv
```

## 复现命令

先生成 visual CSV：

```bash
python -X utf8 "可视化交付包/scripts/build_visual_data.py"
```

再生成 Nature 风格 P0 核心图：

```bash
python -X utf8 "可视化交付包/scripts/build_p0_figures.py"
```

绘图脚本会清理旧版 P0 图像输出，然后重新导出当前推荐的 6 张核心图。

## 当前 P0 核心图

| 图号 | 文件名 | 论证作用 |
|---|---|---|
| Fig. 1 | `fig_q1_route_tradeoff` | 展示 Q1 覆盖、缓冲、红色压力日与两类成功率之间的取舍 |
| Fig. 2 | `fig_q2_gateway_cost_rule` | 展示 Q2 固定起讫、多口岸费用与外部差价阈值 |
| Fig. 3 | `fig_q3_minmax_evidence` | 展示 Q3 三组 MinMax 完成时间与 fixed-policy exact gap |
| Fig. 4 | `fig_q4_capacity_evidence` | 展示 Q4 旧9线路、基准需求、极端策略和18线路容量对比 |
| Fig. 5 | `fig_q4_quality_audit_dotplot` | 展示 Q4 线路质量审计，区分可直接投放与需微调线路 |
| Fig. 6 | `fig_q4_bottleneck_shadow_price` | 展示 Q4 主要瓶颈影子价格，替代低信息量热力图 |

图像索引位于：

```text
可视化交付包/figure_index.csv
```

每张图均导出：

- `06_paper_ready/png/`：论文和通用预览 PNG；
- `06_paper_ready/svg/`：后期可编辑矢量图；
- `06_paper_ready/pdf/`：论文插图 PDF；
- `07_ppt_ready/`：答辩 PPT 可直接插入 PNG；
- `05_p0_figures/`：P0 快速预览 PNG。

## 执行原则

1. 只画能支撑结论的证据图。
2. 不使用网格线，不用大面积色块制造视觉重量。
3. 成功率必须拆分为运营成功率与严格舒适成功率。
4. 离散策略不画连续折线。
5. Q4 不再用大面积空白热力图作为主图，瓶颈用 Top 约束排序表达。
6. 中文标题和坐标轴使用宋体加粗，数字和英文使用 Times New Roman。
