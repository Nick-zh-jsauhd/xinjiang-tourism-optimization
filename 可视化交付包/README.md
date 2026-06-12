# 新疆旅游线路优化可视化交付包

本目录用于承接四问模型冻结后的可视化制作。当前阶段先建立稳定的数据底座和图表执行清单，后续论文图、PPT 图和展示图都从 `01_visual_data/` 中的派生表生成。

## 当前阶段

已完成：

- 统一视觉风格规范；
- P0/P1/P2 图表执行清单；
- 可视化数据派生脚本；
- 第一批图表输入 CSV。
- P0 核心图 PNG/SVG/PDF 导出。

尚未完成：

- PPT 专用版布局；
- P1/P2 地图、甘特、策略热力图等增强图；
- 论文附录图表排版。

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
└── scripts/
    ├── build_visual_data.py
    └── build_p0_figures.py
```

## 复现命令

先生成 visual CSV：

```bash
python -X utf8 "可视化交付包/scripts/build_visual_data.py"
```

再生成 P0 核心图：

```bash
python -X utf8 "可视化交付包/scripts/build_p0_figures.py"
```

两个脚本只读取四问已经冻结的最终输出和 visual CSV，不改变任何模型结果。

## P0 图表输出

P0 图表索引位于：

```text
可视化交付包/figure_index.csv
```

当前已导出 9 张核心图，每张均包含：

- `06_paper_ready/png/`：论文和通用预览 PNG；
- `06_paper_ready/svg/`：后期可编辑矢量图；
- `06_paper_ready/pdf/`：论文插图 PDF；
- `07_ppt_ready/`：PPT 直接插入 PNG；
- `05_p0_figures/`：P0 快速预览 PNG。

## 执行原则

1. 先生成 visual CSV，再画图。
2. P0 图优先服务答辩主叙事。
3. 每张图只回答一个问题，标题使用结论句。
4. Q1 必须区分运营成功率和严格舒适成功率。
5. Q2 所有交通方式构成图必须按代表方案过滤。
6. Q3 缓冲策略使用离散点图，不用连续趋势线误导。
7. Q4 服务率和严格资源通过必须分开表达。
