# 新疆旅游线路优化可视化交付包

本目录用于承接四问模型冻结后的可视化制作。当前阶段先建立稳定的数据底座和图表执行清单，后续论文图、PPT 图和展示图都从 `01_visual_data/` 中的派生表生成。

## 当前阶段

已完成：

- 统一视觉风格规范；
- P0/P1/P2 图表执行清单；
- 可视化数据派生脚本；
- 第一批图表输入 CSV。

尚未完成：

- PNG/SVG/PDF 成图导出；
- PPT 专用版布局；
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
├── 06_paper_ready/
├── 07_ppt_ready/
└── scripts/
    └── build_visual_data.py
```

## 复现命令

在仓库根目录运行：

```bash
python -X utf8 "可视化交付包/scripts/build_visual_data.py"
```

脚本只读取四问已经冻结的最终输出，不改变任何模型结果。

## 执行原则

1. 先生成 visual CSV，再画图。
2. P0 图优先服务答辩主叙事。
3. 每张图只回答一个问题，标题使用结论句。
4. Q1 必须区分运营成功率和严格舒适成功率。
5. Q2 所有交通方式构成图必须按代表方案过滤。
6. Q3 缓冲策略使用离散点图，不用连续趋势线误导。
7. Q4 服务率和严格资源通过必须分开表达。

