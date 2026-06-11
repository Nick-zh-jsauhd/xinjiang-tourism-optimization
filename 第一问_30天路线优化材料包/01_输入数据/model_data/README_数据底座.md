# 新疆旅游运筹优化建模数据底座

源文件：新疆旅游全品类数据明细.xlsx

## 生成表
- `spot_clean.csv`：40 行，30 列
- `hub_clean.csv`：25 行，6 列
- `spot_hub_map.csv`：40 行，8 列
- `edge_road_directed.csv`：56 行，13 列
- `hotel_options_clean.csv`：29 行，10 列
- `hotel_place_default.csv`：19 行，7 列
- `transport_cost_clean.csv`：33 行，14 列
- `scenario_parameters.csv`：10 行，5 列
- `data_quality_issues.csv`：20 行，7 列
- `source_sites.csv`：12 行，4 列
- `data_dictionary.csv`：9 行，3 列

## 建模使用建议
- 交通主图使用 `hub_clean` 与 `edge_road_directed`，不要直接用景点名建图。
- 景点访问变量使用 `spot_clean.spot_id`，再通过 `spot_hub_map.hub_id` 挂接到交通图。
- 第一问费用估算应区分 `per_person` 与 `per_vehicle/day`，住宿是双人房房费，不按人数乘二。
- 第四问仍缺容量数据，只能先保留参数化模型。
- 建模前先阅读 `data_quality_issues.csv`，其中 High 级问题需要在报告中显式说明或补数。