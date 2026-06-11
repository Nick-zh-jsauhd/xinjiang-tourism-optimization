# 第一问 Q1-V2 建模修改优化方案

## 0. 当前第一版模型的定位与问题诊断

当前第一问已经不应被理解为普通最短路或普通 TSP，而更接近：

\[
\textbf{Time-Windowed Multimodal Prize-Collecting Orienteering Problem}
\]

即：**带时间窗、多模式交通、收益选择、住宿可行性和风险扰动的奖励收集定向游问题**。

第一版主结果 `HYBRID_30D_ACO_ALNS_SA` 已经给出一条 30 天硬约束可行路线：覆盖 32 个景点，排程 30 天，时间窗违规为 0，住宿受限夜为 0，除餐饮外总代理成本约 14548.53 元，但仍存在 2 个超强度活动日。进一步的 Monte Carlo 扰动检验显示，这条路线虽然确定性可行，但现实波动下稳定性不足：模拟平均天数超过 30 天，且预约失败、酒店满房、道路延误等风险会显著影响可执行性。

因此，下一轮第一问的核心不是继续追求“再多塞几个景点”，而是将模型从**极限路线求解**升级为**真实游客可执行的路线方案族设计**。

---

## 1. Q1-V2 总体目标

下一版模型建议命名为：

\[
\textbf{Q1-V2: Humanized Robust Multimodal Prize-Collecting Orienteering Model}
\]

中文表述为：

> 面向真实游客体验的鲁棒多模式奖励收集定向游模型。

其目标是：

> 在 30 天暑期旅行周期内，为王先生夫妇设计一组兼顾景点覆盖、费用控制、交通可达、住宿落地、体力舒适、偏好满足和风险缓冲的旅游路线方案族。

路线输出不再只有一条，而应形成如下方案族：

| 方案类型 | 建模定位 | 适用场景 | 是否主推 |
|---|---|---|---|
| 极限覆盖版 | 保留 32 景点，展示硬 30 天覆盖上限 | 算法能力展示、对照基准 | 否 |
| 均衡稳健版 | 适当删减低收益点，保留 1—2 天缓冲 | 普通成年人现实自由行 | 是 |
| 亲子舒适版 | 降低每日强度，减少长转场和高温户外暴露 | 亲子、轻松游 | 扩展方案 |
| 长者慢游版 | 控制高海拔、长交通、晚到酒店和连续疲劳 | 长者、保守稳健游 | 扩展方案 |

---

# 2. 改进一：把单路线输出改成 Pareto 方案族

## 2.1 当前问题

第一版主结果给出的是单条路线：

\[
R^{(1)}=\text{32景点，30天，总成本14548.53元}
\]

但题目中的“花最少的钱游尽可能多的地方”天然是多目标问题。景点数量、费用、舒适度、风险之间存在冲突。若只输出一条路线，会导致以下问题：

- 为什么 32 个景点就是合理的？
- 少去 2—4 个低收益点是否能显著降低风险？
- 多花一点费用是否能换来更舒适的铁路、航班或包车组合？
- 普通游客、亲子游客、长者游客是否应该共用同一条路线？
- 当前 32 景点路线是否只是“极限压缩”，而不是“最推荐”？

因此，下一版应将单路线输出改为 **Pareto 方案族输出**。

## 2.2 多目标定义

设路线 \(R\) 的四个核心指标为：

\[
F_1(R)=\text{旅游覆盖价值}
\]

\[
F_2(R)=\text{除餐饮外总费用}
\]

\[
F_3(R)=\text{平均舒适度}
\]

\[
F_4(R)=\text{风险暴露}
\]

其中：

\[
F_1(R)=\sum_{i\in V}w_i y_i
\]

\[
F_2(R)=C_{\text{transport}}+C_{\text{ticket}}+C_{\text{hotel}}+C_{\text{misc}}
\]

\[
F_3(R)=\frac{1}{D}\sum_{d=1}^{D}comfort_d
\]

\[
F_4(R)=P(T(R,\omega)>30)+P(\text{预约失败})+P(\text{酒店满房})+P(\text{严重交通延误})
\]

其中：

- \(y_i=1\)：表示访问景点 \(i\)；
- \(w_i\)：景点综合价值；
- \(D\)：排程天数；
- \(\omega\)：扰动情景。

多目标模型为：

\[
\max F_1(R),\quad \min F_2(R),\quad \max F_3(R),\quad \min F_4(R)
\]

## 2.3 Pareto 有效性

路线 \(R_a\) 支配路线 \(R_b\)，当且仅当：

\[
F_1(R_a)\ge F_1(R_b),\quad F_2(R_a)\le F_2(R_b),\quad F_3(R_a)\ge F_3(R_b),\quad F_4(R_a)\le F_4(R_b)
\]

且至少一个目标严格更优。所有不被其他路线支配的路线构成 Pareto 前沿。

## 2.4 \(\varepsilon\)-约束求解方式

由于景点数量和覆盖价值具有整数特征，费用和风险是连续/半连续指标，建议采用 \(\varepsilon\)-约束法，而不是直接加权和。

设最低覆盖水平为 \(q\)，求解：

\[
\min \quad F_2(R)+\lambda_1F_4(R)-\lambda_2F_3(R)
\]

\[
\text{s.t.}\quad F_1(R)\ge q
\]

\[
T(R)\le 30
\]

\[
\text{时间窗、住宿、准入、偏好组约束成立}
\]

枚举：

\[
q=20,22,24,26,28,30,32
\]

分别求解，得到不同覆盖水平下的最优路线。

## 2.5 Pareto 输出表设计

下一轮应输出如下表格：

| route_id | 方案类型 | 覆盖景点数 | 区域覆盖数 | 除餐饮外费用 | 排程天数 | 缓冲日 | 平均舒适度 | 红色日 | 扰动可行率 | CVaR损失 | 推荐等级 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q1-A | 极限覆盖版 | 32 | 高 | 中 | 30 | 0 | 中 | 多 | 低 | 高 | 对照 |
| Q1-B | 均衡稳健版 | 28—30 | 高 | 中 | 28—29 | 1—2 | 较高 | 少 | 中高 | 中 | 主推 |
| Q1-C | 舒适推荐版 | 26—28 | 中高 | 中 | 27—29 | 2 | 高 | 很少 | 高 | 低 | 可选 |
| Q1-D | 长者慢游版 | 20—24 | 中 | 较低 | 23—26 | 2+ | 高 | 低 | 高 | 低 | 保守 |

## 2.6 主推方案选择规则

下一版报告应明确：

> 32 景点路线保留为极限覆盖版，不作为唯一主推。正文主推方案应从 Pareto 前沿中选择“均衡稳健解”。

可定义：

\[
R^*=\arg\max_R \left[
\frac{F_1(R)}{F_1^{max}}
-\alpha\frac{F_2(R)}{F_2^{max}}
+\beta\frac{F_3(R)}{100}
-\gamma F_4(R)
\right]
\]

同时满足：

\[
P(T(R,\omega)\le 30)\ge 0.8
\]

\[
\sum_d red_d\le 1
\]

\[
\sum_d buffer_d\ge 1
\]

该规则把路线选择从“最多景点”变为“综合推荐”。

---

# 3. 改进二：把鲁棒性从后验仿真前移到主模型

## 3.1 当前问题

第一版逻辑更接近：

\[
\text{先求确定性路线}\rightarrow\text{再做鲁棒性仿真}\rightarrow\text{再给策略修正}
\]

若路线本身已经压满 30 天，后验仿真发现高风险时，只能被动删点或拆段。下一版应改为：

\[
\text{扰动情景进入主模型}\rightarrow\text{直接求稳健路线}\rightarrow\text{再做仿真验证}
\]

也就是把鲁棒性从“后验检测”前移为“先验约束”。

## 3.2 情景集合定义

沿用现有数字孪生情景：

\[
\Omega=\{\text{常规暑期},\text{热浪高温},\text{雨洪道路延误},\text{预约收紧},\text{暑期客流高峰},\text{复合极端冲击}\}
\]

在每个情景 \(\omega\in\Omega\) 下，交通时间、服务时间、预约失败概率、酒店可得性、道路封闭概率都会变化：

\[
t_{ij}^{k}(\omega),\quad s_i(\omega),\quad p_i^{fail}(\omega),\quad h_i^{full}(\omega),\quad road_{ij}^{delay}(\omega)
\]

## 3.3 机会约束模型

设路线 \(R\) 在情景 \(\omega\) 下的总天数为：

\[
T(R,\omega)
\]

加入 30 天机会约束：

\[
P_{\omega}\left(T(R,\omega)\le 30\right)\ge 1-\epsilon_T
\]

例如 \(\epsilon_T=0.2\)，表示路线在扰动情景下至少有 80% 的概率能在 30 天内完成。

预约失败约束：

\[
P_{\omega}\left(N_{\text{reservation\_fail}}(R,\omega)\le 2\right)\ge 1-\epsilon_A
\]

酒店满房约束：

\[
P_{\omega}\left(N_{\text{hotel\_full}}(R,\omega)=0\right)\ge 1-\epsilon_H
\]

道路严重延误约束：

\[
P_{\omega}\left(N_{\text{severe\_delay}}(R,\omega)\le 1\right)\ge 1-\epsilon_D
\]

综合写成：

\[
P_{\omega}\left(
T(R,\omega)\le 30,
N_A\le 2,
N_H=0,
N_D\le 1
\right)\ge 1-\epsilon
\]

## 3.4 CVaR 鲁棒模型

定义路线在情景 \(\omega\) 下的损失函数：

\[
L(R,\omega)=
a_1\max(0,T(R,\omega)-30)
+a_2N_{\text{reservation\_fail}}(R,\omega)
+a_3N_{\text{hotel\_full}}(R,\omega)
+a_4N_{\text{red\_days}}(R,\omega)
+a_5C_{\text{extra}}(R,\omega)
\]

目标函数改为：

\[
\max \quad F_{\text{value}}(R)-\alpha C(R)-\beta E_{\omega}[L(R,\omega)]-\lambda CVaR_{75\%}(L(R,\omega))
\]

其中：

\[
CVaR_{75\%}(L)=\min_{\zeta}\left[\zeta+\frac{1}{1-0.75}E_{\omega}\max(0,L(R,\omega)-\zeta)\right]
\]

若采用样本平均近似，设情景样本为 \(\omega_s,s=1,\ldots,S\)，则：

\[
CVaR_{75\%}(L)=\min_{\zeta,\xi_s}\left[\zeta+\frac{1}{0.25S}\sum_{s=1}^{S}\xi_s\right]
\]

\[
\xi_s\ge L(R,\omega_s)-\zeta,\quad \xi_s\ge 0
\]

## 3.5 三档鲁棒模型输出

| 模型版本 | 约束/目标 | 用途 |
|---|---|---|
| Deterministic | 只要求确定性 30 天可行 | 对照基准 |
| Chance-Constrained | 要求扰动可行率达到阈值 | 正文主推 |
| CVaR-Robust | 控制尾部极端风险 | 保守方案 |

这样可以明确说明：不是模型求不出更多景点，而是当真实世界扰动进入模型后，过度压缩的路线不再是最优推荐。

---

# 4. 改进三：把 `self_drive` 拆成现实交通方式

## 4.1 当前问题

当前多模式数据虽然包含自驾、铁路、航班、接驳、景区交通等，但第一版主结果仍以道路接驳和景区接驳为主。这说明 `self_drive` 在模型里可能承担了过多现实交通含义。

它可能同时代表：

- 租车自驾；
- 包车；
- 拼车；
- 长途汽车；
- 旅游专线；
- 出租/网约车接驳；
- 景区区间车；
- 地图估计道路通行。

如果不拆分，模型会把道路交通当作低成本、灵活、可无限使用的万能方式，从而压制铁路、夜车、航班和公共交通。

## 4.2 交通方式集合重定义

下一轮将交通方式集合改为：

\[
K=\{\text{rail},\text{air},\text{coach},\text{tourist\_bus},\text{rental\_car},\text{charter\_car},\text{carpool},\text{taxi\_transfer},\text{scenic\_shuttle}\}
\]

| 方式 | 中文含义 | 适用场景 | 成本口径 |
|---|---|---|---|
| rail | 铁路/动车/普速/夜车 | 长距离跨区域转移 | 两人票价 |
| air | 航班 | 超长距离快速转移 | 两人机票+机场接驳 |
| coach | 长途客运 | 城市间低成本移动 | 两人票价 |
| tourist_bus | 旅游专线 | 热门景区接驳 | 两人票价 |
| rental_car | 租车自驾 | 区域内多点游 | 车辆日租+油费+过路费 |
| charter_car | 包车 | 偏远景点或公共交通不足 | 日包车费+司机成本 |
| carpool | 拼车 | 半公共交通 | 两人拼车价+不确定性 |
| taxi_transfer | 出租/网约车短接驳 | 站点、酒店、景点间 | 单车费用 |
| scenic_shuttle | 景区区间车 | 景区内部必要交通 | 两人票价 |

## 4.3 费用函数改造

原交通费用：

\[
C_{\text{transport}}=\sum_{i,j,k}c_{ij}^k x_{ij}^k
\]

改为分类费用：

\[
C_{\text{transport}}=
C_{\text{rail}}+C_{\text{air}}+C_{\text{coach}}+C_{\text{tourist\_bus}}+C_{\text{rental}}+C_{\text{charter}}+C_{\text{carpool}}+C_{\text{taxi}}+C_{\text{scenic}}
\]

### 铁路费用

\[
C_{ij}^{rail}=2\cdot fare_{ij}^{rail}
\]

若为夜车，可减少住宿费用：

\[
C_{\text{hotel}}=C_{\text{hotel}}-hotel\_saving_{ij}^{rail}
\]

但增加睡眠质量惩罚：

\[
P_{\text{sleep}}=\theta_{\text{night\_rail}}
\]

### 航班费用

\[
C_{ij}^{air}=2\cdot fare_{ij}^{air}+2\cdot airport\_transfer
\]

时间应包括：

\[
t_{ij}^{air}=flight\_time+airport\_access+security\_waiting+airport\_exit
\]

### 长途客运费用

\[
C_{ij}^{coach}=2\cdot fare_{ij}^{coach}
\]

\[
t_{ij}^{coach}=ride\_time+station\_waiting
\]

### 租车自驾费用

\[
C_{ij}^{rental}=rental\_daily\_fee\cdot days+fuel\_cost+toll\_cost+parking\_cost+one\_way\_return\_fee
\]

注意：租车费用通常按车计，不按人数乘 2。

### 包车费用

\[
C_{ij}^{charter}=charter\_daily\_fee\cdot days+driver\_lodging+driver\_meal\_subsidy
\]

### 拼车费用

\[
C_{ij}^{carpool}=2\cdot fare_{ij}^{carpool}
\]

但其风险应高于固定班次公共交通：

\[
risk_{ij}^{carpool}>risk_{ij}^{coach}
\]

## 4.4 长距离疲劳惩罚

定义交通疲劳函数：

\[
fatigue_{ij}^{k}=\theta_k\cdot t_{ij}^{k}+\mu_k\cdot \max(0,t_{ij}^{k}-T_{\text{safe}}^{k})^2
\]

其中：

- \(T_{\text{safe}}^{k}\)：交通方式 \(k\) 的舒适阈值；
- \(\theta_k\)：单位时间疲劳系数；
- \(\mu_k\)：长距离非线性疲劳惩罚。

建议参数：

| 方式 | 舒适阈值 \(T_{\text{safe}}\) | 疲劳特征 |
|---|---:|---|
| rental_car | 4 小时 | 驾驶疲劳高 |
| charter_car | 6 小时 | 乘坐疲劳中 |
| coach | 6 小时 | 乘坐疲劳中高 |
| rail | 8 小时 | 长途但可休息 |
| air | 5 小时总流程 | 快但换乘麻烦 |
| scenic_shuttle | 2 小时 | 景区内拥挤疲劳 |

## 4.5 交通方式约束

### 长距离道路限制

若道路转移超过 6 小时：

\[
longroad_d=1
\]

限制连续出现：

\[
longroad_d+longroad_{d+1}\le 1
\]

### 自驾总时长限制

\[
\sum_{i,j}t_{ij}^{rental}x_{ij}^{rental}\le T_{\text{drive\_max}}
\]

例如：

\[
T_{\text{drive\_max}}=40\text{小时}
\]

### 偏远景区交通限制

对无公共交通或接驳不稳定的景点：

\[
x_{ij}^{coach}=0
\]

必须使用：

\[
charter\_car \quad \text{or} \quad carpool
\]

### 长距离铁路优先奖励

对于跨区域长距离转移：

\[
bonus_{ij}^{rail}=\begin{cases}\psi,& distance_{ij}>500km\\0,& otherwise\end{cases}
\]

从而鼓励模型在长距离移动中选择铁路/夜车，而不是连续道路转移。

---

# 5. 改进四：把“游更多地方”改成“区域覆盖 + 偏好满足 + 内容多样性”

## 5.1 当前问题

如果目标函数直接最大化景点数：

\[
\max \sum_i y_i
\]

模型会倾向于选择交通成本低、距离密集的景区小点。例如吐鲁番区域内多个景点之间距离近，算法容易把它们全部加入，从而“刷高景点数”。

这会导致：

- 景点数量上升，但真实体验不一定更丰富；
- 区域覆盖不足；
- 文化/自然/民俗主题结构失衡；
- 题面偏好可能被形式覆盖，而非实质满足。

因此，下一轮应将“游更多地方”改造为：

\[
\text{旅游价值}=\text{区域覆盖}+\text{题面偏好满足}+\text{内容多样性}+\text{核心景点质量}
\]

## 5.2 三级旅游结构

建立三级结构：

\[
\text{spot}\rightarrow\text{cluster}\rightarrow\text{region}
\]

示例：

| region | cluster | spot |
|---|---|---|
| 东疆 | 吐鲁番文化圈 | 交河故城、坎儿井、葡萄沟、火焰山、高昌故城 |
| 北疆 | 天山北坡 | 天山天池、江布拉克 |
| 伊犁 | 伊犁河谷 | 赛里木湖、那拉提、喀拉峻、昭苏、夏塔 |
| 南疆 | 龟兹文化 | 库车王府、克孜尔石窟、天山神秘大峡谷 |
| 南疆 | 喀什帕米尔 | 喀什古城、白沙湖、石头城、奥依塔克 |
| 南疆 | 和田文化 | 和田博物馆、千里葡萄长廊等 |

## 5.3 区域覆盖变量

定义：

\[
r_m=\begin{cases}1,& \text{若覆盖区域 }m\\0,& \text{否则}\end{cases}
\]

\[
c_l=\begin{cases}1,& \text{若覆盖景点簇 }l\\0,& \text{否则}\end{cases}
\]

约束：

\[
r_m\le \sum_{i\in V_m}y_i
\]

\[
c_l\le \sum_{i\in V_l}y_i
\]

## 5.4 区域覆盖目标

旅游价值函数改为：

\[
F_{\text{value}}=\sum_i w_i y_i+\sum_l \alpha_l c_l+\sum_m \beta_m r_m
\]

其中：

- \(\sum_i w_i y_i\)：景点本身价值；
- \(\sum_l \alpha_l c_l\)：景点簇覆盖价值；
- \(\sum_m \beta_m r_m\)：区域覆盖价值。

这样模型不再只偏好局部密集点，而会鼓励跨区域体验。

## 5.5 同区域边际收益递减

对每个区域 \(m\)，设访问数量为：

\[
n_m=\sum_{i\in V_m}y_i
\]

设置推荐上限：

\[
\bar n_m
\]

若超过推荐上限，则引入过度打卡惩罚：

\[
P_{\text{overcluster}}=\sum_m \theta_m\max(0,n_m-\bar n_m)
\]

或使用边际收益递减函数：

\[
value_m(n_m)=a_m(1-e^{-b_m n_m})
\]

这表示：

- 第 1 个景点带来较大区域覆盖收益；
- 第 2—3 个景点丰富区域体验；
- 超过建议数量后边际收益快速下降。

## 5.6 题面偏好组建模

题面偏好为：

\[
G=\{\text{天池组},\text{达坂城组},\text{吐鲁番组},\text{楼兰文化组},\text{伊犁组}\}
\]

定义：

\[
u_g=\begin{cases}1,& \text{偏好组 }g\text{ 被满足}\\0,& \text{否则}\end{cases}
\]

约束：

\[
u_g\le \sum_{i\in V_g}y_i
\]

可以设置硬约束：

\[
u_g=1,\quad \forall g\in G
\]

也可以设置软约束：

\[
\sum_{g\in G}u_g\ge 4
\]

并在目标函数中加入偏好奖励：

\[
F_{\text{preference}}=\sum_{g\in G}\phi_g u_g
\]

对于“楼兰古城”这类现实可达性复杂的点，应设置为“楼兰文化组”，而不是单点硬约束。候选点可以包括楼兰文化相关博物馆、若羌文化节点、米兰遗址等。如果某些点普通游客不可达，则通过特殊准入约束剔除或惩罚。

## 5.7 内容多样性约束

给每个景点设置主题标签：

\[
tag_i\in\{\text{自然},\text{历史},\text{民族民俗},\text{宗教文化},\text{丝路遗址},\text{城市休闲},\text{高原边境}\}
\]

定义主题覆盖变量：

\[
s_h=\begin{cases}1,& \text{主题 }h\text{ 被覆盖}\\0,& \text{否则}\end{cases}
\]

约束：

\[
s_h\le \sum_{i:tag_i=h}y_i
\]

目标增加：

\[
F_{\text{diversity}}=\sum_h \delta_h s_h
\]

并设置最低多样性要求：

\[
\sum_h s_h\ge H_{\min}
\]

例如至少覆盖自然、历史、民族民俗、丝路文化四类主题。

---

# 6. 改进五：把日程排程变成人性化排程

## 6.1 当前问题

第一版已经有逐日排程和修复，但仍然存在超强度活动日，并且扰动后平均天数会超过 30 天。下一轮应把“可排程”升级为“可旅行”。

现实旅游排程不仅要满足：

\[
scheduled\_days\le 30
\]

还要满足：

- 每日活动不能过载；
- 长转场后次日应降低强度；
- 中午高温时段应减少户外暴露；
- 晚到酒店会显著降低体验；
- 应有机动缓冲日；
- 连续多天赶路会导致疲劳累积；
- 亲子、长者、普通成年人需要不同日强度阈值。

## 6.2 每日活动强度模型

定义第 \(d\) 天活动强度：

\[
A_d=T_d^{travel}+T_d^{service}+T_d^{wait}+\chi_d^{heat}+\chi_d^{altitude}+\chi_d^{crowd}
\]

其中：

- \(T_d^{travel}\)：当天交通时间；
- \(T_d^{service}\)：游览时间；
- \(T_d^{wait}\)：排队、换乘、候车时间；
- \(\chi_d^{heat}\)：高温惩罚；
- \(\chi_d^{altitude}\)：高海拔惩罚；
- \(\chi_d^{crowd}\)：拥挤惩罚。

不同游客画像设置不同阈值：

普通体力型：

\[
A_d\le 8.5+u_d
\]

亲子舒适型：

\[
A_d\le 7.0+u_d
\]

长者慢游型：

\[
A_d\le 6.5+u_d
\]

其中 \(u_d\) 是超强度松弛变量，但必须强惩罚。

## 6.3 红黄绿日约束

定义：

\[
green_d=1 \quad \text{if } A_d\le 7
\]

\[
yellow_d=1 \quad \text{if } 7<A_d\le 8.5
\]

\[
red_d=1 \quad \text{if } A_d>8.5
\]

约束：

\[
green_d+yellow_d+red_d=1
\]

限制红色日数量：

\[
\sum_d red_d\le R_{\max}
\]

普通体力型：

\[
R_{\max}=1
\]

亲子和长者路线：

\[
R_{\max}=0
\]

禁止连续红色高压日：

\[
red_d+red_{d+1}\le 1
\]

禁止红色日之后立刻黄色高压日：

\[
red_d+yellow_{d+1}\le 1
\]

## 6.4 缓冲日约束

定义：

\[
buffer_d=\begin{cases}1,& \text{第 }d\text{ 天为机动缓冲/轻量日}\\0,& \text{否则}\end{cases}
\]

要求：

\[
\sum_{d=1}^{30}buffer_d\ge B
\]

不同版本设置：

| 版本 | 缓冲日数量 \(B\) |
|---|---:|
| 极限覆盖版 | 0 |
| 均衡稳健版 | 1 |
| 舒适推荐版 | 2 |
| 长者慢游版 | 2+ |

缓冲日可用于：

- 交通误点；
- 景区预约失败后的替代；
- 天气导致的顺延；
- 体力恢复；
- 城市轻游；
- 洗衣补给；
- 调整住宿。

## 6.5 午休和高温避让约束

新疆暑期存在高温和强日照问题，尤其吐鲁番、南疆部分区域。定义户外暴露时间：

\[
O_{d,t}
\]

其中 \(t\) 表示一天中的时间段。

对高温区域设置：

\[
O_{d,12:00-16:00}\le O_{\max}
\]

若某景点为高温户外型，则优先安排：

\[
start_i\in[8:00,11:00]\cup[17:00,20:00]
\]

每日固定午休块：

\[
lunch\_break_d\ge 1\text{小时}
\]

## 6.6 晚到住宿惩罚

定义当日到达住宿点时间：

\[
arrival^{hotel}_d
\]

若：

\[
arrival^{hotel}_d>20:00
\]

则：

\[
late_d=1
\]

约束：

\[
\sum_d late_d\le 2
\]

并在目标函数中加入惩罚：

\[
P_{\text{late}}=\theta_{\text{late}}\sum_d late_d
\]

对亲子和长者版本，可设：

\[
\sum_d late_d=0
\]

## 6.7 长转场后降强度约束

若第 \(d\) 天交通时间超过 6 小时：

\[
longtransfer_d=1
\]

则第 \(d+1\) 天活动强度应下降：

\[
A_{d+1}\le B_{\text{persona}}-\Delta
\]

例如：

\[
\Delta=1.5
\]

限制连续长转场：

\[
longtransfer_d+longtransfer_{d+1}\le 1
\]

这能避免“今天长途赶路，明天继续高强度游览”的不现实安排。

---

# 7. Q1-V2 总体数学模型

综合上述五类改进，下一版模型可以写成：

\[
\max \quad
F(R)=
F_{\text{spot}}(R)+F_{\text{region}}(R)+F_{\text{preference}}(R)+F_{\text{diversity}}(R)
-\alpha C(R)-\beta Fatigue(R)-\gamma Risk(R)-\lambda CVaR_{75\%}(L(R,\omega))
\]

其中：

\[
F_{\text{spot}}(R)=\sum_i w_i y_i
\]

\[
F_{\text{region}}(R)=\sum_m \beta_m r_m+\sum_l \alpha_l c_l
\]

\[
F_{\text{preference}}(R)=\sum_g \phi_g u_g
\]

\[
F_{\text{diversity}}(R)=\sum_h \delta_h s_h
\]

\[
C(R)=C_{\text{transport}}+C_{\text{ticket}}+C_{\text{hotel}}+C_{\text{misc}}
\]

\[
Fatigue(R)=\sum_d u_d+\sum_d red_d+\sum_d longtransfer_d+\sum_d late_d
\]

\[
Risk(R)=P(T>30)+P(A_{\text{fail}}>2)+P(H_{\text{full}}>0)+P(D_{\text{delay}}>1)
\]

主约束包括：

\[
T(R)\le 30
\]

\[
P_{\omega}(T(R,\omega)\le 30)\ge 1-\epsilon
\]

\[
\sum_d buffer_d\ge B
\]

\[
\sum_d red_d\le R_{\max}
\]

\[
longtransfer_d+longtransfer_{d+1}\le 1
\]

\[
\sum_{g\in G}u_g\ge G_{\min}
\]

\[
\sum_h s_h\ge H_{\min}
\]

\[
y_i=0,\quad \forall i\notin V_o
\]

其中 \(V_o\) 是普通游客可达景点集合。

---

# 8. 下一轮算法实现方案

## 8.1 阶段一：交通标签生成

对每对景点 \(i,j\)，生成非支配交通标签：

\[
L_{ij}=\{(k,t,c,r,fatigue,path)\}
\]

若某标签 \(a\) 被标签 \(b\) 支配，即：

\[
t_a\ge t_b,\quad c_a\ge c_b,\quad r_a\ge r_b
\]

且至少一个严格大于，则删除标签 \(a\)。

输出文件建议：

```text
enhanced_od_labels_v2.csv
```

字段包括：

| 字段 | 含义 |
|---|---|
| from_spot | 起点 |
| to_spot | 终点 |
| mode_combo | 交通组合 |
| time_hours | 时间 |
| cost_yuan_for_two | 两人费用 |
| fatigue_score | 疲劳 |
| risk_score | 风险 |
| is_night_transport | 是否夜车/夜航 |
| requires_transfer | 是否换乘 |
| label_rank | 标签编号 |

## 8.2 阶段二：Pareto 主问题搜索

枚举覆盖水平：

\[
q=20,22,24,26,28,30,32
\]

每个 \(q\) 求解：

\[
\max F(R)
\]

\[
\text{s.t.}\quad F_{\text{coverage}}(R)\ge q
\]

输出：

```text
q1_v2_pareto_routes.csv
```

## 8.3 阶段三：逐日排程与人性化修复

对每条候选路线执行：

1. 时间窗检查；
2. 住宿落点检查；
3. 午休块插入；
4. 缓冲日插入；
5. 长转场后降强度；
6. 红黄绿日标记；
7. 预约失败替代点预置；
8. 高温户外时段避让；
9. 晚到酒店惩罚计算。

输出：

```text
q1_v2_daily_itinerary.csv
```

## 8.4 阶段四：鲁棒性内嵌评估

对每条 Pareto 路线计算：

\[
P(T\le 30),\quad CVaR_{75\%}(L),\quad P(\text{预约失败}),\quad P(\text{酒店满房}),\quad P(\text{严重道路延误})
\]

输出：

```text
q1_v2_robustness_summary.csv
```

---

# 9. 代码层修改建议

## 9.1 数据预处理模块

新增：

```text
build_transport_labels_v2.py
```

功能：

- 拆分 `self_drive`；
- 生成多交通方式标签；
- 删除被支配标签；
- 加入疲劳、风险、夜车、换乘字段。

## 9.2 主求解模块

在现有：

```text
hybrid_30day_metaheuristic_optimizer.py
```

基础上新增：

```text
q1_v2_pareto_humanized_optimizer.py
```

功能：

- 枚举覆盖阈值 \(q\)；
- 引入区域覆盖、偏好组、主题多样性；
- 加入缓冲日和强度惩罚；
- 输出 Pareto 方案族。

## 9.3 排程模块

在现有：

```text
build_daily_itinerary_layer.py
repair_daily_itinerary_layer.py
```

基础上新增：

```text
humanized_daily_scheduler_v2.py
```

功能：

- 午休；
- 晚到酒店惩罚；
- 长转场后降强度；
- 红黄绿日标记；
- persona-specific daily limit。

## 9.4 鲁棒性模块

在现有：

```text
digital_twin_robustness_lab.py
risk_aware_policy_engine.py
```

基础上新增：

```text
q1_v2_cvar_robust_selector.py
```

功能：

- 将仿真结果前移到主模型评价；
- 计算机会约束可行率；
- 计算 CVaR；
- 选择均衡稳健主方案。

---

# 10. 最终输出设计

## 10.1 Pareto 路线总表

| route_id | route_type | spots | regions | cost | days | buffer_days | comfort | success_prob | cvar_loss | recommendation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q1-A | 极限覆盖版 | 32 | 高 | 14548.53 | 30 | 0 | 中 | 低 | 高 | 对照 |
| Q1-B | 均衡稳健版 | 28—30 | 高 | 待求解 | 28—29 | 1—2 | 较高 | 中高 | 中 | 主推 |
| Q1-C | 舒适推荐版 | 26—28 | 中高 | 待求解 | 27—29 | 2 | 高 | 高 | 低 | 可选 |
| Q1-D | 长者慢游版 | 20—24 | 中 | 待求解 | 23—26 | 2+ | 高 | 高 | 低 | 保守 |

## 10.2 路线产品卡片

每条路线生成一张“旅游产品卡”：

```text
路线名称：均衡稳健版
适用人群：普通体力成年人
景点数：30
排程天数：29 + 1天缓冲
费用：xxxx元
平均舒适度：xx
风险提示：预约敏感点x个，长转场x段
推荐理由：兼顾题面偏好、区域覆盖和扰动可行性
```

## 10.3 每日行程表

| day | 上午 | 下午 | 晚上 | 住宿 | 活动小时 | 交通小时 | 舒适度 | 风险等级 |
|---:|---|---|---|---|---:|---:|---:|---|
| 1 | 待排程 | 待排程 | 待排程 | 待定 | 待算 | 待算 | 待算 | 绿/黄/红 |

## 10.4 体验曲线图

横轴为天数，纵轴为舒适度：

\[
comfort_d=100-aA_d-bRisk_d-cCrowd_d
\]

标出：

- 红色日；
- 黄色日；
- 缓冲日；
- 长转场日；
- 高温户外日；
- 晚到酒店日。

## 10.5 风险策略矩阵

| 情景 | 极限覆盖版 | 均衡稳健版 | 亲子舒适版 | 长者慢游版 |
|---|---|---|---|---|
| 常规暑期 | 可执行但不主推 | 主推 | 可执行 | 可执行 |
| 热浪 | 午后避让 | 主推 | 删点/午休 | 缩短户外 |
| 雨洪 | 高风险 | 加缓冲 | 拆段 | 拆段 |
| 预约收紧 | 删点 | 错峰预约 | 删点 | 替代点 |
| 复合极端 | 不推荐 | 重排 | 拆段 | 延期/拆段 |

---

# 11. 报告推荐表述

下一版报告可以这样写：

> 第一问不再将“一个月内游尽可能多的地方”简单处理为景点数量最大化，而是构建“区域覆盖—偏好满足—内容多样性—成本控制—舒适度—鲁棒性”共同约束下的多模式奖励收集定向游模型。现有 32 景点路线保留为极限覆盖方案，用于展示硬 30 天约束下的可行边界；正文主推方案则从 Pareto 前沿中选择均衡稳健解，在略微减少低边际收益景点的同时，引入 1—2 天缓冲、控制连续长转场和红色高强度日，并通过机会约束或 CVaR 风险项降低 30 天超期、预约失败、酒店满房和道路延误风险。

---

# 12. 下一步执行清单

## 12.1 数据层

- [ ] 拆分 `self_drive` 为租车、自驾、包车、拼车、客运、旅游专线；
- [ ] 为长距离交通补充铁路/夜车候选标签；
- [ ] 为每个景点补充区域、景点簇、主题标签；
- [ ] 建立偏好组表：天池、达坂城、吐鲁番、楼兰文化、伊犁；
- [ ] 为每条交通边增加疲劳分、风险分、是否夜间、是否长转场；
- [ ] 更新住宿数据，区分可住宿、舒适住宿、晚到惩罚。

## 12.2 模型层

- [ ] 从单目标路线优化改为 Pareto 方案族；
- [ ] 用 \(\varepsilon\)-约束枚举覆盖水平；
- [ ] 引入机会约束或 CVaR 鲁棒项；
- [ ] 加入区域覆盖、偏好满足、主题多样性；
- [ ] 加入缓冲日、午休、晚到、连续高强度约束。

## 12.3 算法层

- [ ] 新建交通标签生成器；
- [ ] 新建 Q1-V2 Pareto 优化器；
- [ ] 新建人性化逐日排程器；
- [ ] 新建 CVaR 鲁棒选择器；
- [ ] 保留现有 32 景点路线作为对照基准。

## 12.4 结果层

- [ ] 输出极限覆盖版、均衡稳健版、亲子舒适版、长者慢游版；
- [ ] 输出 Pareto 曲线；
- [ ] 输出每日舒适度曲线；
- [ ] 输出风险策略矩阵；
- [ ] 明确主推方案为“均衡稳健版”，而不是单纯 32 景点极限版。

---

# 13. 最终一句话总结

下一轮第一问的核心不是“再求一条更长路线”，而是：

> **把 32 景点 30 天路线从唯一答案降级为极限覆盖基准，把均衡稳健路线升级为正文主方案；通过 Pareto 方案族、鲁棒机会约束/CVaR、多模式真实交通、区域多样性和人性化排程，把第一问从图论路径题升级为真实游客可执行的旅游路线优化系统。**
