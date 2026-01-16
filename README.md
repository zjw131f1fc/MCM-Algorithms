# MCM-Algorithms

美赛常用算法代码库

## 评价与决策

### topsis
```python
from evaluation.topsis import topsis
scores, ranks = topsis(data, weights=None, criteria_types=None, normalize='vector')
```
- `data`: 决策矩阵 (m方案 × n指标)
- `weights`: 权重向量，None则等权
- `criteria_types`: 1正向/-1负向，None全正向
- `normalize`: 'vector' | 'minmax'
- 返回: (scores, ranks) 得分和排名

### entropy_weight
```python
from evaluation.entropy_weight import entropy_weight
weights = entropy_weight(data, criteria_types=None)
```
- `data`: 决策矩阵 (m样本 × n指标)
- `criteria_types`: 1正向/-1负向，None全正向
- 返回: weights 各指标权重

### ahp
```python
from evaluation.ahp import ahp
weights, CR, is_consistent = ahp(judgment_matrix)
```
- `judgment_matrix`: 判断矩阵 (n × n)，标度1-9
- 返回: (weights, CR, is_consistent) 权重、一致性比率、是否通过检验

### grey_relation
```python
from evaluation.grey_relation import grey_relation
scores, ranks = grey_relation(data, reference=None, rho=0.5)
```
- `data`: 数据矩阵 (m样本 × n指标)
- `reference`: 参考序列，None取各列最大值
- `rho`: 分辨系数，默认0.5
- 返回: (scores, ranks) 关联度和排名

### fuzzy_evaluation
```python
from evaluation.fuzzy_evaluation import fuzzy_evaluation, build_membership_matrix
B, grade = fuzzy_evaluation(R, weights, method='M(·,⊕)')
```
- `R`: 模糊关系矩阵 (m因素 × n等级)
- `weights`: 权重向量
- `method`: 'M(·,⊕)' | 'M(∧,∨)' | 'M(·,∨)'
- 返回: (B, grade) 综合评价向量和最优等级

### dea_ccr
```python
from evaluation.dea import dea_ccr
efficiencies, is_efficient = dea_ccr(inputs, outputs, orientation='input')
```
- `inputs`: 投入指标矩阵 (n决策单元 × m投入)
- `outputs`: 产出指标矩阵 (n决策单元 × k产出)
- `orientation`: 'input'投入导向 或 'output'产出导向
- 返回: (efficiencies, is_efficient) 效率值和是否有效

## 预测与时间序列

### gm11
```python
from prediction.gm11 import gm11
x0_pred, x0_forecast, params = gm11(x0, n_predict=1)
```
- `x0`: 原始时间序列，至少4个点
- `n_predict`: 预测未来步数，默认1
- 返回: (x0_pred, x0_forecast, params) 拟合值、预测值、模型参数

### markov_predict
```python
from prediction.markov import markov_predict
P, probs, predictions = markov_predict(data, states=None, n_steps=1)
```
- `data`: 历史状态序列
- `states`: 所有可能状态列表，None则自动提取
- `n_steps`: 预测未来步数，默认1
- 返回: (P, probs, predictions) 转移矩阵、概率分布、最可能状态

### arima_forecast
```python
from prediction.arima import arima_forecast
model, forecast, conf_int = arima_forecast(data, order=(1,1,1), n_forecast=1)
```
- `data`: 时间序列数据
- `order`: ARIMA模型阶数(p,d,q)，默认(1,1,1)
- `n_forecast`: 预测未来步数，默认1
- 返回: (model, forecast, conf_int) 拟合模型、预测值、置信区间

## 优化算法

### linear_program
```python
from optimization.linear_programming import linear_program
x, obj_value, success, message = linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None)
```
- `c`: 目标函数系数向量 (求最小值)
- `A_ub`: 不等式约束系数矩阵 (A_ub @ x <= b_ub)
- `b_ub`: 不等式约束右端向量
- `A_eq`: 等式约束系数矩阵 (A_eq @ x == b_eq)
- `b_eq`: 等式约束右端向量
- `bounds`: 变量边界 [(min, max), ...], 默认 (0, None)
- 返回: (x, obj_value, success, message) 最优解、目标值、是否成功、状态信息

### aco_tsp
```python
from optimization.aco import aco_tsp
path, distance, history = aco_tsp(dist_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, rho=0.5)
```
- `dist_matrix`: 距离矩阵 (n × n)
- `n_ants`: 蚂蚁数量，默认20
- `n_iterations`: 迭代次数，默认100
- `alpha`: 信息素重要程度，默认1.0
- `beta`: 启发函数重要程度，默认2.0
- `rho`: 信息素挥发系数，默认0.5
- 返回: (path, distance, history) 最优路径、距离、迭代历史

### pso_optimize
```python
from optimization.pso import pso_optimize
best_position, best_cost = pso_optimize(objective_func, bounds, n_particles=30, n_iterations=100, w=0.9, c1=0.5, c2=0.3)
```
- `objective_func`: 目标函数，接受 (n_particles, n_dimensions) 数组，返回 (n_particles,) 目标值（求最小值）
- `bounds`: 变量边界 (lower_bounds, upper_bounds)，例如 (np.array([0, 0]), np.array([10, 10]))
- `n_particles`: 粒子数量，默认30
- `n_iterations`: 迭代次数，默认100
- `w`: 惯性权重，默认0.9
- `c1`: 认知参数，默认0.5
- `c2`: 社会参数，默认0.3
- 返回: (best_position, best_cost) 最优解和最优值

### 动态规划模板

#### knapsack_01
```python
from optimization.dynamic_programming import knapsack_01
max_value, selected = knapsack_01(weights, values, capacity)
```
- `weights`: 物品重量数组
- `values`: 物品价值数组
- `capacity`: 背包容量
- 返回: (max_value, selected) 最大价值和选中物品索引

#### knapsack_complete
```python
from optimization.dynamic_programming import knapsack_complete
max_value = knapsack_complete(weights, values, capacity)
```
- `weights`: 物品重量数组
- `values`: 物品价值数组
- `capacity`: 背包容量
- 返回: max_value 最大价值

#### lcs
```python
from optimization.dynamic_programming import lcs
length, sequence = lcs(s1, s2)
```
- `s1`: 序列1
- `s2`: 序列2
- 返回: (length, sequence) 最长公共子序列长度和序列

#### edit_distance
```python
from optimization.dynamic_programming import edit_distance
distance = edit_distance(s1, s2)
```
- `s1`: 源字符串
- `s2`: 目标字符串
- 返回: distance 最小编辑距离

#### lis
```python
from optimization.dynamic_programming import lis
length, sequence = lis(arr)
```
- `arr`: 输入序列
- 返回: (length, sequence) 最长递增子序列长度和序列

## 图论

### astar
```python
from graph.astar import astar
path = astar(grid, start, goal, heuristic='manhattan', allow_diagonal=True)
```
- `grid`: 网格地图 (0可通行, 1障碍物)
- `start`: 起点坐标 (row, col)
- `goal`: 终点坐标 (row, col)
- `heuristic`: 'manhattan' | 'euclidean'
- `allow_diagonal`: 是否允许对角线移动
- 返回: 路径坐标列表或None

### dijkstra
```python
from graph.network_algorithms import dijkstra
distances = dijkstra(G, source=0)  # 单源最短路径
path, length = dijkstra(G, source=0, target=3)  # 指定目标
```
- `G`: networkx图对象，边权重通过'weight'属性指定
- `source`: 源节点
- `target`: 目标节点，None则返回到所有节点的距离
- 返回: 距离字典或(路径, 长度)元组

### minimum_spanning_tree
```python
from graph.network_algorithms import minimum_spanning_tree
mst = minimum_spanning_tree(G, algorithm='kruskal')
```
- `G`: networkx无向图对象
- `algorithm`: 'kruskal' | 'prim'
- 返回: 最小生成树图对象

### floyd_warshall
```python
from graph.network_algorithms import floyd_warshall
distances = floyd_warshall(G)
```
- `G`: networkx图对象
- 返回: 嵌套字典 {源节点: {目标节点: 最短距离}}

### maximum_flow
```python
from graph.network_algorithms import maximum_flow
max_flow_value, flow_dict = maximum_flow(G, source=0, target=3)
```
- `G`: networkx有向图对象
- `source`: 源节点
- `target`: 汇节点
- 返回: (最大流量值, 流量分配字典)

## 机理分析

### seir
```python
from mechanism.seir import seir
S, E, I, R = seir(N, E0, I0, R0, beta, sigma, gamma, t)
```
- `N`: 总人口数
- `E0, I0, R0`: 初始潜伏者、感染者、康复者数量
- `beta`: 接触率
- `sigma`: 潜伏者转感染者速率(1/潜伏期)
- `gamma`: 康复率(1/感染期)
- `t`: 时间点数组
- 返回: (S, E, I, R) 各时刻四类人群数量

### cellular_automaton
```python
from mechanism.cellular_automaton import CellularAutomaton, conway_game_of_life
ca = CellularAutomaton(grid_size=(50, 50))
ca.set_random(density=0.3)
ca.run(steps=100)
```
- `grid_size`: 网格大小 (rows, cols)
- `rule`: 自定义规则函数,默认Conway生命游戏
- `boundary`: 'periodic'(周期) 或 'fixed'(固定)
- 返回: CellularAutomaton对象,包含grid和history属性

### queueing
```python
from mechanism.queueing import mm1, mmc, mm1k
metrics = mm1(lambda_rate=3, mu_rate=5)  # M/M/1模型
metrics = mmc(lambda_rate=8, mu_rate=3, c=4)  # M/M/c模型
metrics = mm1k(lambda_rate=4, mu_rate=5, K=10)  # M/M/1/K模型
```
- `lambda_rate`: 平均到达率
- `mu_rate`: 平均服务率
- `c`: 服务台数量(M/M/c)
- `K`: 系统容量(M/M/1/K)
- 返回: dict包含L(平均顾客数)、Lq(平均队长)、W(平均逗留时间)、Wq(平均等待时间)等指标

### monte_carlo
```python
from mechanism.monte_carlo import monte_carlo, estimate_pi, option_pricing, integration
# 通用框架
results, stats = monte_carlo(func, n_simulations=10000, seed=None)
# 估计π
pi_est, error = estimate_pi(n_simulations=10000, seed=None)
# 期权定价
price, std_error = option_pricing(S0=100, K=105, T=1, r=0.05, sigma=0.2, n_simulations=10000, option_type='call')
# 数值积分
integral, std_error = integration(func, bounds=[(0,1), (0,1)], n_simulations=10000)
```
- `monte_carlo`: 通用蒙特卡洛框架,接受模拟函数,返回结果和统计信息
- `estimate_pi`: 估计圆周率π
- `option_pricing`: 欧式期权定价(Black-Scholes模型)
- `integration`: 多维数值积分

## 机器学习

### linear_regression
```python
from ml.regression import linear_regression
result = linear_regression(X, y, method='ols', alpha=1.0)
```
- `X`: 特征矩阵 (n_samples × n_features)
- `y`: 目标变量 (n_samples,)
- `method`: 'ols'(普通最小二乘) | 'ridge'(岭回归) | 'lasso'(Lasso回归)
- `alpha`: 正则化强度 (仅用于ridge和lasso)
- 返回: dict包含model(模型对象)、coef(系数)、intercept(截距)、r2(R²)、mse(均方误差)、predictions(预测值)

### kmeans
```python
from ml.clustering import kmeans
result = kmeans(X, n_clusters=3, random_state=None)
```
- `X`: 特征矩阵 (n_samples × n_features)
- `n_clusters`: 簇的数量，默认3
- `random_state`: 随机种子
- 返回: dict包含model(模型对象)、labels(簇标签)、centers(簇中心)、inertia(簇内平方和)、n_iter(迭代次数)

### dbscan
```python
from ml.clustering import dbscan
result = dbscan(X, eps=0.5, min_samples=5)
```
- `X`: 特征矩阵 (n_samples × n_features)
- `eps`: 邻域半径，默认0.5
- `min_samples`: 核心点的最小邻居数，默认5
- 返回: dict包含model(模型对象)、labels(簇标签，-1为噪声)、n_clusters(簇数)、n_noise(噪声点数)、core_sample_indices(核心样本索引)

### hierarchical_clustering
```python
from ml.clustering import hierarchical_clustering
result = hierarchical_clustering(X, n_clusters=3, linkage='ward')
```
- `X`: 特征矩阵 (n_samples × n_features)
- `n_clusters`: 簇的数量，默认3
- `linkage`: 链接准则 'ward' | 'complete' | 'average' | 'single'，默认'ward'
- 返回: dict包含model(模型对象)、labels(簇标签)、n_clusters(簇数)、n_leaves(叶节点数)、children(层次树结构)

### random_forest
```python
from ml.ensemble import random_forest
result = random_forest(X, y, task='classification', n_estimators=100, random_state=None)
```
- `X`: 特征矩阵 (n_samples × n_features)
- `y`: 目标变量 (n_samples,)
- `task`: 'classification'(分类) | 'regression'(回归)
- `n_estimators`: 决策树数量，默认100
- `random_state`: 随机种子
- 返回: dict包含model、predictions、accuracy/r2、mse(回归)、feature_importances

### xgboost
```python
from ml.ensemble import xgboost
result = xgboost(X, y, task='classification', n_estimators=100, learning_rate=0.1)
```
- `X`: 特征矩阵 (n_samples × n_features)
- `y`: 目标变量 (n_samples,)
- `task`: 'classification'(分类) | 'regression'(回归)
- `n_estimators`: 提升树数量，默认100
- `learning_rate`: 学习率，默认0.1
- 返回: dict包含model、predictions、accuracy/r2、mse(回归)、feature_importances
