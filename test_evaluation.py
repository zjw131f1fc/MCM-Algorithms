"""测试评价与决策类算法"""
import numpy as np
from evaluation.topsis import topsis
from evaluation.entropy_weight import entropy_weight
from evaluation.ahp import ahp
from evaluation.grey_relation import grey_relation
from evaluation.fuzzy_evaluation import fuzzy_evaluation
from evaluation.dea import dea_ccr

print("=" * 50)
print("测试评价与决策类算法")
print("=" * 50)

# 1. TOPSIS
print("\n1. TOPSIS")
data = np.array([
    [80, 90, 85, 70],
    [75, 85, 80, 85],
    [90, 80, 75, 90],
    [85, 95, 90, 80]
])
scores, ranks = topsis(data)
print(f"得分: {scores}")
print(f"排名: {ranks}")

# 2. 熵权法
print("\n2. 熵权法")
weights = entropy_weight(data)
print(f"权重: {weights}")

# 3. AHP
print("\n3. AHP")
judgment_matrix = np.array([
    [1, 2, 5],
    [1/2, 1, 3],
    [1/5, 1/3, 1]
])
weights, CR, is_consistent = ahp(judgment_matrix)
print(f"权重: {weights}")
print(f"一致性比率CR: {CR:.4f}")
print(f"是否通过一致性检验: {is_consistent}")

# 4. 灰色关联分析
print("\n4. 灰色关联分析")
scores, ranks = grey_relation(data)
print(f"关联度: {scores}")
print(f"排名: {ranks}")

# 5. 模糊综合评价
print("\n5. 模糊综合评价")
R = np.array([
    [0.3, 0.4, 0.2, 0.1],
    [0.2, 0.5, 0.2, 0.1],
    [0.4, 0.3, 0.2, 0.1]
])
weights = np.array([0.4, 0.3, 0.3])
B, grade = fuzzy_evaluation(R, weights)
print(f"综合评价向量: {B}")
print(f"最优等级: {grade}")

# 6. DEA
print("\n6. DEA")
inputs = np.array([
    [5, 3],
    [4, 4],
    [6, 2],
    [3, 5]
])
outputs = np.array([
    [8, 6],
    [7, 7],
    [9, 5],
    [6, 8]
])
efficiencies, is_efficient = dea_ccr(inputs, outputs)
print(f"效率值: {efficiencies}")
print(f"是否有效: {is_efficient}")

print("\n" + "=" * 50)
print("评价与决策类算法测试完成")
print("=" * 50)
