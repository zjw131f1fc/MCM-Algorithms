"""测试优化算法"""
import numpy as np
from optimization.linear_programming import linear_program
from optimization.aco import aco_tsp
from optimization.pso import pso_optimize
from optimization.dynamic_programming import knapsack_01, knapsack_complete, lcs, edit_distance, lis

print("=" * 50)
print("测试优化算法")
print("=" * 50)

# 1. 线性规划
print("\n1. 线性规划")
c = np.array([2, 3])  # 目标函数系数
A_ub = np.array([[1, 1], [2, 1]])  # 不等式约束
b_ub = np.array([10, 15])
x, obj_value, success, message = linear_program(c, A_ub=A_ub, b_ub=b_ub)
print(f"最优解: {x}")
print(f"目标值: {obj_value:.4f}")
print(f"求解状态: {message}")

# 2. 蚁群算法TSP
print("\n2. 蚁群算法TSP")
dist_matrix = np.array([
    [0, 2, 3, 4],
    [2, 0, 5, 3],
    [3, 5, 0, 2],
    [4, 3, 2, 0]
])
path, distance, history = aco_tsp(dist_matrix, n_ants=10, n_iterations=50)
print(f"最优路径: {path}")
print(f"路径长度: {distance:.4f}")
print(f"收敛历史(前5代): {history[:5]}")

# 3. 粒子群优化
print("\n3. 粒子群优化")
def sphere_function(X):
    return np.sum(X**2, axis=1)

bounds = (np.array([-5, -5]), np.array([5, 5]))
best_position, best_cost = pso_optimize(sphere_function, bounds, n_particles=20, n_iterations=50)
print(f"最优解: {best_position}")
print(f"最优值: {best_cost:.6f}")

# 4. 0-1背包
print("\n4. 0-1背包")
weights = np.array([2, 3, 4, 5])
values = np.array([3, 4, 5, 6])
capacity = 8
max_value, selected = knapsack_01(weights, values, capacity)
print(f"最大价值: {max_value}")
print(f"选中物品索引: {selected}")

# 5. 完全背包
print("\n5. 完全背包")
max_value = knapsack_complete(weights, values, capacity)
print(f"最大价值: {max_value}")

# 6. 最长公共子序列
print("\n6. 最长公共子序列")
s1 = "ABCDGH"
s2 = "AEDFHR"
length, sequence = lcs(s1, s2)
print(f"序列1: {s1}")
print(f"序列2: {s2}")
print(f"LCS长度: {length}")
print(f"LCS序列: {sequence}")

# 7. 编辑距离
print("\n7. 编辑距离")
s1 = "kitten"
s2 = "sitting"
distance = edit_distance(s1, s2)
print(f"源字符串: {s1}")
print(f"目标字符串: {s2}")
print(f"编辑距离: {distance}")

# 8. 最长递增子序列
print("\n8. 最长递增子序列")
arr = [10, 9, 2, 5, 3, 7, 101, 18]
length, sequence = lis(arr)
print(f"输入序列: {arr}")
print(f"LIS长度: {length}")
print(f"LIS序列: {sequence}")

print("\n" + "=" * 50)
print("优化算法测试完成")
print("=" * 50)
