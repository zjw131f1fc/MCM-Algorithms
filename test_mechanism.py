"""测试机理分析算法"""
import numpy as np
from mechanism.seir import seir
from mechanism.cellular_automaton import CellularAutomaton
from mechanism.queueing import mm1, mmc, mm1k
from mechanism.monte_carlo import estimate_pi, option_pricing, integration

print("=" * 50)
print("测试机理分析算法")
print("=" * 50)

# 1. SEIR模型
print("\n1. SEIR传染病模型")
N = 10000
E0, I0, R0 = 10, 5, 0
beta, sigma, gamma = 0.5, 0.2, 0.1
t = np.linspace(0, 100, 101)
S, E, I, R = seir(N, E0, I0, R0, beta, sigma, gamma, t)
print(f"初始状态: S={S[0]:.0f}, E={E[0]:.0f}, I={I[0]:.0f}, R={R[0]:.0f}")
print(f"最终状态: S={S[-1]:.0f}, E={E[-1]:.0f}, I={I[-1]:.0f}, R={R[-1]:.0f}")
print(f"感染峰值: {I.max():.0f} (第{t[I.argmax()]:.0f}天)")

# 2. 元胞自动机
print("\n2. 元胞自动机(Conway生命游戏)")
ca = CellularAutomaton(grid_size=(10, 10))
ca.set_random(density=0.3)
initial_alive = np.sum(ca.grid)
ca.run(steps=10)
final_alive = np.sum(ca.grid)
print(f"初始活细胞数: {initial_alive}")
print(f"10步后活细胞数: {final_alive}")
print(f"历史记录长度: {len(ca.history)}")

# 3. 排队论
print("\n3. 排队论")
print("  M/M/1模型:")
metrics = mm1(lambda_rate=3, mu_rate=5)
print(f"    平均顾客数L: {metrics['L']:.4f}")
print(f"    平均队长Lq: {metrics['Lq']:.4f}")
print(f"    平均逗留时间W: {metrics['W']:.4f}")
print(f"    平均等待时间Wq: {metrics['Wq']:.4f}")

print("  M/M/c模型:")
metrics = mmc(lambda_rate=8, mu_rate=3, c=4)
print(f"    平均顾客数L: {metrics['L']:.4f}")
print(f"    平均队长Lq: {metrics['Lq']:.4f}")

print("  M/M/1/K模型:")
metrics = mm1k(lambda_rate=4, mu_rate=5, K=10)
print(f"    平均顾客数L: {metrics['L']:.4f}")
print(f"    损失率: {metrics['PK']:.4f}")

# 4. 蒙特卡洛方法
print("\n4. 蒙特卡洛方法")
print("  估计π:")
pi_est, error = estimate_pi(n_simulations=10000, seed=42)
print(f"    估计值: {pi_est:.6f}")
print(f"    误差: {error:.6f}")

print("  期权定价:")
price, std_error = option_pricing(S0=100, K=105, T=1, r=0.05, sigma=0.2, n_simulations=10000)
print(f"    看涨期权价格: {price:.4f}")
print(f"    标准误差: {std_error:.4f}")

print("  数值积分(∫∫x^2+y^2 dxdy, x∈[0,1], y∈[0,1]):")
def func(x):
    return x[0]**2 + x[1]**2
integral, std_error = integration(func, bounds=[(0, 1), (0, 1)], n_simulations=10000)
print(f"    积分值: {integral:.6f}")
print(f"    标准误差: {std_error:.6f}")
print(f"    理论值: {2/3:.6f}")

print("\n" + "=" * 50)
print("机理分析算法测试完成")
print("=" * 50)
