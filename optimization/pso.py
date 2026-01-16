"""
粒子群优化算法 (Particle Swarm Optimization, PSO)

基于 pyswarms 库的封装，用于求解连续优化问题。
"""

import numpy as np
from pyswarms.single import GlobalBestPSO


def pso_optimize(objective_func, bounds, n_particles=30, n_iterations=100,
                 w=0.9, c1=0.5, c2=0.3, seed=None):
    """
    粒子群优化算法求解连续优化问题

    Parameters
    ----------
    objective_func : callable
        目标函数，接受形状为 (n_particles, n_dimensions) 的数组，
        返回形状为 (n_particles,) 的目标值数组（求最小值）
    bounds : tuple of array-like
        变量边界 (lower_bounds, upper_bounds)，每个为长度为 n_dimensions 的数组
        例如: (np.array([0, 0]), np.array([10, 10]))
    n_particles : int, optional
        粒子数量，默认30
    n_iterations : int, optional
        迭代次数，默认100
    w : float, optional
        惯性权重，控制粒子保持原速度的程度，默认0.9
    c1 : float, optional
        认知参数，控制粒子向自身最优位置移动的程度，默认0.5
    c2 : float, optional
        社会参数，控制粒子向全局最优位置移动的程度，默认0.3
    seed : int, optional
        随机种子

    Returns
    -------
    best_position : ndarray
        最优解向量 (n_dimensions,)
    best_cost : float
        最优目标函数值

    Examples
    --------
    >>> # 求解 Sphere 函数: f(x) = sum(x^2)
    >>> def sphere(x):
    ...     return np.sum(x**2, axis=1)
    >>> bounds = (np.array([-5, -5]), np.array([5, 5]))
    >>> best_pos, best_cost = pso_optimize(sphere, bounds, n_particles=30, n_iterations=100)
    >>> print(f"最优解: {best_pos}, 最优值: {best_cost:.6f}")

    >>> # 求解 Rosenbrock 函数: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    >>> def rosenbrock(x):
    ...     return (1 - x[:, 0])**2 + 100 * (x[:, 1] - x[:, 0]**2)**2
    >>> bounds = (np.array([-2, -2]), np.array([2, 2]))
    >>> best_pos, best_cost = pso_optimize(rosenbrock, bounds, n_particles=50, n_iterations=200)
    >>> print(f"最优解: {best_pos}, 最优值: {best_cost:.6f}")

    Notes
    -----
    - 目标函数必须接受二维数组输入 (n_particles, n_dimensions)
    - 算法求解最小化问题，若求最大值，目标函数返回负值
    - 参数调优建议:
      * w: 0.4-0.9，较大值利于全局搜索，较小值利于局部搜索
      * c1, c2: 通常取 0-2，c1+c2 < 4
      * n_particles: 通常为问题维度的10-30倍
    """
    if seed is not None:
        np.random.seed(seed)

    options = {'c1': c1, 'c2': c2, 'w': w}

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options=options,
        bounds=bounds
    )

    best_cost, best_position = optimizer.optimize(objective_func, iters=n_iterations)

    return best_position, best_cost


if __name__ == '__main__':
    # 示例1: Sphere函数 (全局最优解: x=[0,0], f(x)=0)
    def sphere(x):
        return np.sum(x**2, axis=1)

    print("示例1: Sphere函数优化")
    bounds = (np.array([-5, -5]), np.array([5, 5]))
    best_pos, best_cost = pso_optimize(
        sphere, bounds,
        n_particles=30,
        n_iterations=100,
        seed=42
    )
    print(f"最优解: {best_pos}")
    print(f"最优值: {best_cost:.6f}\n")

    # 示例2: Rosenbrock函数 (全局最优解: x=[1,1], f(x)=0)
    def rosenbrock(x):
        return (1 - x[:, 0])**2 + 100 * (x[:, 1] - x[:, 0]**2)**2

    print("示例2: Rosenbrock函数优化")
    bounds = (np.array([-2, -2]), np.array([2, 2]))
    best_pos, best_cost = pso_optimize(
        rosenbrock, bounds,
        n_particles=50,
        n_iterations=200,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42
    )
    print(f"最优解: {best_pos}")
    print(f"最优值: {best_cost:.6f}")
