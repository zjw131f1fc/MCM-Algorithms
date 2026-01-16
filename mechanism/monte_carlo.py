"""
蒙特卡洛模拟(Monte Carlo Simulation)

通过大量随机抽样来估计数值解或概率分布的统计方法。
适用于复杂系统、不确定性分析、风险评估等场景。
"""

import numpy as np


def monte_carlo(func, n_simulations=10000, seed=None):
    """
    通用蒙特卡洛模拟框架

    Parameters
    ----------
    func : callable
        模拟函数,每次调用返回一个模拟结果
        函数签名: func() -> float or array_like
    n_simulations : int, optional
        模拟次数,默认10000
    seed : int, optional
        随机种子,用于结果可复现

    Returns
    -------
    results : ndarray
        所有模拟结果数组
    stats : dict
        统计信息,包含mean、std、min、max、percentiles等

    Examples
    --------
    >>> def coin_flip():
    ...     return np.random.choice([0, 1])
    >>> results, stats = monte_carlo(coin_flip, n_simulations=1000)
    >>> print(f"Mean: {stats['mean']:.3f}")
    """
    if seed is not None:
        np.random.seed(seed)

    results = np.array([func() for _ in range(n_simulations)])

    stats = {
        'mean': np.mean(results),
        'std': np.std(results),
        'min': np.min(results),
        'max': np.max(results),
        'median': np.median(results),
        'percentile_5': np.percentile(results, 5),
        'percentile_95': np.percentile(results, 95)
    }

    return results, stats


def estimate_pi(n_simulations=10000, seed=None):
    """
    蒙特卡洛估计圆周率π

    原理: 在单位正方形内随机投点,落在1/4圆内的概率为π/4

    Parameters
    ----------
    n_simulations : int, optional
        模拟次数,默认10000
    seed : int, optional
        随机种子

    Returns
    -------
    pi_estimate : float
        π的估计值
    error : float
        与真实值的误差

    Examples
    --------
    >>> pi_est, err = estimate_pi(n_simulations=100000)
    >>> print(f"π ≈ {pi_est:.6f}, error: {err:.6f}")
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(0, 1, n_simulations)
    y = np.random.uniform(0, 1, n_simulations)
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside) / n_simulations
    error = abs(pi_estimate - np.pi)

    return pi_estimate, error


def option_pricing(S0, K, T, r, sigma, n_simulations=10000, option_type='call', seed=None):
    """
    蒙特卡洛期权定价(欧式期权)

    使用几何布朗运动模拟股价路径,计算期权价值

    Parameters
    ----------
    S0 : float
        初始股价
    K : float
        行权价
    T : float
        到期时间(年)
    r : float
        无风险利率
    sigma : float
        波动率
    n_simulations : int, optional
        模拟路径数,默认10000
    option_type : {'call', 'put'}, optional
        期权类型,默认'call'
    seed : int, optional
        随机种子

    Returns
    -------
    price : float
        期权价格
    std_error : float
        标准误差

    Examples
    --------
    >>> price, se = option_pricing(S0=100, K=105, T=1, r=0.05, sigma=0.2, n_simulations=10000)
    >>> print(f"Call option price: {price:.2f} ± {se:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)

    return price, std_error


def integration(func, bounds, n_simulations=10000, seed=None):
    """
    蒙特卡洛数值积分

    使用随机采样估计多维积分

    Parameters
    ----------
    func : callable
        被积函数,接受array_like参数,返回标量
        函数签名: func(x) -> float, 其中x为d维向量
    bounds : list of tuples
        各维度积分区间 [(a1,b1), (a2,b2), ...]
    n_simulations : int, optional
        采样点数,默认10000
    seed : int, optional
        随机种子

    Returns
    -------
    integral : float
        积分估计值
    std_error : float
        标准误差

    Examples
    --------
    >>> # 计算 ∫∫ x^2 + y^2 dx dy, x∈[0,1], y∈[0,1]
    >>> def f(x):
    ...     return x[0]**2 + x[1]**2
    >>> integral, se = integration(f, [(0,1), (0,1)], n_simulations=10000)
    >>> print(f"Integral ≈ {integral:.4f} ± {se:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)

    bounds = np.array(bounds)
    d = len(bounds)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])

    samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_simulations, d))
    values = np.array([func(x) for x in samples])

    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_simulations)

    return integral, std_error


if __name__ == '__main__':
    print("=" * 60)
    print("蒙特卡洛模拟示例")
    print("=" * 60)

    # 示例1: 估计π
    print("\n1. 估计圆周率π")
    pi_est, err = estimate_pi(n_simulations=100000, seed=42)
    print(f"   模拟次数: 100,000")
    print(f"   π估计值: {pi_est:.6f}")
    print(f"   真实值:   {np.pi:.6f}")
    print(f"   误差:     {err:.6f}")

    # 示例2: 期权定价
    print("\n2. 欧式看涨期权定价")
    S0, K, T, r, sigma = 100, 105, 1, 0.05, 0.2
    price, se = option_pricing(S0, K, T, r, sigma, n_simulations=10000, seed=42)
    print(f"   初始股价: {S0}")
    print(f"   行权价:   {K}")
    print(f"   到期时间: {T}年")
    print(f"   无风险利率: {r}")
    print(f"   波动率:   {sigma}")
    print(f"   期权价格: {price:.2f} ± {se:.2f}")

    # 示例3: 数值积分
    print("\n3. 数值积分 ∫∫(x²+y²)dxdy, x∈[0,1], y∈[0,1]")
    def f(x):
        return x[0]**2 + x[1]**2
    integral, se = integration(f, [(0, 1), (0, 1)], n_simulations=10000, seed=42)
    true_value = 2/3  # 解析解
    print(f"   估计值:   {integral:.4f} ± {se:.4f}")
    print(f"   真实值:   {true_value:.4f}")
    print(f"   误差:     {abs(integral - true_value):.4f}")

    # 示例4: 通用框架 - 投骰子
    print("\n4. 通用框架 - 投骰子平均值")
    def roll_dice():
        return np.random.randint(1, 7)
    results, stats = monte_carlo(roll_dice, n_simulations=10000, seed=42)
    print(f"   模拟次数: 10,000")
    print(f"   平均值:   {stats['mean']:.3f}")
    print(f"   标准差:   {stats['std']:.3f}")
    print(f"   中位数:   {stats['median']:.1f}")
    print(f"   理论均值: 3.5")
