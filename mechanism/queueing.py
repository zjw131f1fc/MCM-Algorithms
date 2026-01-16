"""
排队论（Queueing Theory）模型

实现经典排队模型的性能指标计算，包括M/M/1、M/M/c等模型。

依赖: numpy
"""

import numpy as np
import math


def mm1(lambda_rate, mu_rate):
    """
    M/M/1排队模型：单服务台、泊松到达、指数服务

    Parameters
    ----------
    lambda_rate : float
        平均到达率（单位时间到达顾客数）
    mu_rate : float
        平均服务率（单位时间服务顾客数）

    Returns
    -------
    dict
        包含以下性能指标：
        - rho: 服务强度（利用率）
        - L: 系统平均顾客数
        - Lq: 队列平均顾客数
        - W: 系统平均逗留时间
        - Wq: 队列平均等待时间
        - P0: 系统空闲概率

    Notes
    -----
    稳定条件: lambda < mu (rho < 1)

    Examples
    --------
    >>> metrics = mm1(lambda_rate=3, mu_rate=5)
    >>> print(f"平均队长: {metrics['Lq']:.2f}")
    """
    if lambda_rate >= mu_rate:
        raise ValueError("系统不稳定: lambda必须小于mu")

    rho = lambda_rate / mu_rate
    P0 = 1 - rho
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu_rate - lambda_rate)
    Wq = rho / (mu_rate - lambda_rate)

    return {
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'P0': P0
    }


def mmc(lambda_rate, mu_rate, c):
    """
    M/M/c排队模型：c个服务台、泊松到达、指数服务

    Parameters
    ----------
    lambda_rate : float
        平均到达率
    mu_rate : float
        单个服务台平均服务率
    c : int
        服务台数量

    Returns
    -------
    dict
        包含以下性能指标：
        - rho: 服务强度
        - L: 系统平均顾客数
        - Lq: 队列平均顾客数
        - W: 系统平均逗留时间
        - Wq: 队列平均等待时间
        - P0: 系统空闲概率
        - Pw: 需要等待的概率（Erlang C公式）

    Notes
    -----
    稳定条件: lambda < c*mu (rho < 1)

    Examples
    --------
    >>> metrics = mmc(lambda_rate=8, mu_rate=3, c=4)
    >>> print(f"等待概率: {metrics['Pw']:.2%}")
    """
    if lambda_rate >= c * mu_rate:
        raise ValueError("系统不稳定: lambda必须小于c*mu")

    rho = lambda_rate / (c * mu_rate)

    # 计算P0
    sum_term = sum((lambda_rate / mu_rate)**n / math.factorial(n) for n in range(c))
    last_term = (lambda_rate / mu_rate)**c / (math.factorial(c) * (1 - rho))
    P0 = 1 / (sum_term + last_term)

    # Erlang C公式：需要等待的概率
    Pw = ((lambda_rate / mu_rate)**c / math.factorial(c)) * (1 / (1 - rho)) * P0

    Lq = Pw * rho / (1 - rho)
    L = Lq + lambda_rate / mu_rate
    Wq = Lq / lambda_rate
    W = Wq + 1 / mu_rate

    return {
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'P0': P0,
        'Pw': Pw
    }


def mm1k(lambda_rate, mu_rate, K):
    """
    M/M/1/K排队模型：单服务台、系统容量K（含服务台）

    Parameters
    ----------
    lambda_rate : float
        平均到达率
    mu_rate : float
        平均服务率
    K : int
        系统最大容量（队列+服务台）

    Returns
    -------
    dict
        包含以下性能指标：
        - rho: 服务强度
        - L: 系统平均顾客数
        - Lq: 队列平均顾客数
        - W: 系统平均逗留时间
        - Wq: 队列平均等待时间
        - P0: 系统空闲概率
        - PK: 系统满概率（拒绝率）
        - lambda_eff: 有效到达率

    Notes
    -----
    有限容量系统，总是稳定的

    Examples
    --------
    >>> metrics = mm1k(lambda_rate=4, mu_rate=5, K=10)
    >>> print(f"拒绝率: {metrics['PK']:.2%}")
    """
    rho = lambda_rate / mu_rate

    if abs(rho - 1) < 1e-10:
        P0 = 1 / (K + 1)
        PK = P0
        L = K / 2
    else:
        P0 = (1 - rho) / (1 - rho**(K + 1))
        PK = P0 * rho**K
        L = rho / (1 - rho) - (K + 1) * rho**(K + 1) / (1 - rho**(K + 1))

    lambda_eff = lambda_rate * (1 - PK)
    Lq = L - (1 - P0)
    W = L / lambda_eff if lambda_eff > 0 else 0
    Wq = Lq / lambda_eff if lambda_eff > 0 else 0

    return {
        'rho': rho,
        'L': L,
        'Lq': Lq,
        'W': W,
        'Wq': Wq,
        'P0': P0,
        'PK': PK,
        'lambda_eff': lambda_eff
    }


if __name__ == '__main__':
    print("=== M/M/1 模型示例 ===")
    m1 = mm1(lambda_rate=3, mu_rate=5)
    print(f"服务强度: {m1['rho']:.2f}")
    print(f"平均队长: {m1['Lq']:.2f}")
    print(f"平均等待时间: {m1['Wq']:.2f}\n")

    print("=== M/M/c 模型示例 ===")
    mc = mmc(lambda_rate=8, mu_rate=3, c=4)
    print(f"等待概率: {mc['Pw']:.2%}")
    print(f"平均队长: {mc['Lq']:.2f}")
    print(f"平均等待时间: {mc['Wq']:.2f}\n")

    print("=== M/M/1/K 模型示例 ===")
    m1k = mm1k(lambda_rate=4, mu_rate=5, K=10)
    print(f"拒绝率: {m1k['PK']:.2%}")
    print(f"有效到达率: {m1k['lambda_eff']:.2f}")
    print(f"平均队长: {m1k['Lq']:.2f}")
