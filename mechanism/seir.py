"""
SEIR传染病模型

SEIR模型将人群分为四类:
- S (Susceptible): 易感者
- E (Exposed): 潜伏者(已感染但无传染性)
- I (Infected): 感染者(有传染性)
- R (Recovered): 康复者(免疫)

使用常微分方程(ODE)描述各类人群随时间的变化。
"""

import numpy as np
from scipy.integrate import odeint


def seir(N, E0, I0, R0, beta, sigma, gamma, t):
    """
    SEIR传染病模型求解

    Parameters
    ----------
    N : int
        总人口数
    E0 : int
        初始潜伏者数量
    I0 : int
        初始感染者数量
    R0 : int
        初始康复者数量
    beta : float
        接触率(易感者与感染者接触后被感染的概率)
    sigma : float
        潜伏者转为感染者的速率(1/潜伏期)
    gamma : float
        康复率(1/感染期)
    t : array_like
        时间点数组

    Returns
    -------
    S : ndarray
        各时刻易感者数量
    E : ndarray
        各时刻潜伏者数量
    I : ndarray
        各时刻感染者数量
    R : ndarray
        各时刻康复者数量

    Examples
    --------
    >>> import numpy as np
    >>> from mechanism.seir import seir
    >>> N = 10000
    >>> E0, I0, R0 = 10, 1, 0
    >>> beta, sigma, gamma = 0.5, 0.2, 0.1
    >>> t = np.linspace(0, 160, 160)
    >>> S, E, I, R = seir(N, E0, I0, R0, beta, sigma, gamma, t)
    """
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]

    def deriv(y, t):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    ret = odeint(deriv, y0, t)
    S, E, I, R = ret.T
    return S, E, I, R


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 10000
    E0, I0, R0 = 10, 1, 0
    beta = 0.5
    sigma = 0.2
    gamma = 0.1
    t = np.linspace(0, 160, 160)

    S, E, I, R = seir(N, E0, I0, R0, beta, sigma, gamma, t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Peak infected: {I.max():.0f} at day {t[I.argmax()]:.0f}")
