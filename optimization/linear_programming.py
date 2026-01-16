"""
线性规划 (Linear Programming) - scipy.optimize.linprog 封装

用于求解线性规划问题的标准形式：
    min c^T x
    s.t. A_ub x <= b_ub
         A_eq x = b_eq
         bounds
"""

import numpy as np
from scipy.optimize import linprog


def linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs'):
    """
    线性规划求解器

    求解标准形式的线性规划问题：
        minimize:    c^T x
        subject to:  A_ub @ x <= b_ub
                     A_eq @ x == b_eq
                     bounds[i][0] <= x[i] <= bounds[i][1]

    Parameters
    ----------
    c : array_like
        目标函数系数向量 (n,)，求最小值
    A_ub : array_like, optional
        不等式约束系数矩阵 (m_ub × n)
    b_ub : array_like, optional
        不等式约束右端向量 (m_ub,)
    A_eq : array_like, optional
        等式约束系数矩阵 (m_eq × n)
    b_eq : array_like, optional
        等式约束右端向量 (m_eq,)
    bounds : sequence of tuples, optional
        变量边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
        默认 (0, None) 表示 x >= 0
    method : str, optional
        求解方法，默认 'highs'
        可选: 'highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex', 'simplex'

    Returns
    -------
    x : ndarray
        最优解向量 (n,)，若无解返回 None
    obj_value : float
        最优目标函数值，若无解返回 None
    success : bool
        是否成功求解
    message : str
        求解状态信息

    Examples
    --------
    示例1: 简单线性规划
    >>> # min -x1 - 2x2
    >>> # s.t. x1 + x2 <= 4
    >>> #      2x1 + x2 <= 5
    >>> #      x1, x2 >= 0
    >>> c = [-1, -2]
    >>> A_ub = [[1, 1], [2, 1]]
    >>> b_ub = [4, 5]
    >>> x, obj, success, msg = linear_program(c, A_ub, b_ub)
    >>> print(f"最优解: x = {x}, 目标值 = {obj}")

    示例2: 含等式约束
    >>> # min 2x1 + 3x2
    >>> # s.t. x1 + x2 = 5
    >>> #      x1 <= 3
    >>> #      x1, x2 >= 0
    >>> c = [2, 3]
    >>> A_ub = [[1, 0]]
    >>> b_ub = [3]
    >>> A_eq = [[1, 1]]
    >>> b_eq = [5]
    >>> x, obj, success, msg = linear_program(c, A_ub, b_ub, A_eq, b_eq)

    示例3: 自定义变量边界
    >>> # min x1 + x2
    >>> # s.t. x1 + 2x2 >= 3  (转为 -x1 - 2x2 <= -3)
    >>> #      -1 <= x1 <= 2
    >>> #      x2 无上界
    >>> c = [1, 1]
    >>> A_ub = [[-1, -2]]
    >>> b_ub = [-3]
    >>> bounds = [(-1, 2), (0, None)]
    >>> x, obj, success, msg = linear_program(c, A_ub, b_ub, bounds=bounds)

    Notes
    -----
    - 若求最大值，将目标函数系数取负：max c^T x = -min (-c)^T x
    - 不等式约束 >= 需转为 <=：a^T x >= b  =>  -a^T x <= -b
    - 默认变量非负 (x >= 0)，若需无约束变量，设置 bounds=(-np.inf, np.inf)
    - 复杂度: O(n^3) ~ O(n^4)，取决于约束数量和求解方法
    """
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=method)

    if result.success:
        return result.x, result.fun, True, result.message
    else:
        return None, None, False, result.message


if __name__ == '__main__':
    print("=" * 60)
    print("示例1: 简单线性规划")
    print("=" * 60)
    print("目标: min -x1 - 2x2")
    print("约束: x1 + x2 <= 4")
    print("      2x1 + x2 <= 5")
    print("      x1, x2 >= 0\n")

    c = [-1, -2]
    A_ub = [[1, 1], [2, 1]]
    b_ub = [4, 5]

    x, obj, success, msg = linear_program(c, A_ub, b_ub)

    if success:
        print(f"最优解: x1 = {x[0]:.4f}, x2 = {x[1]:.4f}")
        print(f"目标值: {obj:.4f}")
        print(f"最大值: {-obj:.4f}")
    else:
        print(f"求解失败: {msg}")

    print("\n" + "=" * 60)
    print("示例2: 含等式约束")
    print("=" * 60)
    print("目标: min 2x1 + 3x2")
    print("约束: x1 + x2 = 5")
    print("      x1 <= 3")
    print("      x1, x2 >= 0\n")

    c = [2, 3]
    A_ub = [[1, 0]]
    b_ub = [3]
    A_eq = [[1, 1]]
    b_eq = [5]

    x, obj, success, msg = linear_program(c, A_ub, b_ub, A_eq, b_eq)

    if success:
        print(f"最优解: x1 = {x[0]:.4f}, x2 = {x[1]:.4f}")
        print(f"目标值: {obj:.4f}")
    else:
        print(f"求解失败: {msg}")

    print("\n" + "=" * 60)
    print("示例3: 生产计划问题")
    print("=" * 60)
    print("某工厂生产A、B两种产品")
    print("A产品利润: 30元/件, B产品利润: 40元/件")
    print("资源约束:")
    print("  原料: A需2kg, B需3kg, 共12kg")
    print("  工时: A需3h, B需2h, 共12h")
    print("求最大利润生产方案\n")

    # max 30A + 40B => min -30A - 40B
    c = [-30, -40]
    A_ub = [[2, 3],   # 原料约束
            [3, 2]]   # 工时约束
    b_ub = [12, 12]

    x, obj, success, msg = linear_program(c, A_ub, b_ub)

    if success:
        print(f"最优方案: 生产A产品 {x[0]:.2f} 件, B产品 {x[1]:.2f} 件")
        print(f"最大利润: {-obj:.2f} 元")
        print(f"原料使用: {2*x[0] + 3*x[1]:.2f} kg")
        print(f"工时使用: {3*x[0] + 2*x[1]:.2f} h")
    else:
        print(f"求解失败: {msg}")
