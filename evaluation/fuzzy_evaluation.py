"""
Fuzzy Comprehensive Evaluation
模糊综合评价

用于处理具有模糊性的多因素决策问题，通过模糊关系矩阵和模糊合成运算得到综合评价结果。
"""

import numpy as np


def fuzzy_evaluation(R, weights, method='M(·,⊕)'):
    """
    模糊综合评价

    Parameters
    ----------
    R : array-like, shape (m, n)
        模糊关系矩阵，m个因素，n个评价等级
        R[i,j]表示第i个因素对第j个评价等级的隶属度
    weights : array-like, shape (m,)
        权重向量，各因素权重，和为1
    method : str, optional
        模糊合成算子，默认 'M(·,⊕)'
        - 'M(·,⊕)': 加权平均型 (主算子)
        - 'M(∧,∨)': 取小取大型
        - 'M(·,∨)': 乘积取大型

    Returns
    -------
    B : ndarray, shape (n,)
        综合评价向量，B[j]表示对第j个评价等级的隶属度
    grade : int
        最大隶属度对应的评价等级索引

    Examples
    --------
    >>> import numpy as np
    >>> # 3个因素，4个评价等级(优、良、中、差)
    >>> R = np.array([
    ...     [0.3, 0.5, 0.2, 0.0],  # 因素1
    ...     [0.4, 0.4, 0.1, 0.1],  # 因素2
    ...     [0.2, 0.3, 0.4, 0.1]   # 因素3
    ... ])
    >>> weights = [0.5, 0.3, 0.2]
    >>> B, grade = fuzzy_evaluation(R, weights)
    >>> print(f"综合评价向量: {B}")
    >>> print(f"评价等级: {grade}")
    """
    R = np.array(R, dtype=float)
    A = np.array(weights, dtype=float)

    if method == 'M(·,⊕)':
        # 加权平均型: B = A · R
        B = A @ R
    elif method == 'M(∧,∨)':
        # 取小取大型: B[j] = max_i(min(A[i], R[i,j]))
        B = np.max(np.minimum(A[:, np.newaxis], R), axis=0)
    elif method == 'M(·,∨)':
        # 乘积取大型: B[j] = max_i(A[i] * R[i,j])
        B = np.max(A[:, np.newaxis] * R, axis=0)
    else:
        raise ValueError(f"未知的合成算子: {method}")

    grade = np.argmax(B)
    return B, grade


def build_membership_matrix(data, levels, membership_func='trapezoidal'):
    """
    构建模糊关系矩阵

    Parameters
    ----------
    data : array-like, shape (m,)
        各因素的实际值
    levels : array-like, shape (n+1,) or (n, 4)
        评价等级边界
        - 若为(n+1,): 等级分界点，自动构建梯形隶属函数
        - 若为(n, 4): 每个等级的梯形参数[a,b,c,d]
    membership_func : str, optional
        隶属函数类型，默认 'trapezoidal'
        - 'trapezoidal': 梯形隶属函数

    Returns
    -------
    R : ndarray, shape (m, n)
        模糊关系矩阵

    Examples
    --------
    >>> data = [85, 72, 90]
    >>> levels = [0, 60, 70, 80, 90, 100]  # 差、中、良、优
    >>> R = build_membership_matrix(data, levels)
    """
    data = np.array(data, dtype=float)
    levels = np.array(levels, dtype=float)
    m = len(data)

    if levels.ndim == 1:
        # 从分界点构建梯形参数
        n = len(levels) - 1
        trapezoids = []
        for i in range(n):
            if i == 0:
                trapezoids.append([levels[0], levels[0], levels[1], levels[2]])
            elif i == n - 1:
                trapezoids.append([levels[-3], levels[-2], levels[-1], levels[-1]])
            else:
                trapezoids.append([levels[i], levels[i+1], levels[i+1], levels[i+2]])
        trapezoids = np.array(trapezoids)
    else:
        trapezoids = levels
        n = len(trapezoids)

    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            a, b, c, d = trapezoids[j]
            x = data[i]
            if x <= a or x >= d:
                R[i, j] = 0.0
            elif a < x <= b:
                R[i, j] = (x - a) / (b - a) if b > a else 1.0
            elif b < x <= c:
                R[i, j] = 1.0
            else:  # c < x < d
                R[i, j] = (d - x) / (d - c) if d > c else 1.0

    return R


def multilevel_fuzzy_evaluation(factors, weights_list, R_list, method='M(·,⊕)'):
    """
    多层次模糊综合评价

    Parameters
    ----------
    factors : list of list
        因素层级结构，factors[i]为第i个一级因素包含的二级因素索引
    weights_list : list of array-like
        各层权重，weights_list[0]为一级权重，weights_list[1:]为各一级因素的二级权重
    R_list : list of array-like
        各二级因素的模糊关系矩阵
    method : str, optional
        模糊合成算子，默认 'M(·,⊕)'

    Returns
    -------
    B : ndarray
        最终综合评价向量
    grade : int
        最大隶属度对应的评价等级

    Examples
    --------
    >>> # 2个一级因素，第1个包含2个二级因素，第2个包含1个二级因素
    >>> factors = [[0, 1], [2]]
    >>> weights_list = [[0.6, 0.4], [0.5, 0.5], [1.0]]
    >>> R_list = [
    ...     np.array([[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]]),
    ...     np.array([[0.2, 0.3, 0.5]])
    ... ]
    >>> B, grade = multilevel_fuzzy_evaluation(factors, weights_list, R_list)
    """
    # 一级评价
    B_first = []
    for i, factor_indices in enumerate(factors):
        R_sub = np.vstack([R_list[j] for j in factor_indices])
        w_sub = weights_list[i + 1]
        B_sub, _ = fuzzy_evaluation(R_sub, w_sub, method)
        B_first.append(B_sub)

    # 二级评价
    R_first = np.array(B_first)
    w_first = weights_list[0]
    B, grade = fuzzy_evaluation(R_first, w_first, method)

    return B, grade


if __name__ == '__main__':
    print("=== 模糊综合评价示例 ===\n")

    # 示例1: 基本模糊综合评价
    print("示例1: 学生综合评价")
    R = np.array([
        [0.3, 0.5, 0.2, 0.0],  # 学习成绩
        [0.4, 0.4, 0.1, 0.1],  # 品德表现
        [0.2, 0.3, 0.4, 0.1]   # 体育成绩
    ])
    weights = [0.5, 0.3, 0.2]

    print(f"模糊关系矩阵:\n{R}")
    print(f"权重: {weights}\n")

    for method in ['M(·,⊕)', 'M(∧,∨)', 'M(·,∨)']:
        B, grade = fuzzy_evaluation(R, weights, method)
        print(f"{method}: B = {B.round(3)}, 等级 = {grade}")

    # 示例2: 构建隶属度矩阵
    print("\n示例2: 从实际数据构建隶属度矩阵")
    data = [85, 72, 90]
    levels = [0, 60, 70, 80, 90, 100]
    R_auto = build_membership_matrix(data, levels)
    print(f"实际数据: {data}")
    print(f"等级边界: {levels}")
    print(f"隶属度矩阵:\n{R_auto.round(3)}")

    B, grade = fuzzy_evaluation(R_auto, weights)
    print(f"综合评价: B = {B.round(3)}, 等级 = {grade}")
