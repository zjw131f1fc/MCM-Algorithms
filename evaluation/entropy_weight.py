"""
熵权法 (Entropy Weight Method)

客观赋权方法，根据指标数据的离散程度确定权重。
离散程度越大，熵值越小，权重越大。
"""

import numpy as np


def entropy_weight(data, criteria_types=None):
    """
    熵权法计算指标权重

    Parameters
    ----------
    data : array-like, shape (m, n)
        决策矩阵，m个样本，n个指标
    criteria_types : array-like, shape (n,), optional
        指标类型，1正向（越大越好），-1负向（越小越好）
        默认全为正向指标
        用于正向化处理，确保所有指标越大越好

    Returns
    -------
    weights : ndarray, shape (n,)
        各指标权重，和为1

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([
    ...     [8, 7, 6],
    ...     [7, 8, 8],
    ...     [9, 6, 7],
    ...     [6, 9, 9]
    ... ])
    >>> weights = entropy_weight(data)
    >>> print(f"权重: {weights}")

    >>> # 第三个指标为负向
    >>> weights = entropy_weight(data, criteria_types=[1, 1, -1])

    Notes
    -----
    计算步骤：
    1. 正向化处理（负向指标取倒数或极差变换）
    2. 归一化（每列除以列和）
    3. 计算熵值 e_j = -k * sum(p * ln(p))
    4. 计算权重 w_j = (1 - e_j) / sum(1 - e)
    """
    X = np.array(data, dtype=float)
    m, n = X.shape

    # 正向化处理
    if criteria_types is not None:
        criteria_types = np.array(criteria_types)
        for j in range(n):
            if criteria_types[j] == -1:
                X[:, j] = X[:, j].max() - X[:, j] + X[:, j].min()

    # 归一化（比重）
    X_sum = X.sum(axis=0)
    X_sum[X_sum == 0] = 1
    P = X / X_sum

    # 计算熵值
    k = 1 / np.log(m)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)

    # 计算权重
    D = 1 - E  # 差异系数
    weights = D / D.sum()

    return weights


if __name__ == '__main__':
    data = np.array([
        [8, 7, 6],
        [7, 8, 8],
        [9, 6, 7],
        [6, 9, 9]
    ])

    print("=== 熵权法示例 ===")
    print(f"数据矩阵:\n{data}\n")

    weights = entropy_weight(data)
    print(f"权重: {weights.round(4)}")
    print(f"权重和: {weights.sum():.4f}")
