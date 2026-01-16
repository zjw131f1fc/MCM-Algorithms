"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
逼近理想解排序法

用于多属性决策问题，通过计算各方案与正负理想解的距离进行排序。
"""

import numpy as np


def topsis(data, weights=None, criteria_types=None, normalize='vector'):
    """
    TOPSIS评价方法

    Parameters
    ----------
    data : array-like, shape (m, n)
        决策矩阵，m个方案，n个指标
    weights : array-like, shape (n,), optional
        权重向量，默认等权重
    criteria_types : array-like, shape (n,), optional
        指标类型，1表示正向指标（越大越好），-1表示负向指标（越小越好）
        默认全为正向指标
    normalize : str, optional
        标准化方法，'vector'（向量标准化）或 'minmax'（极差标准化）
        默认 'vector'

    Returns
    -------
    scores : ndarray, shape (m,)
        各方案的贴近度得分，范围[0,1]，越大越优
    ranks : ndarray, shape (m,)
        各方案排名，1为最优

    Examples
    --------
    >>> import numpy as np
    >>> # 4个方案，3个指标
    >>> data = np.array([
    ...     [8, 7, 6],
    ...     [7, 8, 8],
    ...     [9, 6, 7],
    ...     [6, 9, 9]
    ... ])
    >>> scores, ranks = topsis(data)
    >>> print(f"得分: {scores}")
    >>> print(f"排名: {ranks}")

    >>> # 指定权重和指标类型
    >>> weights = [0.4, 0.3, 0.3]
    >>> criteria_types = [1, 1, -1]  # 前两个越大越好，第三个越小越好
    >>> scores, ranks = topsis(data, weights, criteria_types)
    """
    # 转换为numpy数组
    X = np.array(data, dtype=float)
    m, n = X.shape

    # 默认参数
    if weights is None:
        weights = np.ones(n) / n
    weights = np.array(weights)

    if criteria_types is None:
        criteria_types = np.ones(n)
    criteria_types = np.array(criteria_types)

    # 标准化
    if normalize == 'vector':
        norm = np.sqrt(np.sum(X ** 2, axis=0))
        norm[norm == 0] = 1  # 避免除零
        X_norm = X / norm
    elif normalize == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        denom = X_max - X_min
        denom[denom == 0] = 1
        X_norm = (X - X_min) / denom
    else:
        raise ValueError(f"未知的标准化方法: {normalize}")

    # 加权
    X_weighted = X_norm * weights

    # 确定正负理想解
    positive_ideal = np.where(criteria_types == 1, X_weighted.max(axis=0), X_weighted.min(axis=0))
    negative_ideal = np.where(criteria_types == 1, X_weighted.min(axis=0), X_weighted.max(axis=0))

    # 计算到正负理想解的距离
    d_positive = np.sqrt(np.sum((X_weighted - positive_ideal) ** 2, axis=1))
    d_negative = np.sqrt(np.sum((X_weighted - negative_ideal) ** 2, axis=1))

    # 计算贴近度
    denom = d_positive + d_negative
    denom[denom == 0] = 1
    scores = d_negative / denom

    # 计算排名
    ranks = np.argsort(np.argsort(-scores)) + 1

    return scores, ranks


if __name__ == '__main__':
    # 示例：评价4个方案
    data = np.array([
        [8, 7, 6],
        [7, 8, 8],
        [9, 6, 7],
        [6, 9, 9]
    ])

    print("=== TOPSIS 示例 ===")
    print(f"决策矩阵:\n{data}\n")

    # 等权重，全正向
    scores, ranks = topsis(data)
    print("等权重，全正向指标:")
    print(f"得分: {scores.round(4)}")
    print(f"排名: {ranks}\n")

    # 自定义权重和指标类型
    weights = [0.4, 0.3, 0.3]
    criteria_types = [1, 1, -1]
    scores, ranks = topsis(data, weights, criteria_types)
    print(f"权重: {weights}, 指标类型: {criteria_types}")
    print(f"得分: {scores.round(4)}")
    print(f"排名: {ranks}")
