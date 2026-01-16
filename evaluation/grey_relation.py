"""
灰色关联分析 (Grey Relational Analysis)

用于分析各因素与参考序列的关联程度，适用于小样本、贫信息系统。
"""

import numpy as np


def grey_relation(data, reference=None, rho=0.5):
    """
    灰色关联分析

    Parameters
    ----------
    data : array-like, shape (m, n)
        数据矩阵，m个样本，n个指标
    reference : array-like, shape (n,), optional
        参考序列，默认取每列最大值
    rho : float, optional
        分辨系数，范围(0,1)，默认0.5，越小分辨力越强

    Returns
    -------
    scores : ndarray, shape (m,)
        各样本的灰色关联度
    ranks : ndarray, shape (m,)
        排名，1为最优

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([
    ...     [8, 7, 6],
    ...     [7, 8, 8],
    ...     [9, 6, 7],
    ...     [6, 9, 9]
    ... ])
    >>> scores, ranks = grey_relation(data)
    >>> print(f"关联度: {scores}")
    >>> print(f"排名: {ranks}")

    Notes
    -----
    计算步骤：
    1. 无量纲化（均值化或初值化）
    2. 计算差序列 |x0(k) - xi(k)|
    3. 计算关联系数 ξ = (min + ρ*max) / (Δ + ρ*max)
    4. 计算关联度（关联系数均值）
    """
    X = np.array(data, dtype=float)
    m, n = X.shape

    # 参考序列（理想解）
    if reference is None:
        reference = X.max(axis=0)
    reference = np.array(reference, dtype=float)

    # 均值化无量纲化
    X_mean = X.mean(axis=0)
    X_mean[X_mean == 0] = 1
    X_norm = X / X_mean
    ref_norm = reference / X_mean

    # 差序列
    delta = np.abs(X_norm - ref_norm)

    # 两级极差
    delta_min = delta.min()
    delta_max = delta.max()

    # 关联系数
    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max)

    # 关联度（等权平均）
    scores = xi.mean(axis=1)

    # 排名
    ranks = np.argsort(np.argsort(-scores)) + 1

    return scores, ranks


if __name__ == '__main__':
    data = np.array([
        [8, 7, 6],
        [7, 8, 8],
        [9, 6, 7],
        [6, 9, 9]
    ])

    print("=== 灰色关联分析示例 ===")
    print(f"数据矩阵:\n{data}\n")

    scores, ranks = grey_relation(data)
    print(f"关联度: {scores.round(4)}")
    print(f"排名: {ranks}")
