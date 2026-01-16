"""
AHP (Analytic Hierarchy Process)
层次分析法

主观赋权方法，通过构建判断矩阵，利用成对比较确定权重。
"""

import numpy as np


# 随机一致性指标 RI
RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(judgment_matrix):
    """
    AHP层次分析法计算权重

    Parameters
    ----------
    judgment_matrix : array-like, shape (n, n)
        判断矩阵，满足 a_ij * a_ji = 1
        标度含义：1同等重要，3稍微重要，5明显重要，7强烈重要，9极端重要
        2,4,6,8为相邻标度中间值

    Returns
    -------
    weights : ndarray, shape (n,)
        各指标权重，和为1
    CR : float
        一致性比率，CR < 0.1 表示通过一致性检验
    is_consistent : bool
        是否通过一致性检验

    Examples
    --------
    >>> import numpy as np
    >>> # 3个指标的判断矩阵
    >>> A = np.array([
    ...     [1,   2,   5],
    ...     [1/2, 1,   3],
    ...     [1/5, 1/3, 1]
    ... ])
    >>> weights, CR, is_consistent = ahp(A)
    >>> print(f"权重: {weights}")
    >>> print(f"CR: {CR:.4f}, 一致性: {is_consistent}")

    Notes
    -----
    计算步骤：
    1. 计算判断矩阵最大特征值及对应特征向量
    2. 特征向量归一化得到权重
    3. 计算一致性指标 CI = (λmax - n) / (n - 1)
    4. 计算一致性比率 CR = CI / RI
    """
    A = np.array(judgment_matrix, dtype=float)
    n = A.shape[0]

    # 特征值法求权重
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    weights = eigenvectors[:, max_idx].real
    weights = weights / weights.sum()

    # 一致性检验
    if n <= 2:
        CI = 0
        CR = 0
    else:
        CI = (lambda_max - n) / (n - 1)
        RI = RI_TABLE.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0

    is_consistent = CR < 0.1

    return weights, CR, is_consistent


if __name__ == '__main__':
    # 示例：3个指标的判断矩阵
    A = np.array([
        [1,   2,   5],
        [1/2, 1,   3],
        [1/5, 1/3, 1]
    ])

    print("=== AHP 示例 ===")
    print(f"判断矩阵:\n{A.round(3)}\n")

    weights, CR, is_consistent = ahp(A)
    print(f"权重: {weights.round(4)}")
    print(f"CR: {CR:.4f}")
    print(f"一致性检验: {'通过' if is_consistent else '未通过'}")
