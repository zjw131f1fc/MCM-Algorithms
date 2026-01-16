"""
DEA (Data Envelopment Analysis)
数据包络分析

用于评价多投入多产出决策单元（DMU）的相对效率，基于线性规划求解。
"""

import numpy as np
from scipy.optimize import linprog


def dea_ccr(inputs, outputs, orientation='input'):
    """
    DEA-CCR模型（规模报酬不变）

    Parameters
    ----------
    inputs : array-like, shape (n_dmus, n_inputs)
        投入指标矩阵，n_dmus个决策单元，n_inputs个投入指标
    outputs : array-like, shape (n_dmus, n_outputs)
        产出指标矩阵，n_dmus个决策单元，n_outputs个产出指标
    orientation : str, optional
        模型导向，'input'（投入导向）或 'output'（产出导向）
        默认 'input'

    Returns
    -------
    efficiencies : ndarray, shape (n_dmus,)
        各决策单元的效率值，范围(0,1]，1表示有效
    is_efficient : ndarray, shape (n_dmus,)
        各决策单元是否有效（效率值=1）

    Examples
    --------
    >>> import numpy as np
    >>> # 5个决策单元，2个投入，1个产出
    >>> inputs = np.array([
    ...     [3, 5],
    ...     [4, 3],
    ...     [5, 4],
    ...     [2, 6],
    ...     [6, 2]
    ... ])
    >>> outputs = np.array([
    ...     [8],
    ...     [7],
    ...     [9],
    ...     [6],
    ...     [8]
    ... ])
    >>> efficiencies, is_efficient = dea_ccr(inputs, outputs)
    >>> print(f"效率值: {efficiencies}")
    >>> print(f"有效单元: {np.where(is_efficient)[0]}")

    Notes
    -----
    CCR模型假设规模报酬不变（CRS），适用于评价整体效率。
    投入导向：在产出不变下最小化投入
    产出导向：在投入不变下最大化产出
    """
    X = np.array(inputs, dtype=float)
    Y = np.array(outputs, dtype=float)
    n_dmus, n_inputs = X.shape
    n_outputs = Y.shape[1]

    efficiencies = np.zeros(n_dmus)

    for k in range(n_dmus):
        if orientation == 'input':
            # 投入导向：min θ
            # s.t. θX_k - Xλ >= 0, Yλ >= Y_k, λ >= 0
            c = np.concatenate([[1], np.zeros(n_dmus)])
            A_ub = np.vstack([
                np.hstack([-X[k:k+1].T, X.T]),
                np.hstack([np.zeros((n_outputs, 1)), -Y.T])
            ])
            b_ub = np.concatenate([np.zeros(n_inputs), -Y[k]])
            bounds = [(0, None)] + [(0, None)] * n_dmus

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            efficiencies[k] = res.fun if res.success else 0

        else:  # output
            # 产出导向：max φ
            # s.t. Xλ <= X_k, φY_k - Yλ <= 0, λ >= 0
            c = np.concatenate([[-1], np.zeros(n_dmus)])
            A_ub = np.vstack([
                np.hstack([np.zeros((n_inputs, 1)), X.T]),
                np.hstack([Y[k:k+1].T, -Y.T])
            ])
            b_ub = np.concatenate([X[k], np.zeros(n_outputs)])
            bounds = [(0, None)] + [(0, None)] * n_dmus

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            efficiencies[k] = 1 / (-res.fun) if res.success and res.fun < 0 else 0

    is_efficient = np.isclose(efficiencies, 1, atol=1e-6)

    return efficiencies, is_efficient


if __name__ == '__main__':
    # 示例：评价5个决策单元
    inputs = np.array([
        [3, 5],
        [4, 3],
        [5, 4],
        [2, 6],
        [6, 2]
    ])
    outputs = np.array([
        [8],
        [7],
        [9],
        [6],
        [8]
    ])

    print("=== DEA-CCR 示例 ===")
    print(f"投入矩阵:\n{inputs}\n")
    print(f"产出矩阵:\n{outputs}\n")

    # 投入导向
    efficiencies, is_efficient = dea_ccr(inputs, outputs, orientation='input')
    print("投入导向:")
    for i, (eff, is_eff) in enumerate(zip(efficiencies, is_efficient)):
        status = "有效" if is_eff else "无效"
        print(f"DMU{i+1}: 效率={eff:.4f} ({status})")

    print(f"\n有效单元: DMU{np.where(is_efficient)[0] + 1}")
