"""
GM(1,1) Grey Prediction Model
灰色预测GM(1,1)模型

适用于小样本、贫信息的短期预测，通过累加生成弱化随机性。
"""

import numpy as np


def gm11(x0, n_predict=1):
    """
    GM(1,1)灰色预测模型

    Parameters
    ----------
    x0 : array-like, shape (n,)
        原始时间序列数据，至少4个点
    n_predict : int, optional
        预测未来的步数，默认1

    Returns
    -------
    x0_pred : ndarray, shape (n,)
        原始序列的拟合值
    x0_forecast : ndarray, shape (n_predict,)
        未来n_predict步的预测值
    params : dict
        模型参数 {'a': 发展系数, 'b': 灰色作用量}

    Examples
    --------
    >>> import numpy as np
    >>> # 原始数据
    >>> x0 = np.array([24.5, 25.8, 27.2, 29.5, 32.1])
    >>> x0_pred, x0_forecast, params = gm11(x0, n_predict=3)
    >>> print(f"拟合值: {x0_pred}")
    >>> print(f"预测值: {x0_forecast}")
    >>> print(f"参数: a={params['a']:.4f}, b={params['b']:.4f}")
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    if n < 4:
        raise ValueError("数据点至少需要4个")

    # 1-AGO累加生成
    x1 = np.cumsum(x0)

    # 紧邻均值生成
    z1 = (x1[:-1] + x1[1:]) / 2

    # 构造数据矩阵B和常数向量Y
    B = np.column_stack([-z1, np.ones(n - 1)])
    Y = x0[1:]

    # 最小二乘求参数 [a, b]
    params_vec = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = params_vec

    # 时间响应函数预测x1
    k = np.arange(n + n_predict)
    x1_pred = (x0[0] - b / a) * np.exp(-a * k) + b / a

    # 累减还原得x0
    x0_all = np.diff(np.concatenate([[0], x1_pred]))

    x0_pred = x0_all[:n]
    x0_forecast = x0_all[n:]

    return x0_pred, x0_forecast, {'a': a, 'b': b}


if __name__ == '__main__':
    # 示例：预测某指标未来趋势
    x0 = np.array([24.5, 25.8, 27.2, 29.5, 32.1])

    print("=== GM(1,1) 灰色预测示例 ===")
    print(f"原始数据: {x0}\n")

    x0_pred, x0_forecast, params = gm11(x0, n_predict=3)

    print(f"模型参数: a={params['a']:.4f}, b={params['b']:.4f}")
    print(f"拟合值: {x0_pred.round(2)}")
    print(f"预测值: {x0_forecast.round(2)}")

    # 计算拟合误差
    error = np.abs(x0 - x0_pred) / x0 * 100
    print(f"\n相对误差(%): {error.round(2)}")
    print(f"平均相对误差: {error.mean():.2f}%")
