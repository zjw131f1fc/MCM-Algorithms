"""
回归分析 (Regression Analysis)

封装 sklearn.linear_model 的常用回归方法，提供统一接口。

依赖:
    - numpy
    - scikit-learn

复杂度:
    - 训练: O(n*p^2) 其中n为样本数，p为特征数
    - 预测: O(p)
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


def linear_regression(X, y, method='ols', alpha=1.0):
    """
    线性回归统一接口

    支持普通最小二乘(OLS)、岭回归(Ridge)、Lasso回归三种方法。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            目标变量
        method : str, default='ols'
            回归方法:
            - 'ols': 普通最小二乘回归
            - 'ridge': 岭回归 (L2正则化)
            - 'lasso': Lasso回归 (L1正则化)
        alpha : float, default=1.0
            正则化强度 (仅用于ridge和lasso)
            - alpha越大，正则化越强
            - 对ols无效

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的模型对象
            - 'coef': 回归系数 (shape: n_features,)
            - 'intercept': 截距项
            - 'r2': R²决定系数
            - 'mse': 均方误差
            - 'predictions': 训练集预测值

    示例:
        >>> import numpy as np
        >>> from ml.regression import linear_regression
        >>>
        >>> # 生成示例数据
        >>> np.random.seed(42)
        >>> X = np.random.randn(100, 3)
        >>> y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.1
        >>>
        >>> # OLS回归
        >>> result = linear_regression(X, y, method='ols')
        >>> print(f"R²: {result['r2']:.4f}")
        >>> print(f"系数: {result['coef']}")
        >>>
        >>> # 岭回归
        >>> result_ridge = linear_regression(X, y, method='ridge', alpha=1.0)
        >>> print(f"Ridge R²: {result_ridge['r2']:.4f}")
        >>>
        >>> # Lasso回归
        >>> result_lasso = linear_regression(X, y, method='lasso', alpha=0.1)
        >>> print(f"Lasso R²: {result_lasso['r2']:.4f}")
        >>>
        >>> # 预测新数据
        >>> X_new = np.array([[1, 2, 3]])
        >>> y_pred = result['model'].predict(X_new)
        >>> print(f"预测值: {y_pred[0]:.4f}")

    注意:
        - 建议在使用前对数据进行标准化，特别是使用正则化方法时
        - Ridge适合特征间存在多重共线性的情况
        - Lasso可以进行特征选择，将不重要特征的系数压缩为0
        - alpha参数需要通过交叉验证选择最优值
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if method == 'ols':
        model = LinearRegression()
    elif method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError(f"不支持的方法: {method}. 请选择 'ols', 'ridge' 或 'lasso'")

    model.fit(X, y)
    y_pred = model.predict(X)

    return {
        'model': model,
        'coef': model.coef_,
        'intercept': model.intercept_,
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'predictions': y_pred
    }


if __name__ == '__main__':
    # 示例：比较三种回归方法
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_coef = np.array([2, -3, 0.5, 0, 1.5])
    y = X @ true_coef + np.random.randn(100) * 0.5

    print("=" * 60)
    print("回归分析示例")
    print("=" * 60)
    print(f"真实系数: {true_coef}")
    print()

    for method in ['ols', 'ridge', 'lasso']:
        result = linear_regression(X, y, method=method, alpha=1.0)
        print(f"{method.upper()} 回归:")
        print(f"  系数: {result['coef']}")
        print(f"  截距: {result['intercept']:.4f}")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  MSE: {result['mse']:.4f}")
        print()
