"""
集成学习 (Ensemble Learning)

封装 sklearn.ensemble 和 xgboost 的常用集成方法，提供统一接口。

依赖:
    - numpy
    - scikit-learn
    - xgboost

复杂度:
    - 随机森林训练: O(n_trees * n * log(n) * m) 其中n为样本数，m为特征数
    - XGBoost训练: O(n_trees * n * m * log(n))
    - 预测: O(n_trees * depth)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
try:
    import xgboost as xgb
except ImportError:
    xgb = None


def random_forest(X, y, task='classification', n_estimators=100, random_state=None):
    """
    随机森林统一接口

    支持分类和回归任务，基于sklearn.ensemble.RandomForest。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            目标变量
        task : str, default='classification'
            任务类型:
            - 'classification': 分类任务
            - 'regression': 回归任务
        n_estimators : int, default=100
            决策树数量
        random_state : int or None, default=None
            随机种子，用于结果复现

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的模型对象
            - 'predictions': 训练集预测值
            - 'accuracy': 准确率 (仅分类任务)
            - 'r2': R²决定系数 (仅回归任务)
            - 'mse': 均方误差 (仅回归任务)
            - 'feature_importances': 特征重要性

    示例:
        >>> import numpy as np
        >>> from ml.ensemble import random_forest
        >>>
        >>> # 分类任务
        >>> np.random.seed(42)
        >>> X_cls = np.random.randn(100, 4)
        >>> y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
        >>>
        >>> result = random_forest(X_cls, y_cls, task='classification', n_estimators=50)
        >>> print(f"准确率: {result['accuracy']:.4f}")
        >>> print(f"特征重要性: {result['feature_importances']}")
        >>>
        >>> # 回归任务
        >>> X_reg = np.random.randn(100, 3)
        >>> y_reg = 2*X_reg[:, 0] - 3*X_reg[:, 1] + 0.5*X_reg[:, 2]
        >>>
        >>> result = random_forest(X_reg, y_reg, task='regression', n_estimators=50)
        >>> print(f"R²: {result['r2']:.4f}")
        >>> print(f"MSE: {result['mse']:.4f}")
        >>>
        >>> # 预测新数据
        >>> X_new = np.array([[1, 2, 3]])
        >>> y_pred = result['model'].predict(X_new)

    注意:
        - 随机森林对特征缩放不敏感，无需标准化
        - n_estimators越大模型越稳定，但训练时间更长
        - 可通过feature_importances_查看特征重要性
        - 适合处理高维数据和非线性关系
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if task == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy_score(y, y_pred),
            'feature_importances': model.feature_importances_
        }
    elif task == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'model': model,
            'predictions': y_pred,
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'feature_importances': model.feature_importances_
        }
    else:
        raise ValueError(f"不支持的任务类型: {task}. 请选择 'classification' 或 'regression'")


def xgboost(X, y, task='classification', n_estimators=100, learning_rate=0.1):
    """
    XGBoost统一接口

    支持分类和回归任务，基于xgboost库。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            目标变量
        task : str, default='classification'
            任务类型:
            - 'classification': 分类任务
            - 'regression': 回归任务
        n_estimators : int, default=100
            提升树数量
        learning_rate : float, default=0.1
            学习率，控制每棵树的贡献

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的模型对象
            - 'predictions': 训练集预测值
            - 'accuracy': 准确率 (仅分类任务)
            - 'r2': R²决定系数 (仅回归任务)
            - 'mse': 均方误差 (仅回归任务)
            - 'feature_importances': 特征重要性

    示例:
        >>> import numpy as np
        >>> from ml.ensemble import xgboost
        >>>
        >>> # 分类任务
        >>> np.random.seed(42)
        >>> X_cls = np.random.randn(100, 4)
        >>> y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
        >>>
        >>> result = xgboost(X_cls, y_cls, task='classification', n_estimators=50, learning_rate=0.1)
        >>> print(f"准确率: {result['accuracy']:.4f}")
        >>> print(f"特征重要性: {result['feature_importances']}")
        >>>
        >>> # 回归任务
        >>> X_reg = np.random.randn(100, 3)
        >>> y_reg = 2*X_reg[:, 0] - 3*X_reg[:, 1] + 0.5*X_reg[:, 2]
        >>>
        >>> result = xgboost(X_reg, y_reg, task='regression', n_estimators=50, learning_rate=0.1)
        >>> print(f"R²: {result['r2']:.4f}")
        >>> print(f"MSE: {result['mse']:.4f}")
        >>>
        >>> # 预测新数据
        >>> X_new = np.array([[1, 2, 3]])
        >>> y_pred = result['model'].predict(X_new)

    注意:
        - XGBoost通常比随机森林更准确但训练时间更长
        - learning_rate越小需要更多n_estimators，但泛化能力更强
        - 建议对特征进行标准化以提升性能
        - 支持GPU加速(需安装GPU版本xgboost)
        - 需要先安装xgboost: pip install xgboost
    """
    if xgb is None:
        raise ImportError("需要安装xgboost库: pip install xgboost")

    X = np.asarray(X)
    y = np.asarray(y)

    if task == 'classification':
        model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy_score(y, y_pred),
            'feature_importances': model.feature_importances_
        }
    elif task == 'regression':
        model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X, y)
        y_pred = model.predict(X)
        return {
            'model': model,
            'predictions': y_pred,
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'feature_importances': model.feature_importances_
        }
    else:
        raise ValueError(f"不支持的任务类型: {task}. 请选择 'classification' 或 'regression'")


if __name__ == '__main__':
    # 示例：比较随机森林和XGBoost
    np.random.seed(42)

    # 分类任务
    X_cls = np.random.randn(200, 5)
    y_cls = ((X_cls[:, 0] + X_cls[:, 1] - X_cls[:, 2]) > 0).astype(int)

    print("=" * 60)
    print("分类任务示例")
    print("=" * 60)

    rf_result = random_forest(X_cls, y_cls, task='classification', n_estimators=50, random_state=42)
    print(f"随机森林准确率: {rf_result['accuracy']:.4f}")
    print(f"特征重要性: {rf_result['feature_importances']}")
    print()

    if xgb is not None:
        xgb_result = xgboost(X_cls, y_cls, task='classification', n_estimators=50, learning_rate=0.1)
        print(f"XGBoost准确率: {xgb_result['accuracy']:.4f}")
        print(f"特征重要性: {xgb_result['feature_importances']}")
    print()

    # 回归任务
    X_reg = np.random.randn(200, 4)
    y_reg = 3*X_reg[:, 0] - 2*X_reg[:, 1] + X_reg[:, 2] + np.random.randn(200)*0.5

    print("=" * 60)
    print("回归任务示例")
    print("=" * 60)

    rf_result = random_forest(X_reg, y_reg, task='regression', n_estimators=50, random_state=42)
    print(f"随机森林 R²: {rf_result['r2']:.4f}, MSE: {rf_result['mse']:.4f}")
    print()

    if xgb is not None:
        xgb_result = xgboost(X_reg, y_reg, task='regression', n_estimators=50, learning_rate=0.1)
        print(f"XGBoost R²: {xgb_result['r2']:.4f}, MSE: {xgb_result['mse']:.4f}")
