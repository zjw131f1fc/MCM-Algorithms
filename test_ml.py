"""测试机器学习算法"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from ml.regression import linear_regression
from ml.clustering import kmeans, dbscan, hierarchical_clustering
from ml.ensemble import random_forest, xgboost

print("=" * 50)
print("测试机器学习算法")
print("=" * 50)

# 1. 线性回归
print("\n1. 线性回归")
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
result = linear_regression(X, y, method='ols')
print(f"  OLS回归:")
print(f"    系数: {result['coef']}")
print(f"    截距: {result['intercept']:.4f}")
print(f"    R^2: {result['r2']:.4f}")
print(f"    MSE: {result['mse']:.4f}")

result = linear_regression(X, y, method='ridge', alpha=1.0)
print(f"  Ridge回归:")
print(f"    R^2: {result['r2']:.4f}")
print(f"    MSE: {result['mse']:.4f}")

# 2. K-Means聚类
print("\n2. K-Means聚类")
X, _ = make_classification(n_samples=150, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
result = kmeans(X, n_clusters=3, random_state=42)
print(f"  簇标签: {np.unique(result['labels'], return_counts=True)}")
print(f"  簇中心:\n{result['centers']}")
print(f"  簇内平方和: {result['inertia']:.4f}")
print(f"  迭代次数: {result['n_iter']}")

# 3. DBSCAN聚类
print("\n3. DBSCAN聚类")
result = dbscan(X, eps=0.5, min_samples=5)
print(f"  簇数量: {result['n_clusters']}")
print(f"  噪声点数: {result['n_noise']}")
print(f"  簇标签分布: {np.unique(result['labels'], return_counts=True)}")

# 4. 层次聚类
print("\n4. 层次聚类")
result = hierarchical_clustering(X, n_clusters=3, linkage='ward')
print(f"  簇数量: {result['n_clusters']}")
print(f"  簇标签分布: {np.unique(result['labels'], return_counts=True)}")
print(f"  叶节点数: {result['n_leaves']}")

# 5. 随机森林
print("\n5. 随机森林")
X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                           n_redundant=2, random_state=42)
result = random_forest(X, y, task='classification', n_estimators=50, random_state=42)
print(f"  分类准确率: {result['accuracy']:.4f}")
print(f"  特征重要性(前5): {result['feature_importances'][:5]}")

X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
result = random_forest(X, y, task='regression', n_estimators=50, random_state=42)
print(f"  回归R^2: {result['r2']:.4f}")
print(f"  回归MSE: {result['mse']:.4f}")

# 6. XGBoost
print("\n6. XGBoost")
X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                           n_redundant=2, random_state=42)
result = xgboost(X, y, task='classification', n_estimators=50, learning_rate=0.1)
print(f"  分类准确率: {result['accuracy']:.4f}")
print(f"  特征重要性(前5): {result['feature_importances'][:5]}")

X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
result = xgboost(X, y, task='regression', n_estimators=50, learning_rate=0.1)
print(f"  回归R^2: {result['r2']:.4f}")
print(f"  回归MSE: {result['mse']:.4f}")

print("\n" + "=" * 50)
print("机器学习算法测试完成")
print("=" * 50)
