"""
聚类分析 (Clustering Analysis)

封装 sklearn.cluster 的常用聚类方法，提供统一接口。

依赖:
    - numpy
    - scikit-learn

复杂度:
    - K-Means: O(n*k*i*d) 其中n为样本数，k为簇数，i为迭代次数，d为特征维度
    - DBSCAN: O(n*log(n)) 使用空间索引
    - 层次聚类: O(n²*log(n)) 或 O(n³) 取决于linkage方法
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


def kmeans(X, n_clusters=3, random_state=None):
    """
    K-Means聚类

    基于距离的聚类算法，将数据划分为k个簇，使簇内平方和最小。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        n_clusters : int, default=3
            簇的数量
        random_state : int, default=None
            随机种子，用于可重复性

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的KMeans模型对象
            - 'labels': 每个样本的簇标签 (shape: n_samples,)
            - 'centers': 簇中心坐标 (shape: n_clusters, n_features)
            - 'inertia': 簇内平方和
            - 'n_iter': 迭代次数

    示例:
        >>> import numpy as np
        >>> from ml.clustering import kmeans
        >>>
        >>> # 生成示例数据
        >>> np.random.seed(42)
        >>> X = np.vstack([
        ...     np.random.randn(50, 2) + [2, 2],
        ...     np.random.randn(50, 2) + [-2, -2],
        ...     np.random.randn(50, 2) + [2, -2]
        ... ])
        >>>
        >>> # K-Means聚类
        >>> result = kmeans(X, n_clusters=3, random_state=42)
        >>> print(f"簇内平方和: {result['inertia']:.2f}")
        >>> print(f"迭代次数: {result['n_iter']}")
        >>> print(f"簇中心:\n{result['centers']}")
        >>>
        >>> # 预测新数据
        >>> X_new = np.array([[2, 2], [-2, -2]])
        >>> labels_new = result['model'].predict(X_new)
        >>> print(f"新数据簇标签: {labels_new}")

    注意:
        - 需要预先指定簇数k，可用肘部法则或轮廓系数选择
        - 对初始中心敏感，建议设置random_state保证可重复性
        - 假设簇为凸形且大小相近，不适合非凸簇
        - 建议对数据进行标准化处理
    """
    X = np.asarray(X)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    return {
        'model': model,
        'labels': labels,
        'centers': model.cluster_centers_,
        'inertia': model.inertia_,
        'n_iter': model.n_iter_
    }


def dbscan(X, eps=0.5, min_samples=5):
    """
    DBSCAN密度聚类

    基于密度的聚类算法，能发现任意形状的簇，自动识别噪声点。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        eps : float, default=0.5
            邻域半径，两点被视为邻居的最大距离
        min_samples : int, default=5
            核心点的最小邻居数（包括自身）

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的DBSCAN模型对象
            - 'labels': 每个样本的簇标签 (shape: n_samples,)
                       -1表示噪声点，>=0表示簇编号
            - 'n_clusters': 发现的簇数量（不含噪声）
            - 'n_noise': 噪声点数量
            - 'core_sample_indices': 核心样本的索引

    示例:
        >>> import numpy as np
        >>> from ml.clustering import dbscan
        >>>
        >>> # 生成示例数据（包含噪声）
        >>> np.random.seed(42)
        >>> X = np.vstack([
        ...     np.random.randn(50, 2) * 0.3 + [0, 0],
        ...     np.random.randn(50, 2) * 0.3 + [3, 3],
        ...     np.random.uniform(-2, 5, (10, 2))  # 噪声点
        ... ])
        >>>
        >>> # DBSCAN聚类
        >>> result = dbscan(X, eps=0.5, min_samples=5)
        >>> print(f"发现簇数: {result['n_clusters']}")
        >>> print(f"噪声点数: {result['n_noise']}")
        >>> print(f"核心点数: {len(result['core_sample_indices'])}")
        >>>
        >>> # 查看各簇样本数
        >>> labels = result['labels']
        >>> for i in range(result['n_clusters']):
        ...     n_samples = np.sum(labels == i)
        ...     print(f"簇 {i}: {n_samples} 个样本")

    注意:
        - 无需预先指定簇数，自动发现簇
        - eps和min_samples需要根据数据调整，可用k-距离图辅助选择
        - 对不同密度的簇效果可能不佳
        - 标签-1表示噪声点/离群点
        - 适合发现任意形状的簇
    """
    X = np.asarray(X)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    return {
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'core_sample_indices': model.core_sample_indices_
    }


def hierarchical_clustering(X, n_clusters=3, linkage='ward'):
    """
    层次聚类

    自底向上的聚合聚类算法，构建树状聚类结构。

    参数:
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        n_clusters : int, default=3
            最终簇的数量
        linkage : str, default='ward'
            链接准则:
            - 'ward': 最小化簇内方差（仅适用于欧氏距离）
            - 'complete': 最大距离（最远邻）
            - 'average': 平均距离
            - 'single': 最小距离（最近邻）

    返回:
        dict : 包含以下键值:
            - 'model': 训练好的AgglomerativeClustering模型对象
            - 'labels': 每个样本的簇标签 (shape: n_samples,)
            - 'n_clusters': 簇的数量
            - 'n_leaves': 叶节点数量
            - 'children': 层次树结构 (shape: n_samples-1, 2)

    示例:
        >>> import numpy as np
        >>> from ml.clustering import hierarchical_clustering
        >>>
        >>> # 生成示例数据
        >>> np.random.seed(42)
        >>> X = np.vstack([
        ...     np.random.randn(30, 2) * 0.5 + [0, 0],
        ...     np.random.randn(30, 2) * 0.5 + [3, 0],
        ...     np.random.randn(30, 2) * 0.5 + [1.5, 2.5]
        ... ])
        >>>
        >>> # 层次聚类
        >>> result = hierarchical_clustering(X, n_clusters=3, linkage='ward')
        >>> print(f"簇数: {result['n_clusters']}")
        >>> print(f"各簇样本数: {np.bincount(result['labels'])}")
        >>>
        >>> # 尝试不同链接方法
        >>> for method in ['ward', 'complete', 'average', 'single']:
        ...     result = hierarchical_clustering(X, n_clusters=3, linkage=method)
        ...     print(f"{method}: {np.bincount(result['labels'])}")

    注意:
        - 需要预先指定簇数，但可通过树状图辅助决策
        - ward方法通常效果最好，但仅适用于欧氏距离
        - complete和average对离群点较鲁棒
        - single容易产生链式效应
        - 时间复杂度较高，不适合大规模数据
    """
    X = np.asarray(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    return {
        'model': model,
        'labels': labels,
        'n_clusters': model.n_clusters_,
        'n_leaves': model.n_leaves_,
        'children': model.children_
    }


if __name__ == '__main__':
    # 示例：比较三种聚类方法
    np.random.seed(42)

    # 生成三簇数据
    X = np.vstack([
        np.random.randn(50, 2) * 0.5 + [0, 0],
        np.random.randn(50, 2) * 0.5 + [3, 0],
        np.random.randn(50, 2) * 0.5 + [1.5, 2.5]
    ])

    print("=" * 60)
    print("聚类分析示例")
    print("=" * 60)
    print(f"数据形状: {X.shape}")
    print()

    # K-Means
    result_km = kmeans(X, n_clusters=3, random_state=42)
    print("K-Means聚类:")
    print(f"  簇内平方和: {result_km['inertia']:.2f}")
    print(f"  迭代次数: {result_km['n_iter']}")
    print(f"  各簇样本数: {np.bincount(result_km['labels'])}")
    print()

    # DBSCAN
    result_db = dbscan(X, eps=0.5, min_samples=5)
    print("DBSCAN聚类:")
    print(f"  发现簇数: {result_db['n_clusters']}")
    print(f"  噪声点数: {result_db['n_noise']}")
    print(f"  核心点数: {len(result_db['core_sample_indices'])}")
    if result_db['n_clusters'] > 0:
        print(f"  各簇样本数: {[np.sum(result_db['labels'] == i) for i in range(result_db['n_clusters'])]}")
    print()

    # 层次聚类
    result_hc = hierarchical_clustering(X, n_clusters=3, linkage='ward')
    print("层次聚类 (Ward):")
    print(f"  簇数: {result_hc['n_clusters']}")
    print(f"  各簇样本数: {np.bincount(result_hc['labels'])}")
    print()
