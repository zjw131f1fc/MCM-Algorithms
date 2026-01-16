"""
蚁群算法 (Ant Colony Optimization, ACO) - TSP问题求解

用于求解旅行商问题(Traveling Salesman Problem)的启发式优化算法。
通过模拟蚂蚁觅食行为，利用信息素机制寻找最优路径。
"""

import numpy as np


def aco_tsp(dist_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0,
            rho=0.5, Q=100, seed=None):
    """
    蚁群算法求解TSP问题

    Parameters
    ----------
    dist_matrix : ndarray
        距离矩阵 (n_cities × n_cities)，dist_matrix[i][j]表示城市i到j的距离
    n_ants : int, optional
        蚂蚁数量，默认20
    n_iterations : int, optional
        迭代次数，默认100
    alpha : float, optional
        信息素重要程度因子，默认1.0
    beta : float, optional
        启发函数重要程度因子，默认2.0
    rho : float, optional
        信息素挥发系数 (0-1)，默认0.5
    Q : float, optional
        信息素强度，默认100
    seed : int, optional
        随机种子

    Returns
    -------
    best_path : list
        最优路径（城市索引序列）
    best_distance : float
        最优路径长度
    history : list
        每次迭代的最优距离

    Examples
    --------
    >>> dist = np.array([[0, 2, 3, 4],
    ...                  [2, 0, 4, 5],
    ...                  [3, 4, 0, 3],
    ...                  [4, 5, 3, 0]])
    >>> path, dist, history = aco_tsp(dist, n_ants=10, n_iterations=50)
    >>> print(f"最优路径: {path}, 距离: {dist:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    n_cities = len(dist_matrix)
    pheromone = np.ones((n_cities, n_cities))
    eta = 1.0 / (dist_matrix + np.eye(n_cities))  # 启发信息(能见度)

    best_path = None
    best_distance = float('inf')
    history = []

    for iteration in range(n_iterations):
        paths = []
        distances = []

        for ant in range(n_ants):
            path = _construct_path(pheromone, eta, alpha, beta, n_cities)
            distance = _calculate_distance(path, dist_matrix)
            paths.append(path)
            distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_path = path

        # 更新信息素
        pheromone *= (1 - rho)  # 挥发
        for path, distance in zip(paths, distances):
            for i in range(n_cities):
                j = (i + 1) % n_cities
                pheromone[path[i]][path[j]] += Q / distance

        history.append(best_distance)

    return best_path, best_distance, history


def _construct_path(pheromone, eta, alpha, beta, n_cities):
    """构造单只蚂蚁的路径"""
    path = [np.random.randint(n_cities)]
    unvisited = set(range(n_cities)) - {path[0]}

    while unvisited:
        current = path[-1]
        probs = []
        cities = list(unvisited)

        for city in cities:
            prob = (pheromone[current][city] ** alpha) * (eta[current][city] ** beta)
            probs.append(prob)

        probs = np.array(probs)
        probs /= probs.sum()

        next_city = np.random.choice(cities, p=probs)
        path.append(next_city)
        unvisited.remove(next_city)

    return path


def _calculate_distance(path, dist_matrix):
    """计算路径总距离"""
    distance = sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
    distance += dist_matrix[path[-1]][path[0]]  # 回到起点
    return distance


if __name__ == '__main__':
    # 示例：5城市TSP问题
    dist = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ])

    print("蚁群算法求解TSP问题")
    print(f"城市数量: {len(dist)}")
    print(f"距离矩阵:\n{dist}\n")

    path, distance, history = aco_tsp(
        dist,
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        seed=42
    )

    print(f"最优路径: {path}")
    print(f"最优距离: {distance:.2f}")
    print(f"初始距离: {history[0]:.2f}")
    print(f"最终距离: {history[-1]:.2f}")
    print(f"改进率: {(history[0]-history[-1])/history[0]*100:.1f}%")
