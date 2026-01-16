"""
NetworkX 图论算法封装

封装 networkx 常用图算法，提供简洁接口。
"""

import networkx as nx
from typing import Union, Optional, Dict, Any, Tuple


def dijkstra(G: nx.Graph, source: Any, target: Optional[Any] = None) -> Union[Dict, Tuple[list, float]]:
    """
    Dijkstra 最短路径算法

    计算从源节点到目标节点的最短路径，或到所有节点的最短路径。

    Parameters
    ----------
    G : nx.Graph
        输入图，边权重通过 'weight' 属性指定，默认为1
    source : Any
        源节点
    target : Any, optional
        目标节点。若为 None，返回到所有节点的最短路径长度

    Returns
    -------
    Union[Dict, Tuple[list, float]]
        - 若 target 为 None: 返回 dict {节点: 最短距离}
        - 若指定 target: 返回 (path, length)，路径节点列表和路径长度

    Examples
    --------
    >>> import networkx as nx
    >>> from graph.network_algorithms import dijkstra
    >>>
    >>> # 创建加权图
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([
    ...     (0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)
    ... ])
    >>>
    >>> # 单源最短路径
    >>> distances = dijkstra(G, source=0)
    >>> print(distances)  # {0: 0, 1: 3, 2: 1, 3: 4}
    >>>
    >>> # 指定目标节点
    >>> path, length = dijkstra(G, source=0, target=3)
    >>> print(path)    # [0, 2, 1, 3]
    >>> print(length)  # 4

    Notes
    -----
    - 时间复杂度: O((V + E) log V)，V为节点数，E为边数
    - 不支持负权边，负权边请使用 Bellman-Ford 算法
    - 基于 networkx.dijkstra_path 和 networkx.single_source_dijkstra_path_length
    """
    if target is None:
        return nx.single_source_dijkstra_path_length(G, source, weight='weight')
    else:
        path = nx.dijkstra_path(G, source, target, weight='weight')
        length = nx.dijkstra_path_length(G, source, target, weight='weight')
        return path, length


def minimum_spanning_tree(G: nx.Graph, algorithm: str = 'kruskal') -> nx.Graph:
    """
    最小生成树 (Minimum Spanning Tree, MST)

    计算无向连通图的最小生成树，使所有节点连通且边权重和最小。

    Parameters
    ----------
    G : nx.Graph
        输入无向图，边权重通过 'weight' 属性指定，默认为1
    algorithm : str, default='kruskal'
        算法选择: 'kruskal' | 'prim'

    Returns
    -------
    nx.Graph
        最小生成树图对象，包含原图的节点和选中的边

    Examples
    --------
    >>> import networkx as nx
    >>> from graph.network_algorithms import minimum_spanning_tree
    >>>
    >>> # 创建加权图
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([
    ...     (0, 1, 4), (0, 2, 1), (1, 2, 2), (1, 3, 5), (2, 3, 3)
    ... ])
    >>>
    >>> # 计算最小生成树
    >>> mst = minimum_spanning_tree(G)
    >>> print(list(mst.edges(data=True)))
    >>> # [(0, 2, {'weight': 1}), (1, 2, {'weight': 2}), (2, 3, {'weight': 3})]
    >>>
    >>> # 计算总权重
    >>> total_weight = sum(d['weight'] for u, v, d in mst.edges(data=True))
    >>> print(total_weight)  # 6

    Notes
    -----
    - 时间复杂度: Kruskal O(E log E)，Prim O(E log V)
    - 仅适用于无向连通图
    - 基于 networkx.minimum_spanning_tree
    """
    return nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)


def floyd_warshall(G: nx.Graph) -> Dict[Any, Dict[Any, float]]:
    """
    Floyd-Warshall 全源最短路径算法

    计算图中所有节点对之间的最短路径长度。

    Parameters
    ----------
    G : nx.Graph
        输入图，边权重通过 'weight' 属性指定，默认为1

    Returns
    -------
    Dict[Any, Dict[Any, float]]
        嵌套字典 {源节点: {目标节点: 最短距离}}

    Examples
    --------
    >>> import networkx as nx
    >>> from graph.network_algorithms import floyd_warshall
    >>>
    >>> # 创建有向图
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([
    ...     (0, 1, 3), (0, 2, 8), (1, 2, 1), (2, 3, 2), (1, 3, 5)
    ... ])
    >>>
    >>> # 计算全源最短路径
    >>> distances = floyd_warshall(G)
    >>> print(distances[0][3])  # 0到3的最短距离: 6
    >>>
    >>> # 遍历所有节点对
    >>> for u in G.nodes():
    ...     for v in G.nodes():
    ...         if u != v:
    ...             print(f"{u} -> {v}: {distances[u][v]}")

    Notes
    -----
    - 时间复杂度: O(V³)，V为节点数
    - 空间复杂度: O(V²)
    - 支持负权边，但不支持负权环
    - 适用于稠密图，稀疏图建议多次调用 Dijkstra
    - 基于 networkx.floyd_warshall
    """
    return dict(nx.floyd_warshall(G, weight='weight'))


def maximum_flow(G: nx.DiGraph, source: Any, target: Any,
                 capacity: str = 'capacity') -> Tuple[float, Dict]:
    """
    最大流算法

    计算从源节点到汇节点的最大流量。

    Parameters
    ----------
    G : nx.DiGraph
        输入有向图，边容量通过 capacity 参数指定的属性名获取
    source : Any
        源节点（流的起点）
    target : Any
        汇节点（流的终点）
    capacity : str, default='capacity'
        边容量属性名，默认为 'capacity'

    Returns
    -------
    Tuple[float, Dict]
        (max_flow_value, flow_dict)
        - max_flow_value: 最大流量值
        - flow_dict: 流量分配字典 {u: {v: flow}}

    Examples
    --------
    >>> import networkx as nx
    >>> from graph.network_algorithms import maximum_flow
    >>>
    >>> # 创建流网络
    >>> G = nx.DiGraph()
    >>> G.add_edge(0, 1, capacity=10)
    >>> G.add_edge(0, 2, capacity=5)
    >>> G.add_edge(1, 2, capacity=15)
    >>> G.add_edge(1, 3, capacity=10)
    >>> G.add_edge(2, 3, capacity=10)
    >>>
    >>> # 计算最大流
    >>> max_flow_value, flow_dict = maximum_flow(G, source=0, target=3)
    >>> print(f"最大流量: {max_flow_value}")  # 15
    >>>
    >>> # 查看流量分配
    >>> for u in flow_dict:
    ...     for v in flow_dict[u]:
    ...         if flow_dict[u][v] > 0:
    ...             print(f"{u} -> {v}: {flow_dict[u][v]}")

    Notes
    -----
    - 时间复杂度: O(V²E)，使用 Edmonds-Karp 算法
    - 仅适用于有向图
    - 容量必须为非负数
    - 基于 networkx.maximum_flow (默认使用 Edmonds-Karp 算法)
    """
    max_flow_value, flow_dict = nx.maximum_flow(G, source, target, capacity=capacity)
    return max_flow_value, flow_dict
