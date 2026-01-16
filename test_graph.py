"""测试图论算法"""
import numpy as np
import networkx as nx
from graph.astar import astar
from graph.network_algorithms import dijkstra, minimum_spanning_tree, floyd_warshall, maximum_flow

print("=" * 50)
print("测试图论算法")
print("=" * 50)

# 1. A*算法
print("\n1. A*算法")
grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0]
])
start = (0, 0)
goal = (4, 4)
path = astar(grid, start, goal)
print(f"起点: {start}")
print(f"终点: {goal}")
print(f"路径: {path}")

# 2. Dijkstra最短路径
print("\n2. Dijkstra最短路径")
G = nx.Graph()
G.add_weighted_edges_from([
    (0, 1, 4), (0, 2, 2),
    (1, 2, 1), (1, 3, 5),
    (2, 3, 8), (2, 4, 10),
    (3, 4, 2)
])
distances = dijkstra(G, source=0)
print(f"从节点0到各节点的最短距离: {distances}")
path, length = dijkstra(G, source=0, target=4)
print(f"从节点0到节点4的最短路径: {path}, 长度: {length}")

# 3. 最小生成树
print("\n3. 最小生成树")
mst = minimum_spanning_tree(G, algorithm='kruskal')
print(f"MST边: {list(mst.edges(data=True))}")
print(f"MST总权重: {sum(d['weight'] for u, v, d in mst.edges(data=True))}")

# 4. Floyd-Warshall全源最短路径
print("\n4. Floyd-Warshall全源最短路径")
distances = floyd_warshall(G)
print(f"全源最短路径距离矩阵:")
for src in distances:
    print(f"  节点{src}: {distances[src]}")

# 5. 最大流
print("\n5. 最大流")
G_flow = nx.DiGraph()
G_flow.add_edge(0, 1, capacity=10)
G_flow.add_edge(0, 2, capacity=5)
G_flow.add_edge(1, 2, capacity=15)
G_flow.add_edge(1, 3, capacity=10)
G_flow.add_edge(2, 3, capacity=10)
max_flow_value, flow_dict = maximum_flow(G_flow, source=0, target=3)
print(f"最大流量: {max_flow_value}")
print(f"流量分配: {flow_dict}")

print("\n" + "=" * 50)
print("图论算法测试完成")
print("=" * 50)
