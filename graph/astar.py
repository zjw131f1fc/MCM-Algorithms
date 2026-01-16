"""
A* 启发式搜索算法

用于网格路径规划，支持自定义启发函数（曼哈顿距离/欧氏距离）。
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          heuristic: str = 'manhattan', allow_diagonal: bool = True) -> Optional[List[Tuple[int, int]]]:
    """
    A* 路径搜索算法

    Parameters
    ----------
    grid : np.ndarray
        网格地图，0表示可通行，1表示障碍物
    start : Tuple[int, int]
        起点坐标 (row, col)
    goal : Tuple[int, int]
        终点坐标 (row, col)
    heuristic : str, optional
        启发函数类型，'manhattan' 或 'euclidean'，默认 'manhattan'
    allow_diagonal : bool, optional
        是否允许对角线移动，默认 True

    Returns
    -------
    path : List[Tuple[int, int]] or None
        路径坐标列表，若无路径则返回 None

    Examples
    --------
    >>> grid = np.array([[0, 0, 0, 0],
    ...                  [0, 1, 1, 0],
    ...                  [0, 0, 0, 0]])
    >>> path = astar(grid, (0, 0), (2, 3))
    >>> print(path)
    [(0, 0), (0, 1), (0, 2), (1, 3), (2, 3)]
    """
    rows, cols = grid.shape

    # 启发函数
    if heuristic == 'manhattan':
        h_func = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    else:  # euclidean
        h_func = lambda p: np.sqrt((p[0] - goal[0])**2 + (p[1] - goal[1])**2)

    # 邻居方向
    if allow_diagonal:
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        costs = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    else:
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        costs = [1, 1, 1, 1]

    # 初始化
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h_func(start)}

    while open_set:
        # 选择 f 值最小的节点
        current = min(open_set, key=lambda p: f_score.get(p, float('inf')))

        if current == goal:
            # 重建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        open_set.remove(current)

        # 遍历邻居
        for (dr, dc), cost in zip(directions, costs):
            neighbor = (current[0] + dr, current[1] + dc)

            # 边界检查
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue

            # 障碍物检查
            if grid[neighbor] == 1:
                continue

            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h_func(neighbor)
                open_set.add(neighbor)

    return None  # 无路径


if __name__ == '__main__':
    # 示例：网格路径规划
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ])

    start = (0, 0)
    goal = (4, 4)

    # 曼哈顿距离 + 对角线移动
    path1 = astar(grid, start, goal, heuristic='manhattan', allow_diagonal=True)
    print("曼哈顿距离路径:", path1)
    print("路径长度:", len(path1) if path1 else 0)

    # 欧氏距离 + 仅四方向
    path2 = astar(grid, start, goal, heuristic='euclidean', allow_diagonal=False)
    print("\n欧氏距离路径:", path2)
    print("路径长度:", len(path2) if path2 else 0)

    # 可视化路径
    if path1:
        grid_vis = grid.copy()
        for r, c in path1:
            if (r, c) != start and (r, c) != goal:
                grid_vis[r, c] = 2
        grid_vis[start] = 3
        grid_vis[goal] = 4
        print("\n路径可视化 (0:空地 1:障碍 2:路径 3:起点 4:终点):")
        print(grid_vis)
