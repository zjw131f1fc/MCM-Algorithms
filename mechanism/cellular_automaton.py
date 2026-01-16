"""
元胞自动机 (Cellular Automaton)

元胞自动机是一种离散模型,由规则的网格和每个网格上的元胞组成。
每个元胞根据邻居状态和转换规则更新自身状态。

经典应用:
- Conway生命游戏
- 森林火灾模拟
- 交通流模拟
- 疾病传播模拟
"""

import numpy as np


class CellularAutomaton:
    """
    元胞自动机类

    Parameters
    ----------
    grid_size : tuple
        网格大小 (rows, cols)
    rule : callable, optional
        自定义规则函数 func(grid, i, j) -> new_state
        默认使用Conway生命游戏规则
    boundary : str, optional
        边界条件: 'periodic'(周期) 或 'fixed'(固定为0), 默认'periodic'

    Attributes
    ----------
    grid : ndarray
        当前网格状态
    history : list
        历史状态记录

    Examples
    --------
    >>> from mechanism.cellular_automaton import CellularAutomaton
    >>> ca = CellularAutomaton((50, 50))
    >>> ca.set_random(density=0.3)
    >>> ca.run(steps=100)
    >>> print(f"Alive cells: {ca.count_alive()}")
    """

    def __init__(self, grid_size, rule=None, boundary='periodic'):
        self.rows, self.cols = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.rule = rule if rule else self._conway_rule
        self.boundary = boundary
        self.history = []

    def set_random(self, density=0.3):
        """随机初始化网格"""
        self.grid = (np.random.random((self.rows, self.cols)) < density).astype(int)

    def set_pattern(self, pattern, position=(0, 0)):
        """设置特定图案"""
        r, c = position
        pr, pc = pattern.shape
        self.grid[r:r+pr, c:c+pc] = pattern

    def _get_neighbors(self, i, j):
        """获取邻居状态(Moore邻域,8邻居)"""
        if self.boundary == 'periodic':
            neighbors = [
                self.grid[(i-1) % self.rows, (j-1) % self.cols],
                self.grid[(i-1) % self.rows, j],
                self.grid[(i-1) % self.rows, (j+1) % self.cols],
                self.grid[i, (j-1) % self.cols],
                self.grid[i, (j+1) % self.cols],
                self.grid[(i+1) % self.rows, (j-1) % self.cols],
                self.grid[(i+1) % self.rows, j],
                self.grid[(i+1) % self.rows, (j+1) % self.cols]
            ]
        else:
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbors.append(self.grid[ni, nj])
        return neighbors

    def _conway_rule(self, grid, i, j):
        """Conway生命游戏规则"""
        neighbors = self._get_neighbors(i, j)
        alive_neighbors = sum(neighbors)

        if grid[i, j] == 1:  # 活细胞
            return 1 if alive_neighbors in [2, 3] else 0
        else:  # 死细胞
            return 1 if alive_neighbors == 3 else 0

    def step(self):
        """执行一步演化"""
        new_grid = np.zeros_like(self.grid)
        for i in range(self.rows):
            for j in range(self.cols):
                new_grid[i, j] = self.rule(self.grid, i, j)
        self.grid = new_grid

    def run(self, steps, record_history=False):
        """运行多步"""
        if record_history:
            self.history = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            if record_history:
                self.history.append(self.grid.copy())

    def count_alive(self):
        """统计活细胞数量"""
        return np.sum(self.grid)


def conway_game_of_life(grid_size=(50, 50), density=0.3, steps=100):
    """
    Conway生命游戏快捷函数

    Parameters
    ----------
    grid_size : tuple
        网格大小
    density : float
        初始活细胞密度
    steps : int
        演化步数

    Returns
    -------
    ca : CellularAutomaton
        演化后的元胞自动机对象

    Examples
    --------
    >>> ca = conway_game_of_life(grid_size=(50, 50), density=0.3, steps=100)
    >>> print(f"Final alive: {ca.count_alive()}")
    """
    ca = CellularAutomaton(grid_size)
    ca.set_random(density)
    ca.run(steps, record_history=True)
    return ca


if __name__ == '__main__':
    # 示例1: Conway生命游戏
    print("=== Conway生命游戏 ===")
    ca = CellularAutomaton((50, 50))
    ca.set_random(density=0.3)

    print(f"初始活细胞: {ca.count_alive()}")
    ca.run(steps=100)
    print(f"100步后活细胞: {ca.count_alive()}")

    # 示例2: 经典图案 - Glider
    print("\n=== Glider图案 ===")
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    ca2 = CellularAutomaton((20, 20))
    ca2.set_pattern(glider, position=(5, 5))
    print(f"初始状态:\n{ca2.grid[3:11, 3:11]}")
    ca2.run(steps=4)
    print(f"4步后:\n{ca2.grid[3:11, 3:11]}")

    # 可视化(需要matplotlib)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        print("\n=== 生成动画 ===")
        ca3 = CellularAutomaton((50, 50))
        ca3.set_random(density=0.3)
        ca3.run(steps=100, record_history=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(ca3.history[0], cmap='binary', interpolation='nearest')
        ax.set_title('Conway生命游戏')
        ax.axis('off')

        def update(frame):
            im.set_array(ca3.history[frame])
            ax.set_title(f'Conway生命游戏 - Step {frame}')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(ca3.history),
                           interval=100, blit=True, repeat=True)
        plt.show()

    except ImportError:
        print("安装matplotlib可查看动画: pip install matplotlib")
