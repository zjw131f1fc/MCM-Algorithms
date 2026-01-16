"""
动态规划(Dynamic Programming)常用模板

提供经典DP问题的实现模板：
- 0/1背包问题
- 完全背包问题
- 最长公共子序列(LCS)
- 编辑距离
- 最长递增子序列(LIS)
"""

import numpy as np


def knapsack_01(weights, values, capacity):
    """
    0/1背包问题

    每个物品只能选择一次，求在容量限制下的最大价值。

    Parameters
    ----------
    weights : array-like
        物品重量数组
    values : array-like
        物品价值数组
    capacity : int
        背包容量

    Returns
    -------
    max_value : int
        最大价值
    selected : list
        选中的物品索引列表

    Examples
    --------
    >>> weights = [2, 3, 4, 5]
    >>> values = [3, 4, 5, 6]
    >>> capacity = 8
    >>> max_value, selected = knapsack_01(weights, values, capacity)
    >>> print(f"最大价值: {max_value}, 选中物品: {selected}")
    """
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1), dtype=int)

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]

    return int(dp[n][capacity]), selected[::-1]


def knapsack_complete(weights, values, capacity):
    """
    完全背包问题

    每个物品可以选择无限次，求在容量限制下的最大价值。

    Parameters
    ----------
    weights : array-like
        物品重量数组
    values : array-like
        物品价值数组
    capacity : int
        背包容量

    Returns
    -------
    max_value : int
        最大价值

    Examples
    --------
    >>> weights = [2, 3, 4]
    >>> values = [3, 4, 5]
    >>> capacity = 10
    >>> max_value = knapsack_complete(weights, values, capacity)
    >>> print(f"最大价值: {max_value}")
    """
    dp = np.zeros(capacity + 1, dtype=int)

    for i in range(len(weights)):
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return int(dp[capacity])


def lcs(s1, s2):
    """
    最长公共子序列(Longest Common Subsequence)

    Parameters
    ----------
    s1 : str or list
        序列1
    s2 : str or list
        序列2

    Returns
    -------
    length : int
        最长公共子序列长度
    sequence : str or list
        最长公共子序列

    Examples
    --------
    >>> s1 = "ABCDGH"
    >>> s2 = "AEDFHR"
    >>> length, sequence = lcs(s1, s2)
    >>> print(f"LCS长度: {length}, 序列: {sequence}")
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    i, j = m, n
    result = []
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            result.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    sequence = ''.join(result[::-1]) if isinstance(s1, str) else result[::-1]
    return int(dp[m][n]), sequence


def edit_distance(s1, s2):
    """
    编辑距离(Levenshtein Distance)

    计算将s1转换为s2所需的最少操作次数(插入、删除、替换)。

    Parameters
    ----------
    s1 : str
        源字符串
    s2 : str
        目标字符串

    Returns
    -------
    distance : int
        最小编辑距离

    Examples
    --------
    >>> s1 = "kitten"
    >>> s2 = "sitting"
    >>> distance = edit_distance(s1, s2)
    >>> print(f"编辑距离: {distance}")
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return int(dp[m][n])


def lis(arr):
    """
    最长递增子序列(Longest Increasing Subsequence)

    Parameters
    ----------
    arr : array-like
        输入序列

    Returns
    -------
    length : int
        最长递增子序列长度
    sequence : list
        最长递增子序列

    Examples
    --------
    >>> arr = [10, 9, 2, 5, 3, 7, 101, 18]
    >>> length, sequence = lis(arr)
    >>> print(f"LIS长度: {length}, 序列: {sequence}")
    """
    n = len(arr)
    if n == 0:
        return 0, []

    dp = np.ones(n, dtype=int)
    parent = np.arange(n)

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_len = int(np.max(dp))
    max_idx = int(np.argmax(dp))

    sequence = []
    idx = max_idx
    while True:
        sequence.append(arr[idx])
        if parent[idx] == idx:
            break
        idx = parent[idx]

    return max_len, sequence[::-1]


if __name__ == '__main__':
    print("=== 0/1背包问题 ===")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    max_value, selected = knapsack_01(weights, values, capacity)
    print(f"物品重量: {weights}")
    print(f"物品价值: {values}")
    print(f"背包容量: {capacity}")
    print(f"最大价值: {max_value}")
    print(f"选中物品索引: {selected}")
    print()

    print("=== 完全背包问题 ===")
    weights = [2, 3, 4]
    values = [3, 4, 5]
    capacity = 10
    max_value = knapsack_complete(weights, values, capacity)
    print(f"物品重量: {weights}")
    print(f"物品价值: {values}")
    print(f"背包容量: {capacity}")
    print(f"最大价值: {max_value}")
    print()

    print("=== 最长公共子序列 ===")
    s1 = "ABCDGH"
    s2 = "AEDFHR"
    length, sequence = lcs(s1, s2)
    print(f"序列1: {s1}")
    print(f"序列2: {s2}")
    print(f"LCS长度: {length}")
    print(f"LCS序列: {sequence}")
    print()

    print("=== 编辑距离 ===")
    s1 = "kitten"
    s2 = "sitting"
    distance = edit_distance(s1, s2)
    print(f"源字符串: {s1}")
    print(f"目标字符串: {s2}")
    print(f"编辑距离: {distance}")
    print()

    print("=== 最长递增子序列 ===")
    arr = [10, 9, 2, 5, 3, 7, 101, 18]
    length, sequence = lis(arr)
    print(f"输入序列: {arr}")
    print(f"LIS长度: {length}")
    print(f"LIS序列: {sequence}")
