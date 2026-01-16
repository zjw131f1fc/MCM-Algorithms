"""
Markov Chain Prediction
马尔可夫链预测

适用于状态转移规律明显的系统预测，如天气、市场状态、设备状态等。
"""

import numpy as np


def markov_predict(data, states=None, n_steps=1):
    """
    马尔可夫链预测

    Parameters
    ----------
    data : array-like, shape (n,)
        历史状态序列
    states : array-like, optional
        所有可能的状态列表，None则自动从data提取
    n_steps : int, optional
        预测未来的步数，默认1

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)
        状态转移概率矩阵，P[i,j]表示从状态i转移到状态j的概率
    predictions : list of ndarray
        每步预测的状态概率分布
    most_likely : list
        每步最可能的状态

    Examples
    --------
    >>> import numpy as np
    >>> # 天气状态序列: 0=晴, 1=阴, 2=雨
    >>> data = [0, 0, 1, 2, 1, 0, 0, 1, 1, 2, 2, 1, 0]
    >>> P, probs, states = markov_predict(data, n_steps=3)
    >>> print(f"转移矩阵:\\n{P}")
    >>> print(f"未来3天最可能状态: {states}")
    """
    data = np.array(data)

    if states is None:
        states = np.unique(data)
    else:
        states = np.array(states)

    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # 构建转移计数矩阵
    count_matrix = np.zeros((n_states, n_states))
    for i in range(len(data) - 1):
        from_state = state_to_idx[data[i]]
        to_state = state_to_idx[data[i + 1]]
        count_matrix[from_state, to_state] += 1

    # 转换为转移概率矩阵
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    transition_matrix = count_matrix / row_sums

    # 初始状态分布（当前状态）
    current_state = state_to_idx[data[-1]]
    state_prob = np.zeros(n_states)
    state_prob[current_state] = 1.0

    # 多步预测
    predictions = []
    most_likely = []

    for _ in range(n_steps):
        state_prob = state_prob @ transition_matrix
        predictions.append(state_prob.copy())
        most_likely.append(states[np.argmax(state_prob)])

    return transition_matrix, predictions, most_likely


if __name__ == '__main__':
    # 示例：天气预测
    print("=== 马尔可夫链预测示例 ===")

    # 历史天气数据: 0=晴, 1=阴, 2=雨
    weather_data = [0, 0, 1, 2, 1, 0, 0, 1, 1, 2, 2, 1, 0, 0, 1]
    weather_names = ['晴', '阴', '雨']

    print(f"历史天气: {[weather_names[w] for w in weather_data]}\n")

    P, probs, predictions = markov_predict(weather_data, n_steps=3)

    print("状态转移概率矩阵:")
    print("     晴    阴    雨")
    for i, name in enumerate(weather_names):
        print(f"{name}  {P[i]}")

    print(f"\n未来3天预测:")
    for i, (prob, state) in enumerate(zip(probs, predictions), 1):
        print(f"第{i}天: {weather_names[state]} (概率分布: {prob.round(3)})")
