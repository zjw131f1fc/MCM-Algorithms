"""测试预测与时间序列算法"""
import numpy as np
from prediction.gm11 import gm11
from prediction.markov import markov_predict
from prediction.arima import arima_forecast

print("=" * 50)
print("测试预测与时间序列算法")
print("=" * 50)

# 1. GM(1,1)灰色预测
print("\n1. GM(1,1)灰色预测")
x0 = np.array([100, 105, 112, 120, 130])
x0_pred, x0_forecast, params = gm11(x0, n_predict=3)
print(f"原始数据: {x0}")
print(f"拟合值: {x0_pred}")
print(f"预测值: {x0_forecast}")
print(f"模型参数 a={params['a']:.4f}, b={params['b']:.4f}")

# 2. 马尔可夫链预测
print("\n2. 马尔可夫链预测")
data = ['晴', '晴', '雨', '晴', '阴', '雨', '雨', '晴', '阴', '晴']
P, probs, predictions = markov_predict(data, n_steps=3)
print(f"转移矩阵:\n{P}")
print(f"未来3步概率分布: {probs}")
print(f"最可能状态: {predictions}")

# 3. ARIMA预测
print("\n3. ARIMA预测")
data = np.array([100, 102, 105, 108, 112, 115, 120, 125, 130, 135])
model, forecast, conf_int = arima_forecast(data, order=(1,1,1), n_forecast=3)
print(f"历史数据: {data}")
print(f"预测值: {forecast}")
print(f"置信区间:\n{conf_int}")

print("\n" + "=" * 50)
print("预测与时间序列算法测试完成")
print("=" * 50)
