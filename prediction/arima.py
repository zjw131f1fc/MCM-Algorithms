"""
ARIMA Time Series Forecasting
ARIMA时间序列预测

基于statsmodels库的ARIMA模型封装,适用于平稳或差分后平稳的时间序列预测。
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(data, order=(1, 1, 1), n_forecast=1):
    """
    ARIMA时间序列预测

    Parameters
    ----------
    data : array-like, shape (n,)
        时间序列数据
    order : tuple of int, optional
        ARIMA模型阶数 (p, d, q)，默认(1,1,1)
        - p: 自回归阶数 (AR)
        - d: 差分阶数 (I)
        - q: 移动平均阶数 (MA)
    n_forecast : int, optional
        预测未来的步数，默认1

    Returns
    -------
    model : ARIMAResults
        拟合后的ARIMA模型对象
    forecast : ndarray, shape (n_forecast,)
        未来n_forecast步的预测值
    conf_int : ndarray, shape (n_forecast, 2)
        预测值的置信区间 [下界, 上界]

    Examples
    --------
    >>> import numpy as np
    >>> from prediction.arima import arima_forecast
    >>>
    >>> # 生成示例数据
    >>> np.random.seed(42)
    >>> data = np.cumsum(np.random.randn(100)) + 50
    >>>
    >>> # ARIMA预测
    >>> model, forecast, conf_int = arima_forecast(data, order=(1,1,1), n_forecast=5)
    >>>
    >>> print(f"预测值: {forecast}")
    >>> print(f"置信区间: {conf_int}")
    >>>
    >>> # 模型诊断
    >>> print(f"AIC: {model.aic:.2f}")
    >>> print(f"BIC: {model.bic:.2f}")
    >>>
    >>> # 拟合值
    >>> fitted = model.fittedvalues
    >>> print(f"拟合值: {fitted[-5:]}")

    Notes
    -----
    - 使用前建议进行平稳性检验(ADF检验)
    - 通过AIC/BIC准则选择最优阶数
    - 差分阶数d通常取0-2
    - 置信区间默认95%水平
    """
    data = np.array(data, dtype=float)

    # 拟合ARIMA模型
    model = ARIMA(data, order=order)
    model_fit = model.fit()

    # 预测未来值
    forecast_result = model_fit.forecast(steps=n_forecast)
    forecast = np.array(forecast_result)

    # 获取置信区间
    forecast_obj = model_fit.get_forecast(steps=n_forecast)
    conf_int = forecast_obj.conf_int()

    return model_fit, forecast, conf_int


if __name__ == '__main__':
    # 示例：预测时间序列未来趋势
    np.random.seed(42)
    data = np.cumsum(np.random.randn(100)) + 50

    print("=== ARIMA时间序列预测示例 ===")
    print(f"数据长度: {len(data)}")
    print(f"最后5个观测值: {data[-5:].round(2)}\n")

    # ARIMA(1,1,1)预测
    model, forecast, conf_int = arima_forecast(data, order=(1, 1, 1), n_forecast=5)

    print(f"模型阶数: ARIMA(1,1,1)")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}\n")

    print(f"预测值: {forecast.round(2)}")
    print(f"置信区间下界: {conf_int[:, 0].round(2)}")
    print(f"置信区间上界: {conf_int[:, 1].round(2)}")
