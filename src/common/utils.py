import numpy as np
import pandas as pd


def get_stock_price(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., 1 : 1 + stock_dim]


def get_stock_amount(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., 1 + stock_dim : 1 + 2 * stock_dim]


def get_stock_sentiment(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., -2 * stock_dim : -stock_dim]


def get_stock_risk(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., -stock_dim:]


def cal_portfolio_return(action, current_price, next_price):
    price_changes = next_price / current_price - 1
    return np.sum(action * price_changes, axis=-1)


def cal_alg_metrics(
    data_series: pd.Series, risk_f: float = 0.04, tradeing_days: int = 252
) -> pd.Series:
    """
    根据一列数据计算各项评价指标
    input:
        data_series: 单个模型的日频资产数据，pd.Series
        risk_f: 年化无风险利率
        tradeing_days: 每年的交易日数量
    return：
    总收益率，年化收益率，年化夏普比率，年化波动率，最大回撤
    """
    data = data_series.copy().dropna()
    days = len(data)
    total_return = data.iloc[-1] / data.iloc[0] - 1
    total_return_annel = (data.iloc[-1] / data.iloc[0]) ** (tradeing_days / days) - 1

    daily_returns = data.pct_change().dropna()
    daily_rf = (1 + risk_f) ** (1 / tradeing_days) - 1
    excess_returns = daily_returns - daily_rf
    sharpe_annel = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(tradeing_days)

    volatility_annel = daily_returns.std(ddof=1) * np.sqrt(252)

    cummax = data.cummax()
    drawdown = data / cummax - 1
    max_drawdown = -drawdown.min()

    result = pd.Series(
        name=data.name,
        index=[
            "Total Return",
            "Annel Total Return",
            "Annel Sharpe",
            "Annel Volatility",
            "Max Drawdown",
        ],
        data=[total_return, total_return_annel, sharpe_annel, volatility_annel, max_drawdown],
    )
    return result
