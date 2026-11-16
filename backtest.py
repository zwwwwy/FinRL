from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from finrl.config import INDICATORS

from src.common.BackTest import BackTest
from src.env.env_stocktrading_sentiment_risk import StockTradingEnv


trade_origin = pd.read_csv("dataset/data/trade_data_deepseek_risk_2019_2023.csv")

trade = trade_origin.sort_values(["date", "tic"]).reset_index(drop=True)
trade.index = pd.factorize(trade["date"], sort=True)[0]
trade.index.names = [""]
trade["is_member"] = np.ones(len(trade))
trade.drop(columns=trade.columns[0], inplace=True)
trade[["llm_sentiment", "llm_risk"]] = trade[["llm_sentiment", "llm_risk"]].fillna(3.0)


stock_dimension = len(trade.tic.unique())
# cash + close + holdings + technical indicators + membership
# + LLM sentiment + LLM risk
state_space = (
    1 + 2 * stock_dimension + (len(INDICATORS) + 1) * stock_dimension + 2 * stock_dimension
)
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension


TRAINED_MODEL_DIR = "trained/only_dapo"

# DAPO has a custom required constructor argument that SB3 does not restore
# automatically from the saved model.

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "member_col": "is_member",
}

back_tester = BackTest(trade, TRAINED_MODEL_DIR)
back_tester.backtest(env=StockTradingEnv, env_kwargs=env_kwargs)
back_tester.plot_backtest_result()
back_tester.get_backtest_metrics(use_persent=True, format_str=".2f")
