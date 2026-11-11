import numpy as np


def get_stock_price(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., 1:1 + stock_dim]


def get_stock_amount(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., 1 + stock_dim:1 + 2 * stock_dim]


def get_stock_sentiment(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., -2 * stock_dim:-stock_dim]


def get_stock_risk(state: np.ndarray, stock_dim: int) -> np.ndarray:
    return state[..., -stock_dim:]

def cal_portfolio_return(action, current_price, next_price):
    price_changes = next_price / current_price -1
    return np.sum(action*price_changes, axis=-1)
