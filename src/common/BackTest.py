import pandas as pd
import matplotlib.pyplot as plt
import os
from finrl.agents.stablebaselines3.models import MODELS
from src.env.env_stocktrading_sentiment_risk import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent


class BackTest:
    def __init__(
        self,
        trade_df: pd.DataFrame,
        trained_dir: str,
    ):
        self.trade_df = trade_df
        self.models_dict = dict()
        files = sorted(os.listdir(trained_dir))
        for file in files:
            if "." not in file:
                continue
            file_name, file_ext = file.split(".")
            if file_ext != "zip":
                continue
            for key in MODELS:
                if file_name in key or key in file_name:
                    model = MODELS[key].load(trained_dir + f"/{file_name}")
                    self.models_dict[file_name] = model
                    print(f"{key} model in {trained_dir}/{file_name} has been loaded.")
                else:
                    continue

    def backtest(self, env_kwargs, env=StockTradingEnv):
        e_trade_gym = env(df=self.trade_df, **env_kwargs)
        result_dict = dict()
        for model_name in self.models_dict:
            df_account_value_model, df_actions_model = DRLAgent.DRL_prediction(
                model=self.models_dict[model_name], environment=e_trade_gym
            )
            df_result_model = df_account_value_model.set_index(df_account_value_model.columns[0])
            result_dict[model_name] = df_result_model["account_value"]

        # trade_prices = self.trade_df.pivot(index="date", columns="tic", values="close").sort_index()
        # trade_members = (
        #     self.trade_df.pivot(index="date", columns="tic", values="is_member")
        #     .reindex(index=trade_prices.index, columns=trade_prices.columns)
        #     .astype(bool)
        # )
        # daily_returns = trade_prices.pct_change().where(trade_members).mean(axis=1)
        # equal_weight_value = 1 + daily_returns.fillna(0).cumprod() * 1000000
        # pit_equal_weight = pd.DataFrame({"portfolio_value": equal_weight_value})
        # result_dict["point-in-time equal weight"] = pit_equal_weight["portfolio_value"]

        # print(result_dict)
        self.result = pd.DataFrame(result_dict)

    def plot_backtest_result(
        self,
        plot_origin: bool = True,
        plot_smooth: bool = False,
        smooth_window: int = 20,
    ):

        smoothed_result = self.result.rolling(window=smooth_window, min_periods=1).mean()

        fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
        self.result.plot(ax=axes[0])
        axes[0].set_title("Portfolio Value Over Time")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Portfolio Value ($)")

        smoothed_result.plot(ax=axes[1])
        axes[1].set_title(f"Smoothed Portfolio Value ({smooth_window}-Day Rolling Mean)")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Portfolio Value ($)")

        fig.tight_layout()
        fig.savefig("backtest_result.pdf", bbox_inches="tight")
        plt.close(fig)
        print("\nPlots saved to backtest_result.pdf")
