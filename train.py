import numpy as np
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent

from src.models.sb3 import dapo

from stable_baselines3.common.logger import configure

from finrl.config import INDICATORS

# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from src.env.env_stocktrading_sentiment_risk import StockTradingEnv

train_origin = pd.read_csv("dataset/data/train_data_deepseek_risk_2013_2018.csv")
# %%

train = train_origin.copy()
train.head()
train = train_origin.sort_values(["date", "tic"]).reset_index(drop=True)
train.index = pd.factorize(train["date"], sort=True)[0]
train.index.names = [""]
train["is_member"] = np.ones(len(train))
train.drop(columns=train.columns[0], inplace=True)

# %%
cols = ["llm_sentiment", "llm_risk"]
train[cols] = train[cols].fillna(3.0)


# %%

stock_dim = len(train.tic.unique())
# state = cash + close + position + INDICATORS + is_mem + sentiment + risk
state_space = 1 + 2 * stock_dim + (len(INDICATORS) + 1) * stock_dim + 2 * stock_dim

buy_cost_list = sell_cost_list = [0.001] * stock_dim
num_stock_shares = [0] * stock_dim

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dim,
    "reward_scaling": 1e-4,
    "member_col": "is_member",
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# %%
DAPO_PARAMS = {
    "n_steps": 2000,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": None,
    "stock_dim": stock_dim,
}

agent = DRLAgent(env=env_train)

model_dapo = agent.get_model("dapo", model_kwargs=DAPO_PARAMS)
tmp_path = "result/dapo"
new_logger_dapo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_dapo.set_logger(new_logger_dapo)
trained_dapo = agent.train_model(model=model_dapo, tb_log_name="dapo", total_timesteps=20000)
trained_dapo.save("trained/agent_dapo")


# %%

RESULTS_DIR = "result"
TRAINED_MODEL_DIR = "trained"

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

# --- Agent 1: A2C ---
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")
if if_using_a2c:
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger_a2c)

trained_a2c = (
    agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=20000)
    if if_using_a2c
    else None
)
if if_using_a2c:
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")

# --- Agent 2: DDPG ---
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")
if if_using_ddpg:
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ddpg.set_logger(new_logger_ddpg)

trained_ddpg = (
    agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=20000)
    if if_using_ddpg
    else None
)
if if_using_ddpg:
    trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")

# --- Agent 3: PPO ---
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
if if_using_ppo:
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

trained_ppo = (
    agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=20000)
    if if_using_ppo
    else None
)
if if_using_ppo:
    trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")

# --- Agent 4: TD3 ---
agent = DRLAgent(env=env_train)
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
}
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
if if_using_td3:
    tmp_path = RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_td3.set_logger(new_logger_td3)

trained_td3 = (
    agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=20000)
    if if_using_td3
    else None
)
if if_using_td3:
    trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")

# --- Agent 5: SAC ---
agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
if if_using_sac:
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

trained_sac = (
    agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=20000)
    if if_using_sac
    else None
)
if if_using_sac:
    trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")

print("All agents trained and saved to", TRAINED_MODEL_DIR)
