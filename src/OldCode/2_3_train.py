from envs import two_three_fishing
from envs import growth_functions
from parameters import parameters
import callback_fn
from ray.rllib.algorithms import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch

## GLOBALS:
P = parameters()
_DEFAULT_PARAMS = P.parameters()

iterations = 250
experiment_nr = 1
_DATACODE = "2FISHERY/KLIMIT_RXDRIFT"
_PATH = f"../data/{_DATACODE}/{experiment_nr}"
_FILENAME = f"PPO{iterations}"

## SETTING UP RL ALGO

register_env("two_three_fishing", two_three_fishing.twoThreeFishing)
register_env("two_three_fishing_v2", two_three_fishing.twoThreeFishing_v2)

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="two_three_fishing"
#
config.env_config["parameters"] = _DEFAULT_PARAMS
config.env_config["growth_fn"] = growth_functions.K_limit_rx_drift_growth
config.env_config["fluctuating"] = True
config.env_config["initial_pop"] = parameters().init_state()
# agent = config.build()
agent = PPOTrainer(config=config)
#

## TRAIN
checkpoint = f"cache/checkpoint_{_DATACODE}_iter{iterations}"

for i in range(iterations):
  print(f"iteration nr. {i}", end="\r")
  agent.train()

checkpoint = agent.save(f"cache/{checkpoint}")

## POST TRAINING
from eval_util import generate_episodes, episode_plots, state_policy_plot
from plotnine import ggplot, geom_histogram, geom_bar, aes
stats = agent.evaluate()

config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)

episodes_df = generate_episodes(agent, env, reps = 50)
episodes_df.to_csv(f"{_PATH}/{_FILENAME}.csv.xz", index = False)

mean_rew = (
  episodes_df
  .groupby(["rep"])
  .apply(lambda g: g[g['t'] == g['t'].max()])
  .agg({"reward": "mean"})
)
stdev_rew = (
  episodes_df
  .groupby(["rep"])
  .apply(lambda g: g[g['t'] == g['t'].max()])
  .agg({"reward": "std"})
)
print(f"""
iterations = {iterations}
mean total reward = {mean_rew.values[0]} +/- {stdev_rew.values[0]}
""")

for i in range(5):
  episode_plots(
    episodes_df.loc[episodes_df.rep == i], 
    path_and_filename = f"{_PATH}/{_FILENAME}-eps-{i}.png"
  )
  
state_policy_plot(agent, env, path_and_filename= f"{_PATH}/{_FILENAME}-pol.png")

def get_times_rewards(filename):
  df = pd.read_csv(filename)
  max_t_rew = []
  for rep in range(50):
    rep_df = df.loc[df.rep == rep]
    rep_tmax_df = rep_df.loc[rep_df.t == rep_df.t.max()]
    max_t_rew.append([rep, rep_tmax_df.t.values[0], rep_tmax_df.reward.values[0]])
  return pd.DataFrame(max_t_rew, columns=["rep", "t","reward"])

df_t_rew = get_times_rewards(f"{_PATH}/{_FILENAME}.csv.xz")
t_max_hist = (
  ggplot(data=df_t_rew, mapping=aes(x='rep', weight='t')) 
  + geom_bar()
)
t_max_hist.save(filename=f"{_PATH}/{_FILENAME}_tmax_hist.png")
rew_hist = (
  ggplot(data=df_t_rew, mapping=aes(x='rep',weight='reward')) 
  +geom_bar()
  #+ geom_histogram(binwidth=1)
)
rew_hist.save(filename=f"{_PATH}/{_FILENAME}_rew_hist.png")
