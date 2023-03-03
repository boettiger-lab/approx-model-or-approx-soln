from envs import fish_tipping # setwd to source file location if using repl_python()
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


register_env("fish_tipping",fish_tipping.three_sp)

## We could call env directly without this if only  our envs took a env_config dict argument

config = ppo.PPOConfig()
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
agent = config.build(env="fish_tipping")

iterations = 200
checkpoint = ("cache/checkpoint_000{}".format(iterations))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save("cache")

#agent_restored = config.build(env="threeFishing-v2")
#agent_restored.evaluate()
agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env

# Initialize saved copy of eval environment:
config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)

env.training = False
df = []
for rep in range(20):
  episode_reward = 0
  observation, _ = env.reset()
  for t in range(env.Tmax):
    action = agent.compute_single_action(observation)
    df.append(np.append([t, rep, action[0], episode_reward], observation))
    observation, reward, terminated, done, info = env.step(action)
    episode_reward += reward
    if terminated:
      break
    
cols = ["t", "rep", "action", "reward", "X", "Y", "Z"]
df = pd.DataFrame(df, columns = cols)
df.to_csv(f"data/PPO{iterations}.csv.xz", index = False)

## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries
df = pd.read_csv(f"data/PPO{iterations}.csv.xz")
df2 = (df
       .melt(id_vars=["t", "action", "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             'action': 'mean'})) 
ggplot(df2, aes("t", "value", color="variable")) + geom_line()


## quick policy plot
df = []
obs, _ = env.reset()
states = np.linspace(-1,0.5,101)
episode_reward = 0
for rep in range(5):
  for observation in states:
      obs[0] = observation
      action = agent.compute_single_action(obs)
      df.append([observation, action[0]])
df2 = pd.DataFrame(df, columns =  ["observation","action"])

# transform to natural space & compute escapement
df2["escapement"] = df2["observation"] + 1 - df2["action"]
ggplot(df2, aes("observation", "escapement")) + geom_point()


