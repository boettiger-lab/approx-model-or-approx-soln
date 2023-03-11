
from envs import fish_tipping
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


register_env("fish_tipping",fish_tipping.three_sp)

## We could call env directly without this if only  our envs took a env_config dict argument

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="fish_tipping"
config.env_config["training"] = False
agent = config.build()

iterations = 400
checkpoint = ("cache/checkpoint_000{}".format(iterations))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save("cache")

agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env

# Initialize saved copy of eval environment:
config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)

env.training = False
df = []
for rep in range(50):
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
df2 = (df[df.rep == 3.0]
       .melt(id_vars=["t",  "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             #'action': 'mean'
             })) 
ggplot(df2, aes("t", "value", color="variable")) + geom_line()

## summary stats
reward = df[df.t == max(df.t)].reward
reward.mean()
np.sqrt(reward.var())

## quick policy plot
policy_df = []
states = np.linspace(-1,0.5,101)
for rep in range(10):
  obs, _ = env.reset()
  #obs[2] += .05 * rep
  for state in states:
      obs[0] = state
      action = agent.compute_single_action(obs)
      escapement = max(state + 1 - action[0], 0)
      policy_df.append([state+1, escapement, action[0], rep])
      
policy_df = pd.DataFrame(policy_df, columns=["observation","escapement","action","rep"])
ggplot(policy_df, aes("observation", "escapement", color = "rep")) + geom_point(shape=".")


