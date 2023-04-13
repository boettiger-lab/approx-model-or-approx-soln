from envs import fish_tipping
from envs import growth_functions
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


register_env("fish_tipping",fish_tipping.three_sp)

iterations = 151

## We could call env directly without this if only  our envs took a env_config dict argument

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
#
config.env="fish_tipping"
config.env_config["growth_fn"] = growth_functions.y_abiotic_growth # comment out to use default growth_fn
config.env_config["fluctuating"] = True
_DATACODE = "YABIOTIC"
_PATH = f"../data/{_DATACODE}"
_FILENAME = f"PPO{iterations}"
# config.env_config["parameters"] = growth_functions.params_threeSpHolling3() # comment out to use _DEFALUT_PARAMETERS in fish_tipping
#
agent = config.build()

checkpoint = (f"cache/checkpoint_{_DATACODE}_iter{iterations}")

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
df.to_csv(f"{_PATH}/{_FILENAME}.csv.xz", index = False)

## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries
df = pd.read_csv(f"{_PATH}/{_FILENAME}.csv.xz")
df2 = (df[df.rep == 3.0]
        .melt(id_vars=["t",  "reward", "rep"])
        # .groupby(['t', "variable"], as_index=False)
        # .agg(
        #   {'reward': 'mean',
        #    'value': 'mean',
        #   #'action': 'mean'
        #     }
        #    )
        ) 
value_plot = ggplot(df2, aes("t", "value", color="variable")) + geom_line()
value_plot.save(filename = f"val_{_FILENAME}.png", path = _PATH)

## summary stats
reward = df[df.t == max(df.t)].reward
reward.mean()
np.sqrt(reward.var())

## quick policy plot
policy_df = []
states = np.linspace(-1,0.5,100)
for rep in range(10):
  obs, _ = env.reset()
  #obs[2] += .05 * rep
  for state in states:
      obs[0] = state
      action = agent.compute_single_action(obs)
      escapement = max((state + 1)*(1 - action[0]), 0)
      policy_df.append([state+1, escapement, action[0], rep])
      
policy_df = pd.DataFrame(policy_df, columns=["observation","escapement","action","rep"])
escapement_plot = ggplot(policy_df, aes("observation", "escapement", color = "rep")) + geom_point(shape=".")
escapement_plot.save(filename = f"esc_{_FILENAME}.png", path = _PATH)
action_plot = ggplot(policy_df, aes("observation", "action", color = "rep")) + geom_point(shape=".")
action_plot.save(filename = f"act_{_FILENAME}.png", path = _PATH)


