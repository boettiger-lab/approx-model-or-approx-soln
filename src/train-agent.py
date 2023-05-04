from envs import fish_tipping
from envs import growth_functions
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch
from parameters import parameters

iterations = 150
_DEFAULT_PARAMETERS = parameters().parameters()
_DATACODE = "RXDRIFT/2"
_PATH = f"../data/{_DATACODE}"
_FILENAME = f"PPO{iterations}"

register_env("fish_tipping",fish_tipping.three_sp)

## We could call env directly without this if only  our envs took a env_config dict argument

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
#
config.env="fish_tipping"
config.env_config["growth_fn"] = growth_functions.rx_drift_growth # comment out to use default growth_fn
config.env_config["fluctuating"] = True
# config.env_config["parameters"] = growth_functions.params_threeSpHolling3() # comment out to use _DEFALUT_PARAMETERS in fish_tipping
#
agent = config.build()

checkpoint = (f"cache/checkpoint_{_DATACODE}_iter{iterations}")

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save(f"cache/{checkpoint}")

agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env

# Initialize saved copy of eval environment:
config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)

env.training = False
df = []
repetitions = 50
for rep in range(repetitions):
  episode_reward = 0
  observation, _ = env.reset()
  for t in range(env.Tmax):
    action = agent.compute_single_action(observation)
    escapement = (observation[0] + 1) * (1 - action[0])
    df.append(np.append([t, rep, action[0], episode_reward, escapement], 1+observation))
    observation, reward, terminated, done, info = env.step(action)
    episode_reward += reward
    if terminated:
      break
    
cols = ["t", "rep", "action", "reward", "escapement", "X", "Y", "Z"]
df = pd.DataFrame(df, columns = cols)
df.to_csv(f"{_PATH}/{_FILENAME}.csv.xz", index = False)

## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries
# df = pd.read_csv(f"{_PATH}/{_FILENAME}.csv.xz")
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
max_t_df = df[df.t == max(df.t)]
reward = max_t_df.reward
nr_max_length = len(max_t_df.index)
print(#
f"""
mean reward  = {reward.mean():.3f}
st. dev.     = {reward.std():.3f}
n rep. max t = {nr_max_length}
total reps.  = {repetitions}
"""
)


## quick policy plot
policy_df = []
pops = np.linspace(0,1,100)
for rep in range(10):
  env.reset()
  popY = np.random.choice(df.Y.values)
  popZ = np.random.choice(df.Z.values)
  for popX in pops:
      obs = np.array([popX - 1, popY - 1, popZ - 1], dtype=np.float32)
      action = agent.compute_single_action(obs)
      escapement = max(popX*(1 - action[0]), 0)
      policy_df.append([popX, popY, popZ, escapement, action[0], rep])
      
policy_df = pd.DataFrame(policy_df, columns=["X", "Y", "Z","escapement","action","rep"])
escapement_plot = ggplot(policy_df, aes("X", "escapement", color = "Y")) + geom_point(shape=".")
escapement_plot.save(filename = f"esc_{_FILENAME}.png", path = _PATH)
action_plot = ggplot(policy_df, aes("X", "action", color = "Y")) + geom_point(shape=".")
action_plot.save(filename = f"act_{_FILENAME}.png", path = _PATH)


