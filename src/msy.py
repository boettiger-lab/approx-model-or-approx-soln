import sys
sys.path.insert(0, ".") # rstudio / repl_python doesn't add cwd to path
from src.envs import fish_tipping

import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, action):
  df = []
  for rep in range(30):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, action, episode_reward], observation))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)

env = fish_tipping.three_sp()
env.training = False
actions = np.linspace(0,0.1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in actions]
df = ray.get(parallel)

# convert to data.frame & write to csv
cols = ["t", "rep", "action", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("data/msy.csv.xz", index=False)




