import sys
sys.path.insert(0, ".") # rstudio / repl_python doesn't add cwd to path
from envs import fish_tipping

import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, esc_level):
  df = []
  for rep in range(50):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      population = env.population()
      df.append(np.append([t, rep, esc_level, episode_reward], observation))
      action = np.max([1 - esc_level / population[0], 0]) # action is a mortality rate. 
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)

env = fish_tipping.three_sp()
env.training = True
env.threshold = 0
esc_choices = np.linspace(0,1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in esc_choices]
df = ray.get(parallel)

cols = ["t", "rep", "escapement", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("../data/escapement.csv.xz", index=False)
