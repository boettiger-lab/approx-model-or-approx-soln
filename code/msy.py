import gym_fishing
import gym
import pandas as pd
import numpy as np
import ray
from timebudget import timebudget

@ray.remote
def simulate(env, action):
  df = []
  for rep in range(10):
    episode_reward = 0
    observation = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, action, episode_reward], observation))
      observation, reward, terminated, info = env.step(action)
      episode_reward += reward
  return(df)

# parallelize over actions
@timebudget
def parallel(operation, env, input):
  df = ray.get([operation.remote(env, i) for i in input])
  return(df)


env = gym.make("threeFishing-v2")
actions = np.linspace(0,.1,101)
df = parallel(simulate, env, actions)

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("data/msy.csv.gz", index=False)



