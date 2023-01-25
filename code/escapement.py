import gym_fishing
import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, action):
  df = []
  for rep in range(50):
    episode_reward = 0
    observation = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, action, episode_reward], observation))
      sp1_pop = (observation[0] + 1 ) # natural state-space
      effort = np.max([1 - action / sp1_pop, 0])
      observation, reward, terminated, info = env.step(effort)
      episode_reward += reward
      if terminated:
        break
  return(df)

env = gym.make("threeFishing-v2")
env.training = False
actions = np.linspace(0,1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in actions]
df = ray.get(parallel)

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("data/escapement.csv.gz", index=False)
