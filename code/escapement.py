import gym_fishing
import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, esc_level):
  df = []
  for rep in range(50):
    episode_reward = 0
    observation = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, esc_level, episode_reward], observation))
      sp1_pop = (observation[0] + 1 ) # natural state-space
      action = np.max([1 - esc_level / sp1_pop, 0]) # action is a mortality rate. 
      observation, reward, terminated, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)

env = gym.make("threeFishing-v2")
env.training = False
esc_choices = np.linspace(0,1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in esc_choices]
df = ray.get(parallel)

cols = ["t", "rep", "escapement", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("data/escapement.csv.xz", index=False)
