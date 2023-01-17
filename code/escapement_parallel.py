import gym_fishing
import gym
import pandas as pd
import numpy as np
import ray
from timebudget import timebudget

@ray.remote
def simulate(env, action, rep):    
  df = []
  episode_reward = 0
  observation = env.reset()
  for t in range(200):
    df.append(np.append([t, rep, action, episode_reward], observation))
    effort = np.min(1 - action / observation[0], 0)
    observation, reward, terminated, info = env.step(effort)
    episode_reward += reward
  return(df)

@ray.remote
def replicate_sim(env, action):
  simulate(env, action)

actions = np.linspace(-1,0,11)
env = gym.make("threeFishing-v2")


@timebudget
def parallel(operation, env, input):
  df = ray.get([operation.remote(env, i) for i in input])
  return(df)

#ray.init()
action = 0.1

input = range(10)
df = parallel(simulate, env, input)

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)

