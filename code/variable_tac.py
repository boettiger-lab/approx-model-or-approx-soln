import gym_fishing
import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, fixed_action, threshold_pop = 0.5):
  df = []
  for rep in range(50):
    episode_reward = 0
    observation = env.reset()
    for t in range(env.Tmax):
      action = fixed_action
      if observation[0]+1 < threshold_pop:
        action = action * (observation[0] + 1) / threshold_pop
      df.append(np.append([t, rep, fixed_action, action, episode_reward], observation))
      observation, reward, terminated, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)


env = gym.make("threeFishing-v2")
env.training = False
actions = np.linspace(0,0.2,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in actions]
df = ray.get(parallel)

# convert to data.frame & write to csv
cols = ["t", "rep", "policy_action", "action", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(np.vstack(df), columns = cols)
df2.to_csv("data/var_tac.csv.xz", index=False)

