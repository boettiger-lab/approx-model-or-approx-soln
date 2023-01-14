import gym_fishing
import gym
import pandas as pd
import numpy as np

# Note that state space is mapped to [-1,1]
#
# effort corresponding to constant escapement is
# effort = max(1 - escapement / sp1_pop, 0)

actions = np.linspace(0,1,101)
env = gym.make("threeFishing-v2")
env.reset()
df = []
for action in actions:
  for rep in range(10):
    episode_reward = 0
    observation = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, action, episode_reward], observation))
      sp1_pop = (observation[0] + 1 ) # natural state-space
      effort = np.max([1 - action / sp1_pop, 0])
      observation, reward, terminated, info = env.step(effort)
      if terminated:
        break
      episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
df.to_csv("data/escapement.csv.gz", index=False)


