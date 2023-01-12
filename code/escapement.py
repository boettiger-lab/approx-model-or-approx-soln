import gym_fishing
import gym
import pandas as pd
import numpy as np

# Note that state space is mapped to [-1,1]
#
# effort corresponding to constant escapement is
# effort = np.min(1 - escapement / observation[0], 0)

actions = np.linspace(-1,1,201)
env = gym.make("threeFishing-v2")
df = []
for action in actions:
  for rep in range(10):
    episode_reward = 0
    observation = env.reset()
    for t in range(200):
      df.append(np.append([t, rep, action, episode_reward], observation))
      effort = np.min(1 - action / observation[0], 0)
      observation, reward, terminated, info = env.step(effort)
      episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
df.to_csv("data/escapement.csv.gz", index=False)




