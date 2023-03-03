from envs import fish_tipping

import gym
import pandas as pd
import numpy as np
import ray
env = fish_tipping.three_sp()
env.training = False
df = []
episode_reward = 0
observation, _ = env.reset()
esc_level = 0.6
for t in range(env.Tmax):
  population = env.population()
  df.append(np.append([t, esc_level, episode_reward], observation))
  action = np.max([1 - esc_level / population[0], 0]) # action is a mortality rate. 
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward

print(episode_reward)

cols = ["t", "escapement", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(df,columns=cols)

from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

ggplot(df2, aes("t", "X")) + geom_line()


