from src.envs import fish_tipping

import gym
import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

env = fish_tipping.three_sp()
env.training = False
x = []
esc_level=1
episode_reward = 0
observation, _ = env.reset()
for t in range(env.Tmax):
  population = env.population()
  action = np.max([1 - esc_level / (population[0]+1e-12), 0]) # action is a mortality rate. 
  x.append(np.append([t, esc_level, action, episode_reward], population))
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break

print(episode_reward)

cols = ["t", "escapement", "effort", "reward", "X", "Y", "Z"]
df2 = pd.DataFrame(x,columns=cols)

df = (df2
       .melt(id_vars=["t", "escapement", "effort", "reward"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             'escapement': 'mean',
             'effort': 'mean'})) 
ggplot(df, aes("t", "value", color="variable")) + geom_line()
