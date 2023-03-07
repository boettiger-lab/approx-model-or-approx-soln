from src.envs import fish_tipping

import gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, escapement):
  x = []
  for rep in range(30):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      population = env.population()
      action = np.max([1 - escapement / (population[0] + 1e-12), 0]) # action is a mortality rate. 
      x.append(np.append([t, rep, escapement, action, episode_reward], population))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(x)

env = fish_tipping.three_sp()
env.training = False
esc_choices = np.linspace(0,1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in esc_choices]
x = ray.get(parallel)

cols = ["t", "rep", "escapement", "effort", "reward", "X", "Y", "Z"]
df = pd.DataFrame(np.vstack(x), columns = cols)

# serialize the results
# df.to_csv("data/escapement.csv.xz", index=False)


# Optional visualization & summary stats

    
    
# Determine which escapement scenario maximized mean reward at Tmax:
tmp = (df[df.t == max(df.t)]
 .groupby('escapement', as_index=False)
 .agg({'reward': 'mean'})
 )
best = tmp[tmp.reward == tmp.reward.max()]

 
## Plot averages 
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

df2 = df[df.rep == 4.0] # look at one rep
df4 = (df2[df2.escapement == best.escapement.values[0]]
       .melt(id_vars=["t", "escapement", "effort", "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             'escapement': 'mean',
             'effort': 'mean'})) 
ggplot(df4, aes("t", "value", color="variable")) + geom_line()
best

#ggplot(df4, aes("t", "reward", color="variable")) + geom_line()
#ggplot(df4, aes("t", "effort", color="variable")) + geom_line()



# df.groupby(['rep'], as_index=False).agg({'t': 'max'})
# df.groupby(['escapement'], as_index=False).agg({'reward': 'mean'})
