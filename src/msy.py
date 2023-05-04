from envs import fish_tipping, growth_functions

import gymnasium as gym
import pandas as pd
import numpy as np
import ray

@ray.remote
def simulate(env, action):
  df = []
  for rep in range(30):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      df.append(np.append([t, rep, action, episode_reward], observation))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)

config = {}
config["growth_fn"] = growth_functions.v0_drift_growth
config["fluctuating"] = True
env = fish_tipping.three_sp(
    config = config
)


_DATACODE = "V0DRIFT"
env.training = False
actions = np.linspace(0,0.2,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in actions]
df = ray.get(parallel)

# convert to data.frame & write to csv
cols = ["t", "rep", "action", "reward", "X", "Y", "Z"]
df = pd.DataFrame(np.vstack(df), columns = cols)

df2 = (
  df
  .loc[df.t == df.t.max()]
  .groupby(["action"])
  .agg({"reward": "mean"})
)
print(df2)

tmp = (df[df.t == max(df.t)]
 .groupby('action', as_index=False)
 .agg({'reward': 'mean'})
 )
best = tmp[tmp.reward == tmp.reward.max()]
print(best)

# kinda slow to compress
# df2.to_csv(f"../data/{_DATACODE}/msy.csv.xz", index=False)

## Plot averages 
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

df3 = (df[df.action == best.action.values[0]]
       .melt(id_vars=["t", "action", "reward", "rep"])
       )

opt_msy_plot = ggplot(df3[df3.rep == 4.0], aes("t", "value", color="variable")) + geom_line()
opt_msy_plot.save(path = f"../data/{_DATACODE}", filename = "opt_msy_plot.png")


