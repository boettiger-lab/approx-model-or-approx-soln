from envs import fish_tipping, growth_functions

import gymnasium as gym
import pandas as pd
import numpy as np
import ray
import itertools

repetitions = 100

@ray.remote
def simulate(env, escapement):
  x = []
  for rep in range(repetitions):
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

config = {}
config["growth_fn"] = growth_functions.rx_drift_growth
config["fluctuating"] = True
env = fish_tipping.three_sp(
    config = config
)
env.training = False
esc_choices = np.linspace(0,1,101)

_DATACODE = "RXDRIFT/2"

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in esc_choices]
x = ray.get(parallel)

cols = ["t", "rep", "esc", "act", "reward", "X", "Y", "Z"]
df = pd.DataFrame(np.vstack(x), columns = cols)
print("Done generating episodes.")

df_max_times = pd.DataFrame([], columns = df.columns)
for rep, esc in itertools.product(
  range(repetitions), esc_choices
  ):
  print(f"optimizing rep {rep}, esc = {esc:.3f}", end="\r")
  rep_df = df.loc[
    (df.rep == rep) &
    (df.esc == esc)
  ] # single episode
  df_max_times = pd.concat([
    df_max_times,
    rep_df.loc[rep_df.t == rep_df.t.max()]
  ])

tmp = (
  df_max_times
  .groupby(['esc'], as_index=False)
  .agg({'reward': 'mean'})
)
best = tmp[tmp.reward == tmp.reward.max()]
print(best)

"""
parallel = [simulate.remote(env, i) for i in esc_choices]
x = ray.get(parallel)

cols = ["t", "rep", "escapement", "effort", "reward", "X", "Y", "Z"]
df = pd.DataFrame(np.vstack(x), columns = cols)
"""
# serialize the results
# df.to_csv("data/escapement.csv.xz", index=False)


# Optional visualization & summary stats

 
## Plot averages 
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

df2 = df[df.rep == 4.0] # look at one rep
df4 = (df2[df2.esc == best.esc.values[0]]
       .melt(id_vars=["t", "esc", "act", "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             'esc': 'mean',
             'act': 'mean'})) 
opt_esc_plot = ggplot(df4, aes("t", "value", color="variable")) + geom_line()
opt_esc_plot.save(path = f"../data/{_DATACODE}", filename = "opt_esc_plot.png")

#print(df2["escapement"][0:200])

#ggplot(df4, aes("t", "reward", color="variable")) + geom_line()
#ggplot(df4, aes("t", "effort", color="variable")) + geom_line()



# df.groupby(['rep'], as_index=False).agg({'t': 'max'})
# df.groupby(['escapement'], as_index=False).agg({'reward': 'mean'})
