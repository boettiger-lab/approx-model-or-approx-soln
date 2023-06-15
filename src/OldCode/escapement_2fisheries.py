from envs import two_three_fishing, growth_functions
from parameters import parameters
import gymnasium as gym
import ray
import pandas as pd
import numpy as np
import itertools

_experiment_nr = 1
_DATACODE = "2FISHERY/KLIMIT_RXDRIFT"
_DEFAULT_PARAMETERS = parameters().parameters()

fineness = 51
repetitions = 100

@ray.remote
def simulate(env, esc_x, esc_y):
  x = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      population = env.population()
      act_x = np.max([1 - esc_x / (population[0] + 1e-12), 0])
      act_y = np.max([1 - esc_y / (population[1] + 1e-12), 0])
      action = np.array([act_x, act_y], dtype = np.float32)
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      x.append(np.append([t, rep, esc_x, esc_y, act_x, act_y, episode_reward], population))
      if terminated:
        break
  return(x)

@ray.remote
def simulate_max_t(env, esc_x, esc_y):
  x = []
  path = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    T = 0
    for t in range(env.Tmax):
      population = env.population()
      act_x = np.max([1 - esc_x / (population[0] + 1e-12), 0])
      act_y = np.max([1 - esc_y / (population[1] + 1e-12), 0])
      action = np.array([act_x, act_y], dtype = np.float32)
      path.append(np.append([t, rep, esc_x, esc_y, act_x, act_y, episode_reward], population))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        T = t
        break
      else:
        T = env.Tmax
    x.append(np.append([T, rep, esc_x, esc_y, act_x, act_y, episode_reward], population))
  return(x, path)


config = {}
config["parameters"] = _DEFAULT_PARAMETERS
config["growth_fn"] = growth_functions.K_limit_rx_drift_growth
config["fluctuating"] = True
config["training"] = True
config["initial_pop"] = parameters().init_state()
print(f"initial pop: {[x for x in config['initial_pop']]}")
env = two_three_fishing.twoThreeFishing(
    config = config
)
esc_x_choices = np.linspace(0,1,fineness)
esc_y_choices = np.linspace(0,1,fineness)
esc_choices = itertools.product(esc_x_choices, esc_y_choices)

# define parllel loop and execute
parallel = [simulate_max_t.remote(env, *i) for i in esc_choices]
x = ray.get(parallel) # list of tuples of final point, history
print("done generating data!")

cols = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
X = list(map(list, zip(*x))) # [[final points], [histories]]
df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
print("Done generating episode dataframes.")

"""
df_max_times = pd.DataFrame([], columns = df.columns)
for rep, esc_x, esc_y in itertools.product(
  range(repetitions), esc_x_choices, esc_y_choices
  ):
  print(f"optimizing rep {rep}, esc = {esc_x:.3f}, {esc_y:.3f}", end="\r")
  rep_df = df.loc[
    (df.rep == rep) &
    (df.esc_x == esc_x) & 
    (df.esc_y == esc_y)
  ] # single episode
  df_max_times = pd.concat([
    df_max_times,
    rep_df.loc[rep_df.t == rep_df.t.max()]
  ])
"""

tmp = (
  df_max_times
  .groupby(['esc_x', 'esc_y'], as_index=False)
  .agg({'reward': ['mean','std'] })
)

best = tmp[tmp[('reward','mean')] == tmp[('reward','mean')].max()]
print(best)

"""
## Commented out this block that had the bug, but will keep for now jik.

# Determine which escapement scenario maximized mean reward at Tmax:
tmp = (df[df.t == max(df.t)]
 .groupby(['esc_x', 'esc_y'], as_index=False)
 .agg({'reward': 'mean'})
 )
best = tmp[tmp.reward == tmp.reward.max()]
df_best = df.loc[
  (df.esc_x == best.esc_x.values[0]) &
  (df.esc_y == best.esc_y.values[0])
]
# pedestrian way:
df_best_max_t = pd.DataFrame([], columns = ["rep", "t", "reward"])
for rep in range(30):
  rep_df = df_best.loc[df.rep == rep][["rep", "t", "reward"]]
  # print(rep_df.loc[rep_df.t == rep_df.t.max()][["rep", "t", "reward"]])
  df_best_max_t = pd.concat(
    [
      df_best_max_t,
      rep_df.loc[rep_df.t == rep_df.t.max()]
    ]
  )

print(df_best_max_t)  
"""
## Plot averages 
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_histogram

df2 = df[df.rep == 4.0] # look at one rep
df4 = (df2.loc[
  (df2.esc_x == best[('esc_x','')].values[0]) &
  (df2.esc_y == best[('esc_y','')].values[0])
  ]
       .melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
       ) 
#print(df4[["t","reward"]])
# print(best)
df4.to_csv("opt_esc.csv")
opt_esc_plot = ggplot(df4, aes("t", "value", color="variable")) + geom_line()
opt_esc_plot.save(path = f"../data/{_DATACODE}/{_experiment_nr}", filename = "opt_esc_plot.png")

opt_max_times = df_max_times.loc[
  (df_max_times.esc_x == best[('esc_x','')].values[0]) &
  (df_max_times.esc_y == best[('esc_y','')].values[0])
  ]
opt_max_times.to_csv(f"../data/{_DATACODE}/{_experiment_nr}/esc.csv")
#print(opt_max_times.t)

max_t_vals = opt_max_times[["t"]]
t_hist = ggplot(data=max_t_vals, mapping=aes(x='t')) + geom_histogram(binwidth=1)
t_hist.save(filename=f"../data/{_DATACODE}/{_experiment_nr}/esc_maxtimes_histogram.png")
