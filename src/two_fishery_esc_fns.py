""" functions related to constant escapement policies on 2-fishery scenarios """

import ray
import pandas as pd
import numpy as np
import itertools

@ray.remote
def simulate(env, esc_x, esc_y, repetitions):
  x = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    population = env.population()
    for t in range(env.Tmax):
      act_x = np.max([1 - esc_x / (population[0] + 1e-12), 0])
      act_y = np.max([1 - esc_y / (population[1] + 1e-12), 0])
      action = np.array([act_x, act_y], dtype = np.float32)
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      x.append(np.append([t, rep, esc_x, esc_y, act_x, act_y, episode_reward], population))
      population = env.population()
      if terminated:
        break
  return(x)

@ray.remote
def simulate_max_t(env, esc_x, esc_y, repetitions):
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

def generate_esc_episodes(env, grid_nr=51, repetitions=100):
  esc_x_choices = np.linspace(0,1,grid_nr)
  esc_y_choices = np.linspace(0,1,grid_nr)
  esc_choices = itertools.product(esc_x_choices, esc_y_choices)
  
  # define parllel loop and execute
  parallel = [simulate_max_t.remote(env, *i, repetitions) for i in esc_choices]
  x = ray.get(parallel) # list of tuples of final point, history

  cols = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
  X = list(map(list, zip(*x))) # [[final points], [histories]]
  df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
  df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
  return df_max_times, df

def find_best_esc(env, grid_nr=50, repetitions=100):
  df_max_times, df = generate_esc_episodes(env, grid_nr=grid_nr, repetitions=repetitions)
  tmp = (
    df_max_times
    .groupby(['esc_x', 'esc_y'], as_index=False)
    .agg({'reward': ['mean','std'] })
  )
  best = tmp[tmp[('reward','mean')] == tmp[('reward','mean')].max()]
  return (
    best, df.loc[
  		(df.esc_x == best.esc_x.values[0]) &
  		(df.esc_y == best.esc_y.values[0])
  	]#.melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
	)

def find_best_esc_several_params(
  env, param_name="r_x", param_vals=[1,0.5], grid_nr=51, repetitions=100,
):
  results = {}
  for val in param_vals:
    env.parameters[param_name] = val
    results[val] = find_best_esc(env, grid_nr=grid_nr, repetitions=repetitions)
  return results

def esc_grid_averages(env, grid_nr = 51, repetitions = 100):
  df_max_times, df = generate_esc_episodes(env, grid_nr=grid_nr, repetitions=repetitions)
  df_max_times = df_max_times[df_max_times.t == 199]
  del df
  return (
    df_max_times
    .groupby(['esc_x', 'esc_y'], as_index=False)
    .agg({'reward': 'mean', 'reward_std':'std'})
  )

def esc_grid_heatmap(env, path, filename, grid_nr=51, repetitions=100):
  import seaborn as sns
  import os
  import matplotlib.pyplot as plt
  plt.close()
  plt.figure(figsize=(10,10))
  # ticklabels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  df = esc_grid_averages(env, grid_nr = grid_nr, repetitions = repetitions)
  df = df[
    (df.esc_x >= 0.20) & (df.esc_x <= 0.80) &
    (df.esc_y >= 0.20) & (df.esc_y <= 0.80)
  ]
  df.to_csv(os.path.join(path,"esc_grid.csv.xz"))
  # following: https://stackoverflow.com/questions/74646588/how-to-create (cont.)
  # (cont.) -a-heatmap-in-python-with-3-columns-the-x-and-y-coordinates-and-t
  # stdev_annot = df.pivot(index='esc_x', columns='esc_y', values='reward_std')
  htmp = sns.heatmap(
    df.pivot(index='esc_x', columns='esc_y', values='reward'),
    cmap = plt.cm.get_cmap('viridis'),
    xticklabels = 5,
    yticklabels = 5,
  )
  fig = htmp.get_figure()
  fig.savefig(os.path.join(path,filename))
  
def esc_grid_ep_ends(env, grid_nr = 51, repetitions = 100):
  df_max_times, df = generate_esc_episodes(env, grid_nr=grid_nr, repetitions=repetitions)
  del df
  df_max_times['counter']= 0
  return (
    df_max_times[['esc_x', 'esc_y', 'counter']]
    .groupby(['esc_x', 'esc_y'], as_index=False)
    .count()
  ) # all columns other than esc_x, esc_y are gonna be the count
  
def esc_grid_ep_ends_heatmap(env, path, filename, grid_nr=51, repetitions=100):
  import seaborn as sns
  import os
  import matplotlib.pyplot as plt
  plt.close()
  plt.figure(figsize=(10,10))
  # ticklabels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  df = esc_grid_ep_ends(env, grid_nr = grid_nr, repetitions = repetitions)
  df = df[
    (df.esc_x >= 0.20) & (df.esc_x <= 0.80) &
    (df.esc_y >= 0.20) & (df.esc_y <= 0.80)
  ]
  df_all_full = df[df.counter == 100]
  print(df_all_200[["esc_x", "esc_y", "counter"]])
  
  """
  df.to_csv(os.path.join(path,"esc_grid.csv.xz"))
  # following: https://stackoverflow.com/questions/74646588/how-to-create (cont.)
  # (cont.) -a-heatmap-in-python-with-3-columns-the-x-and-y-coordinates-and-t
  # stdev_annot = df.pivot(index='esc_x', columns='esc_y', values='reward_std')
  htmp = sns.heatmap(
    df.pivot(index='esc_x', columns='esc_y', values='reward'),
    cmap = plt.cm.get_cmap('viridis'),
    xticklabels = 5,
    yticklabels = 5,
  )
  fig = htmp.get_figure()
  fig.savefig(os.path.join(path,filename))
  """
  
	
# OLD:

"""
def generate_esc_episodes(env, grid_nr=30, repetitions=100):
	esc_x_choices = np.linspace(0,1,grid_nr)
	esc_y_choices = np.linspace(0,1,grid_nr)
	esc_choices = itertools.product(esc_x_choices, esc_y_choices)

	# define parllel loop and execute
	parallel = [simulate.remote(env, *i, repetitions=repetitions) for i in esc_choices]
	x = ray.get(parallel)

	cols = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
	df = pd.DataFrame(np.vstack(x), columns = cols)
	print("Done generating episodes.")
	return df
"""
"""
def find_best_esc(env, repetitions=100):
	df = generate_esc_episodes(env)
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
	#
	tmp = (
	df_max_times
	.groupby(['esc_x', 'esc_y'], as_index=False)
	.agg({'reward': 'mean'})
	)
	best = tmp[tmp.reward == tmp.reward.max()]
	return best, df.loc[
		(df.esc_x == best.esc_x.values[0]) &
		(df.esc_y == best.esc_y.values[0])
	].melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
"""
