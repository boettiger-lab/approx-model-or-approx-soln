""" functions related to constant escapement policies on 1-fishery scenarios """

import ray
import pandas as pd
import numpy as np
import itertools

@ray.remote
def simulate_max_t_1fish(env, esc, repetitions):
  x = []
  path = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    T = 0
    for t in range(env.Tmax):
      population = env.population()
      act= np.max([1 - esc / (population[0] + 1e-12), 0])
      action = np.array([act], dtype = np.float32)
      path.append(np.append([t, rep, esc, act, episode_reward], population))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        T = t
        break
      else:
        T = env.Tmax
    x.append(np.append([T, rep, esc, act, episode_reward], population))
  return(x, path)

def generate_esc_episodes_1fish(env, grid_nr=101, repetitions=100, only_max_times = False):
  esc_choices = np.linspace(0,1,grid_nr)

  # define parllel loop and execute
  parallel = [simulate_max_t_1fish.remote(env, i, repetitions) for i in esc_choices]
  x = ray.get(parallel) # list of tuples of final point, history
  
  if env.num_species == 3:
    cols = ["t", "rep", "esc", "act", "reward", "X", "Y", "Z"]
  else: # assume 1 species as default
    cols = ["t", "rep", "esc", "act", "reward", "X"]
    
  X = list(map(list, zip(*x))) # [[final points], [histories]]
  df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
  df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
  if only_max_times:
    return df_max_times
  return df_max_times, df

def find_best_esc_1fish(env, grid_nr=101, repetitions=100):
  df_max_times, df = generate_esc_episodes_1fish(env, grid_nr=grid_nr, repetitions=repetitions)
  tmp = (
    df_max_times
    .groupby(['esc'], as_index=False)
    .agg({'reward': ['mean','std'] })
  )
  best = tmp[tmp[('reward','mean')] == tmp[('reward','mean')].max()]
  return (
    best, df.loc[
  		df.esc == best.esc.values[0]
  	]
	)
	
def esc_grid_averages_1fish(env, grid_nr = 101, repetitions = 100):
  df_max_times = generate_esc_episodes(
    env, grid_nr=grid_nr, repetitions=repetitions, only_max_times=True
  )
  df_max_times['reward_std'] = df_max_times['reward']
  return (
    df_max_times
    .groupby(['esc_x', 'esc_y'], as_index=False)
    .agg({'reward': 'mean', 'reward_std':'std'})
  )
