import ray
import pandas as pd
import numpy as np
import itertools

@ray.remote
def msy_max_t_1fish(env, mortality, repetitions):
  x = []
  path = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    T = 0
    for t in range(env.Tmax):
      population = env.population()
      act= mortality
      action = np.array([act], dtype = np.float32)
      path.append([t, rep, mortality, act, episode_reward, *population])
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        T = t
        break
      else:
        T = env.Tmax
    x.append([T, rep, mortality, act, episode_reward, *population])
  return(x, path)

@ray.remote
def msy_max_t(env, mortality_x, mortality_y, repetitions):
  x = []
  path = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    T = 0
    for t in range(env.Tmax):
      population = env.population()
      act_x, act_y= mortality_x, mortality_y
      action = np.array([act_x, act_y], dtype = np.float32)
      path.append([t, rep, mortality_x, mortality_y, act_x, act_y, episode_reward, *population])
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        T = t
        break
      else:
        T = env.Tmax
    x.append([T, rep, mortality_x, mortality_y, act_x, act_y, episode_reward, *population])
  return(x, path)

def generate_msy_episodes(env, grid_nr=101, repetitions=100, only_max_times = False):
  mortality_choices = np.linspace(0,0.5,grid_nr)

  # define parllel loop and execute
  parallel = [msy_max_t_1fish.remote(env, i, repetitions) for i in mortality_choices]
  x = ray.get(parallel) # list of tuples of final point, history
  
  cols = ["t", "rep", "mortality", "act", "reward", "X", "Y", "Z"] # assume 3 species as default
  if env.num_species == 1:
    cols = ["t", "rep", "mortality", "act", "reward", "X"]
    
  X = list(map(list, zip(*x))) # [[final points], [histories]]
  df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
  df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
  if only_max_times:
    return df_max_times
  return df_max_times, df

def generate_msy_episodes_2fish(env, grid_nr=51, repetitions=100, only_max_times = False):
  mortality_choices = itertools.product(np.linspace(0,0.5,grid_nr), repeat=2)

  # define parllel loop and execute
  parallel = [msy_max_t.remote(env, *i, repetitions) for i in mortality_choices]
  x = ray.get(parallel) # list of tuples of final point, history
  
  cols = ["t", "rep", "mortality_x", "mortality_y", "act_x", "act_y", "reward", "X", "Y", "Z"] 
    
  X = list(map(list, zip(*x))) # [[final points], [histories]]
  df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
  df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
  if only_max_times:
    return df_max_times
  return df_max_times, df

def generate_msy_episodes_1fish(env, grid_nr=101, repetitions=100, only_max_times = False):
  mortality_choices = np.linspace(0,0.5,grid_nr)

  # define parllel loop and execute
  parallel = [msy_max_t_1fish.remote(env, i, repetitions) for i in mortality_choices]
  x = ray.get(parallel) # list of tuples of final point, history
  
  cols = ["t", "rep", "mortality", "act", "reward", "X", "Y", "Z"] # assume 1 species as default
  if env.num_species == 1:
    cols = ["t", "rep", "mortality", "act", "reward", "X"]
    
  X = list(map(list, zip(*x))) # [[final points], [histories]]
  df_max_times = pd.DataFrame(np.vstack(X[0]), columns = cols) # X[0] = [f. pts.]
  df = pd.DataFrame(np.vstack(X[1]), columns = cols) # X[1] = [histories]
  if only_max_times:
    return df_max_times
  return df_max_times, df

def find_msy_1fish(env, grid_nr=101, repetitions=100):
  df_max_times, df = generate_msy_episodes_1fish(env, grid_nr=grid_nr, repetitions=repetitions)
  tmp = (
    df_max_times
    .groupby(['mortality'], as_index=False)
    .agg({'reward': ['mean','std'] })
  )
  best = tmp[tmp[('reward','mean')] == tmp[('reward','mean')].max()]
  return (
    best, df.loc[
  		df.mortality == best.mortality.values[0]
  	]
	)
	
def find_msy_2fish(env, grid_nr=51, repetitions=100):
  df_max_times, df = generate_msy_episodes_2fish(env, grid_nr=grid_nr, repetitions=repetitions)
  tmp = (
    df_max_times
    .groupby(['mortality_x', 'mortality_y'], as_index=False)
    .agg({'reward': ['mean','std'] })
  )
  best = tmp[tmp[('reward','mean')] == tmp[('reward','mean')].max()]
  return (
    best, df.loc[
  		(df.mortality_x == best.mortality_x.values[0]) &
  		(df.mortality_y == best.mortality_y.values[0])
  	]
	)

def msy_performances_1fish(env, grid_nr = 101, repetitions = 100):
  df_max_times = generate_msy_episodes_1fish(
    env, grid_nr=grid_nr, repetitions=repetitions, only_max_times=True
  )
  df_max_times['reward_std'] = df_max_times['reward']
  return (
    df_max_times
    .groupby(['act'], as_index=False)
    .agg({'reward': 'mean', 'reward_std':'std'})
  )
