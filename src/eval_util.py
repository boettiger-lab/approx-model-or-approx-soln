import numpy as np
import pandas as pd
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
import patchworklib as pw
import itertools as it
import ray

# 2 fishery

def generate_episodes(
  agent, env, reps = 50,  
):
  """ 
  Generate a set of env episodes controlled by an RL agent. 
  - agent: PPO RLlib agent, built with PPOConfig().build()
  - env:   two_three_fishing.twoThreeFishing env
  - reps:  number of episodes run
  """
  df_list = []
  for rep in range(reps):
    episode_reward = 0
    observation, _ = env.reset()
    population = env.population()
    for t in range(env.Tmax):
      action = agent.compute_single_action(observation)
      esc_x = population[0] * (1 - action[0])
      esc_y = population[1] * (1 - action[1])
      df_list.append(np.append([t, rep, action[0], action[1], episode_reward, esc_x, esc_y], population))
      observation, reward, terminated, done, info = env.step(action)
      population = env.population()
      episode_reward += reward
      if terminated:
        break
  cols = ["t", "rep", "act_x", "act_y", "reward", "esc_x", "esc_y", "X", "Y", "Z"]
  df = pd.DataFrame(df_list, columns = cols)
  return df

def generate_gpp_episodes(gpp, env, reps=50):
  """ gpp is a gaussian process regression of the RL policy """
  df_list = []
  for rep in range(reps):
    episode_reward = 0
    observation, _ = env.reset()
    population = env.population()
    for t in range(env.Tmax):
      action = gpp.predict([population])[0]
      #print(f"pop = {population}")
      #print(f"action = {action}")
      esc_x = population[0] * (1 - action[0])
      esc_y = population[1] * (1 - action[1])
      
      df_list.append(np.append([t, rep, action[0], action[1], episode_reward, esc_x, esc_y], population))
      
      observation, reward, terminated, done, info = env.step(action)
      population = env.population()
      
      episode_reward += reward
      if terminated:
        break
  cols = ["t", "rep", "act_x", "act_y", "reward", "esc_x", "esc_y", "X", "Y", "Z"]
  df = pd.DataFrame(df_list, columns = cols)
  return df

def episode_plots(df, *, path_and_filename = "path_plots.png"):
  """
  Generates plots for the episodes provided in the df.
  - df:                meant to be df with same columns as those generated with
                       generate_episodes.
  - path_and_filename: the full path and filename which you want to be saved 
                       (i.e. like fname for plt.savefig)
  """
  df2 = df.melt(id_vars=["t", "rep"])
  
  esc_t_plot = ggplot(
    df2.loc[
      (df2.variable == "esc_x") | (df2.variable == "esc_y")
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  act_t_plot = ggplot(
    df2.loc[
      (df2.variable == "act_x") | (df2.variable == "act_y")
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  xyz_t_plot = ggplot(
    df2.loc[
      (df2.variable == "X") | (df2.variable == "Y") | (df2.variable == "Z")
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  rew_t_plot = ggplot(
    df2.loc[
      (df2.variable == "reward")
    ].groupby(["rep"]), 
    aes("t", "value")
  ) + geom_line()
  
  
  pw_esc = pw.load_ggplot(esc_t_plot, figsize = (5,2))
  pw_act = pw.load_ggplot(act_t_plot, figsize = (5,2))
  pw_xyz = pw.load_ggplot(xyz_t_plot, figsize = (5,2))
  pw_rew = pw.load_ggplot(rew_t_plot, figsize = (5,2))
  
  the_plot = (((pw_esc / pw_act) / pw_xyz) / pw_rew)
  the_plot.savefig(path_and_filename)
  
def evaluate_policy(
  agent, 
  range_X = (0,1), 
  range_Y = (0,1), 
  range_Z = (0,1),
  bound = 4,
  ):
  """ ranges in large [0,bound] 'population space' (0,4 in practice) """
  # X - policy
  x_policy = []
  for x in np.linspace(0,1,100):
    for yz in it.product(
      np.linspace(*range_Y,5), 
      np.linspace(*range_Z,5), 
      ):
      pop = np.array([x, *yz], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      x_policy.append([*pop, *action])
  
  x_policy_df = pd.DataFrame(
    x_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
  
  # Y - policy
  y_policy = []
  for y in np.linspace(0,1,100):
    for xz in it.product(
      np.linspace(*range_X,5), 
      np.linspace(*range_Z,5), 
      ):
      x, z = xz
      pop = np.array([x, y, z], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      y_policy.append([*pop, *action])
  
  y_policy_df = pd.DataFrame(
    y_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
      
  # Z - policy
  z_policy = []
  for z in np.linspace(0,1,100):
    for xy in it.product(
      np.linspace(*range_X,5), 
      np.linspace(*range_Y,5), 
      ):
      pop = np.array([*xy, z], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      z_policy.append([*pop, *action])
  
  z_policy_df = pd.DataFrame(
    z_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
  
  return x_policy_df, y_policy_df, z_policy_df
  
def state_policy_plot(agent, env, path_and_filename = None):
  x_policy_df, y_policy_df, z_policy_df = evaluate_policy_in_popular_windows(agent, env)
  
  x_act_plt1 = ggplot(x_policy_df, aes("X", "act_x", color = "Y")) + geom_point(shape=".")
  x_act_plt2 = ggplot(x_policy_df, aes("X", "act_y", color = "Y")) + geom_point(shape=".")
  
  y_act_plt1 = ggplot(y_policy_df, aes("Y", "act_x", color = "X")) + geom_point(shape=".")
  y_act_plt2 = ggplot(y_policy_df, aes("Y", "act_y", color = "X")) + geom_point(shape=".")
  
  x_act_pltz = ggplot(x_policy_df, aes("X", "act_x", color = "Z")) + geom_point(shape=".")
  y_act_pltz = ggplot(y_policy_df, aes("Y", "act_y", color = "Z")) + geom_point(shape=".")
  
  pw_x1 = pw.load_ggplot(x_act_plt1, figsize = (3,2))
  pw_x2 = pw.load_ggplot(x_act_plt2, figsize = (3,2))
  
  pw_y1 = pw.load_ggplot(y_act_plt1, figsize = (3,2))
  pw_y2 = pw.load_ggplot(y_act_plt2, figsize = (3,2))
  
  pw_z1 = pw.load_ggplot(x_act_pltz, figsize = (3,2))
  pw_z2 = pw.load_ggplot(y_act_pltz, figsize = (3,2))
  
  the_plot = (pw_x1 / pw_x2) | (pw_y1 / pw_y2) | (pw_z1 / pw_z2)
  
  if path_and_filename is not None:
    the_plot.savefig(path_and_filename)
  
  return pd.concat([x_policy_df, y_policy_df, z_policy_df])

# Interpolate policy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#@ray.remote
def GaussianProcessPolicy(policy_df, length_scale=10, noise_level=0.1):
  """
  policy_df.columns = [X, Y, Z, act_x, act_y]
                    -> action (act_x, act_y) taken at point (X, Y, Z)
  """
  predictors = policy_df[["X", "Y", "Z"]].to_numpy()
  targets = policy_df[["act_x", "act_y"]].to_numpy()
  kernel = (
    1.0 * RBF(length_scale = length_scale) 
    + WhiteKernel(noise_level=noise_level)
    )
  gpp = (
    GaussianProcessRegressor(kernel=kernel, random_state=0)
    .fit(predictors, targets)
    )
  return gpp


  
def path_policy_plot(agent, path_df, path_and_filename):
  ...
  
def popular_ranges_fixed_size(df, var: str, rel_size = 0.3):
  """ 
  Determines most populated range for var in df. 
  """
  min_val = df[var].min()
  max_val = df[var].max()
  size = rel_size * (max_val - min_val)
  
  histogram =  {}
  for window_start in np.linspace(min_val, max_val - size, 100):
    histogram[window_start] = (
      sum(
        (
          (df[var] >= window_start) &
          (df[var] <= window_start+size)
        )
      )
    )
  
  opt_window_start = max(histogram, key=histogram.get)
  return {opt_window_start: histogram[opt_window_start]}
  
def popular_ranges(df, var: str, min_fraction = 0.7):
  """ vary rel_size parameter in popular_ranges_fixed_size"""
  rel_size_list = [0.8 ** i for i in range(11)]
  N = len(df.index)
  popular_ranges_dict = {}
  for rel_size in rel_size_list:
    fixed_size_opt_range = popular_ranges_fixed_size(df, var, rel_size=rel_size)
    if list(fixed_size_opt_range.values())[0] > min_fraction * N:
      # only add the ones that satisfy the min_fraction condition
      popular_ranges_dict[rel_size] = (
        fixed_size_opt_range
      )
  
  min_popular_range_size = min(popular_ranges_dict.keys())
  return min_popular_range_size, popular_ranges_dict[min_popular_range_size]
  
def evaluate_policy_in_popular_windows(agent, env):
  # generate episodes
  df = generate_episodes(agent, env)
  
  # popular ranges:
  popular_x_size, popular_x_window_dict = (
    popular_ranges(df, var="X")
  )
  X_window_start = list(popular_x_window_dict.keys())[0] # dicts weren't the right choice
  #
  popular_y_size, popular_y_window_dict = (
    popular_ranges(df, var="Y")
  )
  Y_window_start = list(popular_y_window_dict.keys())[0]
  #
  popular_z_size, popular_z_window_dict = (
    popular_ranges(df, var="Z")
  )
  Z_window_start = list(popular_z_window_dict.keys())[0]
  
  ## All of these are in population space [0, env.bound] (=[0,4] for practical purposes)
  
  return evaluate_policy(
  agent, 
  range_X = (X_window_start, X_window_start + popular_x_size), 
  range_Y = (Y_window_start, Y_window_start + popular_y_size), 
  range_Z = (Z_window_start, Z_window_start + popular_z_size)
  )


#### Wrangle data from saved csv's

def get_times_rewards(filename):
  df = pd.read_csv(filename)
  max_t_rew = []
  for rep in range(50):
    rep_df = df.loc[df.rep == rep]
    rep_tmax_df = rep_df.loc[rep_df.t == rep_df.t.max()]
    max_t_rew.append([rep, rep_tmax_df.t.values[0], rep_tmax_df.reward.values[0]])
  return pd.DataFrame(max_t_rew, columns=["rep", "t","reward"])

def values_at_max_t(df, group="rep"):
  """ 
  df separated into different groups labeled by the group argument, 
  and max t row retrieved for each group 
  """
  if (group not in df) or ('t' not in df):
    raise ValueError(f"{df} argument in values_at_max_t() does not have a {group} attribute.")
  max_t_rows_list = []
  for g in range(int(df[group].max()+1)):
    group_df = df.loc[df[group] == g]
    group_df_at_tmax = group_df.loc[group_df.t == group_df.t.max()]
    max_t_rows_list.append(group_df_at_tmax)
  return pd.concat(max_t_rows_list)

## Statistical tests part of evalutation
from scipy.stats import ttest_ind

def ppo_vs_esc_ttest(ppo_file_and_path, esc_file_and_path):
  esc = pd.read_csv(esc_file_and_path) # only final T points saved in esc.csv files right now
  ppo_full = pd.read_csv(ppo_file_and_path) # full paths saved instead
  ppo = values_at_max_t(ppo_full, group="rep")
  return {
    'reward':ttest_ind(ppo['reward'], esc['reward']),
    'tmax':ttest_ind(ppo['t'], esc['t'])
    }
  
  


# 1 fishery

def generate_episodes_1fish(
  agent, env, reps = 50,  
):
  """ 
  Generate a set of env episodes controlled by an RL agent. 
  - agent: PPO RLlib agent, built with PPOConfig().build()
  - env:   two_three_fishing.twoThreeFishing env
  - reps:  number of episodes run
  """
  df_list = []
  for rep in range(reps):
    episode_reward = 0
    observation, _ = env.reset()
    population = env.population()
    for t in range(env.Tmax):
      action = agent.compute_single_action(observation)
      esc = population[0] * (1 - action[0])
      df_list.append(np.append([t, rep, action[0], episode_reward, esc], population))
      observation, reward, terminated, done, info = env.step(action)
      population = env.population()
      episode_reward += reward
      if terminated:
        break
  cols = ["t", "rep", "act", "reward", "esc", "X", "Y", "Z"]
  df = pd.DataFrame(df_list, columns = cols)
  return df



def generate_gpp_episodes_1fish(gpp, env, reps=50):
  """ gpp is a gaussian process regression of the RL policy """
  df_list = []
  for rep in range(reps):
    episode_reward = 0
    observation, _ = env.reset()
    population = env.population()
    for t in range(env.Tmax):
      action = gpp.predict([population])[0]
      #print(f"pop = {population}")
      #print(f"action = {action}")
      esc = population[0] * (1 - action)
      
      df_list.append(np.append([t, rep, action, episode_reward, esc], population))
      
      observation, reward, terminated, done, info = env.step(action)
      population = env.population()
      
      episode_reward += reward
      if terminated:
        break
  cols = ["t", "rep", "act", "reward", "esc", "X", "Y", "Z"]
  df = pd.DataFrame(df_list, columns = cols)
  return df


def episode_plots_1fish(df, *, path_and_filename = "path_plots.png"):
  """
  Generates plots for the episodes provided in the df.
  - df:                meant to be df with same columns as those generated with
                       generate_episodes_1fish.
  - path_and_filename: the full path and filename which you want to be saved 
                       (i.e. like fname for plt.savefig)
  """
  df2 = df.melt(id_vars=["t", "rep"])
  
  esc_t_plot = ggplot(
    df2.loc[
      df2.variable == "esc"
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  act_t_plot = ggplot(
    df2.loc[
      df2.variable == "act"
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  xyz_t_plot = ggplot(
    df2.loc[
      (df2.variable == "X") | (df2.variable == "Y") | (df2.variable == "Z")
    ].groupby(["rep", "variable"]), 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  rew_t_plot = ggplot(
    df2.loc[
      (df2.variable == "reward")
    ].groupby(["rep"]), 
    aes("t", "value")
  ) + geom_line()
  
  
  pw_esc = pw.load_ggplot(esc_t_plot, figsize = (5,2))
  pw_act = pw.load_ggplot(act_t_plot, figsize = (5,2))
  pw_xyz = pw.load_ggplot(xyz_t_plot, figsize = (5,2))
  pw_rew = pw.load_ggplot(rew_t_plot, figsize = (5,2))
  
  the_plot = (((pw_esc / pw_act) / pw_xyz) / pw_rew)
  the_plot.savefig(path_and_filename)


def state_policy_plot_1fish(agent, env, path_and_filename = None):
  x_policy_df, y_policy_df, z_policy_df = evaluate_policy_in_popular_windows_1fish(agent, env)
  
  x_act_plt1 = ggplot(x_policy_df, aes("X", "act", color = "Y")) + geom_point(shape=".")
  x_act_pltz = ggplot(x_policy_df, aes("X", "act", color = "Z")) + geom_point(shape=".")

  y_act_plt = ggplot(y_policy_df, aes("Y", "act", color = "X")) + geom_point(shape=".")
  z_act_plt = ggplot(z_policy_df, aes("Z", "act", color = "X")) + geom_point(shape=".")
  
  pw_x1 = pw.load_ggplot(x_act_plt1, figsize = (3,2))
  pw_x2 = pw.load_ggplot(x_act_pltz, figsize = (3,2))
  
  pw_y = pw.load_ggplot(y_act_plt, figsize = (3,2))
  pw_z = pw.load_ggplot(z_act_plt, figsize = (3,2))
  
  the_plot = (pw_x1 / pw_x2) | (pw_y / pw_z)
  
  if path_and_filename is not None:
    the_plot.savefig(path_and_filename)
  
  return pd.concat([x_policy_df, y_policy_df, z_policy_df])

def evaluate_policy_1fish(
  agent, 
  range_X = (0,1), 
  range_Y = (0,1), 
  range_Z = (0,1),
  bound = 4,
  ):
  """ ranges in large [0,bound] 'population space' (0,4 in practice) """
  # X - policy
  x_policy = []
  for x in np.linspace(0,1,100):
    for yz in it.product(
      np.linspace(*range_Y,5), 
      np.linspace(*range_Z,5), 
      ):
      pop = np.array([x, *yz], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      x_policy.append([*pop, *action])
  
  x_policy_df = pd.DataFrame(
    x_policy,
    columns = ["X", "Y", "Z", "act"]
  )
  
  # Y - policy
  y_policy = []
  for y in np.linspace(0,1,100):
    for xz in it.product(
      np.linspace(*range_X,5), 
      np.linspace(*range_Z,5), 
      ):
      x, z = xz
      pop = np.array([x, y, z], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      y_policy.append([*pop, *action])
  
  y_policy_df = pd.DataFrame(
    y_policy,
    columns = ["X", "Y", "Z", "act"]
  )
      
  # Z - policy
  z_policy = []
  for z in np.linspace(0,1,100):
    for xy in it.product(
      np.linspace(*range_X,5), 
      np.linspace(*range_Y,5), 
      ):
      pop = np.array([*xy, z], dtype = np.float32)
      observation = pop/bound
      action = agent.compute_single_action(observation)
      z_policy.append([*pop, *action])
  
  z_policy_df = pd.DataFrame(
    z_policy,
    columns = ["X", "Y", "Z", "act"]
  )
  
  return x_policy_df, y_policy_df, z_policy_df


def evaluate_policy_in_popular_windows_1fish(agent, env):
  # generate episodes
  df = generate_episodes_1fish(agent, env)
  
  # popular ranges:
  popular_x_size, popular_x_window_dict = (
    popular_ranges(df, var="X")
  )
  X_window_start = list(popular_x_window_dict.keys())[0] # dicts weren't the right choice
  #
  popular_y_size, popular_y_window_dict = (
    popular_ranges(df, var="Y")
  )
  Y_window_start = list(popular_y_window_dict.keys())[0]
  #
  popular_z_size, popular_z_window_dict = (
    popular_ranges(df, var="Z")
  )
  Z_window_start = list(popular_z_window_dict.keys())[0]
  
  ## All of these are in population space [0, env.bound] (=[0,4] for practical purposes)
  
  return evaluate_policy_1fish(
  agent, 
  range_X = (X_window_start, X_window_start + popular_x_size), 
  range_Y = (Y_window_start, Y_window_start + popular_y_size), 
  range_Z = (Z_window_start, Z_window_start + popular_z_size)
  )


def GaussianProcessPolicy_1fish(policy_df, length_scale=10, noise_level=0.1):
  """
  policy_df.columns = [X, Y, Z, act_x, act_y]
                    -> action (act_x, act_y) taken at point (X, Y, Z)
  """
  predictors = policy_df[["X", "Y", "Z"]].to_numpy()
  targets = policy_df[["act"]].to_numpy()
  kernel = (
    1.0 * RBF(length_scale = length_scale) 
    + WhiteKernel(noise_level=noise_level)
    )
  gpp = (
    GaussianProcessRegressor(kernel=kernel, random_state=0)
    .fit(predictors, targets)
    )
  return gpp



