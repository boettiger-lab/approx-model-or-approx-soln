import numpy as np
import pandas as pd
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
import patchworklib as pw
import itertools as it

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
  for _ in range(reps):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      action = agent.compute_single_action(observation)
      esc_x = observation[0] * (1 - action[0])
      esc_y = observation[1] * (1 - action[1])
      df.append(np.append([t, rep, action[0], action[1], episode_reward, esc_x, esc_y], observation))
      observation, reward, terminated, done, info = env.step(action)
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
  df2 = df.melt(id_vars=["t",  "reward", "rep"])
  
  esc_t_plot = ggplot(
    df2[(df2.variable == "esc_x") | (df2.variable == "esc_y")], 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  act_t_plot = ggplot(
    df2[(df2.variable == "act_x") | (df2.variable == "act_y")], 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  xyz_t_plot = ggplot(
    df2[(df2.variable == "X") | (df2.variable == "Y") | (df2.variable == "Z")], 
    aes("t", "value", color="variable")
  ) + geom_line()
  
  rew_t_plot = ggplot(
    df2[(df2.variable == "reward")], 
    aes("t", "value")
  ) + geom_line()
  
  pw_esc = pw.load_ggplot(esc_t_plot, figsize = (5,2))
  pw_act = pw.load_ggplot(act_t_plot, figsize = (5,2))
  pw_xyz = pw.load_ggplot(xyz_t_plot, figsize = (5,2))
  pw_rew = pw.load_ggplot(rew_t_plot, figsize = (5,2))
  
  the_plot = (((pw_esc / pw_act) / pw_xyz) / pw_rew)
  the_plot.savefig(path_and_filename)
  
def evaluate_policy(agent):
  # X - policy
  x_policy = []
  for x in np.linspace(0,1,100):
    for yz in it.product(np.linspace(0,1,5), repeat=2):
      observation = np.array([x, *yz], dtype = np.float32)
      action = agent.compute_single_action(obs)
      x_policy.append([*observation, *action])
  
  x_policy_df = pd.DataFrame(
    x_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
  
  # Y - policy
  y_policy = []
  for y in np.linspace(0,1,100):
    for xz in it.product(np.linspace(0,1,5), repeat=2):
      x, z = xz
      observation = np.array([x, y, z], dtype = np.float32)
      action = agent.compute_single_action(obs)
      y_policy.append([*observation, *action])
  
  y_policy_df = pd.DataFrame(
    y_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
      
  # Z - policy
  z_policy = []
  for z in np.linspace(0,1,100):
    for xy in it.product(np.linspace(0,1,5), repeat=2):
      observation = np.array([*xy, z], dtype = np.float32)
      action = agent.compute_single_action(obs)
      z_policy.append([*observation, *action])
  
  z_policy_df = pd.DataFrame(
    z_policy,
    columns = ["X", "Y", "Z", "act_x", "act_y"]
  )
  
  return x_policy_df, y_policy_df, z_policy_df
  
def state_policy_plot(agent, path_and_filename):
  x_policy_df, y_policy_df, z_policy_df = evaluate_policy(agent)
  
  x_act_plt1 = ggplot(x_policy_df, aes("X", "action", color = "Y")) + geom_point(shape=".")
  x_act_plt2 = ggplot(x_policy_df, aes("X", "action", color = "Z")) + geom_point(shape=".")
  
  y_act_plt1 = ggplot(y_policy_df, aes("Y", "action", color = "X")) + geom_point(shape=".")
  y_act_plt2 = ggplot(y_policy_df, aes("Y", "action", color = "Z")) + geom_point(shape=".")
  
  z_act_plt1 = ggplot(z_policy_df, aes("Z", "action", color = "X")) + geom_point(shape=".")
  z_act_plt2 = ggplot(z_policy_df, aes("Z", "action", color = "Y")) + geom_point(shape=".")
  
  pw_x1 = pw.load_ggplot(x_act_plt1, figsize = (3,2))
  pw_x2 = pw.load_ggplot(x_act_plt2, figsize = (3,2))
  
  pw_y1 = pw.load_ggplot(y_act_plt1, figsize = (3,2))
  pw_y2 = pw.load_ggplot(y_act_plt2, figsize = (3,2))
  
  pw_z1 = pw.load_ggplot(z_act_plt1, figsize = (3,2))
  pw_z2 = pw.load_ggplot(z_act_plt2, figsize = (3,2))
  
  the_plot = (pw_x1 / pw_x2) | (pw_y1 / pw_y2) | (pw_z1 / pw_z2)
  the_plot.savefig(path_and_filename)
  
def path_policy_plot(agent, path_df, path_and_filename):
  ...
  
  
  
  
  


