from parameters import parameters
from envs import two_three_fishing
from train_fns import create_agent, train_agent
from uncontrolled_fns import generate_uncontrolled_timeseries_plot
from two_fishery_esc_fns import find_best_esc
from one_fishery_esc_fns import find_best_esc_1fish
from util import dict_pretty_print, print_params
from eval_util import (
  generate_episodes, episode_plots, state_policy_plot, values_at_max_t,
  GaussianProcessPolicy, generate_gpp_episodes,
  generate_episodes_1fish, episode_plots_1fish, state_policy_plot_1fish,
  GaussianProcessPolicy_1fish, generate_gpp_episodes_1fish, 
  gpp_policy_plot_1fish, gpp_policy_plot_2fish,
)
from msy_fns import find_msy_2fish, find_msy_1fish

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_bar, geom_histogram
)
import ray

"""
Script used to produce and evaluate policies on a three-species system.
Global variables (in all caps at the top of the code) can be used to set the
environment we desire for the control problem in broad strokes:
  
  - ENVCODE ( = '1FISHERY' or '2FISHERY') is used to encode how many fisheries
      our problem has. 1FISHERY defaults to population X being harvested, while
      2FISHERY defaults to populations X and Y being harvested.
  - DATACODE ( = to any option displayed below in the Data-code list) is used
      to choose which growth function is used as the system's 'natural' (i.e. unharvested) 
      dynamics
  - NAME_MODIFIER: optionally may be set to modify the subdirectory in which data and
      plots are saved
  - ITERATIONS: number of iterations over which the DRL agent is trained
  - REPS: number of replicate samples used to optimize and evaluate constant escapement
      and constant mortality strategies. Also number of replicate samples used to
      evaluate DRL policies.
  - ESC_GRID_SIZE:
      if ENVCODE == 1FISHERY: number of choices of constant escapement values 
        on [0, 1] used to evaluate and optimize the strategy. (Resp., number
        of choices of const. mortality values on [0, 0.5] used to evaluate
        and optimize the strategy.)
      if ENVCODE == 2FISHERY: similar to above, but now a 2D grid of 
        ESC_GRID_SIZE x ESC_GRID_SIZE points is laid on [0, 1]^2 (resp.
        [0, 0.5]^2). These are the possible choices of X-escapement and Y-escapement
        (resp. X-mortality and Y-mortality) used to optimize and evaluate the
        cosntant escapement (resp. constant mortality) strategy.
  - DATAPATH: not to be changed. It is the directory where data and plots are saved.
        
"""

'''
Data-code list. These codes are searchable in the envs/growth_fns.py document in 
order to find the source code for the growth function.

"DEFAULT", Default 3-species model as presented in the manuscript (no time-varying 
    parameters).
"RXDRIFT", As DEFAULT, but with a time-varying r_X parameter
"V0DRIFT", As DEFAULT, but with a time-varying v0 parameter (this parameter is 
    called 'c' in manuscript, sorry)
"DDRIFT", As DEFAULT, but with a time-varying D parameter
"KLIMIT", Alternative non-time-varying model. Here DEFAULT's X-Y Lotka-Volterra competition 
    term is replaced by reduction on X's and Y'x carrying capacity
"KLIMIT_RXDRIFT", As KLIMIT, but with a time-varying r_X parameter
"BETADRIFT", As DEFAULT, but with a time-varying beta parameter
"CVDRIFT", As DEFAULT, but with a time-varying X-Y Lotka-Volterra competition parameter, 
    c_V (this parameter is called 'c_XY' in manuscript, sorry). 
"YABIOTIC", Similar to DEFAULT, but Y population is set to vary sinusoidally irrespective
    of the X and Z values.
"ZABIOTIC", Similar to YABIOTIC, but it is Z varying sinusoidally now.
"COUPLFLUC", As DEFAULT but with sinusoidally varying v_0 parameter (this parameter
    is called 'c' in manuscript, sorry)
'''

# globals
NAME_MODIFIER = ""
ITERATIONS = 300
REPS = 100
ESC_GRID_SIZE = 51
DATACODE = "DEFAULT"
FLUCTUATING = False
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE, NAME_MODIFIER)
ENVCODE_TO_ENV = {
  "1FISHERY": two_three_fishing.oneThreeFishing,
  "2FISHERY": two_three_fishing.twoThreeFishing,
}
ENV_CLASS = ENVCODE_TO_ENV[ENVCODE]

os.makedirs(DATAPATH, exist_ok=True)


# escapement
@ray.remote(num_gpus=1, num_cpus=20)
def workflow_twoFishery():
  # define problem
  env_class = ENV_CLASS
  parameters_obj = parameters()
  
  # create agent
  agent = create_agent(
    env_class = env_class,
    parameters_obj=parameters_obj,
    datacode = DATACODE, 
    env_name="twoThreeFishing", 
    fluctuating=FLUCTUATING, 
    training=True,
  )
  
  # create associated env
  env_config = agent.evaluation_config.env_config
  env_config.update({'seed': 42})
  env = agent.env_creator(env_config)
  
  # uncontrolled
  generate_uncontrolled_timeseries_plot(
    env, path_and_filename=os.path.join(DATAPATH, "uncontrolled_timeseries.png"), T=200, reps=1
  )
  
  # escapement
  esc_filename = os.path.join(DATAPATH, f"esc_{REPS}.csv.xz")
  if os.path.exists(esc_filename):
    print("Reading escapement...")
    esc_df = pd.read_csv(esc_filename)
    print(
      values_at_max_t(esc_df)
      .agg({'reward':['mean', 'std']})
    )
  else:
    print("Running escapement...")
    best_esc, esc_df = find_best_esc(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_esc)
    esc_df.to_csv(esc_filename)
  print("Escapement done!")
  
  # msy
  msy_filename = os.path.join(DATAPATH, f"msy_{REPS}.csv.xz")
  if os.path.exists(msy_filename):
    print("Reading msy...")
    msy_df = pd.read_csv(msy_filename)
    print(
      values_at_max_t(msy_df)
      .agg({'reward':['mean', 'std']})
    )
  else:
    print("Running msy...")
    best_msy, msy_df = find_msy_2fish(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_msy)
    msy_df.to_csv(msy_filename)
  print("MSY done!")
  
  # train agent and generate data
  agent = train_agent(agent, iterations=ITERATIONS, path_to_checkpoint="cache")
  print("Finished training RL agent!")
  ppo_df = generate_episodes(agent, env, reps = REPS)
  ppo_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}.csv.xz"), index = False)


  #esc_df, ppo_df = workflow(env, agent)
  
  ### EVALUATION ###
    
  
  ## state policy plots
  
  policy_df = state_policy_plot(
    agent, env, path_and_filename= os.path.join(DATAPATH,f"ppo{ITERATIONS}-pol.png")
  )
  policy_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}_policy_data.csv.xz"))
  
  
  ## Gaussian Process Smoothing:
  
  gpp = GaussianProcessPolicy(policy_df)
  print("Finished fitting GP!")
  gpp_df =  generate_gpp_episodes(gpp, env, reps=REPS)
  gpp_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}_GPP.csv.xz"), index = False)
  
  gpp_policy_df = gpp_policy_plot_2fish(
    gpp, env, path_and_filename = os.path.join(DATAPATH,f"gpp{ITERATIONS}-pol.png")
  )
  gpp_policy_df.to_csv(os.path.join(DATAPATH,f"gpp{ITERATIONS}_policy_data.csv.xz"))
  
  ## data wrangling
  msy_max_t = values_at_max_t(msy_df, group="rep")
  ppo_max_t = values_at_max_t(ppo_df, group="rep")
  gpp_max_t = values_at_max_t(gpp_df, group="rep")
  esc_max_t = values_at_max_t(esc_df, group="rep")
  msy_max_t['strategy']='CMort'
  esc_max_t['strategy']='CEsc'
  ppo_max_t['strategy']='PPO'
  gpp_max_t['strategy']='PPO+GP'
  tot_max_t = pd.concat([msy_max_t, esc_max_t, ppo_max_t, gpp_max_t])
  
  tot_max_t.to_csv(os.path.join(DATAPATH,f"comparison_{ITERATIONS}.csv.xz"))
  print("Done with data generation, finishing up on plots...")
  
  
  ## Data analysis
  
  ## reward and time distributions 
  plt.close()
  ax = sns.violinplot(
    data=tot_max_t, 
    x='reward', 
    y='strategy',
  )
  fig = ax.get_figure()
  fig.savefig(os.path.join(DATAPATH, f"rew_violin_plot_{ITERATIONS}.png"), dpi=500)
  
  ## reward and time histograms
  
  # PPO
  ppo_rew_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  ppo_rew_hist.save(os.path.join(DATAPATH,f"ppo{ITERATIONS}_rew_hist.png"))
  #
  ppo_t_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  ppo_t_hist.save(os.path.join(DATAPATH,f"ppo{ITERATIONS}_t_hist.png"))
  
  # GPP
  gpp_rew_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  gpp_rew_hist.save(os.path.join(DATAPATH,f"gpp{ITERATIONS}_rew_hist.png"))
  #
  gpp_t_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  gpp_t_hist.save(os.path.join(DATAPATH,f"gpp{ITERATIONS}_t_hist.png"))
  
  # ESC
  esc_rew_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  esc_rew_hist.save(os.path.join(DATAPATH,"esc_rew_hist.png"))
  #
  esc_t_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  esc_t_hist.save(os.path.join(DATAPATH,"esc_t_hist.png"))
  
  # MSY
  msy_rew_hist = (
    ggplot(data=msy_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  msy_rew_hist.save(os.path.join(DATAPATH,"msy_rew_hist.png"))
  #
  msy_t_hist = (
    ggplot(data=msy_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  msy_t_hist.save(os.path.join(DATAPATH,"msy_t_hist.png"))
  
  ## episode timeseries
  
  for i in range(3):
    episode_plots(
      ppo_df.loc[ppo_df.rep == i], 
      path_and_filename = os.path.join(DATAPATH,f"ppo{ITERATIONS}-eps-{i}.png")
    )
    episode_plots(
      gpp_df.loc[gpp_df.rep == i], 
      path_and_filename = os.path.join(DATAPATH,f"gpp{ITERATIONS}-eps-{i}.png")
    )
    #esc_path = esc_df.melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
    #opt_esc_plot = ggplot(esc_path[esc_df.rep == i], aes("t", "value", color="variable")) + geom_line()
    #opt_esc_plot.save(path = f"{DATAPATH}", filename = f"esc_{i}.png")

  return None

@ray.remote(num_gpus=1, num_cpus=20)
def workflow_oneFishery():
  # define problem
  env_class = ENV_CLASS
  parameters_obj = parameters()
  
  # create agent
  agent = create_agent(
    env_class = env_class,
    parameters_obj=parameters_obj,
    datacode = DATACODE, 
    env_name="oneThreeFishing", 
    fluctuating=FLUCTUATING, 
    training=True,
  )
  
  # create associated env
  env_config = agent.evaluation_config.env_config
  env_config.update({'seed': 42})
  env = agent.env_creator(env_config)
  
  # uncontrolled
  generate_uncontrolled_timeseries_plot(
    env, path_and_filename=os.path.join(DATAPATH,"uncontrolled_timeseries.png"), T=200, reps=1
  )
  
  # Escapement
  esc_filename = os.path.join(DATAPATH,f"esc_{REPS}.csv.xz")
  if os.path.exists(esc_filename):
    print("Reading escapement...")
    esc_df = pd.read_csv(esc_filename)
  else:
    print("Running escapement...")
    best_esc, esc_df = find_best_esc_1fish(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_esc)
    esc_df.to_csv(esc_filename)
  print("Escapement done!")
  
  
  # msy
  msy_filename = os.path.join(DATAPATH,f"msy_{REPS}.csv.xz")
  if os.path.exists(msy_filename):
    print("Reading msy...")
    msy_df = pd.read_csv(msy_filename)
    print(
      values_at_max_t(msy_df)
      .agg({'reward':['mean', 'std']})
    )
  else:
    print("Running msy...")
    best_msy, msy_df = find_msy_1fish(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_msy)
    msy_df.to_csv(msy_filename)
  print("MSY done!")
  
  # train agent and generate data
  agent = train_agent(agent, iterations=ITERATIONS, path_to_checkpoint="cache")
  print("Finished training RL agent!")
  ppo_df = generate_episodes_1fish(agent, env, reps = REPS)
  ppo_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}.csv.xz"), index = False)


  #esc_df, ppo_df = workflow(env, agent)
  
  ### EVALUATION ###
    
  
  ## state policy plots
  
  policy_df = state_policy_plot_1fish(
    agent, env, path_and_filename= os.path.join(DATAPATH,f"ppo{ITERATIONS}-pol.png")
  )
  policy_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}_policy_data.csv.xz"))
  
  
  ## Gaussian Process Smoothing:
  
  gpp = GaussianProcessPolicy_1fish(policy_df)
  print("Finished fitting GP!")
  gpp_df =  generate_gpp_episodes_1fish(gpp, env, reps=REPS)
  gpp_df.to_csv(os.path.join(DATAPATH,f"ppo{ITERATIONS}_GPP.csv.xz"), index = False)
  
  gpp_policy_df = gpp_policy_plot_1fish(
    gpp, env, path_and_filename = os.path.join(DATAPATH,f"gpp{ITERATIONS}-pol.png")
  )
  gpp_policy_df.to_csv(os.path.join(DATAPATH,f"gpp{ITERATIONS}_policy_data.csv.xz"))
  
  ## data wrangling
  msy_max_t = values_at_max_t(msy_df, group="rep")
  esc_max_t = values_at_max_t(esc_df, group="rep")
  ppo_max_t = values_at_max_t(ppo_df, group="rep")
  gpp_max_t = values_at_max_t(gpp_df, group="rep")
  ppo_max_t['strategy']='PPO'
  gpp_max_t['strategy']='PPO+GP'
  esc_max_t['strategy']='CEsc'
  msy_max_t['strategy']='CMort'
  tot_max_t = pd.concat([msy_max_t, esc_max_t, ppo_max_t, gpp_max_t])
  
  tot_max_t.to_csv(os.path.join(DATAPATH,f"comparison_{ITERATIONS}.csv.xz"))
  print("Done with data generation, finishing up on plots...")
  
  
  ## Data analysis
  
  ## reward and time distributions 
  plt.close()
  ax = sns.violinplot(
    data=tot_max_t, 
    x='reward', 
    y='strategy',
  )
  fig = ax.get_figure()
  fig.savefig(os.path.join(DATAPATH,f"rew_violin_plot_{ITERATIONS}.png"), dpi=500)
  
  ## reward and time histograms
  
  # PPO
  ppo_rew_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  ppo_rew_hist.save(os.path.join(DATAPATH,f"ppo{ITERATIONS}_rew_hist.png"))
  #
  ppo_t_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  ppo_t_hist.save(os.path.join(DATAPATH,f"ppo{ITERATIONS}_t_hist.png"))
  
  # GPP
  gpp_rew_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  gpp_rew_hist.save(os.path.join(DATAPATH,f"gpp{ITERATIONS}_rew_hist.png"))
  #
  gpp_t_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  gpp_t_hist.save(os.path.join(DATAPATH,f"gpp{ITERATIONS}_t_hist.png"))
  
  # ESC
  esc_rew_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  esc_rew_hist.save(os.path.join(DATAPATH,"esc_rew_hist.png"))
  #
  esc_t_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  esc_t_hist.save(os.path.join(DATAPATH,"esc_t_hist.png"))
  
  # ESC
  msy_rew_hist = (
    ggplot(data=msy_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  msy_rew_hist.save(os.path.join(DATAPATH,"msy_rew_hist.png"))
  #
  msy_t_hist = (
    ggplot(data=msy_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  msy_t_hist.save(os.path.join(DATAPATH,"msy_t_hist.png"))
  
  ## episode timeseries
  
  for i in range(3):
    episode_plots_1fish(
      ppo_df.loc[ppo_df.rep == i], 
      path_and_filename = os.path.join(DATAPATH,f"ppo{ITERATIONS}-eps-{i}.png")
    )
    episode_plots_1fish(
      gpp_df.loc[gpp_df.rep == i], 
      path_and_filename = os.path.join(DATAPATH,f"gpp{ITERATIONS}-eps-{i}.png")
    )
    #esc_path = esc_df.melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
    #opt_esc_plot = ggplot(esc_path[esc_df.rep == i], aes("t", "value", color="variable")) + geom_line()
    #opt_esc_plot.save(path = f"{DATAPATH}", filename = f"esc_{i}.png")

  return None

def what_to_do(envcode):
  envcode_to_workflow = {
    "1FISHERY": workflow_oneFishery, 
    "2FISHERY": workflow_twoFishery
  }
  _ = ray.get(
    [envcode_to_workflow[envcode].remote() for _ in range(1)]
  )

# run the thing
what_to_do(envcode=ENVCODE)
