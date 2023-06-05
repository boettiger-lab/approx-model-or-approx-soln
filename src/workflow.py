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
)

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_bar, geom_histogram
)
import ray

'''
Data-code list:

"DEFAULT","RXDRIFT","V0DRIFT","DDRIFT","KLIMIT",
"KLIMIT_RXDRIFT","BETADRIFT","CVDRIFT","YABIOTIC",
"ZABIOTIC","COUPLFLUC",
'''

# globals
NAME_MODIFIER = ""
ITERATIONS = 200
REPS = 100
ESC_GRID_SIZE = 51
DATACODE = "KLIMIT_RXDRIFT"
FLUCTUATING = True
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)
ENVCODE_TO_ENV = {
  "1FISHERY": two_three_fishing.oneThreeFishing,
  "2FISHERY": two_three_fishing.twoThreeFishing,
}
ENV_CLASS = ENVCODE_TO_ENV[ENVCODE]

os.makedirs(DATAPATH, exist_ok=True)

def what_to_do(workflow_choice):
  _ = ray.get(
    [workflow_choice.remote() for _ in range(1)]
  )


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
    env, path_and_filename=f"{DATAPATH}/uncontrolled_timeseries.png", T=200, reps=1
  )
  esc_filename = f"{DATAPATH}/esc_{REPS}.csv.xz"
  if os.path.exists(esc_filename):
    print("Reading escapement...")
    esc_df = pd.read_csv(esc_filename)
  else:
    print("Running escapement...")
    best_esc, esc_df = find_best_esc(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_esc)
    esc_df.to_csv(esc_filename)
  print("Escapement done!")
  
  # train agent and generate data
  agent = train_agent(agent, iterations=ITERATIONS, path_to_checkpoint="cache")
  print("Finished training RL agent!")
  ppo_df = generate_episodes(agent, env, reps = REPS)
  ppo_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}.csv.xz", index = False)


  #esc_df, ppo_df = workflow(env, agent)
  
  ### EVALUATION ###
    
  
  ## state policy plots
  
  policy_df = state_policy_plot(agent, env, path_and_filename= f"{DATAPATH}/ppo{ITERATIONS}-pol.png")
  policy_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}_policy_data.csv.xz")
  
  
  ## Gaussian Process Smoothing:
  
  gpp = GaussianProcessPolicy(policy_df)
  print("Finished fitting GP!")
  gpp_df =  generate_gpp_episodes(gpp, env, reps=REPS)
  gpp_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}_GPP.csv.xz", index = False)
  
  ## data wrangling
  ppo_max_t = values_at_max_t(ppo_df, group="rep")
  gpp_max_t = values_at_max_t(gpp_df, group="rep")
  esc_max_t = values_at_max_t(esc_df, group="rep")
  ppo_max_t['strategy']='PPO'
  gpp_max_t['strategy']='PPO+GP'
  esc_max_t['strategy']='CEsc'
  tot_max_t = pd.concat([esc_max_t, ppo_max_t, gpp_max_t])
  
  tot_max_t.to_csv(f"{DATAPATH}/comparison_{ITERATIONS}.csv.xz")
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
  fig.savefig(f"{DATAPATH}/rew_violin_plot_{ITERATIONS}.png", dpi=500)
  
  ## reward and time histograms
  
  # PPO
  ppo_rew_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  ppo_rew_hist.save(f"{DATAPATH}/ppo{ITERATIONS}_rew_hist.png")
  #
  ppo_t_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  ppo_t_hist.save(f"{DATAPATH}/ppo{ITERATIONS}_t_hist.png")
  
  # GPP
  gpp_rew_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  gpp_rew_hist.save(f"{DATAPATH}/gpp{ITERATIONS}_rew_hist.png")
  #
  gpp_t_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  gpp_t_hist.save(f"{DATAPATH}/gpp{ITERATIONS}_t_hist.png")
  
  # ESC
  esc_rew_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  esc_rew_hist.save(f"{DATAPATH}/esc_rew_hist.png")
  #
  esc_t_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  esc_t_hist.save(f"{DATAPATH}/esc_t_hist.png")
  
  ## episode timeseries
  
  for i in range(3):
    episode_plots(
      ppo_df.loc[ppo_df.rep == i], 
      path_and_filename = f"{DATAPATH}/ppo{ITERATIONS}-eps-{i}.png"
    )
    episode_plots(
      gpp_df.loc[gpp_df.rep == i], 
      path_and_filename = f"{DATAPATH}/gpp{ITERATIONS}-eps-{i}.png"
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
    env, path_and_filename=f"{DATAPATH}/uncontrolled_timeseries.png", T=200, reps=1
  )
  esc_filename = f"{DATAPATH}/esc_{REPS}.csv.xz"
  if os.path.exists(esc_filename):
    print("Reading escapement...")
    esc_df = pd.read_csv(esc_filename)
  else:
    print("Running escapement...")
    best_esc, esc_df = find_best_esc_1fish(env, grid_nr=ESC_GRID_SIZE, repetitions=REPS)
    print(best_esc)
    esc_df.to_csv(esc_filename)
  print("Escapement done!")
  
  # train agent and generate data
  agent = train_agent(agent, iterations=ITERATIONS, path_to_checkpoint="cache")
  print("Finished training RL agent!")
  ppo_df = generate_episodes_1fish(agent, env, reps = REPS)
  ppo_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}.csv.xz", index = False)


  #esc_df, ppo_df = workflow(env, agent)
  
  ### EVALUATION ###
    
  
  ## state policy plots
  
  policy_df = state_policy_plot_1fish(agent, env, path_and_filename= f"{DATAPATH}/ppo{ITERATIONS}-pol.png")
  policy_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}_policy_data.csv.xz")
  
  
  ## Gaussian Process Smoothing:
  
  gpp = GaussianProcessPolicy_1fish(policy_df)
  print("Finished fitting GP!")
  gpp_df =  generate_gpp_episodes_1fish(gpp, env, reps=REPS)
  gpp_df.to_csv(f"{DATAPATH}/ppo{ITERATIONS}_GPP.csv.xz", index = False)
  
  ## data wrangling
  ppo_max_t = values_at_max_t(ppo_df, group="rep")
  gpp_max_t = values_at_max_t(gpp_df, group="rep")
  esc_max_t = values_at_max_t(esc_df, group="rep")
  ppo_max_t['strategy']='PPO'
  gpp_max_t['strategy']='PPO+GP'
  esc_max_t['strategy']='CEsc'
  tot_max_t = pd.concat([esc_max_t, ppo_max_t, gpp_max_t])
  
  tot_max_t.to_csv(f"{DATAPATH}/comparison_{ITERATIONS}.csv.xz")
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
  fig.savefig(f"{DATAPATH}/rew_violin_plot_{ITERATIONS}.png", dpi=500)
  
  ## reward and time histograms
  
  # PPO
  ppo_rew_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  ppo_rew_hist.save(f"{DATAPATH}/ppo{ITERATIONS}_rew_hist.png")
  #
  ppo_t_hist = (
    ggplot(data=ppo_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  ppo_t_hist.save(f"{DATAPATH}/ppo{ITERATIONS}_t_hist.png")
  
  # GPP
  gpp_rew_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  gpp_rew_hist.save(f"{DATAPATH}/gpp{ITERATIONS}_rew_hist.png")
  #
  gpp_t_hist = (
    ggplot(data=gpp_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  gpp_t_hist.save(f"{DATAPATH}/gpp{ITERATIONS}_t_hist.png")
  
  # ESC
  esc_rew_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  esc_rew_hist.save(f"{DATAPATH}/esc_rew_hist.png")
  #
  esc_t_hist = (
    ggplot(data=esc_max_t, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  esc_t_hist.save(f"{DATAPATH}/esc_t_hist.png")
  
  ## episode timeseries
  
  for i in range(3):
    episode_plots_1fish(
      ppo_df.loc[ppo_df.rep == i], 
      path_and_filename = f"{DATAPATH}/ppo{ITERATIONS}-eps-{i}.png"
    )
    episode_plots_1fish(
      gpp_df.loc[gpp_df.rep == i], 
      path_and_filename = f"{DATAPATH}/gpp{ITERATIONS}-eps-{i}.png"
    )
    #esc_path = esc_df.melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
    #opt_esc_plot = ggplot(esc_path[esc_df.rep == i], aes("t", "value", color="variable")) + geom_line()
    #opt_esc_plot.save(path = f"{DATAPATH}", filename = f"esc_{i}.png")

  return None


# run the thing
what_to_do(workflow_choice=workflow_twoFishery)
