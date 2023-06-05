from parameters import parameters
from envs import two_three_fishing
from train_fns import create_agent, train_agent
from uncontrolled_fns import (
  generate_uncontrolled_timeseries_plot,
  non_coexistence_fraction,
)
from two_fishery_esc_fns import find_best_esc
from util import dict_pretty_print, print_params
from eval_util import (
  generate_episodes, episode_plots, state_policy_plot, values_at_max_t,
  GaussianProcessPolicy, generate_gpp_episodes
)

import matplotlib.pyplot as plt
import numpy as np
import ray
import os
import statistics
import seaborn as sns
import pandas as pd
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, 
  geom_path, geom_bar, geom_histogram
)

'''
Data-code list:

"DEFAULT","RXDRIFT","V0DRIFT","DDRIFT","KLIMIT",
"KLIMIT_RXDRIFT","BETADRIFT","CVDRIFT","YABIOTIC",
"ZABIOTIC","COUPLFLUC",
'''

# globals
NAME_MODIFIER = "JIGGLE"
ITERATIONS=200
REPS=100
ESC_GRID_SIZE = 51
N = 100
N_noise_steps = 3
DATACODE = "RXDRIFT"
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE, NAME_MODIFIER)

os.makedirs(DATAPATH, exist_ok=True)

# define problem
#env_class = two_three_fishing.twoThreeFishing
#parameters_obj = parameters()

"""
performances_list = []
for jiggle_strength in [0.02*i for i in range(5)]:
  esc_performance_list = []
  ppo_performance_list = []
  gpp_performance_list = []
  for dummy in range(N):
    print(f"jiggle nr. {dummy}, strenght {jiggle_strength}")
    parameters = parameters_obj.jiggle_params(noise_strength = jiggle_strength)
    
    # create agent
    agent = create_agent(
      env_class = env_class,
      parameters_obj=parameters_obj,
      datacode = DATACODE, 
      env_name="twoThreeFishing", 
      fluctuating=True, 
      training=True,
    )
    
    # create associated env
    env_config = agent.evaluation_config.env_config
    env_config.update({'seed': 42})
    env = agent.env_creator(env_config)
    
    
    # escapement
    best_esc, esc_df = find_best_esc(
      env, grid_nr=ESC_GRID_SIZE, repetitions=REPS
    )
    print("esc", end = " . ", flush=True)
    
    
    # train agent and generate data
    agent = train_agent(
      agent, iterations=ITERATIONS, path_to_checkpoint="cache", verbose=False
    )
    ppo_df = generate_episodes(agent, env, reps = REPS)
    print("RL", end = " . ", flush=True)
    
    policy_df = state_policy_plot(agent, env, path_and_filename = None)
    
    ## Gaussian Process Smoothing:
    
    gpp = GaussianProcessPolicy(policy_df)
    gpp_df =  generate_gpp_episodes(gpp, env, reps=REPS)
    print("GP", end = " . ", flush=True)
    
    ## data wrangling
    ppo_max_t = values_at_max_t(ppo_df, group="rep")
    gpp_max_t = values_at_max_t(gpp_df, group="rep")
    esc_max_t = values_at_max_t(esc_df, group="rep")

    esc_performance_list.append(
      esc_max_t.loc[:, "reward"].mean()
    )
    ppo_performance_list.append(
      ppo_max_t.loc[:, "reward"].mean()
    )
    gpp_performance_list.append(
      gpp_max_t.loc[:, "reward"].mean()
    )
  # closing for dummy in range(20)
  performances_list.append(
    [
      jiggle_strength,
      statistics.mean(esc_performance_list), 
      statistics.stdev(esc_performance_list), 
      statistics.mean(ppo_performance_list), 
      statistics.stdev(ppo_performance_list), 
      statistics.mean(gpp_performance_list), 
      statistics.stdev(gpp_performance_list), 
    ]
  )
columns = [
  "noise_strength",
  "CEsc_mean", 
  "CEsc_stdev",
  "PPO_mean", 
  "PPO_stdev",
  "GPPPO_mean", 
  "GPPPO_stdev"
]

performances = pd.DataFrame(
  performances_list, 
  columns = columns
)

fig = plt.figure()
plt.errorbar(
  performances.noise_strength, performances.CEsc_mean, 
  yerr=performances.CEsc_stdev, label="CEsc", fmt="o-"
)
plt.errorbar(
  performances.noise_strength, performances.PPO_mean, 
  yerr=performances.PPO_stdev, label="PPO", fmt="o-"
)
plt.errorbar(
  performances.noise_strength, performances.GPPPO_mean, 
  yerr=performances.GPPPO_stdev, label="GP+PPO", fmt="o-"
)
plt.xlabel("noise strength")
plt.ylabel("performance")
plt.legend(loc="best")
plt.savefig(fname=f"{DATAPATH}/jiggle.png")
"""

#@ray.remote
@ray.remote(num_gpus=0.5, num_cpus=10)
def jiggle_workflow(
  jiggle_strength,
  parameters_obj_arg = parameters(),
  env_class_arg = two_three_fishing.twoThreeFishing,
  train_iterations=ITERATIONS,
  episode_reps=REPS,
  escapement_grid = ESC_GRID_SIZE,
):
  parameters_obj_arg.reset()
  parameters = parameters_obj_arg.jiggle_params(noise_strength = jiggle_strength)
  
  # create agent
  agent = create_agent(
    env_class = env_class_arg,
    parameters_obj=parameters_obj_arg,
    datacode = DATACODE, 
    env_name="twoThreeFishing", 
    fluctuating=True, 
    training=True,
  )
  
  # create associated env
  env_config = agent.evaluation_config.env_config
  env_config.update({'seed': 42})
  env = agent.env_creator(env_config)
  
  # test out coexistence
  NC = non_coexistence_fraction(env)
  if NC > 0.01:
    print(f"Non-coexistence fraction of {NC} at {jiggle_strength}")
    return [10, 10, 10]
  
  print(jiggle_strength)
  
  # escapement
  best_esc, esc_df = find_best_esc(
    env, grid_nr=escapement_grid, repetitions=episode_reps
  )
  print(f"esc {jiggle_strength}")
  esc_max_t = values_at_max_t(esc_df, group="rep")
  del esc_df

  
  # train agent and generate data
  agent = train_agent(
    agent, iterations=train_iterations, 
    path_to_checkpoint="cache", verbose=False
  )
  print(f"RL {jiggle_strength}")
  ppo_df = generate_episodes(agent, env, reps = episode_reps)
  ppo_max_t = values_at_max_t(ppo_df, group="rep")
  print(f"RL episodes {jiggle_strength}")
  del ppo_df

  policy_df = state_policy_plot(agent, env, path_and_filename = None)
  
  ## Gaussian Process Smoothing:
  gpp = GaussianProcessPolicy(policy_df)
  #gpp = ray.get(
  #  GaussianProcessPolicy.remote(policy_df)
  #)
  gpp_df =  generate_gpp_episodes(gpp, env, reps=episode_reps)
  gpp_max_t = values_at_max_t(gpp_df, group="rep")

  print(f"done {jiggle_strength}")
  
  X = [
    jiggle_strength,
    ppo_max_t.loc[:, "reward"].mean()-esc_max_t.loc[:, "reward"].mean(),
    gpp_max_t.loc[:, "reward"].mean()-esc_max_t.loc[:, "reward"].mean(),
  ]
  print(X)
  return X

def get_results(N_noise_steps=10, N_stats=30):
  from itertools import product 
  
  parallel = [
    jiggle_workflow.remote(iterator[0]) for iterator in product(
      np.linspace(0.2 + 0.15/N_noise_steps, 0.35, N_noise_steps),
      range(N_stats)
    )
  ]
  return ray.get(parallel)

performance_differences = get_results(N_noise_steps=N_noise_steps, N_stats=N)
performance_differences = [
  p for p in performance_differences if p != [10,10,10]
] # [10,10,10] -> param samples outside coexistence region
performance_differences_df = pd.DataFrame(
  performance_differences,
  columns=["NoiseStrength", "PPO_vs_CEsc", "GPPPO_vs_CEsc"]
)
performance_differences_df.to_csv(os.path.join(DATAPATH,"jiggle_results.csv.xz"))

# data wrangling

aux = performance_differences_df
aux["PPO_vs_CEsc2"] = aux["PPO_vs_CEsc"] # for ease calculating the standard dev
aux["GPPPO_vs_CEsc2"] = aux["GPPPO_vs_CEsc"]

results = (
  aux
  .groupby(aux.NoiseStrength)
  .agg(
    {
      "PPO_vs_CEsc":"mean",
      "GPPPO_vs_CEsc":"mean",
      "PPO_vs_CEsc2":"std",
      "GPPPO_vs_CEsc2":"std",
    }
  )
)

print(f"""
results:
  
{results}

""")


fig = plt.figure()
plt.errorbar(
  results.index, results.PPO_vs_CEsc, 
  yerr=results.PPO_vs_CEsc2, label="PPO vs CEsc", fmt="o--",
  capsize=5,
  color = "lightcoral",
)
plt.errorbar(
  results.index, results.GPPPO_vs_CEsc, 
  yerr=results.GPPPO_vs_CEsc2, label="GP+PPO vs CEsc", fmt="o--",
  capsize=5,
  color = "cornflowerblue",
)

plt.grid(axis="y")
plt.xlabel("Parametric noise strength")
plt.ylabel("Mean reward difference")
plt.xticks(list(np.linspace(0, 0.2, N_noise_steps+1)))
plt.legend(loc="best")
plt.savefig(fname=f"{DATAPATH}/jiggle.png")






















  
