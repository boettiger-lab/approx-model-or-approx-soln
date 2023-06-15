from two_fishery_esc_fns import find_best_esc_several_params, simulate_max_t
from parameters import parameters
from envs import two_three_fishing, growth_functions
from train_fns import create_agent, train_agent
from uncontrolled_fns import generate_uncontrolled_timeseries_plot
from two_fishery_esc_fns import find_best_esc
from util import dict_pretty_print, print_params
from eval_util import (
  generate_episodes, episode_plots, state_policy_plot, values_at_max_t,
  GaussianProcessPolicy, generate_gpp_episodes
)

import os
import ray
import pandas as pd
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_bar, geom_histogram
)

## GLOBALS

REPS = 100
GRID = 51

DATACODE = "RXDRIFT/CONST_RX_ESC"
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)
os.makedirs(DATAPATH, exist_ok=True)

parameters_obj = parameters()

# ENV DEF

env_config = {}
env_config["parameters"] = parameters_obj.parameters()
env_config["growth_fn"] = growth_functions.default_population_growth
env_config["fluctuating"] = False
env_config["training"] = True
env_config["initial_pop"] = parameters_obj.init_state()

env = two_three_fishing.twoThreeFishing(
  config=env_config
)

# ESC OPTIMIZATION

param_vals = [0.5,1]
param_val_names = ["half_rx", "full_rx"]
best_esc_dict = find_best_esc_several_params(
  env, param_name="r_x", param_vals=param_vals, grid_nr=GRID, repetitions=REPS
)

for val in param_vals:
  print(f"""{val}:
    
  {best_esc_dict[val][0]}
      
  """)

# REDEFINE ENV (to be RXDRIFT)

env_config = {}
env_config["parameters"] = parameters_obj.parameters()
env_config["growth_fn"] = growth_functions.rx_drift_growth
env_config["fluctuating"] = True
env_config["training"] = True
env_config["initial_pop"] = parameters_obj.init_state()

env2 = two_three_fishing.twoThreeFishing(
  config=env_config
)

# RESULTS:
# performance of each of the esc policies trained on fixed RX, but tested on
# an RXDRIFT env.



parallel = (
  [ simulate_max_t.remote(
      env2, 
      best_esc_dict[val][0].esc_x.values[0], best_esc_dict[val][0].esc_y.values[0], 
      REPS
    ) for val in param_vals
  ]
) # [(best at rx=0.5, paths at rx=0.5), (best at rx = 1, ...)]
maxts_and_paths_raw = ray.get(parallel)
maxts_and_paths = list(
  map(list, zip(*maxts_and_paths_raw))
) # [[best at rx=0.5, at rx=1], [paths at rx=0.5, ...]]
maxts = maxts_and_paths[0]
paths = maxts_and_paths[1]

for i, esc in enumerate(paths): 
  maxts_df = pd.DataFrame(
    maxts[i],
    columns = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
    )
  paths_df = pd.DataFrame(
    paths[i],
    columns = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
    )

  # plots
  esc_rew_hist = (
    ggplot(data=maxts_df, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  esc_rew_hist.save(f"{DATAPATH}/esc_rew_hist_{param_val_names[i]}.png")
  #
  esc_t_hist = (
    ggplot(data=maxts_df, mapping=aes(x='rep',weight='t')) 
    +geom_bar()
  )
  esc_t_hist.save(f"{DATAPATH}/esc_t_hist_{param_val_names[i]}.png")
  #
  for j in range(4):
    episode_plots(
      paths_df.loc[paths_df.rep == j], 
      path_and_filename = f"{DATAPATH}/esc_{param_val_names[i]}_eps_{j}.png"
    )
  maxts_df["reward2"]= maxts_df["reward"]
  rew_stats = maxts_df.agg({"reward":"mean", "reward2":"std"})
  print(f"result for r_x = {param_val_names[i]} trained escapement:")
  print(rew_stats)


























