import pandas as pd
import os
from eval_util import (
  generate_episodes, episode_plots, state_policy_plot, values_at_max_t,
  GaussianProcessPolicy, generate_gpp_episodes
)
from parameters import parameters
from envs import two_three_fishing
from two_fishery_esc_fns import esc_grid_heatmap, esc_grid_ep_ends_heatmap
from train_fns import create_agent

DATACODE = "RXDRIFT"
ENVCODE = "2FISHERY"
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)
os.makedirs(DATAPATH, exist_ok=True)

# define problem
env_class = two_three_fishing.twoThreeFishing
parameters_obj = parameters()

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

#esc_grid_heatmap(
esc_grid_ep_ends_heatmap(
  env,
  path = DATAPATH,
  filename ="esc_heatmap.png",
  grid_nr = 51,
  repetitions = 100,
)



"""
esc = pd.read_csv(os.path.join(DATAPATH, "esc.csv.xz"))
esc_max_t = values_at_max_t(esc)

esc_max_t["reward_std"] = esc_max_t["reward"]
result = esc_max_t.agg({"reward":"mean", "reward_std":"std"})

print(result)
"""
