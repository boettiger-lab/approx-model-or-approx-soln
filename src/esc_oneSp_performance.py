from envs import oneSpecies, growth_functions
import one_fishery_esc_fns
import parameters
from train_fns import create_agent

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from plotnine import (
  ggplot, geom_point, aes, geom_line, facet_wrap, geom_path, geom_bar, geom_histogram
)
import ray

def esc_ep_ends(env, grid_nr = 101, repetitions = 100):
  df_max_times, df = one_fishery_esc_fns.generate_esc_episodes_1fish(env, grid_nr=grid_nr, repetitions=repetitions)
  del df
  return df_max_times[["esc", "reward", "rep"]]
  
def esc_ep_ends_performance(env, path, filename, grid_nr=51, repetitions=100):
  import seaborn as sns
  import os
  import matplotlib.pyplot as plt
  plt.close()
  plt.figure(figsize=(10,10))
  # ticklabels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  df = esc_ep_ends(env, grid_nr = grid_nr, repetitions = repetitions)
  df = df[
    (df.esc >= 0.20) & (df.esc <= 0.80)
  ]
  df["reward_std"] = df["reward"]
  df = df.groupby("esc").agg({"reward":"mean", "reward_std":"std"})
  df["esc2"] = df.index
  df.reset_index()
  with pd.option_context("display.max_rows", 1000):
    print(df)
  print(df.columns)
  df.plot(x="esc2", y="reward")
  plt.savefig(os.path.join(DATAPATH, "esc_performance.png"))
  
  
DATACODE = "ONESP"
DATAPATH = os.path.join("../data/results_data", DATACODE)
os.makedirs(DATAPATH, exist_ok=True)

# define problem
env_class = oneSpecies.singleSp
parameters_obj = parameters.parameters_oneSp()

# create agent
agent = create_agent(
  env_class = env_class,
  parameters_obj=parameters_obj,
  datacode = DATACODE, 
  env_name="oneSpFishing", 
  fluctuating=False, 
  training=True,
)

# create associated env
env_config = agent.evaluation_config.env_config
env_config.update({'seed': 42})
env = agent.env_creator(env_config)

#esc_grid_heatmap(
esc_ep_ends_performance(
  env,
  path = DATAPATH,
  filename ="esc_performance.png",
  grid_nr = 101,
  repetitions = 100,
)
