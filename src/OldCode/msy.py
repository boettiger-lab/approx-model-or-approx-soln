from envs import oneSpecies, two_three_fishing, growth_functions
from msy_fns import find_msy_1fish
from train_fns import create_env
from parameters import parameters_oneSp, parameters
from eval_util import values_at_max_t

from plotnine import ggplot, geom_bar, geom_point, geom_line, aes
import gymnasium as gym
import pandas as pd
import numpy as np
import ray
import os

REPS = 100
MSY_GRID_SIZE = 51
ENVCODE = "1FISHERY"
DATACODE = "DEFAULT"
FLUCTUATING = False
DATAPATH = os.path.join("../data/results_data", ENVCODE, DATACODE)
ENV_CLASS = two_three_fishing.oneThreeFishing #oneSpecies.singleSp

parameter_obj = parameters() #parameters_oneSp()

env = create_env(
  env_class = ENV_CLASS, 
  parameter_obj=parameter_obj, 
  datacode = "DEFAULT", 
  fluctuating=False, 
  training=True,
)

best, msy_df = find_msy_1fish(
  env, grid_nr=MSY_GRID_SIZE, repetitions=REPS,
)
msy_df = values_at_max_t(msy_df)

print(best)
print(msy_df.columns)
print(msy_df.head(20))
msy_df.to_csv(os.path.join(DATAPATH, "msy.csv.xz"))

msy_rew_hist = (
  ggplot(data=msy_df, mapping=aes(x='rep',weight='reward')) 
  +geom_bar()
)
msy_rew_hist.save(os.path.join(DATAPATH,"msy_rew_hist.png"))
#
msy_t_hist = (
  ggplot(data=msy_df, mapping=aes(x='rep',weight='t')) 
  +geom_bar()
)
msy_t_hist.save(os.path.join(DATAPATH,"msy_t_hist.png"))
