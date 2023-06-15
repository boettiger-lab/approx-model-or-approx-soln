import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_bar, geom_point, geom_line
import os

from envs import two_three_fishing, oneSpecies
from train_fns import create_env
from parameters import parameters, parameters_oneSp
from msy_fns import csv_to_frac_msy_1fish, csv_to_frac_msy_2fish

_1sp1fish = ["data", "results_data", "ONESP", "high_beta"]
_3sp1fish = ["data", "results_data", "1FISHERY", "DEFAULT"]
_3sp2fish = ["data", "results_data", "2FISHERY", "DEFAULT"]
_3sp2fish_timevar = ["data", "results_data", "2FISHERY", "RXDRIFT"]

parameter_obj_3sp = parameters()
parameter_obj_1sp = parameters_oneSp()

# choosing env to evaluate / corresponding data location
scenario = _1sp1fish
env_class = oneSpecies.singleSp
fname = os.path.join(*scenario, "msy_100.csv.xz")
DATACODE = "ONESP"

env = create_env(
  env_class, 
  parameter_obj, 
  datacode = DATACODE, 
  fluctuating=False, 
  training=True,
)


msy_frac_df, _ = csv_to_frac_msy_1fish(
  env, fname, fraction=0.8, repetitions=100
)
del _

msy_frac_df.to_csv(
  os.path.join(*scenario, "frac_0-8_msy.csv.xz")
)

# plots

rew_hist = (
  ggplot(data=msy_frac_df, mapping=aes(x='rep',weight='reward')) 
  +geom_bar()
)
msy_frac_df.save(
  os.path.join(*scenario, "frac_0-8_msy_rewards.png")
)
