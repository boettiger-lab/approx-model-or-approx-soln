import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_bar, geom_point, geom_line
import os

from envs import two_three_fishing, singleSp
from train_fns import create_env
from parameters import parameters, parameters_oneSp
from msy_fns import csv_to_frac_msy_1fish, csv_to_frac_msy_2fish

#globals
1sp1fish = ["data", "results_data", "ONESP", "high_beta", "msy_100.csv.xz"]
3sp1fish = ["data", "results_data", "1FISHERY", "DEFAULT", "msy_100.csv.xz"]
3sp2fish = ["data", "results_data", "2FISHERY", "DEFAULT", "msy_100.csv.xz"]
3sp2fish_timevar = ["data", "results_data", "2FISHERY", "RXDRIFT", "msy_100.csv.xz"]

env_class = two_three_fishing

parameter_obj_3sp = parameters()
parameter_obj_1sp = parameters_oneSp()

env = create_env(
  env_class = , 
  parameter_obj, 
  datacode = "DEFAULT", 
  fluctuating=True, 
  training=True
)

fname = os.path.join(*1sp1fish)

msy_frac_df = csv_to_frac_msy_1fish()
