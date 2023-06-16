import pandas as pd
import seaborn as sns
from plotnine import ggplot, aes, geom_bar, geom_point, geom_line
import os

from envs import two_three_fishing, oneSpecies
from train_fns import create_env
from parameters import parameters, parameters_oneSp
from msy_fns import csv_to_frac_msy_1fish, csv_to_frac_msy_2fish

_1sp1fish = ["..", "data", "results_data", "ONESP", "high_beta"]
_3sp1fish = ["..", "data", "results_data", "1FISHERY", "DEFAULT"]
_3sp2fish = ["..", "data", "results_data", "2FISHERY", "DEFAULT"]
_3sp2fish_timevar = ["..", "data", "results_data", "2FISHERY", "RXDRIFT"]

parameter_obj_3sp = parameters()
parameter_obj_1sp = parameters_oneSp()

""" envs: two_three_fishing.twoThreeFishing two_three_fishing.oneThreeFishing oneSpecies.singleSp"""
""" fns: csv_to_frac_msy_1fish csv_to_frac_msy_2fish """

# choosing env to evaluate / corresponding data location
scenario = _3sp2fish_timevar
parameter_obj = parameter_obj_3sp
env_class = two_three_fishing.twoThreeFishing
DATACODE = "RXDRIFT"
FLUCTUATING = True
function_to_use = csv_to_frac_msy_2fish

fname = os.path.join(*scenario, "msy_100.csv.xz")

env = create_env(
  env_class, 
  parameter_obj, 
  datacode = DATACODE, 
  fluctuating=FLUCTUATING, 
  training=True,
)

for fraction in [0.8, 0.9, 0.95, 1]:
  msy_frac_df, _ = function_to_use(
    env, fname, fraction=fraction, repetitions=100,
  )
  del _
  
  cols = ["t", "rep", "mortality_x", "mortality_y", "act_x", "act_y", "reward", "X", "Y", "Z"] # assume _3sp2fish by default
  if scenario == _1sp1fish:
    cols = ["t", "rep", "mortality", "act", "reward", "X"]
  if scenario == _3sp1fish:
    cols = ["t", "rep", "mortality", "act", "reward", "X", "Y", "Z"]
    
    
  msy_frac_df = pd.DataFrame(msy_frac_df, columns=cols)
  msy_frac_df['reward_std'] = msy_frac_df['reward']
  df_finished_eps = msy_frac_df[msy_frac_df.t == 200]
  frac_max_t = len(df_finished_eps.index)/len(msy_frac_df.index)
  
  print(f"""fraction = {fraction}
  
  {msy_frac_df.agg({'reward':'mean', 'reward_std':'std'})}
  
  fraction ep len == 200: {frac_max_t}
  
  """)
  
  msy_frac_df.to_csv(
    os.path.join(*scenario, f"frac_{fraction}_msy.csv.xz")
  )
  
  # plots
  
  rew_hist = (
    ggplot(data=msy_frac_df, mapping=aes(x='rep',weight='reward')) 
    +geom_bar()
  )
  rew_hist.save(
    os.path.join(*scenario, f"frac_{fraction}_msy_rewards.png")
  )
