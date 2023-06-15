from envs import fish_tipping, two_three_fishing
from envs import growth_functions
from parameters import parameters

import gymnasium as gym
import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

_DEFAULT_PARAMETERS = parameters().parameters()
_DATACODE = "2FISHERY/KLIMIT_RXDRIFT"
experiment_nr = 1

config = {}
config["parameters"] = _DEFAULT_PARAMETERS
config["growth_fn"] = growth_functions.K_limit_rx_drift_growth
config["initial_pop"] = parameters().init_state()
#config["initial_pop"] = np.random.rand(3).astype(np.float32)
config["fluctuating"] = True

env = fish_tipping.three_sp(
		config=config,
	)

def timeEvolution(env, T=200, reps = 5):
	path = []
	for rep in range(reps):
	  for t in range(T):
		  obs, rew, ter, aux, info = env.step(0)
		  pop = list(env.population())
		  path.append([t, *pop, rep])
	#
	path_df = pd.DataFrame(path, columns = ["t", "X", "Y", "Z", "rep"])
	return path_df

path_df = timeEvolution(env, T=200, reps=1)
with pd.option_context('display.max_rows', None):	
  print(path_df)

path_df_long = path_df.melt(id_vars=["t", "rep"])

ggp = ggplot(
        path_df_long.groupby(["t", "rep"]),
        aes("t", "value", color = "variable")
      ) + geom_line()
      
ggp.save(filename = f"../data/{_DATACODE}/{experiment_nr}/uncontrolled.png")


