import fish_tipping
from growth_functions import (
  rockPaperScissors, params_rockPaperScissors, threeSpHolling3, params_threeSpHolling3, K_fluctuation_growth,
  coupling_fluctuation_growth, 
  competition_fluctuation_growth,
  z_abiotic_growth,
)

import gymnasium as gym
import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

# initial_pop = np.array([np.random.rand(), np.random.rand(), np.random.rand()], dtype=np.float32)
initial_pop = np.array([0.8, 0.1, 0.4], dtype=np.float32)
env = fish_tipping.three_sp(
		config={
		  "growth_fn": z_abiotic_growth, 
		  "initial_pop": initial_pop
		  },
	)
env.fluctuating = True

def timeEvolution(env, T=100):
	path = np.empty((T, 4))
	for t in range(T):
		obs, rew, ter, aux, info = env.step(0)
		pop = list(env.population())
		path[t] = np.array([t, *pop])
	path_df = pd.DataFrame(path, columns = ["t", "X", "Y", "Z"])
	return path_df

path_df = timeEvolution(env)
with pd.option_context('display.max_rows', None):
	print(path_df)
