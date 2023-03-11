import fish_tipping
from growth_functions import rockPaperScissors, params_rockPaperScissors

import gym
import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path

params = params_rockPaperScissors()
env = fish_tipping.three_sp(
		config={"growth_fn": rockPaperScissors, "parameters": params},
	)

def timeEvolution(env, T=100):
	path = np.empty((T, 4))
	for t in range(T):
		obs, rew, ter, aux, info = env.step(0)
		pop = list(env.population())
		path[t] = np.array([t, *pop])
	path_df = pd.DataFrame(path, columns = ["t", "X", "Y", "Z"])
	return path_df

path_df = timeEvolution(env)
print(path_df.head())