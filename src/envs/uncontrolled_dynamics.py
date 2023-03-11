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

