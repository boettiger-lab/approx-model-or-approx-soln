import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line

from eval_util import values_at_max_t

def generate_uncontrolled_timeseries_plot_oneSp(
	  env, path_and_filename="uncontrolled_experiment.png", T=200, reps=1
	) -> None:
	dynamics_df = generate_time_evolution_df(
	  env, T=T, reps=reps, columns = ["t", "X", "rep"]
	 )
	dynamics_df_long = dynamics_df.melt(id_vars=["t", "rep"])
	ggp = ggplot(
        dynamics_df_long.groupby(["t", "rep"]),
        aes("t", "value", color = "variable")
      ) + geom_line()
	ggp.save(filename = path_and_filename)

def generate_uncontrolled_timeseries_plot(
	  env, path_and_filename="uncontrolled_experiment.png", T=200, reps=1
	) -> None:
	dynamics_df = generate_time_evolution_df(env, T=T, reps=reps)
	dynamics_df_long = dynamics_df.melt(id_vars=["t", "rep"])
	ggp = ggplot(
        dynamics_df_long.groupby(["t", "rep"]),
        aes("t", "value", color = "variable")
      ) + geom_line()
	ggp.save(filename = path_and_filename)


def generate_time_evolution_df(env, T=200, reps = 5, columns = ["t", "X", "Y", "Z", "rep"]):
	dynamics = []
	for rep in range(reps):
	  for t in range(T):
		  obs, rew, ter, aux, info = env.step(0)
		  pop = list(env.population())
		  dynamics.append([t, *pop, rep])
		  if ter:
		    break
	dynamics_df = pd.DataFrame(dynamics, columns = columns)
	return dynamics_df

def non_coexistence_fraction(env, T=200, reps=100):
  dynamics_df = generate_time_evolution_df(env, T=T, reps=reps)
  df = values_at_max_t(dynamics_df, group="rep").loc[dynamics_df.t < T-1]
  return len(df.index)/reps
  
  
