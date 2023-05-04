""" functions related to constant escapement policies on 2-fishery scenarios """

import ray
import pandas as pd
import numpy as np
import itertools

@ray.remote
def simulate(env, esc_x, esc_y, repetitions):
  x = []
  for rep in range(repetitions):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      p   h h                                                                                                                     h  opulation = env.population()
      act_x = np.max([1 - esc_x / (population[0] + 1e-12), 0])
      act_y = np.max([1 - esc_y / (population[1] + 1e-12), 0])
      action = np.array([act_x, act_y], dtype = np.float32)
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      x.append(np.append([t, rep, esc_x, esc_y, act_x, act_y, episode_reward], population))
      if terminated:
        break
  return(x)


def generate_esc_episodes(env, grid_nr=30):
	esc_x_choices = np.linspace(0,1,grid_nr)
	esc_y_choices = np.linspace(0,1,grid_nr)
	esc_choices = itertools.product(esc_x_choices, esc_y_choices)

	# define parllel loop and execute
	parallel = [simulate.remote(env, *i) for i in esc_choices]
	x = ray.get(parallel)

	cols = ["t", "rep", "esc_x", "esc_y", "act_x", "act_y", "reward", "X", "Y", "Z"]
	df = pd.DataFrame(np.vstack(x), columns = cols)
	print("Done generating episodes.")
	return df

def find_best_esc(env):
	df = generate_esc_episodes(env)
	df_max_times = pd.DataFrame([], columns = df.columns)
	for rep, esc_x, esc_y in itertools.product(
		range(repetitions), esc_x_choices, esc_y_choices
		):
		print(f"optimizing rep {rep}, esc = {esc_x:.3f}, {esc_y:.3f}", end="\r")
		rep_df = df.loc[
			(df.rep == rep) &
			(df.esc_x == esc_x) & 
			(df.esc_y == esc_y)
		] # single episode
		df_max_times = pd.concat([
			df_max_times,
			rep_df.loc[rep_df.t == rep_df.t.max()]
		])
	#
	tmp = (
	df_max_times
	.groupby(['esc_x', 'esc_y'], as_index=False)
	.agg({'reward': 'mean'})
	)
	best = tmp[tmp.reward == tmp.reward.max()]
	return best, df.loc[
		(df.esc_x == best.esc_x.values[0]) &
		(df.esc_y == best.esc_y.values[0])
	].melt(id_vars=["t", "rep", "reward", "act_x", "act_y"])
