import gym_fishing
import gym
import pandas as pd
import numpy as np

env = gym.make("threeFishing-v2")

df = []

effort * pop = harvest
escapement = pop - harvest
harvest = pop - escapement
effort * pop = pop - escapement
effort = (pop - escapement)/pop

actions = np.linspace(0,1,101)
for escapement in actions:
  episode_reward = 0
  observation = env.reset()
  for rep in range(10):
    for t in range(200):
      df.append(np.append([t, rep, action, episode_reward], observation))
      effort = np.min(1 - escapement / observation[0], 0)
      observation, reward, terminated, info = env.step(effort)
      episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
df.to_csv("msy.csv")



# identify action at associated with highest cumulative mean reward by last timestep
df2 = ( df
        .groupby(['t','action'], as_index=False)
        .agg({'reward': 'mean',
              'sp1': 'mean',
              'sp2': 'mean',
              'sp3': 'mean'})
        .melt(id_vars=["t", "action", "reward"]) 
        )
best_action = df2.query("t==199")[df2.reward == max(df2.reward)].action.values[0]
df3 = df.query("action == " + str(best_action))
                                

