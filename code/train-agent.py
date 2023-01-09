from gym_fishing.envs import FishingCtsEnv
from gym_fishing.envs import forageVVHcont
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env


## We could call env directly without this if only  our envs took a env_config dict argument
register_env("fish-3sp", lambda config: forageVVHcont())

# Configure the algorithm.
config = {
    "env": "fish-3sp",
    "num_workers": 1,
    "num_envs_per_worker": 20,
    "resources": {
      "num_cpus_per_worker": 12,
    },
    "framework": "torch",
    "num_gpus": 1,
    "log_level": "ERROR",
    "create_env_on_local_worker": True  # need this to restore env from checkpoint
}

trainer = PPOTrainer(config=config)
iterations = 250

for _ in range(iterations):
    trainer.train()
    
checkpoint = trainer.save("cache")


# Evaluate a saved agent:
checkpoint = ("cache/checkpoint_000{}".format(iterations))
model = PPOTrainer(config=config)
model.restore(checkpoint)
model.evaluate()

# simulate from the saved agent:
env_config = model.evaluation_config.env_config
env = model.env_creator(env_config)

import pandas as pd
import numpy as np
df = []
episode_reward = 0
observation = env.reset()
for rep in range(10):
  for t in range(200):
    action = model.compute_single_action(observation)
    df.append(np.append([t, rep, action[0], episode_reward], observation))
    observation, reward, terminated, info = env.step(action)
    episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
#df.to_csv("msy.csv")

from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
df2 = (df
       .melt(id_vars=["t", "action", "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',,
             'action': 'mean'})) 


(ggplot(df2, aes("t", "value", color="variable")) +
 geom_line())
(ggplot(df2, aes("t", "action", color="variable")) + geom_line())
(ggplot(df2, aes("t", "reward", color="variable")) + geom_line())

(ggplot(df, aes("sp2", "sp3", color="action", size="sp1")) + geom_point())
(ggplot(df, aes("sp1", "sp2", color="action", size="sp3")) + geom_point())
(ggplot(df, aes("sp1", "sp3", color="action", size="sp2")) + geom_point())





df = []
actions = np.linspace(0,.1,101)
for action in actions:
  episode_reward = 0
  observation = env.reset()
  for rep in range(10):
    for t in range(200):
      df.append(np.append([t, rep, action, episode_reward], observation))
      observation, reward, terminated, info = env.step(action)
      episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
df.to_csv("msy.csv")






df2 = ( df
        .groupby(['t','action'], as_index=False)
        .agg({'reward': 'mean',
              'sp1': 'mean',
              'sp2': 'mean',
              'sp3': 'mean'})
        .melt(id_vars=["t", "action", "reward"]) 
        ) 

best_action = df2.query("t==199")[df2.reward == max(df2.reward)].action.values[0]

q = "action == " + str(best_action)
df3 = df.query(q)
                                
(ggplot(df3, aes("t", "value", group = interaction("rep", "variable"), color="variable")) +
 geom_line())

