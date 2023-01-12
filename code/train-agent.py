from gym_fishing.envs import forageVVHcont
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
import os

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
    "create_env_on_local_worker": True  # needed to restore env from checkpoint
}

agent = PPOTrainer(config=config)
iterations = 250
checkpoint = ("cache/checkpoint_000{}".format(iterations))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
      agent.train()
  checkpoint = agent.save("cache")


# Restore saved agent:
agent = PPOTrainer(config=config)
agent.restore(checkpoint)

# agent.evaluate() # built-in method to evaluate agent on eval env

# Initialize saved copy of eval environment:
env = agent.env_creator(agent.evaluation_config.env_config)

import pandas as pd
import numpy as np
df = []

for rep in range(10):
  episode_reward = 0
  observation = env.reset()
  for t in range(200):
    action = agent.compute_single_action(observation)
    df.append(np.append([t, rep, action[0], episode_reward], observation))
    observation, reward, terminated, info = env.step(action)
    episode_reward += reward

cols = ["t", "rep", "action", "reward", "sp1", "sp2", "sp3"]
df = pd.DataFrame(df, columns = cols)
df.to_csv("PPO.csv")

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




