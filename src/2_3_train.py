from envs import two_three_fishing
from envs import growth_functions
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch

## GLOBALS:
_DEFAULT_PARAMS = {
  "r_x": np.float32(1.0),
  "r_y": np.float32(1.0),
  "K_x": np.float32(1.0),
  "K_y": np.float32(1.0),
  "beta": np.float32(0.3),
  "v0":  np.float32(0.1),
  "D": np.float32(1.1),
  "tau_yx": np.float32(0),
  "tau_xy": np.float32(0),
  "cV": np.float32(0.5), 
  "f": np.float32(0.25), 
  "dH": np.float32(0.45),
  "alpha": np.float32(0.3),
  "sigma_x": np.float32(0.1),
  "sigma_y": np.float32(0.05),
  "sigma_z": np.float32(0.05)
}

_DATACODE = "TwoFisheriesStatic"
_PATH = f"../data/{_DATACODE}"
_FILENAME = f"PPO{iterations}"

## SETTING UP RL ALGO

register_env("two_three_fishing", two_three_fishing.twoThreeFishing)

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="two_three_fishing"
#
config.env_config["parameters"] = _DEFAULT_PARAMS
config.env_config["growth_fn"] = growth_functions.default_population_growth
#
agent = config.build()
#

## TRAIN
iterations = 250
checkpoint = f"cache/checkpoint_{_DATACODE}_iter{iterations}"

for i in range(iterations):
  print(f"iteration nr. {i}", end="\r")
  agent.train()

checkpoint = agent.save(f"cache/{checkpoint}")

## POST TRAINING
stats = agent.evaluate()

config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)
