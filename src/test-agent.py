from envs import fish_tipping
from envs import growth_functions
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


register_env("fish_tipping",fish_tipping.three_sp)

iterations = 451
checkpoint = "cache/checkpoint_000{iterations}"

config = ppo.PPOConfig()
config.training(vf_clip_param = 50.0)
config.num_envs_per_worker=20
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
#
config.env="fish_tipping"
config.env_config["growth_fn"] = growth_functions.y_abiotic_growth
config.env_config["fluctuating"] = True
_DATACODE = "YABIOTIC"
_PATH = f"../data/{_DATACODE}"
_FILENAME = f"PPO{iterations}"
# config.env_config["parameters"] = (
#    growth_functions.params_threeSpHolling3() 
# )

agent = config.build()
agent.restore(checkpoint)



