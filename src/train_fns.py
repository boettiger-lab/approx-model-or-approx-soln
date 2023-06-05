""" gather the functions used for training control methods """

from envs import two_three_fishing
from envs import growth_functions
from parameters import parameters
from util import dict_pretty_print
import callback_fn
from ray.rllib.algorithms import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch

_datacode_to_growth_fn = {
  "DEFAULT": growth_functions.default_population_growth,
  "RXDRIFT": growth_functions.rx_drift_growth,
  "V0DRIFT": growth_functions.v0_drift_growth,
  "DDRIFT": growth_functions.D_drift_growth,
  "KLIMIT": growth_functions.K_limit_growth,
  "KLIMIT_RXDRIFT": growth_functions.K_limit_rx_drift_growth,
  "BETADRIFT": growth_functions.beta_drift_growth,
  "CVDRIFT": growth_functions.cV_drift_growth,
  "YABIOTIC": growth_functions.y_abiotic_growth,
  "ZABIOTIC": growth_functions.z_abiotic_growth,
  "COUPLFLUC": growth_functions.coupling_fluctuation_growth,
}

def create_env(
  env_class, 
  parameter_obj, 
  datacode = "DEFAULT", 
  fluctuating=True, 
  training=True):
	""" 
	INPUT
	env_class: the environment class to be created
	parameter_obj: a parameters.parameters() object
	datacode: encodes which type of growth_fn is used (and, in other functions, the folder where plots are saved)
	fluctuating: does growth_fn take t as an argument?
	training: should we cut the episodes short if they end early? (used mostly for training)
	
	OUTPUT
	env_class type object
	"""
	config = {}
	config["parameters"] = parameter_obj.parameters()
	config["growth_fn"] = _datacode_to_growth_fn[datacode]
	config["fluctuating"] = fluctuating
	config["training"] = training
	config["initial_pop"] = parameters.init_state()
	env = env_class(
	    config = config
	)
	return env

def create_agent(env_class, parameters_obj, *, datacode = "DEFAULT", env_name="testEnvName", fluctuating=True, training=True):
  register_env(env_name, env_class)
  config = ppo.PPOConfig()
  config.training(vf_clip_param = 50.0)
  config.num_envs_per_worker=20
  config = config.resources(num_gpus=torch.cuda.device_count())
  config.framework_str="torch"
  config.create_env_on_local_worker = True
  config.env=env_name
  #
  config.env_config["parameters"] = parameters_obj.parameters()
  config.env_config["growth_fn"] = _datacode_to_growth_fn[datacode]
  config.env_config["fluctuating"] = fluctuating
  config.env_config["training"] = training
  config.env_config["initial_pop"] = parameters_obj.init_state()
  # agent = config.build()
  agent = PPOTrainer(config=config)
  return agent

def train_agent(agent, iterations, path_to_checkpoint="cache", verbose = True):
  for i in range(iterations):
    if verbose:
      print(f"iteration nr. {i}", end="\r")
    agent.train()
  checkpoint = agent.save(os.path.join(path_to_checkpoint, f"PPO{iterations}_checkpoint"))
  return agent
