import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parameters import parameters

def dummy_growth(pop,parameters):
  return pop * (1 - pop)

_DEFAULT_INIT_POP = np.array([0.7], dtype=np.float32)

def default_penalty(t):
  return - 100/(t+1)

class singleSp(gym.Env):
  """ Single species model. """
  def __init__(self, config=None):
    self.config = config or {}
    #
    self.parameters = config.get("parameters", None)
    if not self.parameters:
      print("the config dict argument needs a parameters component!")
    self.growth_fn = config.get("growth_fn", dummy_growth)
    self.initial_pop = config.get("initial_pop", _DEFAULT_INIT_POP)
    self.Tmax = config.get("Tmax", 200)
    self.threshold = config.get("threshold", np.float32(5e-2))
    self.init_sigma = config.get("init_sigma", np.float32(5e-3))
    self.num_species = 1
    #
    self.bound = 2 * self.config.get("bound", 2)
    self.fluctuating = self.config.get("fluctuating", False)
    self.training = self.config.get("training", True)
    self.early_end_penalty = self.config.get("early_end_penalty", default_penalty)
    self.timestep = 0
    #
    self.action_space = spaces.Box(
      np.array([0], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype = np.float32
    )
    self.observation_space = spaces.Box(
      np.array([0], dtype=np.float32),
      np.array([1], dtype=np.float32),
      dtype=np.float32,
    )
    self.reset(seed=config.get("seed", None))
    #################################################################
  
  def reset(self, *, seed = None, options = None):
    # "*" forces keyword args to be named on calls if values are provided
    self.timestep = 0
    self.state = self.update_state(self.initial_pop)
    self.state += np.float32(self.init_sigma * np.random.normal(size=1) )
    info = {}
    return self.state, info
  
  def step(self, action):
    action = np.clip(action, [0], [1])
    # action *= 0.5
    pop = self.population() # current state in natural units
    
    # harvest & recruitment
    pop, reward = self.harvest(pop, action)
    if self.fluctuating:
      pop = self.growth_fn(pop, self.parameters, self.timestep)
    else:
      pop = self.growth_fn(pop, self.parameters)
    pop[0] = max(pop[0], 0 )
    # Conservation goals:
    terminated = False
    if any(pop <= self.threshold) and self.training:
      terminated = True
      reward += self.early_end_penalty(self.timestep)
            
    self.state = self.update_state(pop) # transform into [0, 1] space
    observation = self.state
    self.timestep += 1
    return observation, reward, terminated, False, {}
  
  ## Functions beyond those standard for gym.Env's:
  
  def harvest(self, pop, action):
    harvested_fish = pop[0] * action[0]
    pop[0] = max(pop[0] - harvested_fish, 0)
    return pop, harvested_fish
  
  def update_state(self, pop):
    return pop / self.bound
  
  def population(self):
    return self.state * self.bound
    
