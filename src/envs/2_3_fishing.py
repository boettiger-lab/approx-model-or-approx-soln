import numpy as np
import gymnasium as gym
from gymnasium import spaces

_DEFAULT_INIT_POP = np.array([0.8,0.2,0.3], dtype=np.float32)

class twoThreeFishing(gym.Env):
  """3 species, 2 are fished"""
  def __init__(self, config=None):
    config = config or {}
    #
    self.parameters = config.get("parameters", None)
    self.growth_fn = config.get(
      "growth_fn", 
      lambda x: np.array(x * (1 - x), dtype = np.float32)
    )
    #
    self.initial_pop = config.get("initial_pop", _DEFAULT_INIT_POP)
    self.Tmax = config.get("Tmax", 200)
    self.threshold = config.get("threshold", np.float32(1e-2))
    self.init_sigma = config.get("init_sigma", np.float32(1e-3))
    #
    self.cost = config.get("cost", np.float32([0.0, 0.0]))
    self.relative_weights = config.get("relative_weights", np.float32([1,1]))
    self.training = self.config.get("training", True)
    self.bound = 2 * self.config.get("bound", 2)
    #
    self.action_space = spaces.Box(
      np.array([0, 0], dtype=np.float32),
      np.array([1, 1], dtype=np.float32),
      dtype = np.float32
    )
    self.observation_space = spaces.Box(
      np.array([0, 0, 0], dtype=np.float32),
      np.array([1, 1, 1], dtype=np.float32),
      dtype=np.float32,
    )
    self.reset(seed=config.get("seed", None))
    #################################################################
  
  def reset(self, *, seed = None, options = None):
    # "*" forces keyword args to be named on calls if values are provided
    self.timestep = 0
    self.state = self.update_state(self.initial_pop)
    self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
    info = {}
    return self.state, info
  
  def step(self, action):
    action = np.clip(action, [0, 0], [1, 1])
    pop = self.population() # current state in natural units
    
    # harvest & recruitment
    pop, reward = self.harvest(pop, action)
    pop = self.growth_fn(pop, self.parameters) 
    
    if any(pop <= self.threshold) and self.training:
            terminated = True
            
    self.state = self.update_state(pop) # transform into [0, 1] space
    observation = self.state
    return observation, reward, terminated, False, {}
  
  ## Functions beyond those standard for gym.Env's:
  
  def harvest(pop, action):
    harvested_fish = pop[:-1] * action
    pop[0] -= harvested_fish[0]
    pop[1] -= harvested_fish[1]
    reward_vec = (
      harvested_fish * self.relative_weights - self.cost * action
    )
    total_reward = sum(reward_vec)
    return pop, np.float32(total_reward)
  
  def update_state(self, pop):
    return pop / self.bound
  
  def population(self):
    return self.state * self.bound
