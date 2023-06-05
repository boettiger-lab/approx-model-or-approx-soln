import numpy as np
import gymnasium as gym
from gymnasium import spaces

_DEFAULT_INIT_POP = np.array([0.7,0.6,1], dtype=np.float32)

def default_penalty(t):
  return - 100/(t+1)

class twoThreeFishing(gym.Env):
  """3 species, 2 are fished"""
  def __init__(self, config=None):
    self.config = config or {}
    #
    self.parameters = config.get("parameters", None)
    self.parametrized = True
    if self.parameters == None:
      self.parametrized = False
    self.growth_fn = config.get(
      "growth_fn", 
      lambda x: np.array(x * (1 - x), dtype = np.float32)
    )
    #
    self.initial_pop = config.get("initial_pop", _DEFAULT_INIT_POP)
    self.Tmax = config.get("Tmax", 200)
    self.threshold = config.get("threshold", np.float32(5e-2))
    self.init_sigma = config.get("init_sigma", np.float32(5e-3))
    #
    self.cost = config.get("cost", np.float32([0.0, 0.0]))
    self.relative_weights = config.get("relative_weights", np.float32([1,1]))
    self.bound = 2 * self.config.get("bound", 2)
    self.fluctuating = self.config.get("fluctuating", False)
    self.training = self.config.get("training", True)
    self.early_end_penalty = self.config.get("early_end_penalty", default_penalty)
    self.timestep = 0
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
    # action *= 0.5
    pop = self.population() # current state in natural units
    
    # harvest & recruitment
    pop, reward = self.harvest(pop, action)
    if self.fluctuating:
      pop = self.growth_fn(pop, self.parameters, self.timestep)
    else:
      pop = self.growth_fn(pop, self.parameters)
    
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
    harvested_fish = pop[:-1] * action
    pop[0] = max(pop[0] - harvested_fish[0], 0)
    pop[1] = max(pop[1] - harvested_fish[1], 0)
    reward_vec = (
      harvested_fish * self.relative_weights - self.cost * action
    )
    total_reward = sum(reward_vec)
    return pop, np.float32(total_reward)
  
  def update_state(self, pop):
    return pop / self.bound
  
  def population(self):
    return self.state * self.bound
  

class oneThreeFishing(gym.Env):
  """3 species, 1 is fished"""
  def __init__(self, config=None):
    self.config = config or {}
    #
    self.parameters = config.get("parameters", None)
    self.parametrized = True
    if self.parameters == None:
      self.parametrized = False
    self.growth_fn = config.get(
      "growth_fn", 
      lambda x: np.array(x * (1 - x), dtype = np.float32)
    )
    #
    self.initial_pop = config.get("initial_pop", _DEFAULT_INIT_POP)
    self.Tmax = config.get("Tmax", 200)
    self.threshold = config.get("threshold", np.float32(5e-2))
    self.init_sigma = config.get("init_sigma", np.float32(5e-3))
    #
    self.cost = config.get("cost", np.float32(0.0))
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
    action = np.clip(action, [0], [1])
    # action *= 0.5
    pop = self.population() # current state in natural units
    
    # harvest & recruitment
    pop, reward = self.harvest(pop, action)
    if self.fluctuating:
      pop = self.growth_fn(pop, self.parameters, self.timestep)
    else:
      pop = self.growth_fn(pop, self.parameters)
    
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
    total_reward = harvested_fish - self.cost * action[0]
    return pop, np.float32(total_reward)
  
  def update_state(self, pop):
    return pop / self.bound
  
  def population(self):
    return self.state * self.bound

class twoThreeFishing_v2(twoThreeFishing):
  """ Includes an incentive to preserve Z """
  def harvest(self, pop, action):
    # action = 0.5 * action # fish at most half the fishes.
    harvested_fish = pop[:-1] * action
    pop[0] = max(pop[0] - harvested_fish[0], 0)
    pop[1] = max(pop[1] - harvested_fish[1], 0)
    reward_vec = (
      harvested_fish * self.relative_weights - self.cost * action
    )
    total_reward = sum(reward_vec)
    #if pop[2] < 0.03:
    #  total_reward *= (pop[2]/0.03)
    return pop, np.float32(total_reward)
  
  def reset(self, *, seed = None, options = None):
    # "*" forces keyword args to be named on calls if values are provided
    # randomized
    self.timestep = 0
    initial_pop = np.random.rand(3).astype(np.float32)
    self.state = self.update_state(initial_pop)
    # self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
    info = {}
    return self.state, info
  
