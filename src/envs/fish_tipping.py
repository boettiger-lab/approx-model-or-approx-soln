# NB: It is typical to use float32 precision to benefit from enhanced GPU speeds
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parameters import parameters

_DEFAULT_PARAMETERS = parameters().parameters()

def default_early_end_penalty(t):
  return - 100/t

def default_population_growth(pop, parameters):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    coupling = p["v0"]**2 #+ 0.02 * np.sin(2 * np.pi * self.timestep / 60)
    K_x = p["K_x"] # + 0.01 * np.sin(2 * np.pi * self.timestep / 30)

    pop[0] += (p["r_x"] * X * (1 - X / K_x)
          - p["beta"] * Z * (X**2) / (coupling + X**2)
          - p["cV"] * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (coupling + Y**2)
          - p["cV"] * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )        
    
    # consider adding the handling-time component here too instead of these   
    #Z = Z + p["alpha"] * (Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
    #                      + p["sigma_z"] * Z  * np.random.normal())
                          
    pop = pop.astype(np.float32)
    return(pop)

class three_sp(gym.Env):
    """A 3-species ecosystem model"""
    def __init__(self, config=None):
        config = config or {},
        initial_pop = np.array([0.8396102377828771, 
                                0.05489978383850558,
                                0.3773367609828674],
                                dtype=np.float32)
        # initial_pop = np.array([0.85, 0.06, 0.4], dtype = np.float32)
        
        ## kink due to initializing the env through PPOConfig().build()
        config = config[0]
                                
        ## these parameters may be specified in config                                  
        self.Tmax = config.get("Tmax", 200)
        self.threshold = config.get("threshold", np.float32(5e-3))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", initial_pop)
        self.parameters = config.get("parameters", _DEFAULT_PARAMETERS)
        self.growth_fn = config.get("growth_fn", default_population_growth)
        self.early_end_penalty = config.get("early_end_penalty", default_early_end_penalty)
        self.fluctuating = config.get("fluctuating", False) # do parameters fluctuate with time?
        self.cost = config.get("cost", np.float32(0.0))
        
        # Growth function:
        
        self.bound = 2 * self.parameters.get("K_x", 1)
        
        self.action_space = spaces.Box(
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )        
        self.reset(seed=config.get("seed", None))

    def initial_pop():
        return 
    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.update_state(self.initial_pop)
        self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
        info = {}
        return self.state, info

    def step(self, action):
        action = np.clip(action, [0], [1])
        pop = self.population() # current state in natural units
        
        # harvest and recruitment. 
        pop, reward = self.harvest(pop, action)
        if self.fluctuating:
          pop = self.growth_fn(pop, self.parameters, self.timestep)
        else:
          pop = self.growth_fn(pop, self.parameters)
        
        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)
        
        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward += self.early_end_penalty(self.timestep)
        
        self.state = self.update_state(pop) # transform into [-1, 1] space
        observation = self.observation() # same as self.state
        return observation, reward, terminated, False, {}


    
    def harvest(self, pop, action): 
        harvest = action * pop[0]
        pop[0] = pop[0] - harvest[0]
        
        reward = np.max(harvest[0],0) - self.cost * action
        return pop, np.float32(reward[0])
      

    def observation(self): # perfectly observed case
        return self.state
    
    # inverse of self.population()
    def update_state(self, pop):
        pop = np.clip(pop, 0, np.Inf) # enforce non-negative population first
        self.state = np.array([
          2 * pop[0] / self.bound - 1,
          2 * pop[1] / self.bound - 1,
          2 * pop[2] / self.bound - 1],
          dtype=np.float32)
        return self.state
    
    def population(self):
        pop = np.array(
          [(self.state[0] + 1) * self.bound / 2,
           (self.state[1] + 1) * self.bound / 2,
           (self.state[2] + 1) * self.bound / 2],
           dtype=np.float32)
        return np.clip(pop, 0, np.Inf)
    
    
class two_sp_one_p(three_sp):
    def growth_function(
        pop, 
        parameters, 
        t, 
        center = 0.4,
        amplitude = 0.6,
        frequency = 50,
    ):
        X, Y, Z = pop[0], pop[1], pop[2]
        Y_center = center
        p = parameters

        pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
            - p["beta"] * Z * (X**2) / (p["v0"]**2  + X**2)
            - p["cV"] * X * Y
            + p["sigma_x"] * X * np.random.normal()
            )
    
        pop[1] = (
            Y_center 
            + amplitude * Y_center * np.sin(2 * np.pi * t / frequency)
            )
    
        pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )   
    
        return pop.astype(np.float32)
    
    def step(self, action):
        action = np.clip(action, [0], [1])
        pop = self.population() # current state in natural units
        
        self.center =  0.4
        self.amplitude = 0.6
        self.frequency = 50
        
        # harvest and recruitment. 
        pop, reward = self.harvest(pop, action)
        if self.fluctuating:
          pop = self.growth_function(
              pop, 
              self.parameters, 
              self.timestep,
              self.center,
              self.amplitude,
              self.frequency
              )
        else:
          pop = self.growth_function(
              pop, 
              self.parameters, 
              self.timestep,
              self.center,
              self.amplitude,
              self.frequency
              )
        
        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)
        
        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward -= 50/self.timestep
        
        self.state = self.update_state(pop) # transform into [-1, 1] space
        observation = self.observation() # same as self.state
        return observation, reward, terminated, False, {}
    
    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.update_state(
            np.array(
                
            )
        )
        self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
        info = {}
        return self.state, info


class three_sp_two_fisheries(gym.Env):
    """A 3-species ecosystem model"""
    def __init__(self, config=None):
        config = config or {},
        initial_pop = np.array([0.8396102377828771, 
                                0.05489978383850558,
                                0.3773367609828674],
                                dtype=np.float32)
        # initial_pop = np.array([0.85, 0.06, 0.4], dtype = np.float32)
        
        ## kink due to initializing the env through PPOConfig().build()
        config = config[0]
                                
        ## these parameters may be specified in config                                  
        self.Tmax = config.get("Tmax", 200)
        self.threshold = config.get("threshold", np.float32(5e-3))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", initial_pop)
        self.parameters = config.get("parameters", _DEFAULT_PARAMETERS)
        self.growth_fn = config.get("growth_fn", default_population_growth)
        self.fluctuating = config.get("fluctuating", False) # do parameters fluctuate with time?
        
        # Growth function:
        
        self.bound = 2 * self.parameters.get("K_x", 1)
        
        self.action_space = spaces.Box(
            np.array([0, 0], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )        
        self.reset(seed=config.get("seed", None))


    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.update_state(self.initial_pop)
        self.state += np.float32(self.init_sigma * np.random.normal(size=3) )
        info = {}
        return self.state, info

    def step(self, action):
        action = np.clip(action, [0, 0], [1, 1])
        pop = self.population() # current state in natural units
        
        # harvest and recruitment. 
        pop, reward = self.harvest(pop, action)
        if self.fluctuating:
          pop = self.growth_fn(pop, self.parameters, self.timestep)
        else:
          pop = self.growth_fn(pop, self.parameters)
        
        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)
        
        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward -= 50/self.timestep
        
        self.state = self.update_state(pop) # transform into [-1, 1] space
        observation = self.observation() # same as self.state
        return observation, reward, terminated, False, {}


    
    def harvest(self, pop, action): 
        harvest_X = action[0] * pop[0]
        harvest_Z = action[1] * pop[2]
        pop[0] = pop[0] - harvest_X
        pop[2] = pop[2] - harvest_Z
        
        reward = np.max(harvest_X + harvest_Y, 0)
        return pop, np.float32(reward[0])
      

    def observation(self): # perfectly observed case
        return self.state
    
    # inverse of self.population()
    def update_state(self, pop):
        pop = np.clip(pop, 0, np.Inf) # enforce non-negative population first
        self.state = np.array([
          2 * pop[0] / self.bound - 1,
          2 * pop[1] / self.bound - 1,
          2 * pop[2] / self.bound - 1],
          dtype=np.float32)
        return self.state
    
    def population(self):
        pop = np.array(
          [(self.state[0] + 1) * self.bound / 2,
           (self.state[1] + 1) * self.bound / 2,
           (self.state[2] + 1) * self.bound / 2],
           dtype=np.float32)
        return np.clip(pop, 0, np.Inf)    
