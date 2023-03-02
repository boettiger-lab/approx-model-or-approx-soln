# NB: It is typical to use float32 precision to benefit from enhanced GPU speeds


import numpy as np
import gymnasium as gym
from gymnasium import spaces

class fish_tipping(gym.Env):
    """A 3-species ecosystem model"""
    def __init__(self, config=None):
        config = config or {}
        parameters = {
         "r_x": np.float32(0.1),
         "r_y": np.float32(0.1),
         "K_x": np.float32(1.0),
         "K_y": np.float32(1.0),
         "beta": np.float32(0.3),
         "v0":  np.float32(0.1),
         "D": np.float32(1.1),
         "tau_yx": np.float32(0),
         "tau_xy": np.float32(0),
         "cV": np.float32(0.5), 
         "f": np.float32(0.5), 
         "dH": np.float32(0.45),
         "alpha": np.float32(0.3),
         "sigma_x": np.float32(0.05),
         "sigma_y": np.float32(0.1),
         "sigma_z": np.float32(0.1)
        }
        initial_state = np.array([0.8396102377828771, 
                                    0.05489978383850558,
                                    0.3773367609828674],
                                    dtype=np.float32) 
        ## these parameters may be specified in config                                  
        self.Tmax = config.get("Tmax", 200)
        self.threshhold = config.get("threshhold", np.float32(0.08))
        self.training = config.get("training", True)
        self.initial_state = config.get("initial_state", initial_state)
        self.parameters = config.get("parameters", parameters)
        
        self.bound = 2 * self.parameters["K_x"]
        
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


    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.state = self.initial_state
        return [self.state], {}

    def step(self, action):
        action = np.clip(action, [0], [1])
        pop = self.population() # current state in natural units
        
        # harvest and recruitment
        pop, reward = self.harvest(pop, action)
        pop = self.population_growth(pop)
        
        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)
        
        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            self.reward -= 50/self.timestep
        
        self.state = self.update_state(pop) # transform into [-1, 1] space
        observation = self.observation() # same as self.state
        return observation, reward, terminated, False, {}


    
    def harvest(self, pop, action): 
        harvest = action * pop[0]
        pop[0] = pop[0] - harvest
        return pop, harvest
      
    def population_growth(self, pop):
        X, Y, Z = pop[0], pop[1], pop[2]
        p = self.parameters

        X += (p["r_x"] * X * (1 - X / p["K_x"])
              - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
              - p["cV"] * X * Y
              + p["tau_yx"] * Y - p["tau_xy"] * X  
              + p["sigma_x"] * X * np.random.normal()
             )
        
        Y += (p["r_y"] * Y * (1 - Y / p["K_y"])
              - p["D"] * ["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
              - p["cV"] * X * Y
              - p["tau_yx"] * Y + p["tau_xy"] * X  
              + p["sigma_y"] * Y * np.random.normal()
             )
        Z = Z + p["alpha"] * (Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                              + p["sigma_z"] * Z  * np.random.normal())
                              
        state = np.array(X, Y, Z, dtype=np.float32)
        return(self.state)

    def observation(self): # perfectly observed case
        return(self.state)
    
    # inverse of self.population()
    def update_state(self, pop):
        pop = np.clip(pop, 0, np.Inf) # enforce non-negative population first
        self.state = np.array([
          2 * pop[0] / self.bound - 1,
          2 * pop[1] / self.bound - 1,
          2 * pop[2] / self.bound - 1],
          dtype=np.float32)
        return(self.state)
    
    def population(self):
        return np.array(
          [(self.state[0] + 1) * self.bound / 2,
           (self.state[1] + 1) * self.bound / 2,
           (self.state[2] + 1) * self.bound / 2],
           dtype=np.float32)
    
    
    
