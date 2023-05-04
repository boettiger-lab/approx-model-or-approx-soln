import numpy as np

# Collection of 3-species population growth functions, together with
# functions that generate the necessary parameters
# 
# populations: X, Y, Z


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

""" DATACODE: KLIMIT """
def K_limit_growth(pop, parameters):
  X, Y, Z = pop[0], pop[1], pop[2]
  p = parameters

  pop[0] += (p["r_x"] * X * (1 - X /  (p["K_x"] - p["cV"] * Y))
        - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
        + p["tau_yx"] * Y - p["tau_xy"] * X  
        + p["sigma_x"] * X * np.random.normal()
       )
    
  pop[1] += (p["r_y"] * Y * (1 - Y /  (p["K_y"] - p["cV"] * X) )
        - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
        - p["tau_yx"] * Y + p["tau_xy"] * X  
        + p["sigma_y"] * Y * np.random.normal()
       )

  pop[2] += p["alpha"] * (
      Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
      + p["sigma_z"] * Z  * np.random.normal()
    )
                          
  pop = pop.astype(np.float32)
  return(pop)

""" for fluctuating functions: """
_PERIOD = 50
_AMPLITUDE = 0.1

def rockPaperScissors(pop, p):
	"""
	in: 
		pop = population, np.array(dtype=np.float32), 
		p = (positive) parameters dictionary p["X"], p["XY"], p["XZ"], ..., all > 0

	Lotka-Volterra

	X eats Y, Y eats Z, Z eats X
	"""
	X, Y, Z = pop[0], pop[1], pop[2]

	pop[0] += p["X"] * X * (1 - X) + p["XY"] * X * Y - p["XZ"] * X * Z
	pop[1] += p["Y"] * Y * (1 - Y) + p["YZ"] * Y * Z - p["YX"] * Y * X
	pop[2] += p["Z"] * Z * (1 - Z)+ p["ZX"] * Z * X - p["ZY"] * Z * Y

	return pop

def params_rockPaperScissors(params = None):
	"""
	sets up parameters for equations
	"""
	if params is not None:
		""" for flexibility """
		return params
	c = 0.3
	params = {
		"X": np.float32(c),
		"Y": np.float32(c),
		"Z": np.float32(c),
		"XY": np.float32(c),
		"XZ": np.float32(c),
		"YX": np.float32(c),
		"YZ": np.float32(c),
		"ZX": np.float32(c),
		"ZY": np.float32(c),
	}
	return params

""" CODE: HOLL3 """
def threeSpHolling3(pop, p):
	"""
	in: 
		pop = population, np.array(dtype=np.float32), 
		p = (positive) parameters dictionary p["v0"], p["K_x"], ..., all > 0

	Similar growth model as in the three_sp base env, but 
	with a Holling's type 3 growth rate for the predator
	(so that all predation terms have Holling's type 3)
	"""
	X, Y, Z, = pop[0], pop[1], pop[2]

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

	pop[2] += Z + p["alpha"] * (
							Z * (p["f"] * ( 
											X**2 / (coupling + X**2) 
											+ p["D"] * Y**2 / (coupling + Y**2)
											) - p["dH"]) 
							+ p["sigma_z"] * Z  * np.random.normal()
	                     )   
	return pop.astype(np.float32)


def params_threeSpHolling3(params = None):
	"""
	sets up parameters for equations. This one is not strictly
	necessary, as threeSpHolling3 uses same parameters as three_sp's
	home growth function. Keeping it for future flexibility and 
	standardization.
	"""
	if params is not None:
		""" for flexibility """
		return params
	params = {
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
		"dH": np.float32(0.2),
		"alpha": np.float32(0.3),
		"sigma_x": np.float32(0.05),
		"sigma_y": np.float32(0.05),
		"sigma_z": np.float32(0.05),
		"cost": np.float32(0.0)
	}
	return params

""" CODE: KFLUC """
def K_fluctuation_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters


    
    coupling = p["v0"]**2 #+ 0.02 * np.sin(2 * np.pi * self.timestep / 60)
    K_x = p["K_x"] + _AMPLITUDE * p["K_x"] * np.sin(2 * np.pi * t / _PERIOD)

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
    return pop.astype(np.float32)

""" CODE: COUPLFLUC """
def coupling_fluctuation_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters


    
    coupling = p["v0"]**2 + _AMPLITUDE * (p["v0"]**2) * np.sin(2 * np.pi * t / _PERIOD)

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
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
    return pop.astype(np.float32)

""" CODE: COMPFLUC """
def competition_fluctuation_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters


    competition = p["cV"] + p["cV"] * _AMPLITUDE * np.sin(2 * np.pi * t / _PERIOD)

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2  + X**2)
          - competition * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2  + Y**2)
          - competition * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)

""" CODE: YABIOTIC """
def y_abiotic_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    Y_center = 0.4
    p = parameters

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2  + X**2)
          - p["cV"] * X * Y
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] = Y_center + 1 * Y_center * np.sin(2 * np.pi * t / 50)
    
    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )   
    
    return pop.astype(np.float32)

""" CODE: ZABIOTIC """
def z_abiotic_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2  + X**2)
          - p["cV"] * X * Y
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2  + Y**2)
          - p["cV"] * X * Y
          + p["sigma_y"] * Y * np.random.normal()
         )
    
    pop[2] += 0.02 * Z * np.cos( 2 * np.pi * t / 50)
    
    return pop.astype(np.float32)

""" CODE: DDRIFT"""
def D_drift_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    D = (
      p["D"] 
      - t * abs(p["D"]-1)/100 
      #+ np.random.normal(0,1) * 0.01 * abs(p["D"]-1)
    )

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
          - p["cV"] * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - D * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
          - p["cV"] * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + D * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)
  
""" CODE: V0DRIFT"""
def v0_drift_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    v0 = (
      p["v0"] * max(
        (1 - t/150),
        1/3
        )
    )

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (v0**2 + X**2)
          - p["cV"] * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (v0**2 + Y**2)
          - p["cV"] * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)

""" CODE: CVDRIFT"""
def cV_drift_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    cV = p["cV"] * min(1 + t/100, 2)

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
          - cV * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
          - cV * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)

""" DATA CODE: BETADRIFT """ 
def beta_drift_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    beta = p["beta"] * (1 + t/100)

    pop[0] += (p["r_x"] * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
          - p["cV"] * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
          - p["cV"] * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)

""" CODE: RXDRIFT"""
def rx_drift_growth(pop, parameters, t):
    X, Y, Z = pop[0], pop[1], pop[2]
    p = parameters
    
    r_x = (
      p["r_x"] * max(1 - 0.5 * t/100, 1/2)
    )

    pop[0] += (r_x * X * (1 - X / p["K_x"])
          - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
          - p["cV"] * X * Y
          + p["tau_yx"] * Y - p["tau_xy"] * X  
          + p["sigma_x"] * X * np.random.normal()
         )
    
    pop[1] += (p["r_y"] * Y * (1 - Y / p["K_y"] )
          - p["D"]  * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
          - p["cV"] * X * Y
          - p["tau_yx"] * Y + p["tau_xy"] * X  
          + p["sigma_y"] * Y * np.random.normal()
         )

    pop[2] += p["alpha"] * (
                          Z * (p["f"] * (X + p["D"]  * Y) - p["dH"]) 
                          + p["sigma_z"] * Z  * np.random.normal()
                         )  
    return pop.astype(np.float32)

""" DATACODE: KLIMIT_RXDRIFT """
def K_limit_rx_drift_growth(pop, parameters, t):
  X, Y, Z = pop[0], pop[1], pop[2]
  p = parameters
  
  r_x = (
      p["r_x"] * max(1 - 0.5 * t/100, 1/2)
    )

  pop[0] += (r_x * X * (1 - X /  (p["K_x"] - p["cV"] * Y))
        - p["beta"] * Z * (X**2) / (p["v0"]**2 + X**2)
        + p["tau_yx"] * Y - p["tau_xy"] * X  
        + p["sigma_x"] * X * np.random.normal()
       )
    
  pop[1] += (p["r_y"] * Y * (1 - Y /  (p["K_y"] - p["cV"] * X) )
        - p["D"] * p["beta"] * Z * (Y**2) / (p["v0"]**2 + Y**2)
        - p["tau_yx"] * Y + p["tau_xy"] * X  
        + p["sigma_y"] * Y * np.random.normal()
       )

  pop[2] += p["alpha"] * (
      Z * (p["f"] * (X + p["D"] * Y) - p["dH"]) 
      + p["sigma_z"] * Z  * np.random.normal()
    )
                          
  pop = pop.astype(np.float32)
  return(pop)

