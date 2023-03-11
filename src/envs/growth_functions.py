import numpy as np

# Collection of 3-species population growth functions, together with
# functions that generate the necessary parameters
# 
# populations: X, Y, Z

def rockPaperScissors(pop, p):
	"""
	in: 
		pop = population, np.array(dtype=np.float32), 
		p = (positive) parameters dictionary p["X"], p["XY"], p["XZ"], ..., all > 0

	Lotka-Volterra

	X eats Y, Y eats Z, Z eats X
	"""
	X, Y, Z = pop[0], pop[1], pop[2]

	pop[0] += - p["X"] * X + p["XY"] * X * Y - p["XZ"] * X * Z
	pop[1] += - p["Y"] * Y + p["YZ"] * Y * Z - p["YX"] * Y * X
	pop[2] += - p["Z"] * Z + p["ZX"] * Z * X - p["ZY"] * Z * Y

	return pop

def params_rockPaperScissors(params = None):
	"""
	sets up parameters for equations
	"""
	if params is not None:
		""" for flexibility """
		return params
	params = {
		"X": np.float32(0.),
		"Y": np.float32(0.),
		"Z": np.float32(0.),
		"XY": np.float32(0.5),
		"XZ": np.float32(0.5),
		"YX": np.float32(0.5),
		"YZ": np.float32(0.5),
		"ZX": np.float32(0.5),
		"ZY": np.float32(0.5),
	}
	return params

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
		"dH": np.float32(0.45),
		"alpha": np.float32(0.3),
		"sigma_x": np.float32(0.1),
		"sigma_y": np.float32(0.05),
		"sigma_z": np.float32(0.05),
		"cost": np.float32(0.01)
	}
	return params






