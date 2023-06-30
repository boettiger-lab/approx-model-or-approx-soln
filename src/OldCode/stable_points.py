from scipy.optimize import fsolve
from numpy.random import rand
from pandas import DataFrame
import numpy as np
import time
import sys

def L(v):
    return v * (1 - v)

def F(v,z,beta,c):
    return beta * z * (v ** 2) / ((c ** 2) + (v ** 2))

def X_eq(point):
    X, Y, Z, beta, c, cXY, D, dZ, f, alpha = point
    return L(X) - F(X, Z, beta, c) - cXY * X * Y

def Y_eq(point):
    X, Y, Z, beta, c, cXY, D, dZ, f, alpha = point
    return L(Y) - D * F(Y, Z, beta, c) - cXY * X * Y

def Z_eq(point):
    X, Y, Z, beta, c, cXY, D, dZ, f, alpha = point
    return alpha * (- dZ * Z + f * Z * (X + D * Y))

def alt_Z_eq(point):
    X, Y, Z, beta, c, cXY, D, dZ, f, alpha = point
    return Z - 0.5
    
def set_param(param, param_value):
    return param - param_value

def eqs_params(point, param_vals):
    """
    Args: variables, and then parameters
    
    (vars)
    X, Y, Z, 
    
    (params)
    beta, c, cXY, D, dZ, f
    """
    X, Y, Z, beta, c, cXY, D, dZ, f, alpha = point
    beta_val, c_val, cXY_val, D_val, dZ_val, f_val, alpha_val = param_vals
    
    return (
        X_eq(point),
        Y_eq(point),
        Z_eq(point),  
        # and now the eqs that fix the params:
        set_param(beta, beta_val),
        set_param(c, c_val),
        set_param(cXY, cXY_val),
        set_param(D, D_val),
        set_param(dZ, dZ_val),
        set_param(f, f_val),
        set_param(alpha, alpha_val),
    )

def eqs(param_vals):
    return lambda pt: eqs_params(pt, param_vals)
    

"""
Find solutions
"""

def var_range(varname):
    if varname == "cXY" or varname == "beta":
        return np.linspace(0.1,0.9,170)
    elif varname == "c":
        return np.linspace(0.01,0.5,150)
    elif varname == "D":
        return np.linspace(0.75, 1.25, 150)
    elif varname == "dZ":
        return np.linspace(0.2, 0.8, 150) 
    elif varname == "f":
        return np.linspace(0.05, 0.95, 150)

def fixed_points(varname, var_range, param_inds, param_vals):
    var_list = []
    sol_list = []
    for x in var_range:
        # param_vals[param_inds[varname]] = x
        param_vals = reassign_tuple_entry(param_vals, param_inds[varname], x)
        s = create_sol_list(param_vals)
        var_list += len(s)*[x]
        if s is not None:
            sol_list += s
    return var_list, sol_list
    

def create_sol_list(param_vals, N = 150):
    sol_list = []
    for _ in range(N):
        eqs_at_params = eqs(param_vals)
        sol_tuple = fsolve(eqs_at_params, (
            #vars
            rand(), #0.5, 
            rand(), #0.5, 
            rand(), #0.5,
            # params
            *(p for p in param_vals)
        ))
        sol = list(sol_tuple)
        if (
            check_sol(sol) and not 
            check_sol_in_sol_list(sol, sol_list)
        ):
            sol = set_zero(sol)
            sol = set_one(sol)
            if all([sol[i]>= 0 for i in range(3)]):
                sol_list.append(sol[:3])
    
    return sol_list

"""
###################
Checking functions:
###################
"""
standard_thresh = 1e-5

def iseq(a, b, thr=standard_thresh):
    """
    Are two numbers equal, up to a threshold? Threshold chosen ad-hoc by looking
    at what the typical variations in  the solutions were, and then being a bit
    lenient. I don't think real pairs of solutions will differ by this little.
    """
    return abs(a-b) < thr


def check_sol(sol):
    is_sol = iseq(X_eq(sol), 0) and (iseq(Y_eq(sol), 0) and iseq(Z_eq(sol), 0))
    is_bounded = all([sol[i]<1+standard_thresh for i in range(3)])
    return is_sol and is_bounded

def check_sol_eq(sol1, sol2, thr=standard_thresh):
    """
    Checks whether two solutions are actually the
    same, up to the threshold.
    """
    return all([
        iseq(sol1[i], sol2[i], thr) for i in range(3)
    ])

def check_sol_in_sol_list(sol, sol_list, thr=standard_thresh):
    return any([
        check_sol_eq(solution, sol, thr) for solution in sol_list
    ])

def set_zero(sol):
    for i in range(3):
        if iseq(sol[i], 0):
            sol[i] = 0
    return sol

def set_one(sol):
    for i in range(3):
        if iseq(sol[i], 1):
            sol[i] = 1
    return sol
    
"""
Helper
"""
    
def printy(list_of_lists):
    print("[")
    for list in list_of_lists:
        print(list)
    print("]")

def sort_helper(v):
    return v[0]*100 + v[1]*10 + v[2]

def reassign_tuple_entry(tup, ind, val):
    l = list(tup)
    l[ind] = val
    return tuple(l)

def visualize(sol_list):
    import matplotlib.pyplot as plt

    Xs = []
    Ys = []
    Zs  = []
    for s in sol_list:
        Xs.append(s[0])
        Ys.append(s[1])
        Zs.append(s[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Xs, Ys, Zs)
    plt.show()
    return Xs, Ys, Zs

def individual_plots(Xs, Ys, Zs, param_vals, varname):
    import matplotlib.pyplot as plt
    
    plt.scatter(param_vals, Xs, marker='.')
    plt.title(f"X coordinate of stable points")
    plt.xlabel(f"{varname}")
    plt.ylabel("X")
    plt.show()

    plt.scatter(param_vals, Ys, marker='.')
    plt.title(f"Y coordinate of stable points")
    plt.xlabel(f"{varname}")
    plt.ylabel("Y")
    plt.show()

    plt.scatter(param_vals, Zs, marker='.')
    plt.title(f"Z coordinate of stable points")
    plt.xlabel(f"{varname}")
    plt.ylabel("Z")
    plt.show()

def choose_varying_parameter(name_list):
    prompt = """
    Choose the parameter you wish to vary for the fixed point plots.
    (Options are: beta, c, cXY, D, dZ, f. Case sensitive atm, sorry :D.)
    
    """
    varname = input(prompt)
    if varname in name_list:
        return varname
    else:
        print("Didn't recognize that, so I chose cXY for you >:+)")
        return "cXY"

def main() -> None:
    param_vals = (0.3, 0.1, 0.5, 1.1, 0.45, 0.5, 0.3)
    param_vals_dict = {"beta":0.3, "c":0.1, "cXY":0.5, "D":1.1, "dZ":0.45, "f":0.5, "alpha":0.3}
    print(f"Fixed point for params {param_vals_dict}:")
    sols = create_sol_list(param_vals)
    for sol in sols: print(sol);

    ## Uncomment to plot stable points as function of varying parameter:
    param_inds = {"beta":0, "c":1, "cXY":2, "D":3, "dZ":4, "f":5, "alpha":6}
    param_vals_list = list(param_vals)

    varname = "cXY"
    varname = choose_varying_parameter(param_inds.keys())
    
    var_rng = var_range(varname)
    param_vals, sol_list = fixed_points(varname, var_rng, param_inds, param_vals)
    Xs, Ys, Zs = visualize(sol_list)
    
    individual_plots(Xs, Ys, Zs, param_vals, varname)
    
    df = DataFrame(zip(param_vals,Xs,Ys,Zs), columns = [varname, "X", "Y", "Z"])
    df.to_csv(f'data/FixedPonts_{varname}.csv.xz', index = False)

    df_X = DataFrame(zip(param_vals, Xs), columns = [varname, "X"])
    df.to_csv(f"data/FixedPoints_{varname}_X.csv.xz", index=False)

if __name__ == "__main__":
    main()
