import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import modules from code/
import data
import numpy as np
import matplotlib.pylab as plt
from pysr import PySRRegressor
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

def make_rhs_vectorized(exprs, variables):
    """
    Converts a list of SymPy expressions into a numerical function for a system of ODEs,
    treating constants as symbolic parameters.

    Parameters:
    - exprs: List of SymPy expressions representing dx/dt for each variable.
    - variables: List of SymPy symbols representing state variables.

    Returns:
    - A tuple (dxdt_func, params), where:
        - dxdt_func: A callable function taking (t, x, *params) and returning a vector.
        - params: List of symbolic parameters replacing constants.
    """
    # Collect all constants across all expressions
    constants = set()
    for expr in exprs:
        constants |= {term for term in expr.atoms(sp.Number) if term != 0 and term != 1}

    # Create symbolic parameters for constants
    param_symbols = [sp.Symbol(f'c{i}') for i, _ in enumerate(constants)]
    const_to_param = dict(zip(constants, param_symbols))

    # Replace constants with symbolic parameters in each expression
    modified_exprs = [expr.subs(const_to_param) for expr in exprs]

    # Create a function returning a vector (list) of expressions
    func = sp.lambdify(variables + param_symbols, modified_exprs, 'numpy')

    # Define the callable function
    def dxdt(t, x, *params):
        return np.array(func(*x, *params))

    return dxdt, param_symbols

def extract_sympy_equations(eqs_all):
    """
    Extract sympy equations from PySR output for each species.

    Parameters:
    - eqs_all: list of DataFrames (PySR model.equations_), one per species

    Returns:
    - List of lists, where each inner list contains sympy expressions
      for a single species.
    """
    eqs_by_species = []
    for eqs_spp in eqs_all:
        # Extract all sympy expressions for this species
        sympy_exprs = list(eqs_spp["sympy_format"])
        eqs_by_species.append(sympy_exprs)
    return eqs_by_species

path_name = "../../data/synthetic_simplest"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

#load all datasets
data_fd = data.Data(file_list)
#inds_mono = find_k_nonzero_column_indices(data_jo.abundances, 2)
#stack all replicates vertically for fit
X = np.vstack(data_fd.abundances)
#obtain "observed" derivatives
y = np.diff(X, axis = 0)

model = PySRRegressor(
    maxsize=20,
    niterations=80,  # < Increase me for better results
    binary_operators=["+", "-", "*"]
)

model.fit(X[0:-1,:], y) #we select all the points but the
print(model)
#get list of all equations for all species
eqs_all = model.equations_ 

eqs_grouped = extract_sympy_equations(eqs_all)

# Accessing first equation of species 1:
eq_1_spp_1 = eqs_grouped[0][0]

# Accessing second equation of species 2:
eq_2_spp_2 = eqs_grouped[1][1]
eqs_spp_1 = eqs_all[0] #all equations for species 1
eqs_spp_2 = eqs_all[1] #all equations for species 2

#pair them to form system
#equation 1, species 1
eq_2_spp_1 = eqs_spp_1["sympy_format"][2]
eq_1_spp_2 = eqs_spp_2["sympy_format"][1]


##piece needed for later
#    return array({sp.lambdify(variables + param_syms, [expr.subs({k: v for k, v in zip(param_syms, param_syms)}) for expr in exprs], 'numpy')(*[x[i] for i in range(len(variables))] + [params[i] for i in range(len(param_syms))])})
