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

def make_rhs(expr, variables):
    """
    Converts a SymPy expression into a numerical function where constants are parameters.

    Parameters:
    - expr: SymPy expression representing dx/dt.
    - variables: List of SymPy symbols representing state variables.

    Returns:
    - A tuple (dxdt_func, params), where:
        - dxdt_func: A callable function taking (t, x, *params).
        - params: List of symbolic parameters replacing constants.
    """
    # Identify constants (numbers in the expression)
    constants = {term for term in expr.atoms(sp.Number) if term != 0 and term != 1}
    
    # Create new symbolic parameters for each constant
    param_symbols = [sp.Symbol(f'c{i}') for i, _ in enumerate(constants)]
    
    # Replace constants with symbolic parameters
    const_to_param = dict(zip(constants, param_symbols))
    modified_expr = expr.subs(const_to_param)

    # Convert modified expression into a numerical function
    func = sp.lambdify(variables + param_symbols, modified_expr, 'numpy')

    # Define the wrapper function
    def dxdt(t, x, *params):
        return func(*x, *params)

    return dxdt, param_symbols

#path_name = "../../data/synthetic_simplest"
#file_list = os.listdir(path_name)
#file_list = [os.path.join(path_name, file_name) for file_name in file_list]
#
##load all datasets
#data_fd = data.Data(file_list)
##inds_mono = find_k_nonzero_column_indices(data_jo.abundances, 2)
##stack all replicates vertically for fit
#X = np.vstack(data_fd.abundances)
##obtain "observed" derivatives
#y = np.diff(X, axis = 0)
#
#model = PySRRegressor(
#    maxsize=20,
#    niterations=80,  # < Increase me for better results
#    binary_operators=["+", "-", "*"]
#)
#
#model.fit(X[0:-1,:], y) #we select all the points but the
#print(model)
##get list of all equations for all species
#eqs_all = model.equations_ 
#eqs_spp_1 = eqs_all[0] #all equations for species 1
#eqs_spp_2 = eqs_all[1] #all equations for species 2
#
##pair them to form system
##equation 1, species 1
#eq_2_spp_1 = eqs_spp_1["sympy_format"][2]
#eq_1_spp_2 = eqs_spp_2["sympy_format"][1]
#
#func = make_rhs(eq_2_spp_1, list(eq_2_spp_1.free_symbols))

# Example: Define an ODE dx/dt = 3*x*y + 5
x, y = sp.symbols('x y')
expr = 3 * x * y + 5  # Includes numeric constants

# Create the function and extract parameters
dxdt, params = make_rhs(expr, [x, y])

# Integration settings
t_span = (0, 5)  # Time range
x0 = np.array([2, 1])  # Initial conditions for x and y
param_values = (3, 5)   # Replacing c0 = 3, c1 = 5

# Perform numerical integration
sol = solve_ivp(dxdt, t_span, x0, args=param_values, method='RK45', dense_output=True)

# Plot the result
t_vals = np.linspace(*t_span, 100)
x_vals = sol.sol(t_vals)

plt.plot(t_vals, x_vals.T)
plt.xlabel('Time t')
plt.ylabel('State x')
plt.legend(['x'])
plt.title('Numerical Integration of dx/dt = 3*x*y + 5')
plt.grid()
plt.show()
#extract squeleton using sympy commands

#write function with the two squeletons

#write wrapper for integration with numerical integrating pipeline
