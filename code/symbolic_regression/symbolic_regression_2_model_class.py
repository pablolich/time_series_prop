import os
import sys
import numpy as np
import sympy as sp
from pysr import PySRRegressor
#from your_module import export_models_to_single_file, extract_sympy_equations  # import the necessary functions
# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import modules from code/
import data

def extract_sympy_equations(eqs_all):
    return [list(eqs_spp["sympy_format"]) for eqs_spp in eqs_all]

def make_rhs_vectorized(exprs, variables):
    constants = set()
    for expr in exprs:
        constants |= {term for term in expr.atoms(sp.Number) if term != 0 and term != 1}
    param_symbols = [sp.Symbol(f'c{i}') for i, _ in enumerate(constants)]
    const_to_param = dict(zip(constants, param_symbols))
    modified_exprs = [expr.subs(const_to_param) for expr in exprs]
    func = sp.lambdify(variables + param_symbols, modified_exprs, 'numpy')
    def dxdt(t, x, *params):
        return np.array(func(*x, *params))
    return dxdt, param_symbols

def generate_model_class_code(class_index, exprs, variables, data_source):
    dxdt_func, param_syms = make_rhs_vectorized(exprs, variables)
    class_name = f"Model_{class_index}"
    model_name = f"model_{class_index}_{data_source}"

    import ipdb; ipdb.set_trace(context = 20)
    param_syms_str = ", ".join(str(p) for p in param_syms)

    class_code = f'''
import numpy as np

THRESH = 1e-10  # default threshold to avoid zeros


# Define the function to compute dx/dt
def dxdt_func(t, x, *params):
    from numpy import array
    # Use the actual 'x' and 'params' arguments properly
    return array({sp.lambdify(variables + param_syms, [expr.subs({k: v for k, v in zip(param_syms, param_syms)}) for expr in exprs], 'numpy')(*[variables[i] for i in range(len(variables))] + [param_syms[i] for i in range(len(param_syms))])})

class {class_name}:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: {class_index}
        Data source: {data_source}

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = {len(param_syms)}
        self.model_name = "{model_name}"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        \"\"\"
        Compute dx/dt for model {class_index}.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [{param_syms_str}]
        :return: dx/dt as NumPy array
        \"\"\"
        x = np.maximum(x, THRESH)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return {param_syms}

    def parse_model_parameters(self, pars):
        return pars
    '''
    return class_code.strip()

def export_models_to_single_file(eqs_all, data_source="default", output_file="generated_models.py"):
    # Create or overwrite the output file
    with open(output_file, "w") as f:
        f.write('''import numpy as np
THRESH = 1e-10  # default threshold to avoid zeros\n\n''')
        
        eqs_grouped = extract_sympy_equations(eqs_all)
        variables = [sp.Symbol(f"x{i}") for i in range(len(eqs_grouped))]

        for i, exprs in enumerate(zip(*eqs_grouped)):  # each exprs is one full model (1 per species)
            exprs = list(exprs)
            code = generate_model_class_code(i, exprs, variables, data_source)
            f.write(code + "\n\n")  # Add a newline after each class

    print(f"✅ All models written to: {output_file}")


# Path to the dataset folder
path_name = "../../data/synthetic_simplest"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

# Load all datasets (you should already have data_fd from your code)
data_fd = data.Data(file_list)

# Stack all replicates vertically for fitting
X = np.vstack(data_fd.abundances)

# Compute the "observed" derivatives (dx/dt) from the data
y = np.diff(X, axis=0)

# Define and configure the PySRRegressor model
model = PySRRegressor(
    maxsize=20,
    niterations=80,  # < Increase me for better results
    binary_operators=["+", "-", "*"]
)

# Fit the model with the data (excluding the last point for x and y)
model.fit(X[0:-1,:], y)

# Print the model results
print(model)

# Extract the equations for all species from the fitted model
eqs_all = model.equations_
# Example symbolic equations for two species, just as placeholders for testing
#x1, x2 = sp.symbols('x1 x2')
#
## Sample equations, replace this with actual equations from PySRRegressor
#eq_1 = x1 * (1 + 0.5 * x2)  # dx1/dt equation
#eq_2 = x2 * (0.8 - 0.3 * x1)  # dx2/dt equation
#
## Equations for all species (assuming two species, adjust as per your actual model)
#eqs_all = [
#    {"sympy_format": [eq_1]},  # Equation for species 1
#    {"sympy_format": [eq_2]},  # Equation for species 2
#]

# Extract and group the equations by species
eqs_grouped = extract_sympy_equations(eqs_all)

# Now generate and export all models to a single Python file
export_models_to_single_file(eqs_all, data_source="synthetic_simplest", output_file="generated_models.py")

print("✅ Models written to 'generated_models.py'")

