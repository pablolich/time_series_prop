from fit import Fit
from glv import GLVModel
from ssq_prop import SSQCostFunction
from utils import continuous_step

# Load data and initialize Fit object
fit = Fit(["data/glv_chaos_4spp.csv"], "compiled_data/glv_chaos_4spp")

# Initialize model and cost function
model = GLVModel(fit)
cost_function = SSQCostFunction(fit)

# Generate model predictions
model.initialize_model_parameters()
import ipdb; ipdb.set_trace(context = 20)
model.integrate()

# Compute cost function
cost_value = cost_function.compute_cost(fit.predicted_abundances)
print(f"Cost Value: {cost_value}")

