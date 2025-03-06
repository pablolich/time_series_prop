from glv import GLVModel
from ssq_prop import SSQCostFunction
from data import Data
from fit import Fit
from utils import continuous_step
import os

# Load data and initialize Fit object

#get file names from Davis data
path_name = "data/Davis/"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]
# Initialize data, model and cost function
data = Data(file_list)
model = GLVModel(data)
cost_function = SSQCostFunction()
# Initialize fit
import ipdb; ipdb.set_trace(context = 20)
fit = Fit(data, model, cost_function)

#fit = Fit(["data/glv_chaos_4spp.csv"], "compiled_data/glv_chaos_4spp")
# Generate model predictions
#model.initialize_model_parameters()
#import ipdb; ipdb.set_trace(context = 20)
#model.integrate()
#
## Compute cost function
#cost_value = cost_function.compute_cost(fit.predicted_abundances)
#print(f"Cost Value: {cost_value}")

