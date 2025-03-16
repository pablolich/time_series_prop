from glv import GLVModel
from ssq_prop import SSQCostFunction
from data import Data
from fit import Fit
from optimization_funcs import *
from integration_funcs import *
import os

# Load data and initialize Fit object

#get file names from Davis data
path_name = "data/glv_3spp/"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]
# Initialize data, model and cost function
data = Data(file_list)
model = GLVModel(data.n)
cost_function = SSQCostFunction()
# Initialize fit
fit = Fit(data, model, cost_function)
#initialize parameters (initial conditions, model, and cost funct parameters)
fit.initialize_parameters()
#compute cost for initial parameters
fit = get_predictions(fit)
fit.cost_value = fit.cost.compute_cost(fit.data.observed_abundances, 
        fit.predicted_abundances, 
        fit.data.times)
fit = optimize_k(range(len(fit.pars)), fit)
print(fit.cost_value)
#optimize parameters
