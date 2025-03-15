from glv import GLVModel
from ssq_prop import SSQCostFunction
from data import Data
from fit import Fit
from utils import continuous_step, integrate
import os

# Load data and initialize Fit object

#get file names from Davis data
path_name = "data/Davis/"
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
integrate(fit)
cost = cost_function.compute_cost(fit.data.observed_proportions,
                           fit.predicted_proportions, 
                           fit.data.times)
import ipdb; ipdb.set_trace(context = 20)
#optimize parameters
