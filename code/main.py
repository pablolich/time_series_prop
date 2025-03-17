from models.glv import GLVModel
from cost_functions.ssq_prop import SSQCostFunction
from data.data import Data
from opt_protocols.reveal_optimize_refine import *
from fit import Fit
from optimization_funcs import *
from integration_funcs import *
from plotting_funcs import *
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
#search for good initial parameters 
fit = initialize_random(fit, n_rounds = 100) 
fit = get_predictions(fit)
plot_res_combined(fit)
#run optimization protocol 
weights = np.linspace(10, 1, num=3).tolist() + [0]
fit = reveal_optimize_refine(fit, weights)
plot_res_combined(fit)
fit.save_results()
