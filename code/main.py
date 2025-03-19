from models.glv import GLVModel
from cost_functions.ssq_prop import SSQCostFunction
from cost_functions.dirichlet import DirichletFunction
from cost_functions.log_prop import LogDistCostFunction 
from models.exponential import ExponentialModel
from data.data import Data
from opt_protocols.nelder_bfgs import *
from fit import Fit
from optimization_funcs import *
from integration_funcs import *
import os

# Load data and initialize Fit object

#get file names from Davis data
path_name = "data/exponential_errors_gamma/"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]
# Initialize data, model and cost function
data = Data(file_list)
model = ExponentialModel(data.n)
#cost_function = SSQCostFunction()
cost_function = DirichletFunction(data.n)
#cost_function = LogDistCostFunction()
# Initialize fit
fit = Fit(data, model, cost_function)
#search for good initial parameters 
#fit = initialize_random(fit, n_rounds = 100) 
fit.pars = np.array([10, 5, 1, 1, 2, 3, 1, 1, 1])
fit.get_predictions()
#run optimization protocol 
weights = np.linspace(10, 1, num=5).tolist() + [0]
fit = nelder_bfgs(fit, weights)
import ipdb; ipdb.set_trace(context = 20)
fit.plot_results()
fit.save_results()
