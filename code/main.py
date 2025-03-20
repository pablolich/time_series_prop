#general imports
import os

#import necessary classes
import data, cost_functions, models
import fit

#import auxiliary functions
from aux_integration import *
from aux_optimization import *
from opt_protocols import *

#get file names
path_name = "data/exponential_errors_gamma/"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

# Initialize data, model, cost function and fit object
data = data.Data(file_list, normalize = False)
model = models.Exponential(data.n)
cost_function = cost_functions.Dirichlet(data.n)
fit = fit.Fit(data, model, cost_function)

#search for good initial parameters and update based on new parameters
fit = initialize_random(fit, n_rounds = 100) 
fit.get_predictions()
fit.cost_value = fit.to_minimize(fit.pars, range(fit.n_pars), weight = 0)
fit.plot()

#run optimization protocol 
weights = np.linspace(10, 1, num=5).tolist() + [0]
fit = nelder_bfgs(fit, n_rounds = 1)

#plot and save
print("Final parameters: ", fit.pars)
fit.plot()
fit.save_results()
