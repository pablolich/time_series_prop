"""
This script runs the main code fitting data to model and cost, returning a file
that is saved and can be reloaded to be plotted
"""

# General imports
import os
import importlib

# Import necessary classes
import data, cost_functions, fit, generated_models

# Import auxiliary functions
from aux_integration import *
from aux_optimization import *
from opt_protocols import *

# Get file names
path_name = "../data/synthetic_logistic"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

# Initialize data and cost function
data = data.Data(file_list, normalize=True)
cost_function = cost_functions.LogDist(data.n)

# Dynamically load the generated models from the 'generated_models.py' file

models = [getattr(generated_models, method) for method in dir(generated_models) if method.startswith("Model")]

# Iterate through all models and fit them
for i, model_class in enumerate(models):
    print(f"Fitting model {i + 1}: {model_class.__name__}")

    # Initialize the model
    model = model_class(data.n)  # Adjust based on the model's initialization parameters

    # Initialize fit object
    fit_object = fit.Fit(data, model, cost_function)

    # Search for good initial parameters and update based on new parameters
    fit_object = initialize_random(fit_object, n_rounds=100)

    # Get predictions and calculate cost value
    fit_object.get_predictions()
    fit_object.cost_value = fit_object.to_minimize(fit_object.pars, range(fit_object.n_pars), weight=0)
    
    # Run optimization protocol
    weights = np.linspace(10, 1, num=10).tolist() + [0]
    fit_object = reveal_optimize_refine(fit_object, n_rounds=10, weights=weights)

    # Plot and save the results for the current model
    print("Final parameters: ", fit_object.pars)
    #fit_object.plot()
    fit_object.save_results()
    
print("All models have been fitted and saved.")

