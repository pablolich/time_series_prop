"""
This script contains code defining the class fit, and its methods. An example
of how to instantiate an object fit is called at the end
"""
import numpy as np
import pandas as pd
import os
import pickle

class Fit:
    def __init__(self, file_names, output_file):
        self.file_names = file_names
        self.output_file = output_file
        self.output_name = os.path.splitext(os.path.basename(output_file))[0]
        
        # Observed data
        self.observed_abundances = []
        self.times = []
        
        # Read files and process observed data
        for fn in file_names:
            tmp = pd.read_csv(fn).values
            self.times.append(tmp[:, 0].astype(float))
            self.observed_abundances.append(tmp[:, 1:].astype(float))
        
        # Normalize time
        maxtime = max(map(np.max, self.times))
        mintime = min(map(np.min, self.times))
        self.times = [(t - mintime) / (maxtime - mintime) for t in self.times]
        
        # Normalize abundances
        Tot = np.sum(self.observed_abundances[0][0, :])
        self.observed_abundances = [x / Tot for x in self.observed_abundances]
        self.observed_proportions = [x / np.sum(x, axis=1, keepdims=True) for x in self.observed_abundances]
        
        # Initialize other attributes
        self.n = self.observed_abundances[0].shape[1]  # Number of species
        self.n_time_series = len(self.observed_abundances)  # Number of time series
        self.n_initial = self.n * self.n_time_series  # Initial conditions
        self.type_of_inference = ""  # String to be set later (e.g., "proportions", "abundances")
        self.predicted_abundances = np.zeros_like(self.observed_abundances)  # Same shape as observed_abundances
        self.predicted_proportions = np.zeros_like(self.observed_abundances)  # Same shape as observed_abundances
        self.n_model = 0  # Integer: number of model parameters (to be set later)
        self.n_cost_function = 0  # Integer: number of cost functions (to be set later)
        self.model_name = ""  # String: name of the model (e.g., "GLV")
        self.cost_function_name = ""  # String: name of the cost function (e.g., "SSQ_prop")
        self.cost = 0.0  # Real number: cost value (float)
        self.pars = np.array([])  # Vector of real numbers (initialized empty)
        self.set_true_zeros = np.array([x[0, :] > 0 for x in self.observed_abundances]).astype(int).flatten()
        self.random_seed = 0
    
    def save(self):
        """
        Save the current instance of Fit to a file.
        """
        with open(self.output_file, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        """
        Load a saved Fit instance from a file.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)


# Instantiate the Fit object

fit = Fit(["data/glv_chaos_4spp.csv"], "compiled_data/glv_chaos_4spp")

#save it as pickle
fit.save()
