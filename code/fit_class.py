import numpy as np
import pandas as pd
import os
import pickle

class Fit:
    def __init__(self, file_names, output_file):
        self.file_names = file_names
        self.output_file = output_file
        self.output_name = os.path.splitext(os.path.basename(output_file))[0]
        self.observed_abundances = []
        self.times = []
        
        # Read files
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
        
        # Other attributes
        self.n = self.observed_abundances[0].shape[1]
        self.n_time_series = len(self.observed_abundances)
        self.n_initial = self.n * self.n_time_series
        self.type_of_inference = None
        self.predicted_abundances = None
        self.predicted_proportions = None
        self.n_model = None
        self.n_cost_function = None
        self.model_name = None
        self.cost_function_name = None
        self.cost = None
        self.pars = None
        self.set_true_zeros = np.array([x[0, :] > 0 for x in self.observed_abundances]).astype(int).flatten()
        self.random_seed = 0
    
    def save(self):
        with open(self.output_file, 'wb') as f:
            pickle.dump(self, f)
    
    # Static method: does not require an instance of the class
    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

# Instantiate the Fit object
fit = Fit([
    "data/Davis/VES_1.csv", 
    "data/Davis/ES_1.csv", 
    "data/Davis/VE_1.csv", 
    "data/Davis/VS_1.csv"
], "compiled_data/Davis_1.pkl")

import ipdb; ipdb.set_trace(context = 20)

