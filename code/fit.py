"""
This script contains code defining the class fit, and its methods. An example
of how to instantiate an object fit is called at the end
"""
import numpy as np
import pandas as pd
import os
import pickle

class Fit:
    def __init__(self, data, model, cost):
        #self.file_names = data.file_names
        
        # Initialize inherited attributes from data, model, cost
        self.n = data.n
        self.n_time_series = data.n_time_series
        self.n_initial = self.n * self.n_time_series
        self.observed_abundances = data.observed_abundances
        self.times = data.times
        self.predicted_abundances = np.zeros_like(self.observed_abundances)     
        self.predicted_proportions = np.zeros_like(self.observed_abundances)  

        self.n_model = model.n_model 
        self.model_name = model.model_name

        self.cost_function_name = cost.cost_name
        self.n_cost = cost.n_cost  

        self.cost = 0.0  # Real number: cost value (float)
        self.pars = np.array([])  # Vector of real numbers (initialized empty)
        self.set_true_zeros = np.array([x[0, :] > 0 for x in self.observed_abundances]).astype(int).flatten()
        self.random_seed = 0

    def parse_model_parameters(self):
        """
        Parse parameters from the fit object to extract initial conditions 
        and model coefficients.
        :return: Dictionary with structured parameters.
        """
        pars = self.fit.pars
        p = pars[self.fit.n_initial:self.fit.n_initial + self.n_model]
        init_conds = []
        x0 = np.abs(pars[:self.fit.n_initial])

        for i in range(self.fit.n_time_series):
            tmp = x0[i * self.n:(i + 1) * self.n]
            # Apply hard zeros
            zeros = self.fit.set_true_zeros[i * self.n:(i + 1) * self.n]
            tmp *= zeros
            init_conds.append(tmp)

        params = {
            "r": p[:self.n], 
            "A": p[self.n:].reshape(self.n, self.n), 
            "init_conds": init_conds
        }
        return params

