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
        
        #inherit properties from data, model, cost
        self.data = data
        self.model = model
        self.cost = cost

        self.predicted_abundances = np.zeros_like(self.data.observed_abundances)     
        self.predicted_proportions = np.zeros_like(self.data.observed_abundances)  
        
        #additional parameters
        self.n_initial = self.data.n * self.data.n_time_series
        self.cost_value = 0.0  # Real number: cost value (float)
        #preallocate parameter vector
        total_params = self.n_initial + self.model.n_model + self.cost.n_cost
        self.pars = np.zeros(total_params)
        self.set_true_zeros = np.array([x[0, :] > 0 for x in self.data.observed_abundances]).astype(int).flatten()
        self.random_seed = 0

    def initialize_parameters(self):
        """
        Initialize parameters for model, cost function, and initial conditions
        """
        #set seed
        np.random.seed(self.random_seed)
        #initial conditions

        tmp = np.array([])
        for k in range(self.data.n_time_series):
           tmp = np.append(tmp, self.data.observed_proportions[k][0])  

        self.pars[:self.n_initial] = tmp

        #model parameters
        self.pars[self.n_initial:self.n_initial + self.model.n_model] = np.random.randn(self.model.n_model)

        #goal function parameters
        if self.cost.n_cost > 0:
            self.pars[self.n_initial + self.model.n_model:self.n_initial + self.model.n_model + self.cost.n_cost] = np.random.randn(self.cost.n_cost)



