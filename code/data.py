"""
Script containing the class Data, which reads csv files inside a data folder
and creates a data object. The files need to have all the species, even if
for that particular experiment they where zero.
"""

import numpy as np
import pandas as pd
import os

class Data:
    def __init__(self, file_names, observation_type = "prop", normalize = True):

        self.file_names = file_names
        self.obs_type = observation_type
        
        # Observed data
        self.abundances = []
        self.proportions = []
        self.times = []
        
        # Read files and store contents
        for fn in file_names:
            data = pd.read_csv(fn)
            tmp = data.values
            abundances = tmp[:,1:]
            total = np.sum(abundances, axis = 1)
            proportions = np.array([x / np.sum(x) for x in abundances])
            proportions[np.sum(abundances, axis = 1) == 0] = 0
            times = tmp[:,0]
            #store observation times, abundances and proportions
            self.times.append(times)
            self.abundances.append(abundances)
            self.proportions.append(proportions)
        
        #get names of columns
        self.pop_names = data.columns[1:].tolist()

        #if desired, normalize times and abundances
        if normalize:
            # Normalize time so it goes from 0 to 1
            maxtime = max(map(np.max, self.times))
            mintime = min(map(np.min, self.times))
            self.times = [(t - mintime) / (maxtime - mintime) for t in self.times]
            
            # Normalize abundances by initial total biomas
            Tot = np.sum(self.abundances[0][0, :])
            self.abundances = [x / Tot for x in self.abundances]
            self.proportions = [x / np.sum(x, axis=1, keepdims=True) for x in \
                    self.abundances]
        
        #set rest of parameters (number of species, time series, and initial conditions)
        self.n = self.abundances[0].shape[1]
        self.n_time_series = len(file_names)
        self.n_initial = self.n * self.n_time_series
