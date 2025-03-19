import numpy as np
import pandas as pd
import os

class Data:
    def __init__(self, file_names, observation_type = "prop"):
        """
        Load data from file and store observed abundances and times.
        :param file_name: Path to CSV file
        :param observation_type: Whether observed abundances are relative or
                                 absolute
        """

        self.file_names = file_names
        self.obs_type = observation_type
        
        # Observed data
        self.abundances = []
        self.times = []
        
        # Read files and process observed data
        for fn in file_names:
            data = pd.read_csv(fn)
            tmp = data.values
            self.times.append(tmp[:, 0].astype(float))
            self.abundances.append(tmp[:, 1:].astype(float))
        
        #get names of columns
        self.pop_names = data.columns[1:].tolist()
        # Normalize time
        maxtime = max(map(np.max, self.times))
        mintime = min(map(np.min, self.times))
        self.times = [(t - mintime) / (maxtime - mintime) for t in self.times]
        
        # Normalize abundances
        Tot = np.sum(self.abundances[0][0, :])
        self.abundances = [x / Tot for x in self.abundances]
        self.proportions = [x / np.sum(x, axis=1, keepdims=True) for x in self.abundances]
        
        self.n = self.abundances[0].shape[1]  # Number of species
        self.n_time_series = len(self.abundances)  # Number of time series
        self.n_initial = self.n * self.n_time_series #number of initial conditions
