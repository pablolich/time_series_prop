import numpy as np
import pandas as pd

class Data:
    def __init__(self, file_names):
        """
        Load data from file and store observed abundances and times.
        :param file_name: Path to CSV file
        """

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
        
        self.n = self.observed_abundances[0].shape[1]  # Number of species
        self.n_time_series = len(self.observed_abundances)  # Number of time series
        #self.type_of_inference = ""  # String to be set later (e.g., "proportions", "abundances")
