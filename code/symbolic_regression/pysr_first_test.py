import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import modules from code/
import data
import numpy as np
import matplotlib.pylab as plt
from pysr import PySRRegressor
import numpy as np

def find_k_nonzero_column_indices(array_list, k):
    indices = []
    
    for idx, arr in enumerate(array_list):
        nonzero_columns = np.any(arr != 0, axis=0)  # Boolean mask: True for columns with nonzero values
        if np.sum(nonzero_columns) == k:  # Check if exactly k columns have nonzero values
            indices.append(idx)
    
    return indices

path_name = "../../data/fodelianakis_2018"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

#load all datasets
data_fd = data.Data(file_list)
#inds_mono = find_k_nonzero_column_indices(data_jo.abundances, 2)
#get only the first time series for now
X = data_fd.abundances[0]
t = data_fd.times[0]
#t_target = t[1::8]
#X_target = X[1::8]
#get dxdt
#y = np.diff(X_target, axis = 0)/(np.diff(t_target))
y = np.diff(X, axis = 0)

model = PySRRegressor(
    maxsize=20,
    niterations=80,  # < Increase me for better results
    constraints = {"^":(-1, 1)},
    binary_operators=["+", "-", "*", "^"]
)

#X = X_target[1:].reshape(-1, 1)
X = X[0:-1,:]
model.fit(X, y)

print(model)

import ipdb; ipdb.set_trace(context = 20)

#plt.scatter(X, y, color='blue', label='Actual Data')

## Overlay the predicted lines for each equation in the model
#for i in range(len(model.equations_)):
#    ypred = model.predict(X, i)
#    plt.plot(X, ypred, label=f'Equation {i+1}', linestyle='--')
#
## Labels and legend
#plt.xlabel('X')
#plt.ylabel('y')
#plt.legend()
#plt.title('Scatter plot with predicted lines')
#plt.show()
