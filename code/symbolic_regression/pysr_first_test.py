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

path_name = "../../data/jo"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

#load all datasets
data_jo = data.Data(file_list)
import ipdb; ipdb.set_trace(context = 20)
inds_mono = find_k_nonzero_column_indices(data_jo.abundances, 2)
import ipdb; ipdb.set_trace(context = 20)
#get only the first time series for now
X = data_jo.abundances[0]
import ipdb; ipdb.set_trace(context = 20)
t = data_jo.times[0]
#get dxdt
import ipdb; ipdb.set_trace(context = 20)
y = np.diff(X, axis = 0)/np.diff(t)

model = PySRRegressor(
    maxsize=20,
    niterations=80,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "exp",
        "log10",
        "neg",
        "square",
        "cube",
        "sqrt",
    ],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

X = X[1:].reshape(-1, 1)
model.fit(X, y)

print(model)

plt.scatter(X, y, color='blue', label='Actual Data')

# Overlay the predicted lines for each equation in the model
for i in range(len(model.equations_)):
    ypred = model.predict(X, i)
    plt.plot(X, ypred, label=f'Equation {i+1}', linestyle='--')

# Labels and legend
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Scatter plot with predicted lines')
plt.show()
import ipdb; ipdb.set_trace(context = 20)

#predict to check visually
#next, write down the equation and compare to see if I am getting good results

