import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Now you can import modules from code/
import data
import numpy as np
from pysr import PySRRegressor

path_name = "../../data/jo"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

#load all datasets
data_jo = data.Data(file_list)
#get only the first time series for now
X = data_jo.abundances[0][:,0]
t = data_jo.times[0]
#get dxdt
import ipdb; ipdb.set_trace(context = 20)
y = np.diff(X)/np.diff(t)

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=[
        "exp",
        "log10",
        "neg",
        "square",
        "cube",
        "sqrt",
        "inv"
    ],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

X = X[1:].reshape(-1, 1)
model.fit(X, y)

#next, write down the equation and compare to see if I am getting good results

