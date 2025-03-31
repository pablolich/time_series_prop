import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import data


# Define the Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self, n):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n)
        )

    def forward(self, t, y):
        return self.net(y)

# Define the training function
def train_ode_model(time, true_data, model, learning_rate=0.05, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    n_time_series = len(true_data)
    # Convert data from each time series to torch tensors and store in list
    time_data_list = [torch.tensor(time[i], dtype=torch.float32) for  
            i in range(n_time_series)]
    true_data_list = [torch.tensor(true_data[i], dtype=torch.float32) for 
            i in range(n_time_series)]

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = 0
        for ts in range(1):
            #get corresponding time series
            true_data_i = true_data_list[ts]
            init_conds_i = true_data_i[0]
            times_i = time_data_list[ts]
            # Solve the ODE using the current model for each time series
            pred_data_i = odeint(model, init_conds_i, times_i)  # Using initial_conditions_1 as a start
            
            # Compute the loss (mean squared error) for both time series
            loss_i = criterion(pred_data_i, true_data_i)
            
            # Combine the losses
            loss += loss_i
            
        # Backpropagate the loss once calculated for all time series
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
    
    return model

path_name = "../../data/glv_4_chaos"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

data_glv4 = data.Data(file_list)

# Create the ODE model
ode_model = ODEFunc(data_glv4.n)

# Train the model using both time series
ode_model = train_ode_model(data_glv4.times, data_glv4.proportions, ode_model)

# After training, let's visualize the learned dynamics
with torch.no_grad():
    predicted_solutions = []
    
    for i in range(1):  # Iterate over all initial conditions
        initial_conditions = torch.tensor(data_glv4.abundances[i][0], dtype=torch.float32)
        predicted_solution = odeint(ode_model, initial_conditions, torch.tensor(data_glv4.times[i], dtype=torch.float32))
        predicted_solutions.append(predicted_solution.detach().numpy())

# Convert true data to numpy arrays for easier plotting
true_solutions = [np.array(data_glv4.abundances[i]) for i in range(len(data_glv4.proportions))]

# Get the number of species
n_species = data_glv4.n

# Create subplots: One subplot per species
fig, axes = plt.subplots(n_species, 1, figsize=(8, 4 * n_species), sharex=True)

# If only one species, axes won't be a list, so make it a list for consistency
if n_species == 1:
    axes = [axes]

# Loop over species
for species_idx in range(n_species):
    ax = axes[species_idx]
    
    # Plot all runs (true data)
    for i in range(1):  # Iterate over initial conditions
        ax.plot(data_glv4.times[i], true_solutions[i][:, species_idx], label=f'True Run {i+1}', linestyle='solid')
    
    # Plot all runs (predicted data)
    for i in range(1):  # Iterate over initial conditions
        ax.plot(data_glv4.times[i], predicted_solutions[i][:, species_idx], label=f'Pred Run {i+1}', linestyle='dashed')
    
    ax.set_ylabel(f'Species {species_idx + 1}')
    ax.legend()

axes[-1].set_xlabel('Time')  # Only label x-axis on the last subplot
plt.suptitle("True vs. Predicted Population Dynamics")
plt.tight_layout()
plt.show()

