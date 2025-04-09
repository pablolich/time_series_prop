import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
import sys
import os

# Get the absolute path of the parent directory (code/)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import data

path_name = "../../data/synthetic_logistic"
file_list = os.listdir(path_name)
file_list = [os.path.join(path_name, file_name) for file_name in file_list]

data_glv4 = data.Data(file_list)
n_time_series = len(data_glv4.abundances)

# Neural ODE function
class PopulationODE(nn.Module):
    def __init__(self):
        super(PopulationODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, t, y):
        return self.net(y)

def time_weighting(epoch, total_epochs, seq_length):
    """
    Generate weights for each time step that gradually shift the importance
    from earlier to later time points over the epochs.
    Uses a sigmoid schedule to smoothly transition the weights.
    """
    # Sigmoid weighting function
    fraction_revealed = 1 / (1 + np.exp(-10 * (epoch / total_epochs - 0.5)))  # Sigmoid schedule
    weights = np.linspace(0, 1, seq_length)  # Linear interpolation for simplicity
    weights = torch.tensor(weights, dtype=torch.float32)

    # Gradually shift weights to later points over time (higher weights for later time points)
    weights = weights * fraction_revealed + (1 - fraction_revealed) * torch.flip(weights, [0])
    return weights

# Initialize model
neural_ode_func = PopulationODE()

# Define optimizer and loss
optimizer = torch.optim.Adam(neural_ode_func.parameters(), lr=0.005)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)
# Define learning rate range
eta_initial = 0.01
eta_final = 0.0001
total_epochs = 100000

# Compute adaptive gamma
gamma = np.exp(np.log(eta_final / eta_initial) / total_epochs)

# Define scheduler with adaptive gamma
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

print(f"Adapted gamma: {gamma:.6f}")  # Check the computed gamma value

loss_fn = nn.MSELoss()

# Training loop with progressive data revealing
for epoch in range(total_epochs):
    optimizer.zero_grad()
    total_loss = 0.0

    for i in range(n_time_series):
        X_train = torch.tensor(data_glv4.proportions[i], dtype=torch.float32)
        T_train = torch.tensor(data_glv4.times[i], dtype=torch.float32)
        seq_length = X_train.shape[0]
        
        # Get the weights based on the current epoch
        time_weights = time_weighting(epoch, total_epochs, seq_length)

        
        # Apply the mask (only compare predicted values for revealed parts)
        init_conds = torch.log(X_train[0])
        int_sol = odeint(neural_ode_func, 
                torch.nan_to_num(init_conds, nan=0.0, neginf=0.0),
                T_train)
        #zero out species that are not present in the solution
        ind_absent = torch.where(int_sol[0,:] == 0)
        int_sol = torch.exp(int_sol)
        #int_sol[:,ind_absent[0]] = 0
        pred_props = int_sol / int_sol.sum(dim=1, keepdim=True)

        
        # Compute weighted loss
        # Compute the difference (predicted - true values)
        diff = pred_props - X_train  # Shape: (seq_length, 4)
        
        # Multiply each column (species) by the time_weights
        weighted_diff = diff * time_weights[:, None]  # Broadcasting time_weights across all species
        
        # Compute weighted MSE loss
        weighted_loss = (weighted_diff ** 2).sum() / seq_length  # Sum over all time steps and species

        total_loss += weighted_loss

    avg_loss = total_loss / n_time_series
    avg_loss.backward()
    optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.16f}')


print("Training complete!")
# Step 2: Prediction with Multiple Initializations
# Now, to predict trajectories for all initializations, you can run the ODE solver for each initialization
with torch.no_grad():
    learned_abundances = []
    learned_proportions = []
    for i in range(n_time_series):
        X_train = torch.tensor(data_glv4.abundances[i], dtype=torch.float32) #cheating
        T_train = torch.tensor(data_glv4.times[i], dtype=torch.float32)
        # Apply the mask (only compare predicted values for revealed parts)
        init_conds = torch.log(X_train[0])
        learned_abundance = odeint(neural_ode_func, 
                torch.nan_to_num(init_conds, nan=0.0, neginf=0.0),
                T_train)
        #zero out species that are not present in the solution
        ind_absent = torch.where(learned_abundance[0,:] == 0)
        learned_abundance = torch.exp(learned_abundance)
        learned_abundance[:,ind_absent[0]] = 0

        learned_proportion = learned_abundance/learned_abundance.sum(dim=1, keepdim=True)
        learned_abundances.append(learned_abundance)
        learned_proportions.append(learned_proportion) 

plt.figure(figsize=(18, 5))
labels = ["Species 1", "Species 2"]#, "Species 3"]

for i in range(n_time_series):#loop through initial conditions
    for j in range(2): #loop through species
        plt.subplot(2, n_time_series, i+1) #one system in each plot
        # Plot solid line first and get color
        line, = plt.plot(T_train, data_glv4.proportions[i][:, j], label=f"Init {i+1}")
        plt.plot(T_train, learned_proportions[i][:, j], linestyle="dashed", color=line.get_color())  # Use same color
        
        plt.subplot(2, n_time_series, i+1+n_time_series)
        line, = plt.plot(T_train, data_glv4.abundances[i][:, j], label=f"Init {i+1}")
        plt.plot(T_train, learned_abundances[i][:, j], linestyle="dashed", color=line.get_color())  # Use same color

    plt.ylabel(labels[j])
    plt.xlabel("Time")

plt.legend()
plt.tight_layout()
plt.show()
