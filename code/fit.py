"""
This script contains code defining the class fit, and its methods.
"""

#general imports
import numpy as np
import pandas as pd

#for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#for saving result
import os
import pickle

#auxiliary functions
from aux_integration import *
from aux_optimization import *

class Fit:
    def __init__(self, data, model, cost):
        
        #inherit properties from data, model, cost
        self.data = data
        self.model = model
        self.cost = cost

        #initialize time series predictions
        self.predicted_abundances = []
        self.predicted_proportions = []
        
        #initialize cost function value, vector of parameters and random seed
        self.cost_value = 0.0  
        self.n_pars = self.data.n_initial + \
                self.model.n_model + \
                self.cost.n_cost
        self.pars = np.zeros(self.n_pars)
        self.set_true_zeros = np.array([x[0, :] > 0 for x in self.data.abundances]).astype(int).flatten()
        self.random_seed = 0

        #indices of each parameter group
        self.par_ix_data = np.arange(self.data.n_initial)
        self.par_ix_model = np.arange(self.model.n_model) + self.data.n_initial
        self.par_ix_cost = np.arange(self.cost.n_cost) + self.data.n_initial + self.model.n_model

    def initialize_parameters(self):
        """
        Initialize parameters for model, cost function, and initial conditions
        """
        #set seed
        #np.random.seed(self.random_seed)
        #initial conditions

        tmp = np.array([])
        for k in range(self.data.n_time_series):
           tmp = np.append(tmp, self.data.proportions[k][0])  

        self.pars[:self.data.n_initial] = tmp

        #model parameters
        self.pars[self.data.n_initial:self.data.n_initial + self.model.n_model] = np.random.randn(self.model.n_model)

        #goal function parameters
        if self.cost.n_cost > 0:
            self.pars[self.data.n_initial + self.model.n_model:self.data.n_initial + self.model.n_model + self.cost.n_cost] = np.repeat(1, self.cost.n_cost) #np.random.randn(self.cost.n_cost)

    def get_predictions(self, my_method = "RK23"):
        """
        Updates predicted_abundances and predicted_proportions given parameters.
        """
        #get parameters necessary for numerical integration
        params = parse_parameters_dynamics(self)
        predicted_abundances = []
        predicted_proportions = []

        #get predictions for each time series separately
        for i in range(self.data.n_time_series):
            times = self.data.times[i]
            init_conds = params["init_conds"][i]
            #look at whether the differential equation has analytic solution
            if self.model.dynamics_type == "dxdt":
                #integrate dynamics
                sol = solve_ivp(
                    fun=lambda t, y: self.model.dynamics(t, y, params),
                    t_span=(times[0], times[-1]),
                    y0=init_conds,
                    t_eval=times,
                    method=my_method
                )
                #process result
                abundances = process_integration_result(sol, times)
            elif self.model.dynamics_type == "x_t":
                #evaluate abundances at sampled times
                abundances = self.model.dynamics(times, init_conds, params)
            else:
                print("Type of dynamics not supported")

            # Handle zero observations
            abundances[self.data.proportions[i] == 0] = 0 
            #compute corresponding proportions and store
            proportions = abundances / np.sum(abundances, axis=1, keepdims=True)
            predicted_abundances.append(abundances)
            predicted_proportions.append(proportions)

        self.predicted_abundances = predicted_abundances
        self.predicted_proportions = predicted_proportions

    def to_minimize(self, values, positions, weight):
        """
        Updates fit parameters at specified positions and computes the goal 
        function.
        
        Parameters:
        values (array): New values for parameters.
        positions (array): Indices of parameters to update.
        weight (optional): Weighting factor for goal function.
        
        Returns:
        float: Computed goal function value.
        """
        self.pars[positions] = values
        #evaluate dynamics forward
        self.get_predictions()
        #extract parameters used to evaluate cost function
        pars_cost = self.pars[-self.cost.n_cost:]
        #score current parameters
        cost_value = self.cost.compute_cost(self.data.proportions, 
                self.predicted_abundances, 
                self.data.times,
                pars_cost,
                self.data.obs_type, 
                weight
                )
        return cost_value

    def optimize(self, positions, method='Nelder-Mead', weight=None):
        """
        Optimizes fit parameters indexed by position using a specified 
        optimization method.
        
        Parameters:
        positions (array): Indices of parameters to optimize.
        method (str): Optimization method to use (default: 'Nelder-Mead').
        weight (optional): Weighting factor for goal function.
        
        Returns:
        Fit: Fit object with updated parameters only if cost is reduced.
        """
        print("optimizing using", method)
        print(f"Initial cost: {self.cost_value}")
        initial_values = self.pars[positions]
        initial_goal = self.to_minimize(self.pars, range(self.n_pars), weight)
        
        res = minimize(
            fun=self.to_minimize,
            x0=initial_values,
            args=(positions, weight),
            method=method,
            options={'maxiter': 250, 'disp': False}
        )
        
        new_goal = self.to_minimize(res.x, positions, weight)
        
        if new_goal < initial_goal:
            self.pars[positions] = res.x
            self.get_predictions()
            self.cost_value = new_goal

    def plot(self):
        """
        Plot observed vs. predicted proportions and abundances over time.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
        markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', 'X', '+']  # Different markers for different time series
        colors = plt.cm.get_cmap("tab10", self.data.proportions[0].shape[1])  # Assign consistent colors per species
        
        # Proportions plot
        for i in range(self.data.n_time_series):
            times = self.data.times[i]
            for j in range(self.data.proportions[i].shape[1]):
                marker = markers[i % len(markers)]
                color = colors(j)  # Assign the same color per species
                axes[0].scatter(times, self.data.proportions[i][:, j], alpha=0.5, marker=marker, color=color)
                axes[0].plot(times, self.predicted_proportions[i][:, j], color=color)
        axes[0].set_title("Proportion Trends")
        axes[0].legend(["Observed", "Predicted"], loc='upper right')
        #axes[0].set_yscale("log")
        
        # Abundances plot
        for i in range(self.data.n_time_series):
            times = self.data.times[i]
            for j in range(self.data.abundances[i].shape[1]):
                marker = markers[i % len(markers)]
                color = colors(j)  # Assign the same color per species
                axes[1].scatter(times, self.data.abundances[i][:, j], alpha=0.5, marker=marker, color=color)
                axes[1].plot(times, self.predicted_abundances[i][:, j], color=color)
        axes[1].set_title("Abundance Trends")
        #axes[1].set_yscale("log")
        
        plt.tight_layout()
        plt.show()

    def save_results(self):
            
        fname = os.path.splitext(os.path.basename(self.data.file_names[0]))[0]
        fname = f"{fname}_{self.cost.cost_name}_{round(self.cost_value, 3)}.pkl"
        
        save_path = os.path.join("../results", fname)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        
        print(f"Results saved to {save_path}")

