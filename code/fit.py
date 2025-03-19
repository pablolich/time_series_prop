"""
This script contains code defining the class fit, and its methods. An example
of how to instantiate an object fit is called at the end
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
#auxiliary files
from integration_funcs import *
from optimization_funcs import *

class Fit:
    def __init__(self, data, model, cost):
        
        #inherit properties from data, model, cost
        self.data = data
        self.model = model
        self.cost = cost

        #initialize time series predictions
        self.predicted_abundances = np.zeros_like(self.data.abundances)     
        self.predicted_proportions = np.zeros_like(self.data.abundances)  
        
        #initialize cost function value, vector of parameters and random seed
        self.cost_value = 0.0  
        #preallocate parameter vector
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
        np.random.seed(self.random_seed)
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

    def get_predictions(self):
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
                    method='RK45'
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
        cost_value = self.cost.compute_cost(self.data.abundances, 
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

    def plot_results(self):
        """
        Plot fit
        """
        nts = self.data.n_time_series
        dt_prop = []
        dt_abund = []
        
        for i in range(nts):
            # Proportions
            tmp = pd.DataFrame(self.data.proportions[i])
            tmp['time'] = self.data.times[i]
            tmp['type'] = 'proportion'
            tmp['state'] = 'observed'
            tmp['time_series'] = i + 1
            tmp['community'] = '-'.join(
                np.array(self.data.pop_names)[self.data.proportions[i][0, :] > 0]
            )
            dt_prop.append(tmp)
            
            tmp = pd.DataFrame(self.predicted_proportions[i])
            tmp['time'] = self.data.times[i]
            tmp['type'] = 'proportion'
            tmp['state'] = 'predicted'
            tmp['time_series'] = i + 1
            tmp['community'] = '-'.join(
                np.array(self.data.pop_names)[self.data.proportions[i][0, :] > 0]
            )
            dt_prop.append(tmp)
            
            # Abundances
            tmp = pd.DataFrame(self.data.abundances[i])
            tmp['time'] = self.data.times[i]
            tmp['type'] = 'abundance'
            tmp['state'] = 'unobserved'
            tmp['time_series'] = i + 1
            tmp['community'] = '-'.join(
                np.array(self.data.pop_names)[self.data.abundances[i][0, :] > 0]
            )
            dt_abund.append(tmp)
            
            tmp = pd.DataFrame(self.predicted_abundances[i])
            tmp['time'] = self.data.times[i]
            tmp['type'] = 'abundance'
            tmp['state'] = 'predicted'
            tmp['time_series'] = i + 1
            tmp['community'] = '-'.join(
                np.array(self.data.pop_names)[self.data.abundances[i][0, :] > 0]
            )
            dt_abund.append(tmp)
        
        dt_prop = pd.concat(dt_prop, ignore_index=True).melt(
            id_vars=['time', 'type', 'state', 'time_series', 'community'], 
            var_name='species', 
            value_name='x'
        )
        
        dt_abund = pd.concat(dt_abund, ignore_index=True).melt(
            id_vars=['time', 'type', 'state', 'time_series', 'community'], 
            var_name='species', 
            value_name='x'
        )
        
        dt_abund['x'] = dt_abund.groupby(['state', 'time_series'])['x'].transform(lambda x: x / x.mean())
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
        
        # Proportions plot
        sns.lineplot(data=dt_prop, x='time', y='x', hue='species', style='state', ax=axes[0])
        axes[0].set_title('Proportion Trends')
        
        # Abundances plot
        sns.lineplot(data=dt_abund, x='time', y='x', hue='species', style='state', ax=axes[1])
        axes[1].set_title('Abundance Trends')
        
        plt.tight_layout()
        plt.show()


    def save_results(self):
            
        fname = os.path.splitext(os.path.basename(self.data.file_names[0]))[0]
        fname = f"{fname}_{self.cost.cost_name}_{round(self.cost_value, 3)}.pkl"
        
        save_path = os.path.join("../results", fname)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        
        print(f"Results saved to {save_path}")

