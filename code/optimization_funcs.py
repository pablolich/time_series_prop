"""
This script contains functions to perform optimization on fit objects
"""
from integration_funcs import *

from scipy.optimize import minimize
import numpy as np
import itertools #for optimization of specific parameter combinations
import random

def to_minimize_k(values, positions, fit, weight):
    """
    Updates fit parameters at specified positions and computes the goal function.
    
    Parameters:
    values (array): New values for parameters.
    positions (array): Indices of parameters to update.
    fit (Fit): Fit object containing model data.
    weight (optional): Weighting factor for goal function.
    
    Returns:
    float: Computed goal function value.
    """
    fit.pars[positions] = values
    fit = get_predictions(fit)
    return fit.cost.compute_cost(fit.data.observed_abundances, fit.predicted_abundances, fit.data.times)

def optimize_k(positions, fit, method='Nelder-Mead', weight=None):
    """
    Optimizes fit parameters using a specified optimization method.
    
    Parameters:
    positions (array): Indices of parameters to optimize.
    fit (Fit): Fit object containing model data.
    method (str): Optimization method to use (default: 'Nelder-Mead').
    weight (optional): Weighting factor for goal function.
    
    Returns:
    Fit: Fit object with updated parameters only if cost is reduced.
    """
    initial_goal = fit.cost.compute_cost(fit.data.observed_abundances, fit.predicted_abundances, fit.data.times)
    print(initial_goal)
    initial_values = fit.pars[positions]
    
    res = minimize(
        fun=to_minimize_k,
        x0=initial_values,
        args=(positions, fit, weight),
        method=method,
        options={'maxiter': 250, 'disp': False}
    )
    
    new_goal = to_minimize_k(res.x, positions, fit, weight)
    
    if new_goal < initial_goal:
        fit.pars[positions] = res.x
        fit = get_predictions(fit)
        fit.goal = new_goal
    
    return fit

def all_k(positions, fit, k=2, weight=None):
    """
    Optimizes all combinations of k sets of parameters for a given 
    `fit` object.
    
    Parameters:
    positions (array-like): List or array of parameter positions to consider 
                            for optimization.
    k (int, optional): The number of positions to combine at each step for 
                       optimization (default is 2).
    fit (Fit): The `Fit` object containing the model data and current parameters.
    weight (float, optional): An optional weighting factor to apply to the 
                              goal function during optimization.

    Returns:
    Fit: The updated `Fit` object with optimized parameters after processing 
         all combinations.
    """
    # Generate all combinations of `k` positions from the `positions` list
    combinations = list(itertools.combinations(positions, k))
    ncombos = len(combinations)

    # Shuffle the combinations
    random.shuffle(combinations)

    # Iterate over the shuffled combinations and optimize
    for i, combo in enumerate(combinations):
        # Call optimize_k for the current combination of parameters
        fit = optimize_k(positions=list(combo), fit=fit, weight=weight)
        
        # Print iteration information: weight, iteration number, total combinations, and the current goal value
        print(f"{weight}, {i + 1}, {ncombos}, {fit.cost_value}")  

    return fit

def hc_k(positions, fit, weight=None, hc_steps=100, hc_dec=0.9):
    """
    Performs hill-climbing optimization on fit parameters.
    
    Parameters:
    positions (array): Indices of parameters to optimize.
    fit (Fit): Fit object containing model data.
    weight (optional): Weighting factor for goal function.
    hc_steps (int): Number of hill-climbing iterations (default: 100).
    hc_dec (float): Decay factor for perturbation (default: 0.9).
    perturb (float): Initial perturbation factor (default: 1.0).
    
    Returns:
    Fit: Fit object with optimized parameters if cost is reduced.
    """
    initial_goal = fit.cost.compute_cost(fit.data.observed_abundances, fit.predicted_abundances, fit.data.times)
    initial_values = fit.pars[positions]
    #set perturbation magnitude at 1
    perturb=1.0
    for _ in range(hc_steps):
        tmp = fit
        tmp.pars[positions] = tmp.pars[positions] * (1 + perturb * np.random.randn(len(positions)))
        tmp.goal = fit.cost.compute_cost(tmp.data.observed_abundances, tmp.predicted_abundances, tmp.data.times)
        
        if tmp.goal < initial_goal:
            fit = tmp
            initial_goal = tmp.goal
        
        perturb *= hc_dec
    
    return fit
