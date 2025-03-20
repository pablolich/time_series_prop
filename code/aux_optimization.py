"""
This script contains functions to perform different optimization techniques 
on fit objects
"""
from aux_integration import *

from scipy.optimize import minimize
import numpy as np
import itertools #for optimization of specific parameter combinations
import random
import copy

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
    print("weights inside all_k: ", weight)
    for i, combo in enumerate(combinations):
        # Call optimize_k for the current combination of parameters
        fit.optimize(positions=list(combo), weight=weight)
        
        # Print iteration information: weight, iteration number, total combinations, and the current goal value
        print(f"{weight}, {i + 1}, {ncombos}, {fit.cost_value}")  

    return fit

def hc_k(positions, fit, weight=None, hc_steps=100, hc_dec=0.9):
    """
    Performs hill-climbing optimization on model parameters.
    
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
    initial_goal = fit.cost_value
    initial_values = fit.pars[positions]
    #set perturbation magnitude at 1
    perturb=1.0
    for _ in range(hc_steps):
        tmp = copy.deepcopy(fit)
        #perturb parameters and propagate abundnaces
        tmp.pars[positions] = tmp.pars[positions] * (1 + perturb * np.random.randn(len(positions)))
        tmp.get_predictions()
        tmp.cost_value = tmp.to_minimize(tmp.pars, range(tmp.n_pars), weight)        
        if tmp.cost_value < initial_goal:
            fit = copy.deepcopy(tmp)
            initial_goal = tmp.cost_value
        
        perturb *= hc_dec
    
    return fit

def initialize_random(fit, n_rounds=10, init_weight=None):
    """
    Performs random initialization of parameters and optimizes using 
    hill-climbing.

    This function initializes the model, cost function, and observed 
    initial conditions randomly over multiple attempts (`n_rounds`). It applies
    hill-climbing optimization (`hc_k`) to refine the parameters and selects 
    the best fit based on the goal function.

    Parameters:
    ----------
    fit : Fit
        An instance of the Fit class containing data, model, and cost function.
    n_rounds : int, optional
        Number of random initialization attempts (default: 10).
    init_weight : float, optional
        Weighting factor for the goal function during optimization.

    Returns:
    -------
    Fit
        The best `Fit` instance found after nrounds of initialization and
        nrounds of hill climbing optimization.
    """
    best_fit = copy.deepcopy(fit)
    best_goal = None

    for ntry_random in range(n_rounds):
        print(f"Random init {ntry_random}, Best goal: {best_goal}")

        # Initialize at the observed initial conditions after setting new seed
        fit.random_seed = ntry_random
        fit.initialize_parameters()
        fit.get_predictions()
        #calculate cost with initial parameters
        fit.cost_value = fit.to_minimize(fit.pars, range(fit.n_pars), 
                weight = init_weight)
        # Perform one round of (100 steps of) hill-climbing optimization
        fit = hc_k(fit.par_ix_model, fit, weight=init_weight)
        cur_goal = fit.cost_value

        # Track the best fit based on the goal function value
        if best_goal is None or cur_goal < best_goal:
            best_goal = cur_goal
            best_fit = copy.deepcopy(fit)
            print(f"Iteration {ntry_random}: New best goal = {best_goal}")

    return best_fit
