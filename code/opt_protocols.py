"""
File containing optimization protocols
"""

from aux_integration import *
from aux_optimization import *

import itertools
import random

def nelder_bfgs(fit, weights=None, n_rounds=10):
    """
    Performs a optimization protocol on the Fit object.

    This function iterates over given weights, optimizing fit using the 
    mehtods Nelder-Mead and BFGS alternatively

    Parameters:
    ----------
    fit : Fit
        The Fit object containing model data, parameters, and cost function.
    weights : list
        A list of weight values used for optimization.
    n_rounds : int, optional
        Number of single optimization steps to perform (default: 100).

    Returns:
    -------
    Fit
        The optimized Fit object after all optimization steps.
    """
    round_i = 0
#    fit = hc_k(fit.par_ix_model, fit)
    while round_i < n_rounds:
        fit.optimize(np.concatenate((fit.par_ix_data, fit.par_ix_model)))
        fit.optimize(np.concatenate((fit.par_ix_data, fit.par_ix_model)), 
            method = "BFGS")
        #fit.optimize(fit.par_ix_model)
        #fit.optimize(fit.par_ix_model, method = 'BFGS')
        round_i += 1
    return fit

def reveal_optimize_refine(fit, weights, n_rounds=100):
    """
    Performs a optimization protocol on the Fit object.

    This function iterates over given weights, optimizing parameters using 
    different optimization techniques including `hc_k`, and `all_k`. 

    Parameters:
    ----------
    fit : Fit
        The Fit object containing model data, parameters, and cost function.
    weights : list
        A list of weight values used for optimization.
    n_rounds : int, optional
        Number of single optimization steps to perform (default: 100).

    Returns:
    -------
    Fit
        The optimized Fit object after all optimization steps.
    """

    for weight in weights:
        #Adjust weights for the log function if they exist
        if len(fit.par_ix_cost) > 0:
            fit.optimize(fit.par_ix_cost) 
        # Adjust ODE parameters through several rounds of NM and BFGS
        fit = nelder_bfgs(fit, weight, n_rounds = 1)
        #Adjust both parameter groups simultaneously
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

        for ss in range(n_rounds):
            fit = hc_k(fit.par_ix_model, fit, weight=weight)
            if ss % 25 == 0:
                print(f"Step: {ss}, Goal: {fit.cost_value}")

        # Optimize combinations of parameters
        fit = all_k(fit.par_ix_model, fit, k=2, weight=weight)
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

        # Optimize again
        fit.optimize(fit.par_ix_model, weight=weight)
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

    # Final optimization on all parameters
    fit.optimize(list(range(fit.n_pars))) #weight is 0 here
    print(f"Weight: 0, Goal: {fit.cost_value}")

    return fit
