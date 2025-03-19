import itertools
import random
from optimization_funcs import *

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
        # Adjust ODE parameters
        fit.optimize(fit.par_ix_model, weight=weight)
        #Adjust both parameter groups simultaneously
        fit.optimize(np.concatenate((fit.par_ix_model, fit.par_ix_cost)),
                weight=weight)

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
