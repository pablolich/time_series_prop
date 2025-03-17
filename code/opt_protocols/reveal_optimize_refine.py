import itertools
import random
from optimization_funcs import *

def reveal_optimize_refine(fit, weights, n_rounds=100):
    """
    Performs a optimization protocol on the Fit object.

    This function iterates over given weights, optimizing parameters using 
    different optimization techniques including `optimize_k`, `hc_k`, and `all_k`. 

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
        # Adjust ODE parameters
        fit = optimize_k(fit.par_ix_model, fit, weight=weight)
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

        for ss in range(n_rounds):
            fit = hc_k(fit.par_ix_model, fit, weight=weight)
            if ss % 25 == 0:
                print(f"Step: {ss}, Goal: {fit.cost_value}")

        # Optimize combinations of parameters
        fit = all_k(fit.par_ix_model, fit, k=2, weight=weight)
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

        # Optimize again
        fit = optimize_k(fit.par_ix_model, fit, weight=weight)
        print(f"Weight: {weight}, Goal: {fit.cost_value}")

    # Final optimization on all parameters
    fit = optimize_k(list(range(fit.n_pars)), fit, weight=0)
    print(f"Weight: 0, Goal: {fit.cost_value}")

    # Compute predictions
    fit = get_predictions(fit)

    # Update the final goal function value
    fit.cost_value = fit.cost.compute_cost(fit.data.observed_abundances,
            fit.predicted_abundances,
            fit.data.times)

    return fit
