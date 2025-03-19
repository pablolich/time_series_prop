from optimization_funcs import *

def nelder_bfgs(fit, weights, n_rounds=10):
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
    fit = hc_k(fit.par_ix_model, fit)
    while round_i < n_rounds:
        fit.optimize(fit.par_ix_model)
        fit.optimize(fit.par_ix_model, method = "BFGS")
        #fit = hc_k(fit.par_ix_model, fit)
        round_i += 1
    return fit
