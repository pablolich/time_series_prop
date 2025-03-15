#this script contains auxiliary functions to score solutions, integrate
#dynamics, and search parameters

import numpy as np
from scipy.integrate import solve_ivp

##############################################################################
#scoring functions

def continuous_step(x, lambd, k=25):
    """
    Returns the weights given x between 0 and 1.

    :param x: Array-like, values between 0 and 1.
    :param lambd: Transition point.
    :param k: Steepness of transition (default: 25).
    :return: Array of weights.
    """
    x = np.array(x)  # Ensure x is a NumPy array

    # If transition happens at 1, return all ones
    if lambd == 1:
        return np.ones_like(x)
    
    # Compute the step function
    return 1 - 1 / (1 + np.exp(-k * (x - lambd)))

###############################################################################
#integration functions

def parse_integration_parameters(fit):
    """
    Parse parameters from the fit object to extract initial conditions 
    and model coefficients for integration routine
    :return: Dictionary with structured parameters.
    """
    pars = fit.pars
    #get parameters corresponding only to model
    p = pars[fit.n_initial:fit.n_initial + fit.model.n_model]
    #create a dictionary of parameters to be read by the integration routine
    params = fit.model.parse_model_parameters(fit.model.dim, p)

    #initialize initial conditions accounting for true zeros
    init_conds = []
    x0 = np.abs(pars[:fit.n_initial])
    #do this for each time series
    for i in range(fit.data.n_time_series):
        tmp = x0[i * fit.data.n:(i + 1) * fit.data.n]
        # Apply hard zeros
        zeros = fit.set_true_zeros[i * fit.data.n:(i + 1) * fit.data.n]
        tmp *= zeros
        init_conds.append(tmp)

    #append initial conditions
    params["init_conds"] = init_conds

    return params

def process_integration_result(solution, times):

    #get status of integration
    int_status = solution.status
    if int_status == 0: #integration finished
        return solution.y.T
    elif int_status == -1: #integration failed
        #solution is composed of current result plus very high abundances
        n_successful = np.shape(solution.y)[1]
        n_complete = len(times) - n_successful
        high_values = 1e6*np.ones((np.shape(solution.y)[0], n_complete))
        return(np.concatenate([solution.y, high_values], axis = 1).T)
    else:
        print("I don't know what to do with this integration status")


def integrate(fit):
    """
    Integrate dynamics of model given initial conditions and parameters in fit.
    Updates fit.predicted_abundances and fit.predicted_proportions.
    """
    #get parameters necessary for numerical integration
    params = parse_integration_parameters(fit)
    predicted_abundances = []
    predicted_proportions = []

    #integrate each time series separately
    for i in range(fit.data.n_time_series):
        times = fit.data.times[i]
        init_conds = params["init_conds"][i]

        sol = solve_ivp(
            fun=lambda t, y: fit.model.dxdt(t, y, params),
            t_span=(times[0], times[-1]),
            y0=init_conds,
            t_eval=times,
            method='RK45'
        )
        #process result
        abundances = process_integration_result(sol, times)
        abundances[fit.data.observed_proportions[i] == 0] = 0  # Handle zero observations

        proportions = abundances / np.sum(abundances, axis=1, keepdims=True)

        predicted_abundances.append(abundances)
        predicted_proportions.append(proportions)

    fit.predicted_abundances = predicted_abundances
    fit.predicted_proportions = predicted_proportions

##############################################################################
#search functions

def optimize(fit):
    """
    Perform optimization of parameters in fit by integrating dynamics of the 
    underlying model
    """

