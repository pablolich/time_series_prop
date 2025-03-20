"""
This script contains functions used for prediction of abundances given 
parameters
"""

import numpy as np #general use
from scipy.integrate import solve_ivp #for integration

THRESH = 1e-16

def parse_parameters_dynamics(fit):
    """
    Parse parameters from fit necessary for forward  evaluation of dynamics of 
    the model, i.e. initial conditions and model parameters
    :return: Dictionary with structured parameters.
    """
    pars = fit.pars
    #get parameters corresponding only to model
    p = pars[fit.data.n_initial:fit.data.n_initial + fit.model.n_model]
    #create a dictionary of parameters to be read by the integration routine
    params = fit.model.parse_model_parameters(fit.model.dim, p)

    #initialize initial conditions accounting for true zeros
    init_conds = []
    x0 = np.abs(pars[:fit.data.n_initial])
    #do this for each time series
    for i in range(fit.data.n_time_series):
        tmp = x0[i * fit.data.n:(i + 1) * fit.data.n]
        #Apply hard zeros
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
        result = solution.y.T
        # Check for negative values and set them to THRESH
        result[result < 0] = THRESH
        return result 
    elif int_status == -1: #integration failed
        #solution is composed of current result plus very high abundances
        result = solution.y
        result[result < 0] = THRESH
        n_successful = np.shape(result)[1]
        n_complete = len(times) - n_successful
        high_values = 1e6*np.ones((np.shape(result)[0], n_complete))
        return(np.concatenate([result, high_values], axis = 1).T)
    else:
        print("I don't know what to do with this integration status")
