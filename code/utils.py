import numpy as np
from scipy.integrate import solve_ivp


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

def integrate(fit, model):
    """
    Integrate the GLV model using the parameters in fit.
    Updates fit.predicted_abundances and fit.predicted_proportions.
    """
    params = fit.parse_model_parameters()
    predicted_abundances = []
    predicted_proportions = []

    for i in range(fit.n_time_series):
        times = fit.times[i]
        init_conds = params["init_conds"][i]

        import ipdb; ipdb.set_trace(context = 20)
        sol = solve_ivp(
            fun=lambda t, y: model.dxdt(t, y, params),
            t_span=(times[0], times[-1]),
            y0=init_conds,
            t_eval=times,
            method='RK45',
            max_step=0.1
        )

        abundances = sol.y.T
        abundances[fit.observed_proportions[i] == 0] = 0  # Handle zero observations
        abundances[~np.isfinite(abundances)] = 1e6  # Handle NaN/Inf

        proportions = abundances / np.sum(abundances, axis=1, keepdims=True)

        predicted_abundances.append(abundances)
        predicted_proportions.append(proportions)

    fit.predicted_abundances = predicted_abundances
    fit.predicted_proportions = predicted_proportions

