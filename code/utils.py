import numpy as np

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

def integrate(fit):
    """
    Integrates the system for each time series in the Fit object and updates 
    predicted_abundances.
    """
    pars = self.parse_parameters()
    predicted_abundances = []

    for i in range(self.fit.n_time_series):
        # Solve ODE using scipy's odeint
        init_cond = pars["init_conds"][i] if "init_conds" in pars else self.fit.observed_abundances[i][0]
        times = self.fit.times[i]

        out = odeint(self.dxdt, init_cond, times, args=(pars,))
        y = np.array(out)

        # Handle invalid values
        y[np.isnan(y) | np.isinf(y) | (y > 1e5) | (y < -1e-15) | (y < 0)] = PENALIZATION_ABUNDANCE

        predicted_abundances.append(y)

    self.fit.predicted_abundances = predicted_abundances

