import numpy as np
import os
import pickle
from scipy.integrate import odeint, solve_ivp

THRESH = 1e-16  # Threshold for small values
PENALIZATION_ABUNDANCE = 1e-7  # Replacement for NaN values

class GLVModel:
    def __init__(self, fit):
        """
        Generalized Lotka-Volterra (GLV) model class.
        :param fit: Fit object containing data and parameters.
        """
        self.fit = fit
        self.n = fit.n  # Number of species
        self.n_model = self.n * (self.n + 1)  # Number of model parameters
        self.model_name = "GLV"

    def dxdt(self, x, t, pars):
        """
        Compute dx/dt for the GLV model.
        :param x: State variables (species abundances)
        :param t: Time (not used explicitly)
        :param pars: Dictionary containing model parameters
        :return: dx/dt as a NumPy array
        """
        x = np.maximum(x, THRESH)  # Apply threshold
        dx = x * (pars["r"] + np.dot(pars["A"], x))
        return dx

    def parse_parameters(self):
        """
        Parse parameters from the Fit object to extract initial conditions and model coefficients.
        :return: Dictionary with structured parameters.
        """
        pars = self.fit.pars
        p = pars[self.fit.n_initial:self.fit.n_initial + self.n_model]
        init_conds = []
        x0 = np.abs(pars[:self.fit.n_initial])

        for i in range(self.fit.n_time_series):
            tmp = x0[i * self.n:(i + 1) * self.n]
            # Apply hard zeros
            zeros = self.fit.set_true_zeros[i * self.n:(i + 1) * self.n]
            tmp *= zeros
            init_conds.append(tmp)

        params = {
            "r": p[:self.n], 
            "A": p[self.n:].reshape(self.n, self.n), 
            "init_conds": init_conds
        }
        return params

    def initialize_model_parameters(self):
        """
        initialize model parameters randomly and store them in fit.pars.
        """
        import ipdb; ipdb.set_trace(context = 20)
        np.random.seed(self.fit.random_seed)
        self.fit.pars[self.fit.n_initial:self.fit.n_initial + self.n_model] = np.random.randn(self.n_model)

    def integrate(self):
        """
        Integrates the system for each time series in the Fit object and updates predicted_abundances.
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

