import numpy as np
import os
import pickle
from scipy.integrate import odeint, solve_ivp

THRESH = 1e-16  # Threshold for small values
PENALIZATION_ABUNDANCE = 1e-7  # Replacement for NaN values

class GLVModel:
    def __init__(self, data):
        """
        Generalized Lotka-Volterra (GLV) model class.
        :param fit: Fit object containing data and parameters.
        """
        self.n = data.n  # Number of species
        self.n_model = self.n * (self.n + 1)  # Number of model parameters
        self.model_name = "GLV"

    def dxdt(self, t, x, pars):
        """
        Compute dx/dt for the GLV model.
        :param t: Time (not used explicitly)
        :param x: State variables (species abundances)
        :param pars: Dictionary containing model parameters
        :return: dx/dt as a NumPy array
        """
        x = np.maximum(x, THRESH)  # Apply threshold
        dx = x * (pars["r"] + np.dot(pars["A"], x))
        return dx


