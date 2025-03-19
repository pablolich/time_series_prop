import numpy as np
import os
import pickle
from scipy.integrate import odeint, solve_ivp

THRESH = 1e-16  # Threshold for small values
PENALIZATION_ABUNDANCE = 1e-7  # Replacement for NaN values

class GLVModel:
    def __init__(self, dim):
        """
        Generalized Lotka-Volterra (GLV) model class.
        :param dim: dimension of the model (number of species).
        :param n_model: number of model parameters
        :param model_name: model name
        :param dynamics: whether dynamics are expressed in differential
                         equation or analytical solution form
        """
        self.dim = dim # Number of species
        self.n_model = dim * (dim + 1)  # Number of model parameters
        self.model_name = "GLV"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
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

    def parse_model_parameters(self, dim, pars):
        """
        Create a dictionary of parameter names and dimensional shapes
        :param dim: dimension of the model
        :param pars: vector of model parameters
        """
        params = {
            "r": pars[:dim], 
            "A": pars[dim:].reshape(dim, dim)
            }
        return params

