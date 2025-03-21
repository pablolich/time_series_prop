"""
File containing definition of model classes
Current models implemented: 
    1. Glv
    2. Exponential
"""

import numpy as np

THRESH = 1e-16  # Threshold for small values

class GlvComp:
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
        self.model_name = "glv"
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
        dx = x * (pars["r"] - np.dot(pars["A"], x))
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

class Exponential:
    def __init__(self, dim):

        self.dim = dim  # Number of species
        self.n_model = dim  # Number of model parameters (x0 and r)
        self.model_name = "exponential"
        self.dynamics_type = "x_t"

    def dynamics(self, times, x0, pars):
        r = pars["r"]
        output = np.zeros((len(times), len(x0)))
        
        #compute exponential growth clipping very large and very small values
        for i in range(len(x0)):
            density = np.minimum(np.maximum(x0[i] * np.exp(r[i] * times), 
                THRESH),1e6)
            output[:, i] = density
        
        return output

    def parse_model_parameters(self, dim, pars):
        return {"r": pars[:dim].reshape(dim, 1)}
