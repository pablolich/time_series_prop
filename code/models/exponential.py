import numpy as np

THRESH = 1e-16  # Threshold for small values

class ExponentialModel:
    def __init__(self, dim):
        """
        Exponential growth model class.
        :param dim: Number of species.
        :param n_model: number of model parameters
        :param model_name: model name
        :param dynamics: whether dynamics are expressed in differential
                         equation or analytical solution form
        """
        self.dim = dim  # Number of species
        self.n_model = dim  # Number of model parameters (x0 and r)
        self.model_name = "exponential"
        self.dynamics_type = "x_t"

    def dynamics(self, times, x0, pars):
        """
        Compute exponential growth over time.
        :param times: Time points for sampling
        :param pars: Dictionary containing initial values and growth rates
        :return: Matrix of densities over time
        """
        r = pars["r"]
        output = np.zeros((len(times), len(x0)))
        
        for i in range(len(x0)):
            density = np.minimum(np.maximum(x0[i] * np.exp(r[i] * times), 
                                            THRESH),
                                 1e6)
            output[:, i] = density
        
        return output

    def parse_model_parameters(self, dim, pars):
        """
        Create a dictionary of parameter names and dimensional shapes.
        :param dim: Dimension of the model
        :param pars: Vector of model parameters
        :return: Dictionary of parameters
        """
        params = {
                "r": pars[:dim].reshape(dim, 1)
                }
        return params

