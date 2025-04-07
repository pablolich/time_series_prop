import numpy as np
THRESH = 1e-10  # default threshold to avoid zeros

def dxdt_func(t, x, *params):
    from numpy import array
    return array([
        params[1],
        params[0],
    ])


class Model_0:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 0
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_0_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 0.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return [c0, c1]

    def parse_model_parameters(self, pars):
        return pars

def dxdt_func(t, x, *params):
    from numpy import array
    return array([
        params[0]*x[0],
        params[1]*x[0],
    ])


class Model_1:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 1
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_1_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 1.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return [c0, c1]

    def parse_model_parameters(self, pars):
        return pars

def dxdt_func(t, x, *params):
    from numpy import array
    return array([
        params[0]*x[0]*x[1],
        params[1]*x[0] + params[2],
    ])


class Model_2:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 2
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 3
        self.model_name = "model_2_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 2.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return [c0, c1, c2]

    def parse_model_parameters(self, pars):
        return pars

def dxdt_func(t, x, *params):
    from numpy import array
    return array([
        params[0]*params[5] + params[0]*x[0] + params[0]*x[1],
        params[2] + x[0]*(params[1]*x[0] + params[3]*params[4]),
    ])


class Model_3:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 3
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 6
        self.model_name = "model_3_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 3.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return [c0, c1, c2, c3, c4, c5]

    def parse_model_parameters(self, pars):
        return pars

def dxdt_func(t, x, *params):
    from numpy import array
    return array([
        x[0]*(params[0] + params[1]*params[5]*(x[0] + x[1])),
        x[1]*(params[2] + x[0])*(params[3] + 2*params[4]*params[5]*x[0]*x[1]**2 + params[4]*x[0]),
    ])


class Model_4:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 4
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 6
        self.model_name = "model_4_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 4.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def get_param_symbols(self):
        return [c0, c1, c2, c3, c4, c5]

    def parse_model_parameters(self, pars):
        return pars

