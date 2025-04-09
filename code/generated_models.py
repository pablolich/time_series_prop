import numpy as np
THRESH = 1e-10  # default threshold to avoid zeros

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
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[1],
              pars[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

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
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[1],
              pars[0]*x[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

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
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2],
              pars[0] + pars[1]*x[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_3:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 3
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 4
        self.model_name = "model_3_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 3.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[3],
              pars[0] + pars[1]*x[0] + pars[2]*x[0]**2,
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_4:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 4
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_4_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 4.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0],
              pars[1],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_5:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 5
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_5_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 5.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0],
              pars[1]*x[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_6:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 6
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 3
        self.model_name = "model_6_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 6.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0],
              pars[1]*x[0] + pars[2],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_7:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 7
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 4
        self.model_name = "model_7_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 7.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0],
              pars[1] + pars[2]*x[0] + pars[3]*x[0]**2,
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_8:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 8
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_8_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 8.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0]*x[1],
              pars[1],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_9:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 9
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 2
        self.model_name = "model_9_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 9.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0]*x[1],
              pars[1]*x[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_10:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 10
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 3
        self.model_name = "model_10_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 10.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0]*x[1],
              pars[1]*x[0] + pars[2],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_11:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 11
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 4
        self.model_name = "model_11_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 11.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0]*x[1],
              pars[1] + pars[2]*x[0] + pars[3]*x[0]**2,
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_12:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 12
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 3
        self.model_name = "model_12_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 12.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[2]*x[0] + pars[2]*x[1],
              pars[1],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_13:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 13
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 3
        self.model_name = "model_13_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 13.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[2]*x[0] + pars[2]*x[1],
              pars[1]*x[0],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_14:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 14
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 4
        self.model_name = "model_14_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 14.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[3]*x[0] + pars[3]*x[1],
              pars[1]*x[0] + pars[2],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

class Model_15:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 15
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_15_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 15.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[1]*x[0] + pars[1]*x[1],
              pars[2]*x[0] + pars[3]*x[0]**2 + pars[4],
            ])
        
        x = np.maximum(x, 1e-10)
        return dxdt_func(t, x, *pars)

    def parse_model_parameters(self, dim, pars):
        """
        Package model parameters into a dictionary.
        
        :param dim: Number of species
        :param pars: List of parameter values
        :return: Dictionary of parameters
        """
        params = {
            "pars": pars,
        }
        return params

