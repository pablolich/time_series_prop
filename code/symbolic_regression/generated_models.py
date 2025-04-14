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
        self.n_model = 3
        self.model_name = "model_0_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 0.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2],
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
        self.n_model = 4
        self.model_name = "model_1_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 1.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2],
              pars[1]*x[0] + pars[3],
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

class Model_2:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 2
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 4
        self.model_name = "model_2_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 2.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2],
              pars[0]*x[1] + pars[3]*x[1]**2,
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

class Model_3:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 3
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_3_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 3.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[4],
              pars[0] + pars[1]*x[1] + pars[3]*x[1]**2,
              pars[2],
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
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[5],
              pars[0] + pars[1]*x[1] + pars[2]*x[1]**2 + pars[3]*x[0],
              pars[4],
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
        self.n_model = 4
        self.model_name = "model_5_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 5.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[3]*x[0],
              pars[2],
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

class Model_6:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 6
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_6_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 6.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[3]*x[0],
              pars[1]*x[0] + pars[4],
              pars[2],
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
        self.n_model = 5
        self.model_name = "model_7_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 7.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[3]*x[0],
              pars[1]*x[1] + pars[4]*x[1]**2,
              pars[2],
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
        self.n_model = 6
        self.model_name = "model_8_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 8.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[4]*x[0],
              pars[1] + pars[2]*x[1] + pars[5]*x[1]**2,
              pars[3],
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
        self.n_model = 7
        self.model_name = "model_9_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 9.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[6]*x[0],
              pars[1] + pars[2]*x[1] + pars[3]*x[1]**2 + pars[4]*x[0],
              pars[5],
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
        self.n_model = 4
        self.model_name = "model_10_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 10.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[0] + pars[3]*x[0]**2,
              pars[2],
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

class Model_11:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 11
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_11_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 11.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2]*x[0] + pars[4]*x[0]**2,
              pars[1]*x[0] + pars[3],
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

class Model_12:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 12
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_12_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 12.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2]*x[0] + pars[4]*x[0]**2,
              pars[0]*x[1] + pars[3]*x[1]**2,
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
        self.n_model = 6
        self.model_name = "model_13_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 13.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[4]*x[0] + pars[5]*x[0]**2,
              pars[0] + pars[1]*x[1] + pars[3]*x[1]**2,
              pars[2],
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
        self.n_model = 7
        self.model_name = "model_14_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 14.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[5]*x[0] + pars[6]*x[0]**2,
              pars[0] + pars[1]*x[1] + pars[2]*x[1]**2 + pars[3]*x[0],
              pars[4],
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
        self.n_model = 4
        self.model_name = "model_15_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 15.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[3]*x[0]*x[1]**2,
              pars[2],
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

class Model_16:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 16
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_16_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 16.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2]*x[1]**2 + pars[4]*x[0]*x[1]**2,
              pars[1]*x[0] + pars[3],
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

class Model_17:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 17
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 5
        self.model_name = "model_17_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 17.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2]*x[1]**2 + pars[4]*x[0]*x[1]**2,
              pars[0]*x[1] + pars[3]*x[1]**2,
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

class Model_18:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 18
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 6
        self.model_name = "model_18_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 18.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[4]*x[1]**2 + pars[5]*x[0]*x[1]**2,
              pars[0] + pars[1]*x[1] + pars[3]*x[1]**2,
              pars[2],
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

class Model_19:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 19
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 7
        self.model_name = "model_19_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 19.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[5]*x[1]**2 + pars[6]*x[0]*x[1]**2,
              pars[0] + pars[1]*x[1] + pars[2]*x[1]**2 + pars[3]*x[0],
              pars[4],
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

class Model_20:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 20
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 6
        self.model_name = "model_20_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 20.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[1]*x[1] + pars[3]*x[0]*x[1] + pars[5]*x[0]*x[1]**2,
              pars[4],
              pars[2],
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

class Model_21:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 21
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 7
        self.model_name = "model_21_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 21.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[1]*x[1] + pars[4]*x[0]*x[1] + pars[5]*x[0]*x[1]**2,
              pars[2]*x[0] + pars[6],
              pars[3],
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

class Model_22:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 22
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 7
        self.model_name = "model_22_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 22.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[1]*x[1] + pars[4]*x[0]*x[1] + pars[5]*x[0]*x[1]**2,
              pars[2]*x[1] + pars[6]*x[1]**2,
              pars[3],
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

class Model_23:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 23
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 8
        self.model_name = "model_23_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 23.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[1]*x[1] + pars[5]*x[0]*x[1] + pars[7]*x[0]*x[1]**2,
              pars[2] + pars[3]*x[1] + pars[6]*x[1]**2,
              pars[4],
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

class Model_24:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 24
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_24_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 24.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[5]*x[1] + pars[7]*x[0]*x[1] + pars[8]*x[0]*x[1]**2,
              pars[1] + pars[2]*x[1] + pars[3]*x[1]**2 + pars[6]*x[0],
              pars[4],
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

class Model_25:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 25
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 8
        self.model_name = "model_25_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 25.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[2]*x[1] + pars[3]*x[0]*x[1]**2 + pars[4]*x[0]*x[1] + pars[5]*x[0] + pars[7],
              pars[6],
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

class Model_26:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 26
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_26_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 26.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[2]*x[0]*x[1]**2 + pars[3]*x[0] + pars[4] + pars[5]*x[1]**2 + pars[6]*x[0]*x[1] + pars[7]*x[1],
              pars[1]*x[0] + pars[8],
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

class Model_27:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 27
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_27_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 27.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[1]*x[0]*x[1]**2 + pars[2]*x[0] + pars[4] + pars[5]*x[1]**2 + pars[7]*x[0]*x[1] + pars[8]*x[1],
              pars[3]*x[1]**2 + pars[6]*x[1],
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

class Model_28:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 28
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 10
        self.model_name = "model_28_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 28.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[1]*x[0]*x[1]**2 + pars[2]*x[0] + pars[3] + pars[4]*x[1]**2 + pars[8]*x[0]*x[1] + pars[9]*x[1],
              pars[5] + pars[6]*x[1] + pars[7]*x[1]**2,
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

class Model_29:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 29
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 11
        self.model_name = "model_29_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 29.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[10]*x[1] + pars[4]*x[0]*x[1]**2 + pars[5]*x[0] + pars[6] + pars[7]*x[1]**2 + pars[9]*x[0]*x[1],
              pars[0] + pars[1]*x[1] + pars[2]*x[1]**2 + pars[8]*x[0],
              pars[3],
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

class Model_30:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 30
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 8
        self.model_name = "model_30_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 30.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[2]*x[0]*x[1]**2 + pars[3]*x[1] + pars[4] + pars[5]*x[0] + pars[7]*x[0]*x[1],
              pars[6],
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

class Model_31:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 31
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_31_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 31.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[3]*x[0]*x[1]**2 + pars[4]*x[1] + pars[5]*x[0] + pars[6]*x[0]*x[1] + pars[7],
              pars[1]*x[0] + pars[8],
              pars[2],
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

class Model_32:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 32
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_32_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 32.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[2]*x[0]*x[1]**2 + pars[3]*x[1] + pars[4]*x[0] + pars[6]*x[0]*x[1] + pars[8],
              pars[5]*x[1]**2 + pars[7]*x[1],
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

class Model_33:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 33
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 10
        self.model_name = "model_33_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 33.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[2]*x[0]*x[1]**2 + pars[3]*x[1] + pars[4]*x[0] + pars[5]*x[0]*x[1] + pars[9],
              pars[6] + pars[7]*x[1] + pars[8]*x[1]**2,
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

class Model_34:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 34
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 11
        self.model_name = "model_34_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 34.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0]*x[1]**2 + pars[10] + pars[4]*x[0]*x[1]**2 + pars[6]*x[1] + pars[7]*x[0] + pars[8]*x[0]*x[1],
              pars[1] + pars[2]*x[1] + pars[3]*x[1]**2 + pars[9]*x[0],
              pars[5],
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

class Model_35:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 35
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 6
        self.model_name = "model_35_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 35.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[1]*x[0] + pars[3]*x[0]*x[1] + pars[5]*x[1]**2 + x[0]*x[2] + x[2],
              pars[4],
              pars[2],
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

class Model_36:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 36
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 7
        self.model_name = "model_36_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 36.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[1]*x[0] + pars[4]*x[0]*x[1] + pars[6]*x[1]**2 + x[0]*x[2] + x[2],
              pars[2]*x[0] + pars[5],
              pars[3],
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

class Model_37:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 37
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 7
        self.model_name = "model_37_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 37.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[1]*x[0] + pars[4]*x[0]*x[1] + pars[6]*x[1]**2 + x[0]*x[2] + x[2],
              pars[2]*x[1] + pars[5]*x[1]**2,
              pars[3],
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

class Model_38:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 38
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 8
        self.model_name = "model_38_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 38.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[1]*x[0] + pars[6]*x[0]*x[1] + pars[7]*x[1]**2 + x[0]*x[2] + x[2],
              pars[2] + pars[3]*x[1] + pars[4]*x[1]**2,
              pars[5],
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

class Model_39:
    def __init__(self, dim):
        """
        Auto-generated model class from symbolic PySR equations.

        Model number: 39
        Data source: synthetic_simplest

        :param dim: Number of species.
        """
        self.dim = dim
        self.n_model = 9
        self.model_name = "model_39_synthetic_simplest"
        self.dynamics_type = "dxdt"

    def dynamics(self, t, x, pars):
        """
        Compute dx/dt for model 39.

        :param t: Time
        :param x: State variables
        :param pars: List of parameters in order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
        :return: dx/dt as NumPy array
        """
        pars = pars['pars']
        def dxdt_func(t, x, *pars):
            return np.array([
              pars[0] + pars[5]*x[0]*x[1] + pars[6]*x[1]**2 + pars[7]*x[0] + x[0]*x[2] + x[2],
              pars[1] + pars[2]*x[1] + pars[3]*x[1]**2 + pars[8]*x[0],
              pars[4],
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

