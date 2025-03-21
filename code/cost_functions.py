"""
File containing definition of  cost function classes
Current cost functions implemented: 
    1. Ssq
    2. LogProp
    3. Dirichlet
"""

#general imports
import numpy as np
#sample from dirichlet distribution
from scipy.stats import dirichlet

class Ssq:

    def __init__(self, dim):

        self.n_cost = 0 #number of parameters
        self.cost_name = "ssq"

    def compute_cost(self, observed, predicted, times, pars, obs_type, weight=None):
        """
        Compute the SSQ cost function for predicted abundances.
        :param observed: List of matrices of observed proportions
        :param predicted: List of matrices of predicted abundances
        :param times: List of vectors representing times
        :param pars: Dictionary of parameters for the cost function
        :param weight: Optional weighting factor, based on time (for each time point)
        :return: Log of mean SSQ error.
        """
        SSQ = []

        for i in range(len(observed)):
            obs = observed[i]
            if obs_type == "prop":
                #transform predictions to proportions
                pred = predicted[i] / np.sum(predicted[i], axis=1, keepdims=True) 

            # Calculate sum of squared differences
            diff = (obs - pred) ** 2
            weighted_diff = diff
            if weight is not None:
                # Apply the weight based on time for the current time series
                weighted_diff *= np.exp(-weight * times[i])[:, np.newaxis]

            # Flatten the weighted squared differences and add them to SSQ list
            SSQ.extend(weighted_diff.flatten())

        return np.log(np.mean(SSQ))

    def initialize_cost_function_parameters(self):
        # Return None since SSQ has no parameters
        return None

class LogDist:
    """
    Logarithmic Distance.
    """

    def __init__(self):

        self.n_cost = 0
        self.cost_name = "logdist"

    def compute_cost(self, observed, predicted, times, pars,  obs_type, weight=None):
        #initialize goal
        goal = 0

        for i in range(len(observed)):
            obs = observed[i]
            if obs_type == "prop":
                #transform predictions to proportions
                pred = predicted[i] / np.sum(predicted[i], axis=1, keepdims=True) 
            
            # Compute the log ratio (add THRESH to avoid log(0))
            ratio = np.log(pred + THRESH) - np.log(obs + THRESH)
            goal_rs = np.sum(np.abs(ratio), axis=1)  #Sum the absolute values row-wise
            
            if weight is not None:
                goal_rs *= np.exp(-weight * times[i])

            # Add to the overall goal function
            goal += np.sum(goal_rs)

        return goal

    def initialize_cost_function_parameters(self):
        # Return None since LogDist has no parameters
        return None

    def initialize_goal_pars(self):
        # Initialize goal parameters (no parameters for LogDist)
        return []

class Dirichlet:

    def __init__(self, dim):

        self.n_cost = dim
        self.cost_name = "dirichlet"

    def compute_cost(self, observed, predicted, times, pars,  obs_type="prop", weight=None):
        #initialize likelihood
        likelihood = []
        #get weights to transform proportions & abundances
        pars_dict = self.parse_cost_function_parameters(pars)
        w = abs(pars_dict["w"])
        #new abundances
        for i in range(len(observed)):
            #need to renormalize taking w into account
            obs = observed[i]*w / np.sum(observed[i]*w, axis=1, keepdims=True)
            #rescale absolute predictions by w
            pred = predicted[i]*w
            # Calculate likelihood for each time point
            lik_i_vec = np.array([dirichlet.logpdf(obs[row_i], pred[row_i]) for \
                    row_i in range(len(pred))])
            weighted_lik_i_vec = lik_i_vec

            if weight is not None:
                weighted_lik_i_vec *= np.exp(-weight * times[i])

            likelihood.extend(weighted_lik_i_vec.flatten())

        return -np.sum(likelihood)
    
    def parse_cost_function_parameters(self, pars):
        """
        Create a dictionary of parameter names and dimensional shapes
        :param pars: vector of cost function parameters
        """
        params = {
            "w": pars, 
            }
        return params
