import numpy as np
from utils import continuous_step

class SSQCostFunction:
    """
    Sum of Squared Differences (SSQ) for Proportions as Cost Function.
    """

    def __init__(self, weighting = continuous_step):
        """
        Initialize the cost function with the Fit object.
        """
        self.n_cost = 0
        self.cost_name = "SSQ"
        self.weight_func = weighting #computes weights for each time point

    def compute_cost(self, observed, predicted, times, fraction_reveal=0.5):
        """
        Compute the SSQ cost function for predicted abundances.
        :param observed: Matrix of observed abuncances
        :param predicted: Matrix of predicted abundances
        :param times: Vector of times
        :param fraction_reveal: Controls how much of the time series is emphasized.
        :return: Log of mean SSQ error.
        """
        SSQ = []

        for i in range(len(observed)):
            weights = self.weight_func(times[i], fraction_reveal)
            obs = observed[i]
            pred = predicted[i] / np.sum(predicted[i], axis=1, keepdims=True)

            tmp = (weights[:, np.newaxis] * (obs - pred) ** 2).flatten()
            SSQ.extend(tmp)

        return np.log(np.mean(SSQ))

    def initialize_cost_function_parameters(self):
        #return none since ssq has no parameters
        return None
