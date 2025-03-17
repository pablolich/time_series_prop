import numpy as np

class SSQCostFunction:
    """
    Sum of Squared Differences (SSQ) for Proportions as Cost Function.
    """

    def __init__(self):
        """
        Initialize the cost function.
        """
        self.n_cost = 0
        self.cost_name = "SSQ"

    def compute_cost(self, observed, predicted, times, weight=None):
        """
        Compute the SSQ cost function for predicted abundances.
        :param observed: List of matrices of observed abundances
        :param predicted: List of matrices of predicted abundances
        :param times: List of vectors representing times
        :param weight: Optional weighting factor, based on time (for each time point)
        :return: Log of mean SSQ error.
        """
        SSQ = []

        for i in range(len(observed)):
            obs = observed[i]
            pred = predicted[i] / np.sum(predicted[i], axis=1, keepdims=True)  # Normalize predictions

            # Calculate sum of squared differences
            diff = (obs - pred) ** 2
            weighted_diff = diff

            if weight is not None:
                # Apply the weight based on time for the current time series
                weighted_diff *= np.exp(-weight * times[i])

            # Flatten the weighted squared differences and add them to SSQ list
            SSQ.extend(weighted_diff.flatten())

        return np.log(np.mean(SSQ))

    def initialize_cost_function_parameters(self):
        # Return None since SSQ has no parameters
        return None

