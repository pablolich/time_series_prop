import numpy as np

class SSQCostFunction:
    """
    Sum of Squared Differences (SSQ) for Proportions as Cost Function.
    """

    DATATYPE = "proportions"  # Inference type for this cost function

    def __init__(self, fit):
        """
        Initialize the cost function with the Fit object.
        :param fit: Fit object containing observed data and predictions.
        """
        self.fit = fit
        self.fit.n_cost_function = 0
        self.fit.cost_function_name = "SSQ_prop"
        self.fit.type_of_inference = self.DATATYPE

    def compute_cost(self, predicted, fraction_reveal=0.5):
        """
        Compute the SSQ cost function for predicted abundances.
        :param predicted: List of predicted abundance matrices
        :param fraction_reveal: Controls how much of the time series is emphasized.
        :return: Log of mean SSQ error.
        """
        SSQ = []

        for i in range(len(self.fit.observed_proportions)):
            weights = continuous_step(self.fit.times[i], fraction_reveal)
            obs = self.fit.observed_proportions[i]
            pred = predicted[i] / np.sum(predicted[i], axis=1, keepdims=True)

            tmp = (weights * (obs - pred) ** 2).flatten()
            SSQ.extend(tmp)

        return np.log(np.mean(SSQ))

    def initialize_cost_function_parameters(self):
        #return none since ssq has no parameters
        return None
