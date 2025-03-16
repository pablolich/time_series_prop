import numpy as np

THRESH = 1e-6  # A small threshold to avoid log(0) error

class LogDistCostFunction:
    """
    Logarithmic Distance (LOGDIST) for Proportions as Cost Function.
    """

    def __init__(self):
        """
        Initialize the cost function with the Fit object.
        :param weighting: Optional weighting function based on observed time.
        """
        self.n_cost = 0
        self.cost_name = "LOGDIST"

    def compute_cost(self, observed, predicted, times, weight=None):
        """
        Compute the LOGDIST cost function for predicted abundances.
        :param fit: The Fit object containing observed and predicted data.
        :param weight: Optional weighting factor, based on time (for each time point).
        :return: Total goal function value.
        """
        goal = 0

        for i in range(len(observed)):
            # Extract predicted and observed proportions for the ith time series
            pred = predicted[i]
            obs = observed[i]
            
            # Compute the log ratio
            ratio = np.log(pred + THRESH) - np.log(obs + THRESH)
            goal_rs = np.sum(np.abs(ratio), axis=1)  # Sum the absolute values row-wise
            
            if weight is not None:
                # Apply the weight if specified
                goal_rs *= np.exp(-weight * times[i])

            # Add to the overall goal function
            goal += np.sum(goal_rs)

        return goal

    def initialize_cost_function_parameters(self):
        # Return None since LOGDIST has no parameters
        return None

    def initialize_goal_pars(self):
        # Initialize goal parameters (no parameters for LOGDIST)
        return []

