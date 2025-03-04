import numpy as np

def continuous_step(x, lambd, k=25):
    """
    Returns the weights given x between 0 and 1.

    :param x: Array-like, values between 0 and 1.
    :param lambd: Transition point.
    :param k: Steepness of transition (default: 25).
    :return: Array of weights.
    """
    x = np.array(x)  # Ensure x is a NumPy array

    # If transition happens at 1, return all ones
    if lambd == 1:
        return np.ones_like(x)
    
    # Compute the step function
    return 1 - 1 / (1 + np.exp(-k * (x - lambd)))

