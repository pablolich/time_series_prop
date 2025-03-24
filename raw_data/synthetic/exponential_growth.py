import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exp_growth(x0, r, sampling_times, label="x"):
    """
    Simulates exponential growth with gamma-distributed perturbations.
    
    Parameters:
        x0 (list): Initial population sizes.
        r (list): Growth rates.
        sampling_times (array-like): Time points for sampling.
        label (str): Label prefix for populations.

    Returns:
        pd.DataFrame: Data in the format of the input CSV file.
    """
    output = []

    for i, (x0_i, r_i) in enumerate(zip(x0, r), start=1):
        density = x0_i * np.exp(r_i * sampling_times)
        
        # Apply gamma-distributed perturbation
        perturbed_density = np.random.gamma(shape=density, scale=1)
        
        output.append(pd.DataFrame({
            "time": sampling_times,
            f"{i}": perturbed_density  # Column names match the dataset structure
        }))

    # Merge results into a single DataFrame
    result_df = output[0]
    for df in output[1:]:
        result_df = result_df.merge(df, on="time", how="outer")

    return result_df

# Example usage
sampling_times = np.arange(0, 3, 0.08)
x0 = [10, 5, 1]  # Example initial values (can be adjusted)
r = [1, 2, 3]  # Example growth rates (can be adjusted)

# Generate the perturbed exponential growth dataset
exp_growth_df = exp_growth(x0, r, sampling_times)

# Save plot to a CSV file
exp_growth_df.to_csv("../data/exponential_errors_gamma/perturbed_exponential_growth.csv", index=False)

