import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Deterministic exponential growth function
def exp_growth(x0, r, times):
    data = []
    n = len(x0)
    for i in range(n):
        density = x0[i] * np.exp(r[i] * times)
        df = pd.DataFrame({
            "species": f"x{i+1}",
            "time": times,
            "density": density
        })
        data.append(df)
    return pd.concat(data, ignore_index=True)

# 4. Parameters and execution
true_x0 = [10, 20]
true_r = [2/3, -1/3]
times = np.arange(0, 5.1, 0.02)
theta = 5.0  # Higher theta → lower noise

# 5. Generate and sample
true_trajectories = exp_growth(true_x0, true_r, times)
