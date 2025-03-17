import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_res_prop(fit):
    nts = fit.data.n_time_series
    dt = []
    
    for i in range(nts):
        tmp = pd.DataFrame(fit.data.observed_proportions[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'proportion'
        tmp['state'] = 'observed'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_proportions[i][0, :] > 0]
        )
        dt.append(tmp)
        
        tmp = pd.DataFrame(fit.predicted_proportions[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'proportion'
        tmp['state'] = 'predicted'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_proportions[i][0, :] > 0]
        )
        dt.append(tmp)
    
    dt = pd.concat(dt, ignore_index=True).melt(
        id_vars=['time', 'type', 'state', 'time_series', 'community'], 
        var_name='species', 
        value_name='x'
    )
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dt, x='time', y='x', hue='species', style='state')
    #plt.yscale('sqrt')
    plt.title('Proportion Trends')
    plt.show()
    
def plot_res_abundances(fit):
    nts = fit.data.n_time_series
    dt = []
    
    for i in range(nts):
        tmp = pd.DataFrame(fit.data.observed_abundances[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'abundance'
        tmp['state'] = 'unobserved'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_abundances[i][0, :] > 0]
        )
        dt.append(tmp)
        
        tmp = pd.DataFrame(fit.predicted_abundances[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'abundance'
        tmp['state'] = 'predicted'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_abundances[i][0, :] > 0]
        )
        dt.append(tmp)
    
    dt = pd.concat(dt, ignore_index=True).melt(
        id_vars=['time', 'type', 'state', 'time_series', 'community'], 
        var_name='species', 
        value_name='x'
    )
    
    dt['x'] = dt.groupby(['state', 'time_series'])['x'].transform(lambda x: x / x.mean())
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dt, x='time', y='x', hue='species', style='state')
    plt.title('Abundance Trends')
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_res_combined(fit):
    nts = fit.data.n_time_series
    dt_prop = []
    dt_abund = []
    
    for i in range(nts):
        # Proportions
        tmp = pd.DataFrame(fit.data.observed_proportions[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'proportion'
        tmp['state'] = 'observed'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_proportions[i][0, :] > 0]
        )
        dt_prop.append(tmp)
        
        tmp = pd.DataFrame(fit.predicted_proportions[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'proportion'
        tmp['state'] = 'predicted'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_proportions[i][0, :] > 0]
        )
        dt_prop.append(tmp)
        
        # Abundances
        tmp = pd.DataFrame(fit.data.observed_abundances[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'abundance'
        tmp['state'] = 'unobserved'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_abundances[i][0, :] > 0]
        )
        dt_abund.append(tmp)
        
        tmp = pd.DataFrame(fit.predicted_abundances[i])
        tmp['time'] = fit.data.times[i]
        tmp['type'] = 'abundance'
        tmp['state'] = 'predicted'
        tmp['time_series'] = i + 1
        tmp['community'] = '-'.join(
            np.array(fit.data.pop_names)[fit.data.observed_abundances[i][0, :] > 0]
        )
        dt_abund.append(tmp)
    
    dt_prop = pd.concat(dt_prop, ignore_index=True).melt(
        id_vars=['time', 'type', 'state', 'time_series', 'community'], 
        var_name='species', 
        value_name='x'
    )
    
    dt_abund = pd.concat(dt_abund, ignore_index=True).melt(
        id_vars=['time', 'type', 'state', 'time_series', 'community'], 
        var_name='species', 
        value_name='x'
    )
    
    dt_abund['x'] = dt_abund.groupby(['state', 'time_series'])['x'].transform(lambda x: x / x.mean())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    
    # Proportions plot
    sns.lineplot(data=dt_prop, x='time', y='x', hue='species', style='state', ax=axes[0])
    axes[0].set_title('Proportion Trends')
    
    # Abundances plot
    sns.lineplot(data=dt_abund, x='time', y='x', hue='species', style='state', ax=axes[1])
    axes[1].set_title('Abundance Trends')
    
    plt.tight_layout()
    plt.show()

