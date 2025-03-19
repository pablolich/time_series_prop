import numpy as np
from scipy.stats import dirichlet

class DirichletFunction:
    """
    Sum of Squared Differences (SSQ) for Proportions as Cost Function.
    """

    def __init__(self, dim):
        """
        Initialize the cost function.
        :param dim: number of species in the data set
        """
        self.n_cost = dim
        self.cost_name = "dirichlet"

    def compute_cost(self, observed, predicted, times, pars,  obs_type="prop", weight=None):
        """
        Compute the SSQ cost function for predicted abundances.
        :param observed: List of matrices of observed proportions
        :param predicted: List of matrices of predicted abundances
        :param times: List of vectors representing times
        :param weight: Optional weighting factor, based on time (for each time point)
        :return: Log of mean SSQ error.
        """
        likelihood = []
        #get weights to transform proportions & abundances
        pars_dict = self.parse_cost_function_parameters(pars)
        w = abs(pars_dict["w"])
        #new abundances
        for i in range(len(observed)):
            obs = observed[i]*w / np.sum(observed[i]*w, axis=1, keepdims=True)  #multiply each column by the w
            if obs_type == "prop":
                #transform predictions to proportions
                pred = predicted[i]*w / np.sum(predicted[i]*w, axis=1, keepdims=True) 
            else:
                #cant apply dirichlet if non-compositional data is observed
                raise Exception(self.cost_name, 
                                " is not a valid cost function if ", 
                                obs_type, 
                                " are observed")

            # Calculate likelihood for each time point
            #print(obs)
            lik_i_vec = np.zeros(len(pred))
            for j in range(len(pred)):
                try:
                    lik_i_vec[j] = dirichlet.logpdf(pred[j], obs[j]) 
                except:
                    import ipdb; ipdb.set_trace(context = 20)
            #lik_i_vec = np.array([dirichlet.logpdf(pred[i], obs[i]) for \
                    #i in range(len(pred))])
            weighted_lik_i_vec = lik_i_vec

            if weight is not None:
                # Apply the weight based on time for the current time series
                weighted_lik_i_vec *= np.exp(-weight * times[i])

            # Flatten the weighted squared differences and add them to SSQ list
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
