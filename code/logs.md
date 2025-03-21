# TODOS
--------------------------------------------------------------------
3. Prepare functionality to make it easy to do meta fitting with a bunch of models and datasets.
4. Implement jax here
5. Powers of ssq
7. Find datasets and models that can describe the dataset. Maybe the models are very weird
8. When plotting, rescale all plots by initial conditions of observed

PROBLEMS WITH DATA
1. Discovering the models
2. Data transformations that help?
3. Which data set should we concentrate on?

# QUESTIONS
---------------------------------------------------------------------
10. Is how I am dealing with divergences (seting in proportions 1/n each) a good way to deal with that?

--------------------------------------------------------------------
#### Mar 17,2025

I have finished the basic pipeline, and orgainzed things in folders

#### Mar 16,2025

Simplest optimization rutine finished. Next: find a way for things not to diverge. That is, do an optimization protocol.
Questions for Stefano: 
1. what is ix_ode, and also, what is the role of WEIGHTSin the cost function
2. How to penalize for divergences in integration

#### Mar 16,2025

Finished coding forward integration and cost calculation. 
Todos are (1) write the simplest optimization routine, (2) come up with a code that searches for initial conditions, (3) implement saving of results, and possible recyling on new round of fitting, and (4) separate model between intraspecific interactions versus interspecific interactions

#### Mar 08,2025

Integration working, figure out how to handle zero species in the observations. Almost there in the integration

#### Mar 06,2025

Finish seting up Fit object by inheriting all classes. Then integrate, optimize, and we're gucci

#### Mar 04,2025

Model is not changing the parameters of fit. think of a way to build model without fit, and build fit at the end with data, model and cost function

