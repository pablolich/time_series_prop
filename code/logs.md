
#### Apr 21,2025

Tasks for tomorrow

2. Think of smoothing as a way of denoising, ask Peter Lu
3. Model error function with dirichlet distribution. Start with log ration, then do dirichlet
3. Log ratio as cost function try on perfect chaotic
4. Perform parameter refinement by fitting groups of parameters at a time (i.e. initial conditions only, model paramters only, model parameter groups only, etc). 
5. Code general polynomial models 
6. Try on real data
7. Back to encoder hidden state recovery: add a dynamic constrain such that the derivatives from the totals as obtained through the encoder must match the equation for the totals, that is, treat the output of the encoder as an "observation" that must follow also the symbolic model.

#### Apr 20,2025

I have implemented jax-like structure for traditional pipeline. Able to fit lotka volterra perfectly, and can also recover parameters with some noise. 
Next steps: 
1. Fit datasets that are more difficult --Chaos
2. smooth datasets with a neural ode smoother or something this will allow a better fit with model
3. Code a pipeline to fit any polynomial model. If wanted, can add sparisity
4. Possibly go to sindy in jax
5. Try real data
6. Add logratio as cost function 



# TODOS
--------------------------------------------------------------------
1. Prepare functionality to make it easy to do meta fitting with a bunch of models and datasets.
2. Implement jax here
3. Powers of ssq
4. Find datasets and models that can describe the dataset. Maybe the models are very weird
5. When plotting, rescale all plots by initial conditions of observed
6. Make aux_optimization.py a class of its own that will be inherited by fit. In this way its then easy to make optimization protocols with a few clicks.
7. Prepare data for folks: Jo and Davis. For Jo, don't average data, put all replicates (except 7). For Davis, divide by the blank measurements. Generate clean data for our data folder, and create a new folder called raw_data where I put each data set folder, and inside I drop the paper, the original csv file, and the code in R to parse it. 
8. Add functionality to plot the data sets.add a method to the data object
9. Add functionality to change integration method: use euler if data doesn't vary too much. If RK45 is used, then interrupt integration when diverging. 
10. Profile the code


PROBLEMS WITH DATA
1. Discovering the models
2. Data transformations that help?
3. Which data set should we concentrate on?

# QUESTIONS
---------------------------------------------------------------------
10. Is how I am dealing with divergences (seting in proportions 1/n each) a good way to deal with that?

--------------------------------------------------------------------

#### Apr 08,2025

I have to clean up symbolic regression to fully integrate it with current pipeline
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

