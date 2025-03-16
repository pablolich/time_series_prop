
#### Mar 16,2025

Simplest optimization rutine finished. Next: find a way for things not to diverge.
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

