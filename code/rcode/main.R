library(tidyverse)
PENALIZATION_ABUNDANCE <- 10^32
DATATYPE <- NULL
source("refine_search.r")
source("warm_start.r")
source("integrate.R")
source("plotting.R")
source("optimization.R")
source("set_initial_conditions.R")
source("find_good_initial_pars.R")
source("save_output.r")

#return the weights given x between 0 and 1
continuous_step <- function(x, lambda, k = 25) {
  #if the transition happens at 1, then return all ones
  if (lambda == 1){
    return(rep(1, length(x)))
  }
  #otherwise return the step function
  else{
    return( 1-1 / (1 + exp(-k * (x - lambda))) )
    }
}

get_weights <- function(times, stop_reveal){
  #transform the current bout number into a number between 0 and 1
  weights = continuous_step(times, stop_reveal)
  return(weights)
}

fit <- NULL

main <- function(input_file, 
                 model_file, 
                 cost_function_file, 
                 optim_bouts = 10,
                 try_initial = 10,
                 reveal = 0,
                 warm_start = FALSE,
                 refine = FALSE,
                 seed = 151){
  load(input_file)
  # we need the cost function first, because it tells us whether we're dealing with proportions
  # or abundances
  source(cost_function_file)
  source(model_file)
  fit <<- output
  set.seed(seed)
  fit$random_seed <<- seed
  # check whether the input already contains parameters
  if (is.null(fit$pars)){
    initialize_model()
    initialize_cost_function()
    fit$pars <<- rep(0, fit$n_initial + fit$n_model + fit$n_cost_function)
    fit$pars[1:fit$n_initial] <<- initialize_initial_conditions(fit)
    initialize_model_parameters()
    initialize_cost_function_parameters()
    
    #
    predicted_abundances <- integrate_dynamics(fit$pars)
    fit$predicted_abundances <<- predicted_abundances
    fit$predicted_proportions <<- lapply(predicted_abundances, function(x) x / rowSums(x))
    plot_predicted_observed()
    plot_predicted_observed_proportions()
    #
    
    find_initial_pars(try_initial)
    
    #once fit is initialized, perform a warm-start if desired to set 
    #parameters close to solution
    if (warm_start){
      warm_pars = warm_start(input_file, 
                             model_file, 
                             cost_function_file)
      fit$pars <<- warm_pars
    }
  }
  predicted_abundances <- integrate_dynamics(fit$pars)
  fit$predicted_abundances <<- predicted_abundances
  fit$predicted_proportions <<- lapply(predicted_abundances, function(x) x / rowSums(x))
  # cost function for initial parameters
  initial_cost <- cost_function(predicted_abundances, 1)
  print(initial_cost)
  # optimize
  new_pars <- fit$pars
  for (z in 1:optim_bouts){
    print(paste("Bout", z))
    #compute and update the weight vector
    ####################################
    if (reveal != 0){
      stop_reveal = z/optim_bouts
    }
    else{stop_reveal = 1}
    
    new_pars <- optim(par = new_pars, fn = to_minimize, stop_reveal = stop_reveal,
                      method = "BFGS",
                      control = list(maxit = 500, trace = FALSE))$par
    
    new_pars <- optim(par = new_pars, fn = to_minimize, stop_reveal = stop_reveal,
                      method = "Nelder-Mead",
                      control = list(maxit = 5000, trace = FALSE))$par
    
    print(to_minimize(new_pars, stop_reveal))
    
    new_pars <- optim_initial(new_pars = new_pars, stop_reveal = stop_reveal)
    print("After initial")    
    print(to_minimize(new_pars, stop_reveal))
    
    new_pars <- optim_model(new_pars = new_pars, stop_reveal = stop_reveal)
    print("After model")    
    print(to_minimize(new_pars, stop_reveal))
    
    # for (zz in 1:(optim_bouts)) new_pars <- optimk(k = 3, new_pars = new_pars, stop_reveal = stop_reveal)
    # print("After 3 at a time")    
    # print(to_minimize(new_pars, stop_reveal))
    # 
    # for (zz in 1:(optim_bouts)) new_pars <- optimk(k = 4, new_pars = new_pars, stop_reveal = stop_reveal)
    # print("After 4 at a time")    
    # print(to_minimize(new_pars, stop_reveal))
    # 
    for (zz in 1:(optim_bouts)) new_pars <- optimk(k = 5, new_pars = new_pars, stop_reveal = stop_reveal)
    print("After 5 at a time")
    print(to_minimize(new_pars, stop_reveal))

    new_pars <- hillclimb(new_pars, stop_reveal)
    print("After HC")    
    print(to_minimize(new_pars, stop_reveal))
    
    predicted_abundances <- integrate_dynamics(new_pars)
    fit$predicted_abundances <<- predicted_abundances
    fit$predicted_proportions <<- lapply(predicted_abundances, function(x) x / rowSums(x))
    fit$cost <<- cost_function(predicted_abundances, stop_reveal)
    fit$pars <<- new_pars
    #print(parse_parameters(new_pars))
    plot_predicted_observed()
    plot_predicted_observed_proportions()
  }
  if (refine){
    refine_search(input_file, model_file, cost_function_file, seed)
    plot_predicted_observed()
    plot_predicted_observed_proportions()
  }
  output <- fit
  #save the output in an organized manner
  save_output(output)
  return(fit)
}

