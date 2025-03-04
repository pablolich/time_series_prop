#this script initializes the fit object

library(tidyverse)

build_dataset <- function(fn, output_file){
  # read the files
  observed_abundances <- list()
  times <- list()
  for (k in 1:length(fn)) {
    tmp <- read_csv(fn[k]) %>% as.data.frame()
    # separate time
    times[[k]] <- as.numeric(tmp[,1])
    observed_abundances[[k]] <- as.matrix(tmp[,-1])
  }
  
  # normalize time: the minimum time is 0 and the maximum is 1
  maxtime <- max(unlist(times))
  mintime <- min(unlist(times))
  times <- lapply(times, function(x) (x - mintime) / (maxtime - mintime))
  
  # normalize abundances such that the total at t = 0 of the first time series is 1
  Tot <- sum(observed_abundances[[1]][1,])
  observed_abundances <- lapply(observed_abundances, function(x)  x / Tot)
  observed_proportions <- lapply(observed_abundances, function(x)  x / rowSums(x))
  
  
  # build object
  n <- ncol(observed_abundances[[1]])
  n_time_series <- length(observed_abundances)
  fit <- list(
    type_of_inference = NULL,
    observed_abundances = observed_abundances,
    observed_proportions = observed_proportions,
    times = times,
    predicted_abundances = NULL,
    predicted_proportions = NULL,
    n =  n,
    n_time_series = n_time_series,
    n_initial = n * n_time_series,
    n_model = NULL,
    n_cost_function = NULL,
    model_name = NULL,
    cost_function_name = NULL,
    cost = NULL,
    pars = NULL,
    set_true_zeros = (unlist(lapply(observed_abundances, function(x) x[1,])) > 0) * 1,
    file_names = fn,
    output_name = tools::file_path_sans_ext(basename(output_file)),
    random_seed = 0
  )
  
  
  output <- fit
  
  # save
  save(output, file = output_file)
  
}

build_dataset(c("data/Davis/VES_1.csv", 
                "data/Davis/ES_1.csv", 
                "data/Davis/VE_1.csv", 
                "data/Davis/VS_1.csv"), "compiled_data/Davis_1.RData")

