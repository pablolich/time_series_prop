ODEMETHOD <- "euler"
THRESH <- 10^-16
library(deSolve)

integrate_dynamics <- function(pars){
  # for each time series
  # get the right initial conditions
  # get model pars
  # fit the time series
  pp <- parse_parameters(pars)
  predicted_abundances <- list()
  for (i in 1:fit$n_time_series){
    out <- ode(y = pp$init_conds[[i]], 
               times = fit$times[[i]], 
               func = dxdt, 
               parms = pp, 
               method = ODEMETHOD)
    y <- ((out %>% as.data.frame())[,-1]) %>% as.matrix()
    colnames(y) <- colnames(fit$observed_abundances[[i]])
    # deal with inifinities
    y[is.nan(y)] <- NaN
    y[is.infinite(y)] <- NaN
    y[y > 10^5] <- NaN
    #y[(y > 10^(-15)) & (y < 10^(-6))] <- NaN
    y[(y < -10^(-15))] <- NaN
    y[y < 0] <- NaN
    topen <- sum(is.nan(y))
    y[is.nan(y)] <- PENALIZATION_ABUNDANCE #sample(c(PENALIZATION_ABUNDANCE, 10^-7), topen, replace = TRUE)
    if(topen > 0){
      #print(y)
    }
    predicted_abundances[[i]] <- y
  }
  return(predicted_abundances)
}