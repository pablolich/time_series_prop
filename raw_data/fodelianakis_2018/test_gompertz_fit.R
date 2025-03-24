library(tidyverse)
dt <- read_csv("T37C.csv")
  dxdt <- (lead(dt$E310) - dt$E310) / dt$E310 
dxdt <- dxdt[-length(dxdt)]
dtmk <- dt[-nrow(dt),]
