library(readxl)
library(tidyverse)

get_tb <- function(colnum, pop, community){
  density <- rowSums(dt[,colnum + 1])
  return(tibble(time = dt$Time, population = pop, value = density, community = community))
}

# AO ismember(ind,[1,11,21,31,41,51,13,33,53,7,17,37,47,57,38,48,58])
# LP ismember(ind,[12,32,52,5,15,25,35,45,55,27,8,9,19,29,10,20,30])
# LB ismember(ind,[4,14,24,34,44,54,16,36,56,18,28,39,49,59,40,50,60])

dt <- read_xlsx("RawData/Figure 3 -- DM Gut Microbiome CoC.xlsx", range = "B27:BK412")
dt$Time <- seq(0,96,by=0.25)
dt <- dt %>% select(-`T° 600`)
parsed <- tibble()
parsed <- get_tb(c(11,31,51), "AO", "AO-LP") # removed 7 because it is very different
parsed <- rbind(parsed, get_tb(c(13,33,53,17), "AO", "AO-LB") )
parsed <- rbind(parsed, get_tb(c(15,35,55,27), "LP", "LP-LB") )
parsed <- rbind(parsed, get_tb(c(12,32,52,8), "LP", "AO-LP") )
parsed <- rbind(parsed, get_tb(c(14,34,54,18), "LB", "AO-LB") )
parsed <- rbind(parsed, get_tb(c(16,36,56,28), "LB", "LP-LB") )
parsed <- rbind(parsed, get_tb(c(1,21,41), "AO", "AO") )
parsed <- rbind(parsed, get_tb(c(5,25,45), "LP", "LP") )
parsed <- rbind(parsed, get_tb(c(4,24,44), "LB", "LB") )

# rarefy the data
timepoints <- c(seq(2, 96, by = 0.25))
parsed <- parsed %>% filter(time %in% timepoints)

parsed <- parsed %>% pivot_wider(names_from = population, values_from = value, values_fill = 0)
for (cm in sort(unique(parsed$community))){
  p1 <- parsed %>% filter(community == cm) %>% select(-community)
  write_csv(p1, file = paste0(cm, ".csv"))
}

aolb <- parsed %>% filter(community == "AO-LB") %>% select(-LP) %>% select(-community)
write_csv(aolb, file = "AO-LB_2.csv")

aolp <- parsed %>% filter(community == "AO-LP") %>% select(-LB) %>% select(-community)
write_csv(aolp, file = "AO-LP_2.csv")

lblp <- parsed %>% filter(community == "LP-LB") %>% select(-AO) %>% select(-community)
write_csv(lblp, file = "LP-LB_2.csv")