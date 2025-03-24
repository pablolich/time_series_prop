library(readxl)
library(tidyverse)

get_tb <- function(colnum, pop, community){
  density <- rowSums(dt[,colnum + 1])
  return(tibble(time = dt$Time, population = pop, value = density, community = community))
}

get_tb_all = function(colnum, pop, comm){
  densities = dt[,colnum+1]
  select_dt = tibble(time = dt$Time, population = pop, community = comm)
  select_dt = cbind(select_dt, densities)
  
  return(select_dt %>% pivot_longer(-c(time, population, community),
                                    values_to = "density",
                                    names_to = "replicate"))
}
# AO ismember(ind,[1,11,21,31,41,51,13,33,53,7,17,37,47,57,38,48,58])
# LP ismember(ind,[12,32,52,5,15,25,35,45,55,27,8,9,19,29,10,20,30])
# LB ismember(ind,[4,14,24,34,44,54,16,36,56,18,28,39,49,59,40,50,60])

dt <- read_xlsx("Figure 3 -- DM Gut Microbiome CoC.xlsx", range = "B27:BK412")
dt$Time <- seq(0,96,by=0.25)
dt <- dt %>% select(-`T° 600`)
parsed <- tibble()

parsed <- get_tb_all(c(11,31,51), "AO", "AO-LP") # removed 7 because it is very different
parsed <- rbind(parsed, get_tb_all(c(12,32,52,8), "LP", "AO-LP") )

parsed <- rbind(parsed, get_tb_all(c(13,33,53,17), "AO", "AO-LB") )
parsed <- rbind(parsed, get_tb_all(c(14,34,54,18), "LB", "AO-LB") )

parsed <- rbind(parsed, get_tb_all(c(15,35,55,27), "LP", "LP-LB") )
parsed <- rbind(parsed, get_tb_all(c(16,36,56,28), "LB", "LP-LB") )

parsed <- rbind(parsed, get_tb_all(c(1,21,41), "AO", "AO") )
parsed <- rbind(parsed, get_tb_all(c(5,25,45), "LP", "LP") )
parsed <- rbind(parsed, get_tb_all(c(4,24,44), "LB", "LB") )

# rarefy the data
timepoints <- c(seq(2, 96, by = 0.25))
parsed <- parsed %>% filter(time %in% timepoints)

#plot
ggplot(parsed, 
       aes(x = time,
           y = density,
           color = population))+
  geom_line(aes(group = interaction(population, replicate)))+
  facet_wrap(~community)

parsed <- parsed %>% pivot_wider(names_from = population, 
                                 values_from = density, 
                                 values_fill = 0)

for (cm in sort(unique(parsed$community))){
  p1 <- parsed %>% filter(community == cm) 
  for (rp in unique(p1$replicate)){
    p2 = p1 %>% filter(replicate==rp) %>% select(-c(community, replicate))
    write_csv(p2, file = paste0("../../data/jo/", cm, "_", rp, ".csv"))
  }
}
