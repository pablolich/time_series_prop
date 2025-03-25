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
parsed <- rbind(parsed, get_tb_all(c(12,32,52), "LP", "AO-LP") ) #removed 8 because its paired with 7

parsed <- rbind(parsed, get_tb_all(c(13,33,53,17), "AO", "AO-LB") )
parsed <- rbind(parsed, get_tb_all(c(14,34,54,18), "LB", "AO-LB") )

parsed <- rbind(parsed, get_tb_all(c(15,35,55,27), "LP", "LP-LB") )
parsed <- rbind(parsed, get_tb_all(c(16,36,56,28), "LB", "LP-LB") )

parsed <- rbind(parsed, get_tb_all(c(1,21,41), "AO", "AO") )
parsed <- rbind(parsed, get_tb_all(c(5,25,45), "LP", "LP") )
parsed <- rbind(parsed, get_tb_all(c(4,24,44), "LB", "LB") )

parsed <- parsed %>% 
mutate(pair = case_when(
  replicate %in% c("B1", "B2") ~ "Rep1",
  replicate %in% c("D1", "D2") ~ "Rep2",
  replicate %in% c("F1", "F2") ~ "Rep3",
  replicate %in% c("B3", "B4") ~ "Rep4",
  replicate %in% c("D3", "D4") ~ "Rep5",
  replicate %in% c("F3", "F4") ~ "Rep6",
  replicate %in% c("B5", "B6") ~ "Rep7",
  replicate %in% c("D5", "D6") ~ "Rep8",
  replicate %in% c("F5", "F6") ~ "Rep9",
  replicate %in% c("B7", "B8") ~ "Rep10",
  replicate %in% c("C7", "C8") ~ "Rep11",
  replicate %in% c("A1", "A5") ~ "Rep12",
  replicate %in% c("C1", "C5") ~ "Rep13",
  replicate %in% c("E1", "E5") ~ "Rep14",
  replicate %in% c("A4", "A5") ~ "Rep15",
  replicate %in% c("C4", "C5") ~ "Rep16",
  replicate %in% c("E4", "E5") ~ "Rep17",
  TRUE ~ "Unpaired"
)) %>% 
  select(-replicate)


# rarefy the data
timepoints <- c(seq(2, 96, by = 0.25))
parsed <- parsed %>% filter(time %in% timepoints)


#plot
ggplot(parsed, 
       aes(x = time,
           y = density,
           color = population))+
  geom_line(aes(group = interaction(population, pair)))+
  facet_wrap(~community)

parsed_aolb = parsed %>% filter(community == "AO-LB",
                                pair == "Rep4") %>% 
  pivot_wider(names_from = population,
              values_from = density)

species = c("AO", "LB", "LP")
  
for (cm in sort(unique(parsed$community))){
  p1 <- parsed %>% filter(community == cm)
  for (rp in unique(p1$pair)){
    p2 = p1 %>% filter(pair==rp) %>% 
      pivot_wider(names_from = population,
                  values_from = density, 
                  values_fill = 0) %>% 
      select(-c(community, pair))
    write_csv(p2, file = paste0("../../data/jo/", cm, "_", rp, ".csv"))
  }
}
