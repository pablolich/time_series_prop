library(tidyverse)

dt1 <- read_csv("T25C.csv")
dt1$Temp = 25

dt2 <- read_csv("T37C.csv")
dt2$Temp = 37

dt3 <- read_csv("T42C.csv")
dt3$Temp = 42

dt = rbind(dt1, dt2, dt3) %>% 
  pivot_longer(-c(Time, Temp),
              names_to = "species",
               values_to = "density")
  

ggplot(dt, 
       aes(x = Time, 
       y = density,
       ))+
  geom_point(aes(color = species))+
  facet_wrap(~Temp, nrow = 3)+
  scale_y_log10()
