library(tidyverse)
library(readxl)

THRESH = 1E-10

dt <- read_xlsx(path = "Cocultured_data.xlsx", sheet = 1)
dt <- dt %>% select(Time, Experiment, Replicate, Passage, VC, EC, SA) %>% drop_na()
dt <- dt %>% mutate(RP = paste0(Replicate, "-", Passage)) 
dt <- dt %>% pivot_longer(names_to = "Species", values_to = "Fluorescence", cols = -c(Time, Experiment, RP, Replicate, Passage)) %>% 
  filter(Time > 5)
#eiliminate any period of times with negative fluorescences and shift time accordingly
dt_no_negs <- dt %>% 
  group_by(Passage, Experiment) %>% 
  filter(if (any(Fluorescence < 0)) Time > max(Time[Fluorescence < 0], na.rm = TRUE) else TRUE) %>% 
  mutate(Time = Time - min(Time)) %>% 
  arrange(Time) %>% 
  group_by(Passage, Experiment, Time) %>% 
  filter(!all(Time == 0) & !all(Species == 0)) 
#plot to see all datasets
dt_no_negs %>% ggplot(aes(x = Time, y = Fluorescence, group = interaction(RP, Species), colour = Species)) + 
  geom_point() + geom_line() + facet_grid(Passage~Experiment)+ scale_y_log10()

kcomms <- unique(dt$Experiment)
pssgs <- unique(dt_no_negs$Passage)
for (cc in comms){
  for (rr in 1:3){
    for(pp in pssgs){
      tmp <- dt_no_negs %>% filter(Experiment == cc, 
                                   Replicate == rr,
                                   Passage == pp) %>% 
        pivot_wider(names_from = Species, 
                    values_from = Fluorescence) %>% 
        ungroup() %>% 
        select(Time, VC, EC, SA)
      write_csv(tmp, file = paste0("../../data/davis/", cc, "_", 
                                   rr, "_",
                                   pp, ".csv")) 
    }
  }
}
