library(tidyverse)

#load 
data1 = read.table("hiltunen_1.csv", sep = ",", header = T)
colnames(data1) <- c("time", "a", "b", "c")
data1$replicate = 1
data2 = read.table("hiltunen_2.csv", sep = ",", header = T)
colnames(data2) <- c("time","a", "b", "c")
data2$replicate = 2
data3 = read.table("hiltunen_3.csv", sep = ",", header = T) 
colnames(data3) <- c("time","a", "b", "c")
data3$replicate = 3

#merge 
data_all = rbind(data1, data2, data3) %>% 
  pivot_longer(!c(time, replicate),
               names_to = "species",
               values_to = "abundances")

#plot
ggplot(data_all, aes(x = time,
                     y = abundances,
                     color = as.factor(species)))+
  geom_point()+
  geom_line()+
  facet_wrap(~replicate,
             nrow = 3) + scale_color_manual(values = c("red", "darkgreen", "purple"))


