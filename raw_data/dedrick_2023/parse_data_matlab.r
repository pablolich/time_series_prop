library(dplyr)
library(readr)

# ----------------- Load CSV files -------------------

data_1to1 <- read_csv("growth_data_1to1.csv")
data_1to10 <- read_csv("growth_data_1to10.csv")

# ----------------- Process 1:1 Sa:Sna -------------------

totpoints_1to1 <- nrow(data_1to1)

# Calculate proportions and totals
data_1to1_proc <- data_1to1 %>%
  mutate(
    Total_OD = Sa_OD + Sna_OD,
    Prop_Sa = Sa_OD / Total_OD
  )

# Select from 14th time point onward
data_1to1_list <- list(
  times = data_1to1_proc$Time_hr[14:totpoints_1to1],
  proportions = data_1to1_proc$Prop_Sa[14:totpoints_1to1],
  totals = data_1to1_proc$Total_OD[14:totpoints_1to1]
)

# Shift times to start at 0
data_1to1_list$times <- data_1to1_list$times - min(data_1to1_list$times)

# Save as RData
save(data_1to1_list, file = "data_1to1.RData")

# ----------------- Process 1:10 Sa:Sna -------------------

totpoints_1to10 <- nrow(data_1to10)

# Calculate proportions and totals
data_1to10_proc <- data_1to10 %>%
  mutate(
    Total_OD = Sa_OD + Sna_OD,
    Prop_Sa = Sa_OD / Total_OD
  )

# Select from 14th time point onward
data_1to10_list <- list(
  times = data_1to10_proc$Time_hr[14:totpoints_1to10],
  proportions = data_1to10_proc$Prop_Sa[14:totpoints_1to10],
  totals = data_1to10_proc$Total_OD[14:totpoints_1to10]
)

# Shift times to start at 0
data_1to10_list$times <- data_1to10_list$times - min(data_1to10_list$times)

# Save as RData
save(data_1to10_list, file = "data_1to10.RData")

# ----------------- Done! -------------------

# Quick check
load("data_1to1.RData")
str(data_1to1_list)

load("data_1to10.RData")
str(data_1to10_list)
