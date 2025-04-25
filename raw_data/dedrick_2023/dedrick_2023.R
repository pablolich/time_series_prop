# Install and load required package for reading Excel files
install.packages("readxl", dependencies = TRUE)  # if not already installed
library(readxl)
library(tidyr)
library(dplyr)
library(ggplot2)

# Define a helper function to process one co-culture dataset
process_coculture <- function(sheet_name, initial_ratio_Sa_to_KPL) {
  # Read the sheet for the given condition
  data <- read_excel("Compiled Growth Curves.xlsx", sheet = sheet_name)
  # The sheet is assumed to have columns: Time, OD and GFP for each replicate.
  # e.g., Time, OD1, OD2, ... OD6, GFP1, GFP2, ... GFP6 (for 6 replicates).
  
  # Pivot the data longer for easier handling
  data_long <- data %>%
    pivot_longer(cols = -Time, 
                 names_to = c("Measure", "Replicate"), 
                 names_pattern = "([A-Za-z]+)(\\d+)",
                 values_to = "Value")
  # This creates a long table with columns: Time, Measure (OD or GFP), Replicate, Value.
  
  # Split into separate OD and GFP data frames and then recombine by time+replicate
  od_long  <- filter(data_long, Measure == "OD")
  gfp_long <- filter(data_long, Measure == "GFP")
  merged   <- inner_join(od_long, gfp_long, by = c("Time", "Replicate"), suffix = c("_OD", "_GFP"))
  # Now 'merged' has columns: Time, Replicate, Measure_OD, Value_OD, Measure_GFP, Value_GFP for each time & replicate.
  
  # Calculate fraction of total OD that is S. aureus initially, based on the initial ratio.
  # initial_ratio_Sa_to_KPL is like 1/1, 1/10, 1/100 (as numeric ratio of Sa:KPL).
  # First, compute S. aureus fraction of total = Sa / (Sa + KPL).
  frac_Sa <- initial_ratio_Sa_to_KPL / (1 + initial_ratio_Sa_to_KPL)
  
  # Calibrate GFP to OD for S. aureus using the first time point of each replicate
  calibrated <- merged %>%
    group_by(Replicate) %>%
    mutate(
      # Determine S. aureus OD at the first time point (Time == 0 or the earliest reading)
      Sa_initial_OD = frac_Sa * first(Value_OD),         # S. aureus portion of total OD at t0
      GFP_initial   = first(Value_GFP),                  # GFP reading at t0
      calib_factor  = Sa_initial_OD / ifelse(GFP_initial > 0, GFP_initial, NA),
      # Use calib_factor to compute S. aureus OD at all time points from GFP, 
      # and KPL1850 OD as the remainder of total OD.
      Sa_OD     = calib_factor * Value_GFP,
      KPL1850_OD = Value_OD - Sa_OD
    ) %>%
    ungroup()
  
  # Prepare data for plotting: gather Sa and KPL1850 into one column
  plot_data <- calibrated %>%
    select(Time, Replicate, Sa_OD, KPL1850_OD) %>%
    pivot_longer(cols = c(Sa_OD, KPL1850_OD), names_to = "Species", values_to = "OD") %>%
    mutate(Species = ifelse(Species == "Sa_OD", "S. aureus", "KPL1850"))
  
  return(plot_data)
}

# Process each coculture condition
data_1to1   <- process_coculture(sheet_name = "1to1",   initial_ratio_Sa_to_KPL = 1/1)
data_1to10  <- process_coculture(sheet_name = "1to10",  initial_ratio_Sa_to_KPL = 1/10)
data_1to100 <- process_coculture(sheet_name = "1to100", initial_ratio_Sa_to_KPL = 1/100)

# Plotting function for a given dataset and title
plot_coculture <- function(plot_data, title) {
  ggplot(plot_data, aes(x = Time, y = OD, color = Species, 
                        group = interaction(Species, Replicate))) +
    geom_line(alpha = 0.8) +
    labs(title = title, x = "Time (hours)", y = "OD600") +
    theme_minimal()
}

# Generate the plots for each condition
plot_1to1   <- plot_coculture(data_1to1,   "Co-culture 1:1 (S. aureus : KPL1850)")
plot_1to10  <- plot_coculture(data_1to10,  "Co-culture 1:10 (S. aureus : KPL1850)")
plot_1to100 <- plot_coculture(data_1to100, "Co-culture 1:100 (S. aureus : KPL1850)")

# Display the plots (each showing both species growth curves with all replicates)
print(plot_1to1)
print(plot_1to10)
print(plot_1to100)
