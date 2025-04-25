# Load necessary libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(readr)

# Read all lines from the raw text file
lines <- read_lines("Cocultures_GFPSa_1850_1821_MC_Day1_29JUL2019.txt")

# Identify lines that start a new time point
time_lines <- grep("^Time \\d+ \\(", lines)
time_points <- str_extract(lines[time_lines], "\\d+:\\d+:\\d+")

# Function to parse a single time block into a data frame
parse_time_block <- function(start_line, end_line, time_str) {
  block_lines <- lines[(start_line + 1):(end_line - 1)]
  matrix_data <- do.call(rbind, lapply(block_lines, function(l) str_split(l, "\\s+")[[1]]))
  df <- as.data.frame(matrix_data, stringsAsFactors = FALSE)
  df <- df %>%
    mutate(Row = LETTERS[1:n()]) %>%
    pivot_longer(cols = -Row, names_to = "Col", values_to = "OD") %>%
    mutate(
      Well = paste0(Row, Col),
      Time = time_str,
      OD = as.numeric(OD)
    ) %>%
    select(Time, Well, OD)
  return(df)
}

# Combine all time blocks into a single dataframe
od_data <- bind_rows(lapply(seq_along(time_lines), function(i) {
  start <- time_lines[i]
  end <- if (i < length(time_lines)) time_lines[i + 1] else length(lines)
  parse_time_block(start, end, time_points[i])
}))

# Convert Time to minutes since start
od_data <- od_data %>%
  mutate(
    Time_min = as.numeric(hms::as_hms(Time)) / 60
  )

# Example: Define well groupings based on plate layout (you'll adjust this)
# This part depends on how wells map to coculture ratios
group_map <- data.frame(
  Well = c("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"),  # Example wells
  Group = c("1:1", "1:1", "1:10", "1:10", "1:100", "1:100", "control", "control")
)
od_data <- left_join(od_data, group_map, by = "Well")

# Plot OD over time for each group
ggplot(filter(od_data, !is.na(Group)), aes(x = Time_min, y = OD, color = Well)) +
  geom_line() +
  facet_wrap(~ Group) +
  labs(title = "Growth Curves (OD600) by Coculture Ratio",
       x = "Time (minutes)", y = "OD600") +
  theme_minimal()
