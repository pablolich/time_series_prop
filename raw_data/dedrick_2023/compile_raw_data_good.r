library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(readr)
library(hms)

# Read all lines
lines <- read_lines("Cocultures_GFPSa_1850_1821_MC_Day1_29JUL2019.txt")

# Find time headers
time_lines <- grep("^Time \\d+ \\(", lines)
time_numbers <- as.numeric(str_extract(lines[time_lines], "(?<=Time )\\d+"))

# Identify blocks
od_idx <- 1:130        # OD600
gfp_idx <- 146:290     # GFP

# Helper to parse a block
parse_time_block <- function(start_line, end_line, time_number) {
  block_lines <- lines[(start_line + 1):(end_line - 1)]
  block_lines <- block_lines[nzchar(block_lines)]  # remove empty lines
  matrix_data <- do.call(rbind, lapply(block_lines, function(l) str_split(str_trim(l), "\\s+")[[1]]))
  df <- as.data.frame(matrix_data, stringsAsFactors = FALSE)
  colnames(df) <- as.character(1:ncol(df))
  df <- df %>%
    mutate(Row = LETTERS[1:n()]) %>%
    pivot_longer(cols = -Row, names_to = "Col", values_to = "Value") %>%
    mutate(
      Well = paste0(Row, Col),
      Time_number = time_number,
      Value = as.numeric(Value)
    ) %>%
    select(Time_number, Well, Value)
  return(df)
}

# Function to compute OD600 from GFP Fluorescence
fluorescence_to_od600 <- function(F) {
  a <- 0.0283
  b <- 5.20e-6 * (100 - F)
  
  ifelse(
    F < 3000,
    2.97e-5 * (F + 200),
    -a + sqrt(a^2 - b)
  )
}

f_vec = seq(0, 2e4, 10)
od_vec = fluorescence_to_od600(f_vec)
plot(od_vec, f_vec, type = 'b')

# Define correction functions
correct_od600_sa <- function(OD) {
  ifelse(OD > 0.5, (2.7 * OD) / (2.7 - OD), OD)
}

correct_od600_kpl <- function(OD) {
  ifelse(OD > 0.5, (2.5 * OD) / (2.5 - OD), OD)
}


# Parse OD600
od_data <- bind_rows(lapply(od_idx, function(i) {
  start <- time_lines[i]
  end <- if (i < length(time_lines)) time_lines[i + 1] else length(lines)
  parse_time_block(start, end, time_numbers[i])  # use time_numbers[i] directly
})) %>%
  mutate(Measurement = "OD600")

# Parse GFP
gfp_data <- bind_rows(lapply(gfp_idx, function(i) {
  start <- time_lines[i]
  end <- if (i < length(time_lines)) time_lines[i + 1] else length(lines)
  parse_time_block(start, end, time_numbers[i])  # use time_numbers[i] directly
})) %>%
  mutate(Measurement = "GFP")

# Combine
combined_data <- bind_rows(od_data, gfp_data)

# Map wells to experimental conditions
group_map <- data.frame(
  Well = c(
    "A5", "B5", "C5", "D5", "E5", "F5",
    "A6", "B6", "C6", "D6", "E6", "F6",
    "A7", "B7", "C7", "D7", "E7", "F7"
  ),
  Condition = c(
    rep("Sa:Sna 1:1", 6),
    rep("Sa:Sna 1:10", 6),
    rep("Sa:Sna 1:100", 6)
  )
)

# Merge conditions
combined_data <- left_join(combined_data, group_map, by = "Well")

# Split OD600 and GFP
od_cocultures <- combined_data %>%
  filter(Measurement == "OD600", !is.na(Condition))

gfp_cocultures <- combined_data %>%
  filter(Measurement == "GFP", !is.na(Condition))

# ----------------- calibration constants ------------------

# Fraction of OD600 initially due to S. aureus
condition_fraction <- c(
  "Sa:Sna 1:1"   = 1 / (1 + 1),
  "Sa:Sna 1:10"  = 1 / (1 + 10),
  "Sa:Sna 1:100" = 1 / (1 + 100)
)

# ----------------- filter good wells ------------------

# ----------------- Good wells based on OD600 -------------------

good_od_wells <- od_cocultures %>%
  group_by(Well) %>%
  summarize(max_OD = max(Value, na.rm = TRUE)) %>%
  filter(max_OD > 0.1, max_OD < 4) %>%
  pull(Well)

# ----------------- Good wells based on GFP ---------------------

good_gfp_wells <- gfp_cocultures %>%
  group_by(Well) %>%
  filter(Time_number == 1) %>%  # Only time zero
  summarize(initial_GFP = Value) %>%
  filter(initial_GFP > 1000) %>%  # arbitrary threshold (you can adjust)
  pull(Well)

# ----------------- Intersection of both -----------------------

good_wells <- intersect(good_od_wells, good_gfp_wells)

# ----------------- Filter datasets ----------------------------

od_cocultures_filtered <- od_cocultures %>%
  filter(Well %in% good_wells)

gfp_cocultures_filtered <- gfp_cocultures %>%
  filter(Well %in% good_wells)

# ----------------- Plotting GFP fluorescence -------------------

ggplot(gfp_cocultures_filtered, aes(x = Time_number * 10, y = Value, group = Well)) +
  geom_line(alpha = 0.8, color = "darkgreen") +
  facet_wrap(~ Condition, nrow = 3, scales = "free_y") +
  labs(title = "Filtered GFP Fluorescence Curves",
       x = "Time (minutes)", y = "GFP Fluorescence (A.U.)") +
  theme_minimal() +
  theme(legend.position = "none")

# ----------------- calibration per replicate -------------------

# --- merged_data ---
merged_data <- inner_join(
  od_cocultures_filtered %>% rename(OD600 = Value),
  gfp_cocultures_filtered %>% rename(GFP = Value),
  by = c("Time_number", "Well", "Condition")
)

# --- calibrated using the fluorescence_to_od600 function ---
calibrated <- merged_data %>%
  mutate(
    Sa_OD = fluorescence_to_od600(GFP),
    KPL1850_OD = OD600 - Sa_OD)

# ----------------- plotting -------------------

# Prepare for plotting
plot_data <- calibrated %>%
  select(Time_number, Well, Condition, Sa_OD, KPL1850_OD) %>%
  pivot_longer(cols = c(Sa_OD, KPL1850_OD), names_to = "Species", values_to = "OD600") %>%
  mutate(Time_min = Time_number * 10)

ggplot(plot_data, aes(x = Time_min, y = OD600, color = Species, group = interaction(Species, Well))) +
  geom_line(alpha = 0.8) +
  facet_wrap(~ Condition, nrow = 3, scales = "free_y") +
  labs(title = "Corrected OD600 after Lag Alignment (with Nonlinear Adjustment)",
       x = "Time since End of Lag (minutes)", y = "Corrected OD600") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Define lag times (in minutes)
lag_Sa <- 2.35 * 60    # 141 min
lag_KPL1850 <- 2.25 * 60  # 135 min

# Align times
# plot_data_aligned <- plot_data %>%
#   mutate(
#     Time_min_aligned = case_when(
#       Species == "Sa_OD"       ~ Time_min - lag_Sa,
#       Species == "KPL1850_OD"  ~ Time_min - lag_KPL1850,
#       TRUE ~ Time_min
#     )
#   )

# Apply correction after lag and shifting

plot_data_aligned_corrected <- plot_data %>%
  group_by(Well, Species, Condition) %>%
  mutate(
    OD600_corrected = case_when(
      Species == "Sa_OD"        ~ correct_od600_sa(OD600),
      Species == "KPL1850_OD"   ~ correct_od600_kpl(OD600),
      TRUE ~ OD600
    )
  ) %>%
  ungroup()

ggplot(plot_data_aligned_corrected, aes(x = Time_min, y = OD600_corrected, color = Species, group = interaction(Species, Well))) +
  geom_line(alpha = 0.8) +
  facet_wrap(~ Condition, nrow = 3, scales = "free_y") +
  labs(title = "Corrected OD600 after Lag Alignment (with Nonlinear Adjustment)",
       x = "Time since End of Lag (minutes)", y = "Corrected OD600") +
  theme_minimal() +
  theme(legend.position = "bottom")

plot_data_wide_final <- plot_data_aligned_corrected %>%
  select(Time_min_aligned, Well, Condition, Species, OD600_corrected) %>%
  pivot_wider(names_from = Species, values_from = OD600_corrected)

# For each replicate (Well + Condition), clip where both are positive
clipped_data_list <- plot_data_wide_final %>%
  group_by(Condition, Well) %>%
  group_split() %>%
  map(function(df) {
    # Find first time where both OD600s > 0
    valid_idx <- which(df$Sa_OD > 0 & df$KPL1850_OD > 0)
    if (length(valid_idx) == 0) return(NULL)  # skip if no valid points
    start_idx <- min(valid_idx)
    df_clipped <- df[start_idx:nrow(df), ]
    return(df_clipped)
  }) %>%
  compact() 


# Create the output directory
if (!dir.exists("data_sa")) {
  dir.create("data_sa")
}

# Clip to time window [0, 900] minutes after lag
plot_data_aligned_clipped <- plot_data_aligned %>%
  filter(Time_min_aligned >= 0, Time_min_aligned <= 900)

# Split Sa and KPL1850 for easier handling
plot_data_wide <- plot_data_aligned_clipped %>%
  pivot_wider(names_from = Species, values_from = OD600)

# Now for each Well and Condition, save RData object
plot_data_wide %>%
  group_by(Condition, Well) %>%
  group_split() %>%
  purrr::walk(function(df) {
    # Get times
    times <- df$Time_min_aligned
    
    # Compute totals
    totals <- df$Sa_OD + df$KPL1850_OD
    
    # Compute proportions
    proportions <- cbind(
      Sa = df$Sa_OD / totals,
      KPL1850 = df$KPL1850_OD / totals
    )
    
    # Handle any NA from division by zero
    proportions[is.na(proportions)] <- 0
    
    # Make the list
    output_list <- list(
      times = times,
      proportions = proportions,
      totals = totals
    )
    
    # Create a filename
    filename <- paste0("data_sa/", unique(df$Condition), "_", unique(df$Well), ".RData")
    
    # Save
    save(output_list, file = filename)
  })
