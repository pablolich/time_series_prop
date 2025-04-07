library(deSolve)
library(ggplot2)
library(gridExtra)
library(MCMCprecision)  # For Dirichlet distribution sampling
library(tidyverse)

# Lotka-Volterra model function
glv_model <- function(t, state, params) {
  with(as.list(c(state, params)), {
    dS1 <- S1 * (r1 + a11 * S1 + a12 * S2 + a13 * S3)
    dS2 <- S2 * (r2 + a21 * S1 + a22 * S2 + a23 * S3)
    dS3 <- S3 * (r3 + a31 * S1 + a32 * S2 + a33 * S3)
    return(list(c(dS1, dS2, dS3)))
  })
}

# Parameters
time <- seq(0, 25, by = 0.1) # Time sequence
params <- c(
  r1 = 0.5, r2 = 0.3, r3 = 0.4,
  a11 = -1.0, a12 = -0.5, a13 = -0.3,
  a21 = -0.5, a22 = -1.2, a23 = -0.4,
  a31 = -0.3, a32 = -0.4, a33 = -1.5
)

# Function to generate initial conditions from Dirichlet distribution
dirichlet_init <- function(composition) {
  num_present <- sum(composition)
  if (num_present == 0) {
    return(composition)  # Return zero vector if no species present
  }
  sampled_values <- rdirichlet(1, rep(1, num_present))
  init <- composition
  init[composition == 1] <- sampled_values
  return(init)
}

# Compositions
compositions <- list(
  "All_Species" = c(S1 = 1, S2 = 1, S3 = 1),
  "No_S1" = c(S1 = 0, S2 = 1, S3 = 1),
  "No_S2" = c(S1 = 1, S2 = 0, S3 = 1),
  "No_S3" = c(S1 = 1, S2 = 1, S3 = 0)
)

# Store simulation results
all_results <- list()

# Run simulations and save results
for (exp_name in names(compositions)) {
  for (rep in 1:2) { # Two replicates per experiment
    init <- dirichlet_init(compositions[[exp_name]])
    sim <- ode(y = init, times = time, func = glv_model, parms = params)
    
    # Convert to data frame
    sim_data <- as.data.frame(sim)
    colnames(sim_data) <- c("Time", "S1", "S2", "S3")
    sim_data$Experiment <- exp_name
    sim_data$Replicate <- as.factor(rep)
    
    # Ensure zeros when species is not there
    sim_data[sim_data < 1e-6] <- 0
    
    # Store results
    all_results[[paste0(exp_name, "_Rep", rep)]] <- sim_data
    
    # Save CSV
    filename <- paste0("GLV_Simulation_", exp_name, "_Rep", rep, ".csv")
    write.csv(data.frame(sim_data) %>% select(Time, S1, S2, S3), filename, row.names = FALSE)
  }
}

# Combine results for plotting
plot_data <- do.call(rbind, all_results)
plot_data <- reshape2::melt(plot_data, id.vars = c("Time", "Experiment", "Replicate"), 
                            variable.name = "Species", value.name = "Abundance")

# Generate plots
plots <- lapply(unique(plot_data$Experiment), function(exp_name) {
  ggplot(subset(plot_data, Experiment == exp_name), 
         aes(x = Time, y = Abundance, color = Species, linetype = Replicate)) +
    geom_line() +
    labs(title = exp_name) +
    theme_minimal()
})

grid.arrange(grobs = plots, ncol = 4)
