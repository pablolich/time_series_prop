library(deSolve)
library(reshape2)
library(tidyverse)

# Define the GLV system
glv <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dx1 <- r1*x1 * (1 - (a11*x1 + a12*x2 + a13*x3 + a14*x4))
    dx2 <- r2*x2 * (1 - (a21*x1 + a22*x2 + a23*x3 + a24*x4))
    dx3 <- r3*x3 * (1 - (a31*x1 + a32*x2 + a33*x3 + a34*x4))
    dx4 <- r4*x4 * (1 - (a41*x1 + a42*x2 + a43*x3 + a44*x4))
    
    list(c(dx1, dx2, dx3, dx4))
  })
}

# Parameters chosen for chaotic behavior
params <- c(
  r1 = 1, r2 = 0.72, r3 = 1.53, r4 = 1.27,
  a11 = 1, a12 = 1.09, a13 = 1.52, a14 = 0,
  a21 = 0.1, a22 = 1, a23 = 0.44, a24 = 1.36,
  a31 = 2.33, a32 = 0, a33 = 1, a34 = 0.47,
  a41 = 1.21, a42 = 0.51, a43 = 0.35, a44 = 1
)

# Time span
time <- seq(0, 25, by = 0.1)

# Three different initial conditions
library(extraDistr)

# Define the alpha parameters for the Dirichlet distribution
alpha <- c(1, 1, 1, 1)  # Uniform Dirichlet; change values as needed

# Sample 3 sets of initial conditions from the Dirichlet distribution
initial_conditions <- lapply(1:3, function(i) {
  values <- rdirichlet(1, alpha)
  names(values) <- c("x1", "x2", "x3", "x4")
  as.vector(values)
})

# Convert each sampled vector to a named vector
initial_conditions <- lapply(initial_conditions, function(v) {
  setNames(v, c("x1", "x2", "x3", "x4"))
})

# Simulate and save results
for (i in 1:3) {
  out <- ode(y = initial_conditions[[i]], times = time, func = glv, parms = params)
  write.csv(out, paste0("glv_chaos", i, ".csv"), row.names = FALSE)
}


# Function to plot GLV simulation data
plot_glv <- function(file_name, title) {
  data <- read.csv(file_name)
  data_long <- melt(data, id.vars = "time")
  
  ggplot(data_long, aes(x = time, y = value, color = variable)) +
    geom_line() +
    labs(title = title, x = "Time", y = "Population") +
    theme_minimal()
}

# Plot each simulation
plot1 <- plot_glv("glv_chaos1.csv", "GLV Chaotic Simulation - Initial Condition 1")
plot2 <- plot_glv("glv_chaos2.csv", "GLV Chaotic Simulation - Initial Condition 2")
plot3 <- plot_glv("glv_chaos3.csv", "GLV Chaotic Simulation - Initial Condition 3")

plot1
plot2
plot3
