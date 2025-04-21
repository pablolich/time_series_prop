import diffrax
import optimistix as optx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Define the Lotka-Volterra equations
def lotka_volterra(t, state, args):
    x, y = state
    alpha, beta, delta, gamma = args
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return jnp.array([dxdt, dydt])

# Event function to stop integration if the gradient is above 100
def event_fn(t, y, args, **kwargs):
    dydt = lotka_volterra(t, y, args)
    grad_norm = jnp.linalg.norm(dydt)
    # Check if any of the derivatives exceed 100
    return 5 - grad_norm < 0 # event will trigger if the gradient exceeds 100

term = diffrax.ODETerm(lotka_volterra)
# Set up the solver with events handling
solver = diffrax.Tsit5()

# Initial conditions: x=40 (prey), y=9 (predator)
initial_state = jnp.array([40.0, 9.0])

# Time range for integration

# Event: stops integration if grad exceeds threshold
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
# Set up the event function
event = diffrax.Event(event_fn, root_finder)

params = jnp.array([0.1, 0.02, 0.01, 0.1])
t0 = 0
t1 = 15
dt0 = 0.1
save_times = jnp.linspace(t0, t1, num=200)

# Solve the system with event handling
result = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=0.01,
    y0=initial_state,
    args = params,
    saveat=diffrax.SaveAt(ts=save_times),
    event=event
)

# Extract and print the results
time_points = result.ts
populations = result.ys
import ipdb; ipdb.set_trace(context = 20)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot the prey (x) and predator (y) populations
plt.plot(time_points, populations[:, 0], label="Prey", color="b")
plt.plot(time_points, populations[:, 1], label="Predator", color="r")

# Adding labels and title
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka-Volterra Model: Prey vs Predator Populations")

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Print the time and the final populations
print(f"Final populations at t={time_points[-1]}: Prey = {populations[-1, 0]}, Predator = {populations[-1, 1]}")

