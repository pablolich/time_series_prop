
import jax.numpy as jnp
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

def vector_field(t, y, args):
    _, v = y
    return jnp.array([v, -8.0])

def vector_field(t, y, args):
    alpha, beta, delta, gamma = args
    prey, predator = y
    dydt = jnp.array([
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator
    ])
    return dydt


def cond_fn(t, y, args, **kwargs):
    x, _ = y
    return x

# Event condition function (checking if gradient exceeds threshold)
def event_fn(t, y, args, **kwargs):
    dydt = vector_field(t, y, args)
    grad_norm = jnp.linalg.norm(dydt)
    # Return positive when it's above threshold to trigger the event
    return (10. - grad_norm) < 0

y0 = jnp.array([10.0, 5.0])
params = jnp.array([2.0, 1.0, 0.5, 1.0])
t0 = 0
t1 = 13
dt0 = 0.1
save_times = jnp.linspace(t0, t1, num=200)
term = diffrax.ODETerm(vector_field)
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = diffrax.Event(event_fn, root_finder)
solver = diffrax.Tsit5()
sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, args = params,
        saveat=diffrax.SaveAt(ts=save_times),  
        event=event)
print(f"Event time: {sol.ts[0]}") # Event time: 1.58...
print(f"Velocity at event time: {sol.ys[0, 1]}") # Velocity at event time: -12.64...
# Plot the solution
plt.figure(figsize=(12, 6))

# Plot prey and predator populations over time
plt.subplot(1, 2, 1)
plt.plot(sol.ts, sol.ys[:, 0], label="Prey")
plt.plot(sol.ts, sol.ys[:, 1], label="Predator")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.title("Prey and Predator Populations over Time")

# Plot prey proportion over time (relative to total population)
total_population = sol.ys[:, 0] + sol.ys[:, 1]  # Total population
prey_proportion = sol.ys[:, 0] / total_population  # Proportion of prey
plt.subplot(1, 2, 2)
plt.plot(sol.ts, prey_proportion, label="Prey Proportion")
plt.xlabel("Time")
plt.ylabel("Prey Proportion")
plt.legend()
plt.title("Prey Proportion over Time")

plt.tight_layout()
plt.show()
