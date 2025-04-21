import jax
import jax.numpy as jnp
import optax
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
from jax import value_and_grad

# ----------------------------------------
# Lotka-Volterra model
# ----------------------------------------

def lotka_volterra(x, t, params):
    alpha, beta, delta, gamma = params
    prey, predator = x
    dxdt = jnp.array([
        alpha * prey - beta * prey * predator,
        delta * prey * predator - gamma * predator
    ])
    return dxdt

# ----------------------------------------
# Convert absolute population to proportions
# ----------------------------------------

def to_proportions(x):
    totals = jnp.sum(x, axis=1, keepdims=True)
    return x / totals

# ----------------------------------------
# ODE solver with penalty for high gradients
# ----------------------------------------

import diffrax

def integrate_with_event(params, x0, t_eval, grad_threshold=100.0):
    alpha, beta, delta, gamma = params

    # Vector field
    def vf(t, y, args):
        prey, predator = y
        dydt = jnp.array([
            alpha * prey - beta * prey * predator,
            delta * prey * predator - gamma * predator
        ])
        return dydt

    # Gradient norm to detect divergence
    def event_fn(t, y, args, **kwargs):
        dydt = vf(t, y, args)
        grad_norm = jnp.linalg.norm(dydt)
        return 200 - grad_norm < 0  # Trigger when this crosses zero from positive to negative

    term = diffrax.ODETerm(vf)
    solver = diffrax.Tsit5()

    # Event: stops integration if grad exceeds threshold
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(event_fn, root_finder)

    saveat = diffrax.SaveAt(ts=t_eval)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=0.1,
        y0=x0,
        args=params,
        saveat=saveat,
        event=event,
    )
    # Check for infinities and count the number of infinite values in the solution
    inf_penalty = 0.0
    penalty = 0.
    inf_count = jnp.sum(jnp.isinf(sol.ys))
    ys = sol.ys

    inf_count = jnp.sum(jnp.isinf(sol.ys))

    def apply_penalty(_):
        # Shuffle species identities (swap prey <-> predator)
        ys_swapped = sol.ys[:, ::-1]  # just reverses the second axis (prey <-> predator)
        ys_clipped = jnp.where(jnp.isinf(ys_swapped), 1e5, ys_swapped)
        return ys_clipped, 1e2  # apply penalty too

    #def apply_penalty(_):
    #    inf_penalty = 1e2 * inf_count
    #    ys_clipped = jnp.where(jnp.isinf(sol.ys), 1e5, sol.ys)
    #    return ys_clipped, inf_penalty

    def no_penalty(_):
        return sol.ys, 0.0

    ys, penalty = jax.lax.cond(inf_count > 0, apply_penalty, no_penalty, operand=None)

   
    return ys, penalty
# ----------------------------------------
# Curriculum time weights (increasing window)
# ----------------------------------------

def make_time_weights(intensities, times):
    # intensities: (epochs,)
    # times: (ntimes,)
    intensities = jnp.asarray(intensities)[:, None]  # shape (epochs, 1)
    times = jnp.asarray(times)[None, :]              # shape (1, ntimes)
    weights = jnp.exp(-intensities * times)          # shape (epochs, ntimes)
    return weights# normalize per epoch

# ----------------------------------------
# Loss function with weighting
# ----------------------------------------

def loss_fn(params, x0, t_eval, target_props, time_weights):
    x, penalty = integrate_with_event(params, x0, t_eval)
    pred_props = to_proportions(x)
    mse = jnp.mean((pred_props - target_props) ** 2, axis=1)
    weighted = jnp.sum((mse+penalty) * time_weights)
    return weighted# + penalty
# ----------------------------------------
# Define optimizer globally
# ----------------------------------------

optimizer = optax.adam(1e-3)

@jax.jit
def train_step(params, opt_state, x0, t_eval, target_props, time_weights):
    def weighted_loss(p):
        return loss_fn(p, x0, t_eval, target_props, time_weights)

    loss, grads = value_and_grad(weighted_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
# ----------------------------------------
# Training loop
# ----------------------------------------

def fit_model(x0, t_eval, target_props, init_params, steps=100, use_weights=True):
    opt_state = optimizer.init(init_params)
    params = init_params

    if use_weights:
        all_time_weights = make_time_weights(
            jnp.linspace(1.0, 0.0, steps), t_eval
        )
    else:
        all_time_weights = jnp.ones((steps, len(t_eval)))

    for step in range(steps):
        time_weights = all_time_weights[step]
        params, opt_state, loss = train_step(
            params, opt_state, x0, t_eval, target_props, time_weights
        )
        print(f"{'[Weighted]' if use_weights else '[Unweighted]'} Step {step}, Loss: {loss:.4f}")

    return params

# ----------------------------------------
# Plotting function (2-panel)
# ----------------------------------------

def plot_fit(t_eval, x_true, fitted_params, x0):
    x_fitted, _ = integrate_with_event(
        fitted_params, x0, t_eval)
    target_proportions = to_proportions(x_true)
    pred_props = to_proportions(x_fitted)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute populations
    axs[0].plot(t_eval, x_true[:, 0], 'o', label="Prey", color='tab:blue')
    axs[0].plot(t_eval, x_true[:, 1], 'o', label="Predator", color='tab:orange')
    axs[0].plot(t_eval, x_fitted[:, 0], label="Prey", color='tab:blue')
    axs[0].plot(t_eval, x_fitted[:, 1], label="Predator", color='tab:orange')
    axs[0].set_title("Absolute Populations")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Population")
    axs[0].legend()
    axs[0].grid(True)

    # Proportions
    axs[1].plot(t_eval, target_proportions[:, 0], 'o', label="Target Prey", alpha=0.4, color='tab:blue')
    axs[1].plot(t_eval, target_proportions[:, 1], 'o', label="Target Predator", alpha=0.4, color='tab:orange')
    axs[1].plot(t_eval, pred_props[:, 0], '-', label="Fitted Prey", color='tab:blue')
    axs[1].plot(t_eval, pred_props[:, 1], '-', label="Fitted Predator", color='tab:orange')
    axs[1].set_title("Proportions")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Proportion")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
# Create synthetic data
true_params = jnp.array([2.0, 1.0, 0.5, 1.0])
x0 = jnp.array([0.5, 0.5])
t_eval = jnp.linspace(0, 10, 100)
x_true, _ = integrate_with_event(true_params, x0, t_eval)

key = jax.random.PRNGKey(0)
noise_std = 0.2  # Adjust this for more/less noise
noise = jax.random.normal(key, shape=x_true.shape) * noise_std
target_proportions = to_proportions(x_true) + noise

# Initial guess
key = jax.random.PRNGKey(45)  # any seed you like
init_params = jax.random.uniform(key, shape=(4,), minval=0.1, maxval=3.0)

# Train
def train_with_refinement(x0, t_eval, target_props, init_params, first_steps=8000, refine_steps=5000):
    trained = fit_model(x0, t_eval, target_props, init_params, steps=first_steps)
    refined = fit_model(x0, t_eval, target_props, trained, steps=refine_steps, use_weights = False)
    return refined

fitted_params = train_with_refinement(x0, t_eval, target_proportions, init_params, first_steps=5000, refine_steps = 3500)
# Plot result
plot_fit(t_eval, x_true, fitted_params, x0)

