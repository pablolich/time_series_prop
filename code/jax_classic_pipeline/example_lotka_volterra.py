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

#def lotka_volterra(x, t, params):
#    alpha, beta, delta, gamma = params
#    prey, predator = x
#    dxdt = jnp.array([
#        alpha * prey - beta * prey * predator,
#        delta * prey * predator - gamma * predator
#    ])
#    return dxdt
    
def lotka_volterra_general(x, t, params):
    """
    x: (n_species,)
    t: scalar time (unused but required for diffrax)
    params: tuple (r, A)
        - r: (n_species,)
        - A: (n_species, n_species)
    """
    r, A = params
    return x * (r + A @ x)

# ----------------------------------------
# Convert absolute population to proportions
# ----------------------------------------

def to_proportions(x):
    totals = jnp.sum(x, axis=1, keepdims=True)
    return x / totals
    
def flatten_params(r, A):
    return jnp.concatenate([r, A.flatten()])

def unflatten_params(vec, n_species):
    r = vec[:n_species]
    A = vec[n_species:].reshape((n_species, n_species))
    return r, A

# ----------------------------------------
# ODE solver with penalty for high gradients
# ----------------------------------------

import diffrax

def integrate_with_event(params, x0, t_eval, grad_threshold=100.0):

    num_species = x0.shape[0]
    # Split flat params into r and flattened A
    r = params[:num_species]
    A_flat = params[num_species:]
    A = A_flat.reshape((num_species, num_species))
    # Vector field
    #def vf(t, y, args):
    #    alpha, beta, delta, gamma = args
    #    prey, predator = y
    #    dydt = jnp.array([
    #        alpha * prey - beta * prey * predator,
    #        delta * prey * predator - gamma * predator
    #    ])
    #    return dydt
        
    def vf(t, y, args):
        #r, A = args
        dydt = y * (r + A @ y)
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
        #ys_swapped = sol.ys[:, ::-1]  # just reverses the second axis (prey <-> predator)
        ys_clipped = jnp.where(jnp.isinf(sol.ys), 1e5, sol.ys)
        return ys_clipped, 1e2*inf_count  # apply penalty too

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

def loss_fn(flat_params, t_eval, target_props, time_weights):
    num_species = target_props.shape[1]
    x0 = flat_params[:num_species]
    model_params = flat_params[num_species:]
    x, penalty = integrate_with_event(model_params, x0, t_eval)
    pred_props = to_proportions(x)
    mse = jnp.mean((pred_props - target_props) ** 2, axis=1)
    weighted = jnp.sum((mse + penalty) * time_weights)
    return weighted

# ----------------------------------------
# Define optimizer globally
# ----------------------------------------

optimizer = optax.adam(1e-3)


@jax.jit
def train_step(flat_params, opt_state, t_eval, target_props, time_weights):
    def weighted_loss(p):
        return loss_fn(p, t_eval, target_props, time_weights)

    loss, grads = value_and_grad(weighted_loss)(flat_params)
    updates, opt_state = optimizer.update(grads, opt_state, flat_params)
    flat_params = optax.apply_updates(flat_params, updates)
    return flat_params, opt_state, loss

# ----------------------------------------
# Training loop
# ----------------------------------------

from functools import partial
import jax.lax as lax

def try_initializations(num_trials, x0_guess, t_eval, target_props, key):
    n_species = x0_guess.shape[0]
    keys = jax.random.split(key, num_trials)

    @jax.jit
    def evaluate(key):
        r_key, A_key = jax.random.split(key)
        r = jax.random.normal(r_key, shape=(n_species,))
        A = jax.random.normal(A_key, shape=(n_species, n_species))
        model_params = flatten_params(r, A)
        full_params = jnp.concatenate([x0_guess, model_params])
        loss = loss_fn(full_params, t_eval, target_props, jnp.ones_like(t_eval))
        return full_params, loss

    all_params, all_losses = jax.vmap(evaluate)(keys)
    best_idx = jnp.argmin(all_losses)
    return all_params[best_idx]


def fit_model(t_eval, target_props, flat_init_params, steps=100, use_weights=True):
    opt_state = optimizer.init(flat_init_params)
    flat_params = flat_init_params

    if use_weights:
        all_time_weights = make_time_weights(jnp.linspace(1.0, 0.0, steps), t_eval)
    else:
        all_time_weights = jnp.ones((steps, len(t_eval)))

    for step in range(steps):
        time_weights = all_time_weights[step]
        flat_params, opt_state, loss = train_step(
            flat_params, opt_state, t_eval, target_props, time_weights
        )
        print(f"{'[Weighted]' if use_weights else '[Unweighted]'} Step {step}, Loss: {loss:.4f}")

    return flat_params


# ----------------------------------------
# Plotting function (2-panel)
# ----------------------------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_fit(t_eval, x_noisy, fitted_params, true_params=None, x_skeleton=None):
    target_proportions = to_proportions(x_noisy)
    num_species = target_proportions.shape[1]
    x0 = fitted_params[:num_species]
    model_params = fitted_params[num_species:]
    x_fitted, _ = integrate_with_event(model_params, x0, t_eval)
    pred_props = to_proportions(x_fitted)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Define a colormap to assign distinct colors
    color_map = cm.get_cmap("tab10", num_species)
    species_colors = [color_map(i) for i in range(num_species)]

    # Absolute Populations
    for i in range(num_species):
        axs[0].plot(t_eval, x_noisy[:, i], 'o', label=f"Noisy Species {i+1}", color=species_colors[i], alpha=0.4)
        axs[0].plot(t_eval, x_fitted[:, i], '-', label=f"Fitted Species {i+1}", color=species_colors[i])
        if x_skeleton is not None:
            axs[0].plot(t_eval, x_skeleton[:, i], '--', label=f"Skeleton Species {i+1}", color=species_colors[i], alpha=0.7)

    axs[0].set_title("Absolute Populations")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Population")
    axs[0].legend()
    axs[0].grid(True)

    # Proportions
    for i in range(num_species):
        axs[1].plot(t_eval, target_proportions[:, i], 'o', label=f"Noisy Species {i+1}", color=species_colors[i], alpha=0.4)
        axs[1].plot(t_eval, pred_props[:, i], '-', label=f"Fitted Species {i+1}", color=species_colors[i])
        if x_skeleton is not None:
            skel_props = to_proportions(x_skeleton)
            axs[1].plot(t_eval, skel_props[:, i], '--', label=f"Skeleton Species {i+1}", color=species_colors[i], alpha=0.7)

    axs[1].set_title("Proportions")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Proportion")
    axs[1].legend()
    axs[1].grid(True)

    # Param Fit Quality
    if true_params is not None:
        axs[2].scatter(true_params, fitted_params, color='green')
        min_val = float(jnp.minimum(jnp.min(true_params), jnp.min(fitted_params)))
        max_val = float(jnp.maximum(jnp.max(true_params), jnp.max(fitted_params)))
        pad = 0.05 * (max_val - min_val + 1e-8)
        axs[2].plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 'k--', alpha=0.5)
        axs[2].set_xlim(min_val - pad, max_val + pad)
        axs[2].set_ylim(min_val - pad, max_val + pad)
        axs[2].set_title("Fitted vs True Params")
        axs[2].set_xlabel("True")
        axs[2].set_ylabel("Fitted")
        axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# Create synthetic data
#true_params = jnp.array([2.0, 1.0, 0.5, 1.0])
#r = jnp.array([2.0, -1.0])
#A = jnp.array([
#    [0.0, -1.0],
#    [0.5,  0.0]
#])
#x0 = jnp.array([0.5, 0.5])
#model_params = flatten_params(r, A)
#true_params = jnp.concatenate([x0, model_params])
#t_eval = jnp.linspace(0, 10, 100)
#x_true, _ = integrate_with_event(model_params, x0, t_eval)
#load data from file
#data = pd.read_csv("C1.csv")
r_chaos = jnp.array([1, 0.72, 1.53, 1.27])
A_chaos = -1*jnp.array([
    [1, 1.09, 1.52, 0],
    [0.1, 1, 0.44, 1.36],
    [2.33, 0, 1, 0.47],
    [1.21, 0.51, 0.35, 1]
    ])*r_chaos[:, None]
x0_chaos = jnp.array([0.4874672, 0.1488466, 0.2485307, 0.1151555])
model_params = flatten_params(r_chaos, A_chaos)
true_params = jnp.concatenate([x0_chaos, model_params])
t_eval = jnp.linspace(0, 25, 200)
x_true, _ = integrate_with_event(model_params, x0_chaos, t_eval)

key = jax.random.PRNGKey(42)
shape_param = 50.0  # Higher means less variance (more concentrated around the mean)

# Compute scale such that shape * scale = x_true
scale = x_true / shape_param

# Sample from Gamma
key1, key2 = jax.random.split(key)
gamma_noise = jax.random.gamma(key1, shape_param, shape=x_true.shape) * scale

# Rescale so that the total population at t=0 sums to 1
initial_total = jnp.sum(gamma_noise[0])
scaling_factor = 1.0 / initial_total
x_true_noise = gamma_noise * scaling_factor

target_props = to_proportions(x_true_noise)

#rescale
#x_true = x_true/jnp.sum(x_true[0])
#x_true_noise = x_true
#target_props = to_proportions(x_true)

# Train
def train_with_refinement(t_eval, target_props, init_trials=50000, first_steps=50000, refine_steps=50000):
    key = jax.random.PRNGKey(0)
    x0_guess = target_props[0]
    full_init = try_initializations(init_trials, x0_guess, t_eval, target_props, key)

    trained = fit_model(t_eval, target_props, full_init, steps=first_steps)
    unweighted = fit_model(t_eval, target_props, trained, steps=refine_steps, use_weights=False)

    #import numpy as np
    #from scipy.optimize import minimize
    #from jax import value_and_grad

    ## Wrap loss + gradient
    #loss_and_grad_fn = value_and_grad(lambda p: loss_fn(p, t_eval, target_props, jnp.ones_like(t_eval)))

    #def scipy_loss(p_numpy):
    #    # Convert numpy -> jax
    #    p_jax = jnp.array(p_numpy)
    #    loss_val, _ = loss_and_grad_fn(p_jax)
    #    return float(loss_val)#, np.array(grad_val)

    ## Use scipy minimize
    #result = minimize(
    #    fun=scipy_loss,
    #    x0=np.array(unweighted),  # from ADAM result
    #    method="L-BFGS-B",
    #    options=dict(maxiter=500, disp=True)
    #)

    #fitted_params_bfgs = jnp.array(result.x)

    return unweighted


fitted_params = train_with_refinement(t_eval, target_props)
# Plot result
plot_fit(t_eval, x_true_noise, fitted_params, true_params, x_true)

