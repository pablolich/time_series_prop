import jax
import itertools
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import lax

# Load data
data = pd.read_csv("glv_chaos_4spp.csv")
#data = data.iloc[0:50,:]
nspp = np.shape(data)[1] - 1
ntpoints = np.shape(data)[0]
time = data.iloc[:, 0].to_numpy()
species = data.iloc[:, 1:(nspp+1)].to_numpy()

# Compute proportions of all species
totals = np.sum(species, axis=1, keepdims=True)
proportions = species / totals

# Prepare derivatives (assuming all time steps are same)
dp_all = np.gradient(proportions, axis=0) / (time[1] - time[0])

# Padding for delay embedding
pad = 4
p_obs = proportions[pad:-pad]
dp_obs = dp_all[pad:-pad]

# Create sliding windows for encoder
window_size = 2 * pad + 1
p_windows = np.stack([proportions[i - pad:i + pad + 1] for i in range(pad, len(proportions) - pad)])

# Define encoder
def encoder_fn(x):
    N = hk.Sequential([
        hk.Conv1D(16, kernel_shape=9, padding="VALID"),
        jax.nn.softplus,
        hk.Conv1D(16, kernel_shape=1),
        jax.nn.softplus,
        hk.Conv1D(1, kernel_shape=1),
        jax.nn.softplus
    ])(x)
    return N

encoder = hk.without_apply_rng(hk.transform(encoder_fn))

def monomial_exponents(D, o):
    """
    Generate all exponent tuples for monomials of total degree o
    in D variables.
    """
    for c in itertools.combinations(range(o + D - 1), D - 1):
        starts = [0] + [x + 1 for x in c] + [o + D]
        yield tuple(starts[i + 1] - starts[i] - 1 for i in range(D))

def all_monomial_exponents(D, max_order):
    """
    Generate all exponent tuples from order 1 up to max_order.
    If max_order == 0, return an empty list (only constant term used).
    """
    exps = []
    if max_order == 0:
        return exps  # No polynomial terms, only the constant
    for o in range(1, max_order + 1):
        exps.extend(monomial_exponents(D, o))
    return exps

def compute_polynomial_features(x, exponents):
    if len(exponents) == 0:
        return jnp.zeros((x.shape[0], 0))  # No polynomial terms
    features = []
    for exp in exponents:
        term = jnp.prod(x ** jnp.array(exp), axis=1)
        features.append(term)
    return jnp.stack(features, axis=1)

def symbolic_model(x, W, exponents):
    """
    x: (T, D)
    W: (num_features + 1, D)
    exponents: list of exponent tuples
    """
    ones = jnp.ones((x.shape[0], 1))  # (T, 1)
    poly_feats = compute_polynomial_features(x, exponents)  # (T, N_feat)
    all_feats = jnp.concatenate([ones, poly_feats], axis=1)  # (T, N_feat + 1)
    return jnp.dot(all_feats, W)

def p_dot(p, x, W, exponents):
    """
    p: (T, D) — proportions time series
    x: (T, D) — abundances time series
    W: (5, D) — weights for symbolic model
    Returns dp/dt and avg_F(t)
    """
    F = symbolic_model(x, W, exponents)  # (T, D)
    avg_F = jnp.sum(p * F, axis=1, keepdims=True)  # (T, 1)
    dp_dt = p * (F - avg_F)  # First derivative of proportions
    return dp_dt, avg_F

def integrate_N(N_init, avg_F, time_step):
    """
    Integrates N(t) over time using dN/dt = N * avg_F(t).
    N_init: Initial total population (scalar or 1D)
    avg_F: The average force term, (T, 1)
    time_step: Time step size
    """
    def body_fn(i, N):
        # Compute dN/dt
        dN_dt = N * avg_F[i]
        return N + dN_dt * time_step  # Euler integration
    
    # Use lax.scan for efficient looping
    N_final = lax.scan(body_fn, N_init, jnp.arange(len(avg_F), dtype=jnp.int32))  # Ensure that the range is integer type
    return N_final

def integrate_N(N_init, avg_F, time_step):
    """
    Integrates N(t) over time using dN/dt = N * avg_F(t).
    N_init: Initial total population (scalar or 1D)
    avg_F: The average force term, (T, 1)
    time_step: Time step size
    """
    N = jnp.zeros_like(avg_F)  # Initialize the N array with the same shape as avg_F
    N = N.at[0].set(N_init)  # Set the initial value of N(t) at t=0
    
    # Euler integration loop
    for i in range(1, len(avg_F)):
        dN_dt = N[i - 1] * avg_F[i - 1]  # Compute dN/dt based on previous value of N and avg_F
        N = N.at[i].set(N[i - 1] + dN_dt * time_step)  # Update N[i] based on Euler's method
        
    return N

def loss_fn(params, W, p_obs, p_windows, dp_true, exponents):
    """
    Loss function that computes the mean squared error (MSE) 
    for the first derivative of the proportions and compatibility with dynamics.
    """
    # Predict N from encoder
    N_pred = encoder.apply(params, p_windows).squeeze(-1).squeeze(-1)

    # Get absolute abundances given proposed totals
    x = p_obs * N_pred[:, None]

    # Compute dp_pred using p_dot (first derivatives)
    dp_pred, avg_F = p_dot(p_obs, x, W, exponents)

    # Integrate N(t) from N_pred
    N_integrated = integrate_N(N_pred[0], avg_F, time[1] - time[0])

    # MSE between predicted and true proportions (first derivatives)
    mse = jnp.mean((dp_pred - dp_true) ** 2)

    # Add the penalty for N(t) mismatch
    N_loss = jnp.mean((N_integrated - N_pred) ** 2)

    # Total loss includes both MSE and N(t) compatibility
    total_loss = N_loss + mse
    return total_loss

# Adam optimizer with learning rate 1e-3
optimizer = optax.adam(1e-3)

@jax.jit
def update(params, W, opt_state, p_obs, p_windows, dp_true, exponents):
    """
    Performs one step of gradient descent using the Adam optimizer.
    
    params: Current model parameters for the encoder (for predicting N).
    W: Weight matrix for symbolic model.
    opt_state: Current state of the optimizer.
    p_obs: Observed proportions (T, D).
    dp_true: True change in proportions (T, D).
    
    Returns: Updated params, W, opt_state, and the loss.
    """
    # Compute both loss and gradients
    grads_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))  # Compute gradients wrt params and W
    loss, grads = grads_fn(params, W, p_obs, p_windows, dp_true, exponents)  # Loss and gradients
    
    # Apply gradients to update parameters and weights
    updates, opt_state = optimizer.update(grads, opt_state)  # Get updates from optimizer
    params = optax.apply_updates(params, updates[0])  # Update encoder params
    W = optax.apply_updates(W, updates[1])  # Update symbolic model weights
    
    return params, W, opt_state, loss

# Initialize data and parameters in jax and start training
p_windows = jnp.array(p_windows) 
dp_obs = jnp.array(dp_obs)

params = encoder.init(jax.random.PRNGKey(42), p_windows)
order = 0
exponents = all_monomial_exponents(nspp, order)
W = jnp.zeros((len(exponents) + 1, nspp)) 
opt_state = optimizer.init((params, W))

# Training loop
for step in range(10000):
    params, W, opt_state, loss = update(params, W, opt_state, p_obs, p_windows, dp_obs, exponents)
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.5f}")

# Plot results
N_pred = encoder.apply(params, p_windows).squeeze(-1).squeeze(-1)
X_rec = p_obs * N_pred[:, None]  # (T, 4)
X_true = species[pad:-pad]

# Scale both to match total at t=0
true_init_sum = jnp.sum(X_true[0])

X_scaled = X_true / true_init_sum

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Panel 1: Reconstructed populations
for i in range(nspp):
    axes[0].plot(time[pad:-pad], X_scaled[:, i], label=f"True spp {i+1}", color=colors[i])
    axes[0].plot(time[pad:-pad], X_rec[:, i], '--', label=f"Recon spp {i+1}", color=colors[i])
axes[0].legend()
axes[0].set_title("Hidden State Reconstruction for All Species")
axes[0].set_ylabel("Population")

# Panel 2: Proportions
for i in range(nspp):
    axes[1].plot(time[pad:-pad], p_obs[:, i], label=f"Observed prop {i+1}", color=colors[i])
    axes[1].plot(time[pad:-pad], X_rec[:, i] / jnp.sum(X_rec, axis=1), '--', label=f"Recon prop {i+1}", color=colors[i])
axes[1].legend()
axes[1].set_title("Matched Proportions")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Proportion")

plt.tight_layout()
plt.show()

N_pred = encoder.apply(params, p_windows).squeeze(-1).squeeze(-1)
x = p_obs * N_pred[:, None]
_, avg_F = p_dot(p_obs, x, W, exponents)
N_integrated = integrate_N(1., avg_F, time[1] - time[0])

print("N_pred (first 5):", N_pred[:5])
print("N_integrated (first 5):", N_integrated[:5])
plt.plot(N_pred, label="Encoder N_pred")
plt.plot(N_integrated, '--', label="Integrated N(t)")
plt.legend()
plt.title("Total Abundance Over Time")
plt.show()

