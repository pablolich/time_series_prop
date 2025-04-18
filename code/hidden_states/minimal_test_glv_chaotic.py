import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Step 1: Load data from CSV
# ----------------------
data = pd.read_csv("predator_prey.csv")
#data = data.iloc[0:50,:]
#data = data.iloc[30:,:]
nspp = np.shape(data)[1]-1
ntpoints = np.shape(data)[0]
time = data.iloc[:, 0].to_numpy()
species = data.iloc[:, 1:(nspp+1)].to_numpy()  # Use all 4 species

# Compute proportions of all species
species_sum = np.sum(species, axis=1, keepdims=True)
proportions = species / species_sum

# Prepare derivatives
dp_all = np.gradient(proportions, axis=0) / (time[1] - time[0])

pad = 3
p_obs = proportions[pad:-pad]      # (T-2*pad, 4)
dp_obs = dp_all[pad:-pad]          # (T-2*pad, 4)

# Create sliding window input for encoder
window_size = 7
p_windows = np.stack([proportions[i - pad:i + pad + 1] for i in range(pad, len(proportions) - pad)])  # (T-2*pad, 9, 4)

# ----------------------
# Step 2: Haiku Encoder (predicts N)
# ----------------------
def encoder_fn(x):
    return hk.Sequential([
        hk.Conv1D(8, kernel_shape=7, padding="VALID"),
        jax.nn.tanh,
        hk.Conv1D(1, kernel_shape=1),
        jax.nn.softplus,
        #hk.Conv1D(1, kernel_shape=1),
        #jax.nn.relu,
    ])(x)

encoder = hk.without_apply_rng(hk.transform(encoder_fn))

# ----------------------
# Step 3: Symbolic Model (per species)
# ----------------------
def symbolic_model(p, N, W):
    # p: (T, 4), N: (T,), W: (10, 4)
    N_col = N[:, None]  # (T, 1)
    features = jnp.concatenate([
        #jnp.ones_like(N_col),      # (T, 1)
        #p,                         # (T, 4)
        #N_col,                     # (T, 1)
        #p**2,                      # (T, 4)
        p * N_col,                 # (T, 4)
        #N_col**2,                  # (T, 1)
        #p**3,                      # (T, 4)
        #(p**2) * N_col,            # (T, 4)
        p * (N_col**2),            # (T, 4)
        #N_col**3                   # (T, 1)
        (p**2)*(N_col)**2,
        #p*(N_col**3), 
        #(p**2)*(N_col**3),
        #(p**3)*(N_col**3)
    ], axis=1)                    # final shape (T, 26)
    return jnp.einsum('ij,jk->ik', features, W)  # (T, 4)  # (T, 4)

# ----------------------
# Step 4: Loss and Optimizer
# ----------------------
def loss_fn(params, W, x, p_mid, dp_true):
    N_pred = encoder.apply(params, x).squeeze(-1).squeeze(-1)  # (T,)
    dp_pred = symbolic_model(p_mid, N_pred, W)  # (T, 4)
    mse = jnp.mean((dp_pred - dp_true)**2)
    sparsity = jnp.sum(jnp.abs(W))
    return mse #+ 1e-5* sparsity

optimizer = optax.adam(1e-3)

@jax.jit
def update(params, W, opt_state, x, p_mid, dp_true):
    grads_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    loss, grads = grads_fn(params, W, x, p_mid, dp_true)
    updates, opt_state = optimizer.update(grads, opt_state, (params, W))
    params = optax.apply_updates(params, updates[0])
    W = optax.apply_updates(W, updates[1])
    return params, W, opt_state, loss

# ----------------------
# Step 5: Initialization and Training
# ----------------------
p_windows = jnp.array(p_windows)  # (T, 9, 4)
p_obs = jnp.array(p_obs)         # (T, 4)
dp_obs = jnp.array(dp_obs)       # (T, 4)

params = encoder.init(jax.random.PRNGKey(42), p_windows)
W = jnp.zeros((6, nspp))  # one equation per species
opt_state = optimizer.init((params, W))

for step in range(10000):
    params, W, opt_state, loss = update(params, W, opt_state, p_windows, p_obs, dp_obs)
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.5f}")

# ----------------------
# Step 6: Visualization for all species
# ----------------------
N_pred = encoder.apply(params, p_windows).squeeze(-1).squeeze(-1)
X_rec = p_obs * N_pred[:, None]  # (T, 4)
X_true = species[pad:-pad]

# Scale both to match total at t=0
true_init_sum = jnp.sum(X_true[0])
pred_init_sum = jnp.sum(X_rec[0])
scale = true_init_sum / pred_init_sum

X_scaled = X_true
X_rec_scaled = X_rec * scale
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Panel 1: Reconstructed populations
for i in range(nspp):
    axes[0].plot(time[pad:-pad], X_scaled[:, i], label=f"True spp {i+1}", color=colors[i])
    axes[0].plot(time[pad:-pad], X_rec_scaled[:, i], '--', label=f"Recon spp {i+1}", color=colors[i])
axes[0].legend()
axes[0].set_title("Hidden State Reconstruction for All Species")
axes[0].set_ylabel("Population")

# Panel 2: Proportions
for i in range(nspp):
    axes[1].plot(time[pad:-pad], p_obs[:, i], label=f"Observed prop {i+1}", color=colors[i])
    axes[1].plot(time[pad:-pad], X_rec_scaled[:, i] / jnp.sum(X_rec_scaled, axis=1), '--', label=f"Recon prop {i+1}", color=colors[i])
axes[1].legend()
axes[1].set_title("Matched Proportions")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Proportion")

plt.tight_layout()
plt.show()

# Correlation between true and reconstructed populations per species
print("coefficients:")
for i in range(nspp):
    corr = np.corrcoef(X_scaled[:, i], X_rec_scaled[:, i])[0, 1]
    print(f"Species {i+1}: r = {corr:.4f}")

# Print discovered symbolic weights
print("\nDiscovered symbolic weights (columns = species):")
print(jnp.round(W, nspp))


print("Discovered symbolic weights (columns = species):")
print(jnp.round(W, nspp))
# One-to-one plots: reconstruction vs true
fig, axs = plt.subplots(1, 2, figsize=(10, 10), squeeze=False)
for i, ax in enumerate(axs.flat):
    ax.scatter(X_scaled[:, i], X_rec_scaled[:, i], s=5, color=colors[i], alpha=0.6)
    ax.plot([X_scaled[:, i].min(), X_scaled[:, i].max()], [X_scaled[:, i].min(), X_scaled[:, i].max()], 'k--', lw=1)
    ax.set_title(f"Species {i+1}")
    ax.set_xlabel("True")
    ax.set_ylabel("Reconstructed")
    ax.set_aspect('equal', adjustable='box')
plt.suptitle("One-to-One Reconstruction vs True")
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

