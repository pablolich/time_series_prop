import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------
# Step 1: Simulate Predator-Prey Dynamics
# ----------------------
def lotka_volterra(t, z, alpha=1.0, beta=0.5, gamma=1.0, delta=0.5):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

t_eval = np.linspace(0, 30, 1000)
z0 = [2.2, 1.8]
sol = solve_ivp(lotka_volterra, (0, 30), z0, t_eval=t_eval)
X, Y = sol.y
p = X / (X + Y)
N = X + Y
dt = t_eval[1] - t_eval[0]

dp = np.gradient(p, dt)
pad = 4
p_obs = p[pad:-pad]
dp_obs = dp[pad:-pad]

# Create sliding window input
window_size = 9
p_windows = np.stack([p[i - pad:i + pad + 1] for i in range(pad, len(p) - pad)])

# ----------------------
# Step 2: Haiku Encoder
# ----------------------
def encoder_fn(x):
    return hk.Sequential([
        hk.Conv1D(32, kernel_shape=9, padding="VALID"),
        jax.nn.relu,
        hk.Conv1D(32, kernel_shape=1),
        jax.nn.relu,
        hk.Conv1D(1, kernel_shape=1),
    ])(x)

encoder = hk.without_apply_rng(hk.transform(encoder_fn))

# ----------------------
# Step 3: Symbolic Model
# ----------------------
def symbolic_model(p, N_col, w):
    N_col = N[:, None]  # (T, 1)
    features = jnp.column_stack([#jnp.ones_like(N_col),      # (T, 1)
        #p,                         # (T, 4)
        #N_col,                     # (T, 1)
        #p**2,                      # (T, 4)
        p * N_col,                 # (T, 4)
        #N_col**2,                  # (T, 1)
        #p**3,                      # (T, 4)
        #(p**2) * N_col,            # (T, 4)
        p * (N_col**2),            # (T, 4)
        #N_col**3                   # (T, 1)
        (p**2)*(N_col)**2
        #p*(N_col**3), 
        #(p**2)*(N_col**3),
        #(p**3)*(N_col**3)
    ])
    return features @ w

# ----------------------
# Step 4: Loss and Optimizer
# ----------------------
def loss_fn(params, w, x, p_mid, dp_true):
    N_pred = encoder.apply(params, x).squeeze(-1).squeeze(-1)
    dp_pred = symbolic_model(p_mid, N_pred, w)
    mse = jnp.mean((dp_pred - dp_true)**2)
    sparsity = jnp.sum(jnp.abs(w))
    return mse + 1e-4 * sparsity

optimizer = optax.adam(1e-3)

@jax.jit
def update(params, w, opt_state, x, p_mid, dp_true):
    grads_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    loss, grads = grads_fn(params, w, x, p_mid, dp_true)
    updates, opt_state = optimizer.update(grads, opt_state, (params, w))
    params = optax.apply_updates(params, updates[0])
    w = optax.apply_updates(w, updates[1])
    return params, w, opt_state, loss

# ----------------------
# Step 5: Initialization and Training
# ----------------------
p_windows = p_windows[:, :, None]  # (batch, window, 1)
p_obs = jnp.array(p_obs)
dp_obs = jnp.array(dp_obs)
x = jnp.array(p_windows)

params = encoder.init(jax.random.PRNGKey(42), x)
w = jnp.zeros(10)
opt_state = optimizer.init((params, w))

for step in range(5000):
    params, w, opt_state, loss = update(params, w, opt_state, x, p_obs, dp_obs)
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.5f}")

# ----------------------
# Step 6: Visualization
# ----------------------
N_pred = encoder.apply(params, x).squeeze(-1).squeeze(-1)
X_rec = p_obs * N_pred
Y_rec = (1 - p_obs) * N_pred

# Rescale everything so that initial true and predicted populations match
true_init_sum = X[pad] + Y[pad]
pred_init_sum = X_rec[0] + Y_rec[0]
scale = true_init_sum / pred_init_sum

X_scaled = X[pad:-pad]
Y_scaled = Y[pad:-pad]
X_rec_scaled = X_rec * scale
Y_rec_scaled = Y_rec * scale

plt.figure(figsize=(10, 4))
plt.plot(t_eval[pad:-pad], X_scaled, label='True Prey (X)', color='blue')
plt.plot(t_eval[pad:-pad], Y_scaled, label='True Predator (Y)', color='red')
plt.plot(t_eval[pad:-pad], X_rec_scaled, '--', label='Reconstructed X', color='cyan')
plt.plot(t_eval[pad:-pad], Y_rec_scaled, '--', label='Reconstructed Y', color='orange')
plt.legend()
plt.title("Hidden State Reconstruction (Aligned at t=0)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.tight_layout()
plt.show()

# Print discovered symbolic weights
print("\nDiscovered symbolic weights:")
print(w.round(3))

