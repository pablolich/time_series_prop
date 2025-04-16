import jax
import jax.numpy as jnp
import haiku as hk
import optax

# === 1. Data generation ===

def logistic_two_species(t, x, r=jnp.array([1.0, 0.8]), K=10.0):
    return r * x * (1 - (x.sum() / K))

def simulate_logistic_growth(x0, t, dt=0.1):
    xs = [x0]
    for _ in range(t - 1):
        dx = logistic_two_species(_, xs[-1])
        xs.append(xs[-1] + dt * dx)
    return jnp.stack(xs)

# Initial condition
x0 = jnp.array([2.0, 3.0])
data = simulate_logistic_growth(x0, t=100)
total_abundance = data.sum(axis=1)
observed_proportion = data[:, 0] / total_abundance

# === 2. Encoder (same as in paper) ===

def encoder_fn(x):
    return hk.Sequential([
        hk.Conv1D(128, kernel_shape=9, padding="VALID"),
        jax.nn.relu,
        hk.Conv1D(128, kernel_shape=1),
        jax.nn.relu,
        hk.Conv1D(1, kernel_shape=1),  # predict latent scalar: total abundance
    ])(x)

encoder = hk.transform(encoder_fn)

# === 3. Projection function ===

def project(x1_est, total_est):
    return x1_est / total_est

# === 4. Symbolic model (true model, hardcoded) ===

def symbolic_model(x, r=jnp.array([1.0, 0.8]), K=10.0):
    return r * x * (1 - (x.sum() / K))

# === 5. Model pipeline ===

def model_apply(params, seq_obs):
    latent = encoder.apply(params, None, seq_obs)  # shape: [B, 1]
    pad = 4
    x1 = seq_obs[:, pad:-pad, 0]  # center point
    s = latent[:, 0]  # predicted total abundance
    x1_est = x1 * s  # reconstruct full state: x1 known, x2 = s - x1
    x2_est = s - x1_est
    x_est = jnp.stack([x1_est, x2_est], axis=-1)
    dxdt = symbolic_model(x_est)
    return dxdt, x1_est, s

# === 6. Loss: compare predicted proportion derivative to finite difference ===

def loss_fn(params, obs_seq):
    pad = 4
    dxdt, x1_est, s = model_apply(params, obs_seq)
    z_pred = project(x1_est, s)

    z_true = obs_seq[:, pad:-pad, 0]  # center point
    jac_fn = jax.jacrev(lambda x1, s: x1 / s, argnums=(0, 1))
    dzdx1, dzds = jax.vmap(jac_fn)(x1_est, s)

    dx1dt = dxdt[:, 0]
    dsdt = dxdt.sum(axis=1)
    dzdt_pred = (dzdx1.squeeze() * dx1dt + dzds.squeeze() * dsdt)

    import ipdb; ipdb.set_trace(context = 20)
    dzdt_true = (obs_seq[:, 5, 0] - obs_seq[:, 3, 0]) / 2

    return jnp.mean((dzdt_true - dzdt_pred) ** 2)

# === 7. Training loop ===

key = jax.random.PRNGKey(42)
B = 32
X = observed_proportion[:, None]  # [T, 1]
X_seq = jnp.stack([X[i:i+9] for i in range(len(X)-9)])  # [T-9, 9, 1]

init_params = encoder.init(key, X_seq[:B])

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(init_params)

@jax.jit
def update(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
for epoch in range(1000):
    batch = X_seq[:B]
    init_params, opt_state, l = update(init_params, opt_state, batch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {l:.4f}")

