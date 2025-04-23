import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.stats import dirichlet as jax_dirichlet
import numpy as np
from scipy.stats import dirichlet as scipy_dirichlet

# Inputs (same for all backends)
x = jnp.array([0.2, 0.5, 0.3])
alpha = jnp.array([2.0, 3.0, 5.0])

# -------------------------
# Manual log PDF in JAX
# -------------------------
def manual_dirichlet_logpdf(x, alpha):
    log_norm_const = gammaln(jnp.sum(alpha)) - jnp.sum(gammaln(alpha))
    log_pdf = log_norm_const + jnp.sum((alpha - 1.0) * jnp.log(x))
    return log_pdf

# -------------------------
# Compute all versions
# -------------------------
manual_val = manual_dirichlet_logpdf(x, alpha)
jax_val = jax_dirichlet.logpdf(x, alpha)
scipy_val = scipy_dirichlet.logpdf(np.array(x), np.array(alpha))

# -------------------------
# Print and compare
# -------------------------
print("Manual log PDF (JAX) :", manual_val)
print("JAX log PDF          :", jax_val)
print("NumPy/SciPy log PDF  :", scipy_val)
print("JAX == Manual        :", jnp.allclose(jax_val, manual_val))
print("JAX == SciPy         :", jnp.allclose(jax_val, scipy_val))
#shape_param = 200.0

# Compute scale such that shape * scale = x_true
#scale = x_true / shape_param

