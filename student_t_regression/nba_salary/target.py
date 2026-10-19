"""NBA Student-t regression in (beta, v), with v=sinh(log(sigma)).

As in student_t_regression, standardize predictors and response, use no
intercept, and model errors with a t_4 density. The beta prior is product
t_(D+1), where D includes the transformed scale coordinate (82, so df=83);
the prior is flat in u=log(sigma). The nonlinear scale coordinate v makes the
SCS reverse-KL VI objective finite without changing the posterior in (beta,u).
"""
from pathlib import Path

import numpy as np
import pandas as pd

NU = 4                              # Degrees of freedom of the regression errors.
FOLDER = Path(__file__).resolve().parent
RESPONSE = 'sqrt_salary_million'


def load_data():
    data = pd.read_csv(FOLDER / 'nba_salary.csv')
    X = data.drop(columns=RESPONSE).to_numpy(dtype=float)
    y = data[RESPONSE].to_numpy(dtype=float)
    # Same standardization as student_t_regression; no intercept is added.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


X, Y = load_data()
BETA_DIM = X.shape[1]
D = BETA_DIM + 1                   # Beta coefficients followed by v=sinh(log(sigma)).
PRIOR_DF = D + 1


def log_density(theta):
    """Unnormalized density in (beta, v) coordinates for samplers."""
    beta, v = theta[:-1], theta[-1]
    u = np.arcsinh(v)
    residuals = Y - X @ beta
    # logaddexp avoids squaring residual/sigma when sigma is very small.
    with np.errstate(divide='ignore'):
        terms = 2 * np.log(np.abs(residuals)) - np.log(NU) - 2 * u
    log_likelihood = -.5 * (NU + 1) * np.logaddexp(0, terms).sum() - len(Y) * u
    log_prior = -.5 * (PRIOR_DF + 1) * np.log1p(beta * beta / PRIOR_DF).sum()
    # du/dv = 1/sqrt(1+v**2); hypot stays finite for very large |v|.
    log_scale_jacobian = -np.log(np.hypot(1.0, v))
    return log_likelihood + log_prior + log_scale_jacobian


def log_density_jax(theta):
    """The same density in JAX for VI; one length-D vector in, one scalar out.

    The shared run_vi runner enables 64-bit JAX before evaluating this target.
    """
    import jax.numpy as jnp

    beta, v = theta[:-1], theta[-1]
    u = jnp.arcsinh(v)
    residuals = jnp.asarray(Y) - jnp.asarray(X) @ beta
    # Keep the VI gradient finite if a residual is exactly zero.
    safe_abs = jnp.maximum(jnp.abs(residuals), jnp.finfo(residuals.dtype).tiny)
    terms = 2 * jnp.log(safe_abs) - jnp.log(NU) - 2 * u
    log_likelihood = -.5 * (NU + 1) * jnp.logaddexp(0, terms).sum() - len(Y) * u
    log_prior = -.5 * (PRIOR_DF + 1) * jnp.log1p(beta * beta / PRIOR_DF).sum()
    log_scale_jacobian = -jnp.log(jnp.hypot(1.0, v))
    return log_likelihood + log_prior + log_scale_jacobian
