"""Centered correlated Gaussian with covariance Sigma[i,j] = RHO**abs(i-j)."""
import numpy as np

DIMENSIONS = (5, 10, 20, 50, 100)
RHO = 0.9


def get_target_parameters(dimension):
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if not np.isfinite(RHO) or abs(RHO) >= 1:
        raise ValueError("RHO must lie strictly between -1 and 1.")
    indices = np.arange(dimension)
    return np.zeros(dimension), RHO ** np.abs(indices[:, None] - indices[None, :])


def target_info(dimension):
    get_target_parameters(dimension)
    return dict(target="multivariate_normal_ar1", dimension=int(dimension), rho=RHO)


def sample_target(dimension, size, rng):
    mu, covariance = get_target_parameters(dimension)
    return rng.multivariate_normal(mu, covariance, size=size, method="cholesky")


def make_log_density(dimension, *, jax=False):
    """Normalized AR(1) Gaussian density, evaluated using its independent innovations."""
    get_target_parameters(dimension)
    if jax:
        import jax.numpy as xp
    else:
        xp = np
    rho = RHO
    variance = 1 - rho**2
    constant = -.5 * dimension * np.log(2*np.pi) - .5 * (dimension-1) * np.log(variance)

    def log_density(x):
        innovations = x[..., 1:] - rho*x[..., :-1]
        quadratic = x[..., 0]**2 + xp.sum(innovations**2, axis=-1) / variance
        return constant - .5*quadratic

    return log_density
