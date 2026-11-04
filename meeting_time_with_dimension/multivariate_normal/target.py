"""Multivariate normal target; edit mu and Sigma in get_target_parameters."""
import numpy as np

DIMENSIONS = (1, 2, 5, 10, 20, 50, 100)


def get_target_parameters(dimension):
    """Set the mean vector and covariance matrix here, once per dimension."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    mu = np.zeros(dimension)
    Sigma = np.eye(dimension)
    mu, Sigma = np.asarray(mu, dtype=float), np.asarray(Sigma, dtype=float)
    if mu.shape != (dimension,) or Sigma.shape != (dimension, dimension):
        raise ValueError("mu and Sigma must have shapes (d,) and (d,d).")
    if not np.isfinite(mu).all() or not np.isfinite(Sigma).all():
        raise ValueError("mu and Sigma must be finite.")
    if not np.allclose(Sigma, Sigma.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Sigma must be symmetric.")
    np.linalg.cholesky(Sigma)  # Require a positive-definite covariance.
    return mu, Sigma


def target_info(dimension):
    mu, Sigma = get_target_parameters(dimension)
    return {"target": "multivariate_normal", "dimension": int(dimension),
            "mu": mu.tolist(), "Sigma": Sigma.tolist()}


def sample_target(dimension, size, rng):
    """Independent exact N(mu, Sigma) draws, also used for initialization."""
    mu, Sigma = get_target_parameters(dimension)
    return rng.multivariate_normal(mu, Sigma, size=size, method="cholesky")


def make_log_density(dimension, *, jax=False):
    """One vector -> scalar; also supports arrays of vectors on the last axis."""
    mu, Sigma = get_target_parameters(dimension)
    chol = np.linalg.cholesky(Sigma)
    whitening = np.linalg.solve(chol, np.eye(dimension))
    constant = -0.5 * dimension * np.log(2 * np.pi) - np.log(np.diag(chol)).sum()
    if jax:
        import jax.numpy as xp
    else:
        xp = np
    def log_density(x):
        standardized = xp.matmul(x - xp.asarray(mu), xp.asarray(whitening).T)
        return constant - 0.5 * xp.sum(standardized * standardized, axis=-1)

    return log_density
