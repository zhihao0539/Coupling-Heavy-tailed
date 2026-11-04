"""Multivariate Student-t with nu = dimension + 1 and identity scale matrix.

Its unnormalized log density is
    -(nu + dimension)/2 * log(1 + sum_j x_j**2/nu).
"""
import numpy as np

DIMENSIONS = (1, 2, 5, 10, 20, 50, 100)


def target_info(dimension):
    return {"target": "multivariate_student_t", "dimension": int(dimension),
            "df": int(dimension + 1), "scale": "identity"}


def make_log_density(dimension, *, jax=False):
    """One vector -> scalar; also supports arrays of vectors on the last axis."""
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if jax:
        import jax.numpy as xp
    else:
        xp = np
    nu = dimension + 1

    def log_density(x):
        return -0.5 * (nu + dimension) * xp.log1p(xp.sum(x * x, axis=-1) / nu)

    return log_density
