"""Skew-t target: Azzalini-Capitanio skew-t with identity scale matrix.

The unnormalized log density is
    log t_D(x; NU) + log T_{D+NU}(alpha'x * sqrt((D + NU) / (NU + x'x))),
with alpha = (ALPHA, -ALPHA, 0, ..., 0). The same density is given in NumPy
for the coupling runs and in JAX for VI.
"""
import numpy as np
from scipy.special import stdtr

D = 100        # Dimension.
NU = 2.0       # Degrees of freedom.
ALPHA = 100.0  # Skewness vector alpha = (ALPHA, -ALPHA, 0, ..., 0).


def log_density(x):
    """NumPy log density for one length-D vector, used by the coupled samplers."""
    norm2 = x @ x
    skew = ALPHA * (x[0] - x[1]) * np.sqrt((D + NU) / (NU + norm2))
    return -(D + NU) / 2 * np.log1p(norm2 / NU) + np.log(stdtr(D + NU, skew))


def log_density_jax(x):
    """The same density in JAX for fit_vi; one length-D vector in, one scalar out.

    Import generic_algorithms.vi first; it switches JAX to 64-bit arithmetic.
    """
    import jax.numpy as jnp
    from jax.scipy.special import betainc

    norm2 = jnp.sum(x * x, axis=-1)
    skew = ALPHA * (x[..., 0] - x[..., 1]) * jnp.sqrt((D + NU) / (NU + norm2))
    df = D + NU
    # Student-t log CDF through the regularized incomplete beta function.
    w = jnp.clip(df / (df + skew * skew), 0.0, 1.0 - jnp.finfo(jnp.float64).eps)
    b = betainc(df / 2, 0.5, w)
    log_cdf = jnp.where(skew < 0,
                        jnp.log(0.5) + jnp.log(jnp.maximum(b, jnp.finfo(jnp.float64).tiny)),
                        jnp.log1p(-0.5 * b))
    return -(D + NU) / 2 * jnp.log1p(norm2 / NU) + log_cdf
