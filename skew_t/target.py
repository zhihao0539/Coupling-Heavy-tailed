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
    from jax.scipy.special import betainc, gammaln

    norm2 = jnp.sum(x * x, axis=-1)
    skew = ALPHA * (x[..., 0] - x[..., 1]) * jnp.sqrt((D + NU) / (NU + norm2))
    df = D + NU
    # Near zero, the beta formula loses precision and its gradient is singular.
    # A local CDF expansion preserves the nonzero derivative at skew == 0.
    near_zero = jnp.abs(skew) < 1e-4
    safe_skew = jnp.where(near_zero, 1.0, skew)
    w = df / (df + safe_skew * safe_skew)
    b = betainc(df / 2, 0.5, w)
    log_cdf = jnp.where(safe_skew < 0,
                        jnp.log(0.5) + jnp.log(jnp.maximum(b, jnp.finfo(jnp.float64).tiny)),
                        jnp.log1p(-0.5 * b))
    t = jnp.where(near_zero, skew, 0.0)
    pdf_at_zero = jnp.exp(gammaln((df + 1) / 2) - gammaln(df / 2)) / jnp.sqrt(df * jnp.pi)
    central_cdf = 0.5 + pdf_at_zero * t * (1 - (df + 1) * t * t / (6 * df))
    log_cdf = jnp.where(near_zero, jnp.log(central_cdf), log_cdf)
    return -(D + NU) / 2 * jnp.log1p(norm2 / NU) + log_cdf
