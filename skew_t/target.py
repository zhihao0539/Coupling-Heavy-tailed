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


def stdtr_jax(df, t):
    """Student-t CDF: T_df(t) = I_u(df/2, df/2), u = (1 + t/sqrt(df+t²))/2."""
    import jax.numpy as jnp
    from jax.scipy.special import betainc

    u = (1 + t / jnp.sqrt(df + t * t)) / 2
    return betainc(df / 2, df / 2, u)


def log_density_jax(x):
    """The same density in JAX for VI; one length-D vector in, one scalar out.

    The shared run_vi runner enables 64-bit JAX before evaluating this target.
    """
    import jax.numpy as jnp

    norm2 = x @ x
    skew = ALPHA * (x[0] - x[1]) * jnp.sqrt((D + NU) / (NU + norm2))
    return -(D + NU) / 2 * jnp.log1p(norm2 / NU) + jnp.log(stdtr_jax(D + NU, skew))
