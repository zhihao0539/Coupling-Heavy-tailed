"""A small new application: define a target and settings, fit, then couple."""
import numpy as np
import jax.numpy as jnp
from .algorithms import coupled_sampler
from .transformations import default_parameters
from .vi import fit_vi

DIMENSION = 3
DF = 2.


def log_pdf(x):
    """Unnormalized Student-t log density; one d-vector in, one scalar out."""
    return -(DIMENSION + DF) / 2 * jnp.log1p(jnp.sum(x * x) / DF)


def main():
    for method, h in [('scs', .15), ('dcs', .2)]:
        parameters, info, history = fit_vi(
            log_pdf, DIMENSION, method,
            initial_parameters=default_parameters(DIMENSION, method, radius=np.sqrt(DF)),
            seed=42, steps=500, batch_size=256, change_after=350,
            average_last=100, holdout_size=2000,
        )
        # This example uses the same JAX evaluator for VI and sampling. A NumPy
        # version of the same density is faster for long Python sampling loops.
        result = coupled_sampler(
            np.zeros(DIMENSION), np.ones(DIMENSION), log_pdf,
            h=h, n_steps=2000, rng=np.random.default_rng(100),
            method=method, parameters=parameters,
        )
        print(method.upper(), 'radius:', parameters['radius'],
              'holdout objectives:', info['candidate_holdout_losses'],
              'meeting time:', result['meeting_time'])


if __name__ == '__main__':
    main()
