"""Unbiased Poisson asymptotic variance estimation, with scalar test functions."""
from .estimators import (
    sample_signed_measure, integrate, stationary_variance, poisson_estimate,
    upave, run_upave, pilot_lag,
)

# Import native wrappers explicitly from unbiased_poisson.kernels when needed.
# The estimator functions also accept entirely different user-supplied kernels.
