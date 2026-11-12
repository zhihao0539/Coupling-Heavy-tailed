"""Small runnable example: Euclidean RWM for N(0,1), initialized away from pi."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unbiased_poisson import pilot_lag, run_upave
from unbiased_poisson.kernels import euclidean_kernels


def log_pdf(x):
    return -.5 * np.sum(x*x)


kernels = euclidean_kernels(log_pdf, step_size=1.5)


def rinit(rng):
    # Initial law is N(3, 0.5^2), not the target N(0,1).
    return kernels['to_native'](rng.normal(3., .5, size=1))


def h(state):
    # For transformed methods, h still evaluates the original Euclidean state.
    return kernels['to_euclidean'](state)[0]


def main():
    pilot = pilot_lag(kernels['single_kernel'], kernels['coupled_kernel'], rinit,
                      n_pilot=50, quantile=.95, seed=10)
    print(f"Using lag=k={pilot['lag']}, ell={pilot['ell']}")
    result = run_upave(
        kernels['single_kernel'], kernels['coupled_kernel'], rinit, h,
        y=kernels['to_native'](np.zeros(1)),
        lag=pilot['lag'], k=pilot['k'], ell=pilot['ell'], R=10,
        n_replicates=100, seed=20,
    )
    print(f"Estimated v(P,h): {result['mean']:.4f}")
    print(f"Monte Carlo SE:   {result['mcse']:.4f}")
    print(f"Mean cost:        {result['mean_cost']:.1f} marginal transitions")


if __name__ == '__main__':
    main()
