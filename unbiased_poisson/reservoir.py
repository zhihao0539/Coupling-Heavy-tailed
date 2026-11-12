"""UPAVE with uniform reservoir sampling: paper Section 4.4, O(R*d) memory.

Reference: pierrejacob/unbiasedpoisson, sample_unbiasedvar_reservoir.R.
Copyright (c) 2025 Pierre E. Jacob; see LICENSE for the upstream MIT notice.
All sums and weights are the same as in estimators.py; duplicate atoms need
not be combined. R independent reservoir slots sample with replacement.
"""
import time

import numpy as np

from .estimators import _positive_integer, _check_limit, _h, _correction_weight, poisson_estimate


def _add_atom(reservoir, x, weight, h, rng):
    if weight == 0:
        return
    value = _h(h, x)
    reservoir['pi_h'] += weight * value
    reservoir['pi_h2'] += weight * value**2
    reservoir['n_atoms'] += 1
    replace = rng.random(len(reservoir['weights'])) < 1. / reservoir['n_atoms']
    reservoir['atoms'][replace] = x
    reservoir['weights'][replace] = weight


def sample_reservoir_measure(single_kernel, coupled_kernel, rinit, h, rng, *,
                             lag, k, ell, R, max_iterations=None):
    """Accumulate pi_hat(h), pi_hat(h^2), and R uniform atoms without trajectories."""
    _positive_integer(lag, 'lag')
    _positive_integer(k, 'k', 0)
    _positive_integer(ell, 'ell', k)
    _positive_integer(R, 'R')
    if max_iterations is not None:
        _positive_integer(max_iterations, 'max_iterations', max(lag+1, ell))
    # Separate selection randomness from the chain path and initialization.
    selection_rng = np.random.default_rng(rng.integers(0, 2**63))
    x, y = np.array(rinit(rng), copy=True), np.array(rinit(rng), copy=True)
    reservoir = dict(atoms=np.empty((R,) + x.shape, dtype=x.dtype), weights=np.empty(R),
                     n_atoms=0, pi_h=0., pi_h2=0.)
    base_weight = 1. / (ell-k+1)
    for t in range(lag):
        if k <= t <= ell:
            _add_atom(reservoir, x, base_weight, h, selection_rng)
        x = single_kernel(x, rng)
    t, cost, meeting_time = lag, lag, None
    while True:
        if k <= t <= ell:
            _add_atom(reservoir, x, base_weight, h, selection_rng)
        if meeting_time is None:
            weight = _correction_weight(t, lag, k, ell)
            _add_atom(reservoir, x, weight, h, selection_rng)
            _add_atom(reservoir, y, -weight, h, selection_rng)
        if meeting_time is not None and t >= ell:
            break
        _check_limit(t, max_iterations)
        if meeting_time is None:
            x, y = coupled_kernel(x, y, rng)
            cost += 2
        else:
            x = single_kernel(x, rng)
            cost += 1
        t += 1
        if meeting_time is None and np.array_equal(x, y):
            meeting_time = t
    return dict(reservoir, meeting_time=meeting_time, cost=cost)


def _reservoir_poisson_term(measure, other_pi_h, coupled_kernel, h, y, rng, max_iterations):
    total, cost = 0., 0
    for x, weight in zip(measure['atoms'], measure['weights']):
        result = poisson_estimate(coupled_kernel, h, x, y, rng, max_iterations=max_iterations)
        total += measure['n_atoms'] * weight * (_h(h, x)-other_pi_h) * result['estimator']
        cost += result['cost']
    return total / len(measure['weights']), cost


def upave_reservoir(single_kernel, coupled_kernel, rinit, h, y, rng, *,
                    lag, k, ell, R=10, max_iterations=None):
    """One replicate of equation (4.4), retaining only R states per measure.

    Uses exact streaming sums for the stationary expectations and uniform
    atom subsampling for the Poisson terms. Unfinished runs raise; no clipping.
    """
    started = time.perf_counter()
    measures = [sample_reservoir_measure(single_kernel, coupled_kernel, rinit, h, rng,
                lag=lag, k=k, ell=ell, R=R, max_iterations=max_iterations) for _ in range(2)]
    pi_h = np.array([measure['pi_h'] for measure in measures])
    varh = .5*(measures[0]['pi_h2']+measures[1]['pi_h2']) - pi_h[0]*pi_h[1]
    term1, cost1 = _reservoir_poisson_term(measures[0], pi_h[1], coupled_kernel,
                                          h, y, rng, max_iterations)
    term2, cost2 = _reservoir_poisson_term(measures[1], pi_h[0], coupled_kernel,
                                          h, y, rng, max_iterations)
    cost_umcmc = sum(measure['cost'] for measure in measures)
    return dict(estimator=-varh+term1+term2, varh=varh, pi_h=pi_h,
                poisson_terms=np.array([term1, term2]),
                meeting_times=np.array([measure['meeting_time'] for measure in measures]),
                cost_umcmc=cost_umcmc, cost_poisson=cost1+cost2,
                cost=cost_umcmc+cost1+cost2, elapsed_seconds=time.perf_counter()-started)
