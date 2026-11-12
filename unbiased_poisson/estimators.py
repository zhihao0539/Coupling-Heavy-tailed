"""UPAVE, equations (2.4), (3.5), (4.3), and (4.4) of arXiv:2206.05691.

Kernels take NumPy states and an explicit Generator. h is scalar-valued.
Only NumPy is required; no exact target samples are needed.
Algorithm reference: https://github.com/pierrejacob/unbiasedpoisson
Copyright (c) 2025 Pierre E. Jacob; see LICENSE for the upstream MIT notice.
"""
import time

import numpy as np


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _check_limit(t, max_iterations):
    if max_iterations is not None and t >= max_iterations:
        raise RuntimeError("Iteration limit reached. No truncated unbiased estimate is returned.")


def _h(h, x):
    value = float(h(x))
    if not np.isfinite(value):
        raise FloatingPointError("h must return one finite scalar.")
    return value


def _correction_weight(t, lag, k, ell):
    """v_t / (ell-k+1), equations (3.5)-(3.6); zero before k+lag."""
    if t < k + lag:
        return 0.
    ceiling = (max(lag, t - ell) + lag - 1) // lag
    return ((t - k) // lag - ceiling + 1) / (ell - k + 1)


def _prune(atoms, weights):
    """Combine exactly identical states, as in the authors' implementation."""
    atoms, inverse = np.unique(np.asarray(atoms), axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=np.asarray(weights), minlength=len(atoms))
    keep = weights != 0
    return atoms[keep], weights[keep]


def sample_signed_measure(single_kernel, coupled_kernel, rinit, rng, *,
                          lag, k, ell, max_iterations=None):
    """Sample pi_hat_{k:ell}; average X_k,...,X_ell and add lagged corrections.

    single_kernel(x,rng) returns x_new; coupled_kernel(x,y,rng) returns a pair.
    rinit(rng) draws from the chosen initial law pi_0, not necessarily pi.
    Meeting is exact equality of native states. Costs count a pair step as two.
    All nonzero atoms are stored for clarity; this is not reservoir sampling.
    """
    _positive_integer(lag, "lag")
    _positive_integer(k, "k", 0)
    _positive_integer(ell, "ell", k)
    if max_iterations is not None:
        _positive_integer(max_iterations, "max_iterations", max(lag + 1, ell))
    x, y = np.array(rinit(rng), copy=True), np.array(rinit(rng), copy=True)
    atoms, weights = [], []
    base_weight = 1. / (ell - k + 1)
    for t in range(lag):
        if k <= t <= ell:
            atoms.append(x.copy())
            weights.append(base_weight)
        x = single_kernel(x, rng)
    t, cost, meeting_time = lag, lag, None
    while True:
        if k <= t <= ell:
            atoms.append(x.copy())
            weights.append(base_weight)
        weight = _correction_weight(t, lag, k, ell)
        if meeting_time is None and weight != 0:
            atoms.extend((x.copy(), y.copy()))
            weights.extend((weight, -weight))
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
    atoms, weights = _prune(atoms, weights)
    return dict(atoms=atoms, weights=weights, meeting_time=meeting_time, cost=cost)


def integrate(measure, h):
    """Evaluate pi_hat(h) using the signed weights, without renormalizing."""
    return float(measure["weights"] @ np.array([_h(h, x) for x in measure["atoms"]]))


def stationary_variance(measure1, measure2, h):
    """Equation (4.3); the two measures must be independent."""
    means = np.array([integrate(measure1, h), integrate(measure2, h)])
    def h_squared(x):
        return _h(h, x)**2
    second_moment = .5 * (integrate(measure1, h_squared) + integrate(measure2, h_squared))
    return dict(estimator=float(second_moment - means[0]*means[1]), pi_h=means)


def poisson_estimate(coupled_kernel, h, x, y, rng, *, max_iterations=None):
    """G_y(x), equation (2.4): synchronous coupling, including the time-zero term."""
    if max_iterations is not None:
        _positive_integer(max_iterations, "max_iterations")
    x, y = np.array(x, copy=True), np.array(y, copy=True)
    value, t = 0., 0
    while not np.array_equal(x, y):
        _check_limit(t, max_iterations)
        value += _h(h, x) - _h(h, y)
        x, y = coupled_kernel(x, y, rng)
        t += 1
    return dict(estimator=float(value), meeting_time=t, cost=2*t)


def _poisson_term(measure, other_pi_h, coupled_kernel, h, y, R, rng, max_iterations):
    """One side of (4.4), with uniform atom selection xi_n=1/N."""
    total, cost = 0., 0
    n = len(measure["weights"])
    for _ in range(R):
        index = rng.integers(n)
        x = measure["atoms"][index]
        result = poisson_estimate(coupled_kernel, h, x, y, rng, max_iterations=max_iterations)
        total += n * measure["weights"][index] * (_h(h, x) - other_pi_h) * result["estimator"]
        cost += result["cost"]
    return total / R, cost


def upave(single_kernel, coupled_kernel, rinit, h, y, rng, *,
          lag, k, ell, R=1, max_iterations=None):
    """One UPAVE replicate, equation (4.4); y is a fixed reference state.

    No exact samples from pi are used. Each Poisson run uses fresh randomness.
    Negative estimates are retained. A finite iteration limit raises on failure.
    """
    _positive_integer(R, "R")
    started = time.perf_counter()
    measures = [sample_signed_measure(single_kernel, coupled_kernel, rinit, rng,
                lag=lag, k=k, ell=ell, max_iterations=max_iterations) for _ in range(2)]
    variance = stationary_variance(measures[0], measures[1], h)
    term1, cost1 = _poisson_term(measures[0], variance["pi_h"][1], coupled_kernel,
                                h, y, R, rng, max_iterations)
    term2, cost2 = _poisson_term(measures[1], variance["pi_h"][0], coupled_kernel,
                                h, y, R, rng, max_iterations)
    cost_umcmc = sum(measure["cost"] for measure in measures)
    return dict(estimator=-variance["estimator"] + term1 + term2,
                varh=variance["estimator"], pi_h=variance["pi_h"],
                poisson_terms=np.array([term1, term2]),
                meeting_times=np.array([measure["meeting_time"] for measure in measures]),
                cost_umcmc=cost_umcmc, cost_poisson=cost1+cost2,
                cost=cost_umcmc+cost1+cost2, elapsed_seconds=time.perf_counter()-started)


def run_upave(single_kernel, coupled_kernel, rinit, h, y, *,
              lag, k, ell, R=1, n_replicates=100, seed=42,
              max_iterations=None, progress_every=10):
    """Average independent UPAVE replicates; MCSE requires finite estimator variance."""
    _positive_integer(n_replicates, "n_replicates")
    seeds = np.random.SeedSequence(seed).spawn(n_replicates)
    rows = []
    for i, child_seed in enumerate(seeds):
        rows.append(upave(single_kernel, coupled_kernel, rinit, h, y,
            np.random.default_rng(child_seed), lag=lag, k=k, ell=ell, R=R,
            max_iterations=max_iterations))
        if progress_every and ((i+1) % progress_every == 0 or i+1 == n_replicates):
            print(f"UPAVE: finished {i+1}/{n_replicates}", flush=True)
    estimates = np.array([row["estimator"] for row in rows])
    variance = float(estimates.var(ddof=1)) if n_replicates > 1 else np.nan
    mean_cost = float(np.mean([row["cost"] for row in rows]))
    return dict(mean=float(estimates.mean()), mcse=float(np.sqrt(variance/n_replicates)),
                estimates=estimates, mean_cost=mean_cost,
                estimator_inefficiency=mean_cost*variance, replicates=rows)


def pilot_lag(single_kernel, coupled_kernel, rinit, *, n_pilot=100,
              quantile=.95, seed=0, max_iterations=None):
    """Section 4.4 heuristic: lag-one pilots, then lag=k=high quantile, ell=5k.

    Use an independent seed for production. These pilots retain only two states.
    """
    _positive_integer(n_pilot, "n_pilot")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one.")
    if max_iterations is not None:
        _positive_integer(max_iterations, "max_iterations", 2)
    rng = np.random.default_rng(seed)
    meetings = []
    for _ in range(n_pilot):
        x, y = np.array(rinit(rng), copy=True), np.array(rinit(rng), copy=True)
        x, t = single_kernel(x, rng), 1
        while True:
            _check_limit(t, max_iterations)
            x, y = coupled_kernel(x, y, rng)
            t += 1
            if np.array_equal(x, y):
                meetings.append(t)
                break
    lag = max(1, int(np.ceil(np.quantile(meetings, quantile))))
    return dict(lag=lag, k=lag, ell=5*lag, meeting_times=np.array(meetings))
