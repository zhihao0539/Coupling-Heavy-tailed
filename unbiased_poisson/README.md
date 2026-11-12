# Unbiased Poisson asymptotic variance estimation

Simple NumPy implementation of UPAVE, Section 4.3 of
[Solving the Poisson equation using coupled Markov chains](https://arxiv.org/abs/2206.05691).
It implements the paper's general estimator, without requiring exact samples
from the target. The test function `h` is scalar-valued.

The formulas and indexing were checked against the authors'
[implementation](https://github.com/pierrejacob/unbiasedpoisson), commit
`d1c86281ffbb78ab673911559cd134b756a4ae62`, particularly
`sample_unbiasedvar.R`, `sample_unbiasedfishy.R`, and `c_chains_to_measure.cpp`.
The upstream MIT license notice is included in `LICENSE`.

## Files and functions

- `estimators.py`: signed measures, stationary variance, Poisson sums, UPAVE,
  independent repetitions, and pilot-based lag selection.
- `kernels.py`: separate wrappers for Euclidean, stereographic, SCS, DCS
  Mobius ball walk, and DCS matrix ball walk. These reuse proposal functions,
  transformations, and MH decisions in `generic_algorithms`.
- `example.py`: a small complete example with nonstationary initial states.

Run the example from this folder with your scientific Python environment:

```sh
python example.py
```

## Using existing coupling methods

Choose a factory in `kernels.py`, passing the map parameters directly. For example:

```python
import numpy as np
from unbiased_poisson import pilot_lag, run_upave
from unbiased_poisson.kernels import dcs_matrix_ball_kernels

kernels = dcs_matrix_ball_kernels(
    log_pdf, step_size=0.1, mu=mu,
    preconditioning_matrix=L, beta=beta,
)

def rinit(rng):
    return kernels['to_native'](rng.normal(size=len(mu)))

def h(state):
    return kernels['to_euclidean'](state)[0]

pilot = pilot_lag(kernels['single_kernel'], kernels['coupled_kernel'], rinit,
                  n_pilot=100, seed=10)
result = run_upave(
    kernels['single_kernel'], kernels['coupled_kernel'], rinit, h,
    y=kernels['to_native'](mu),
    lag=pilot['lag'], k=pilot['k'], ell=pilot['ell'], R=10,
    n_replicates=100, seed=20,
)
print(result['mean'], result['mcse'])
```

Here `log_pdf`, `mu`, `L`, and `beta` are supplied by your experiment.
The example initial law is illustrative; choose an initial law that meets the
paper's assumptions for your target. A tractable initial law need not equal pi.

The estimator also accepts arbitrary user-defined NumPy-state callbacks:
`single_kernel(x, rng)`, `coupled_kernel(x, y, rng)`, and `rinit(rng)`.
The two marginals of the coupled kernel must match the single kernel;
exactly equal states must remain equal. Kernels must not mutate input states.

## Meaning of the settings and output

- `lag`, `k`, `ell`: the paper's L, k, ell for the unbiased signed measure.
  The averaging window includes both k and ell. Lag is supplied explicitly.
- `R`: number of uniformly selected atoms per signed measure. Each UPAVE
  replicate generates two independent measures and 2R fresh Poisson runs.
- `y`: fixed reference state for G_y(x), chosen before production.
- `pilot_lag`: starts with lag-one pilots, selects a high meeting-time quantile
  (95% by default), and suggests k=L and ell=5L. The paper does not prescribe
  the percentile. Use separate pilot and production seeds, and keep settings fixed.
- `result['mean']`, `result['mcse']`: average UPAVE estimate and its Monte Carlo SE.
- `result['replicates']`: individual estimates, stationary-variance estimates,
  estimated means, meeting times, transition costs, and elapsed times.
- `result['estimator_inefficiency']`: mean cost times the sample variance of
  UPAVE. This measures precision per cost when estimating v(P,h); it is not
  itself the asymptotic variance of the sampler.

All chains stay in native coordinates. Only initialization and evaluations of
the observable need transformations; this avoids losing exact meetings through
repeated forward/inverse rounding. SCS supports either latitude, and both DCS
factories support general positive beta. Keep the VI map parameters and proposal
step size fixed throughout estimation, so the marginal kernel P stays fixed.

The implementation stores signed-measure atoms and combines exact duplicates.
This keeps the code small; very long meetings can require substantial memory.
There is no automatic iteration cutoff. Optional `max_iterations` raises an
error if a run is unfinished, rather than returning a truncated estimate.
Do not discard failures or replace them with zero. Estimates can be negative
and are not clipped. Unbiasedness and finite-variance MCSEs require the paper's
moment and initialization conditions; the code cannot establish these for an
arbitrary supplied kernel and observable.

The supplied factories use the existing private one-step helpers in
`generic_algorithms/algorithms.py`; keep those modules together if moving this code.
