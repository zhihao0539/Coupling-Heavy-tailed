# Multivariate normal coupling comparison

The target is N(mu, Sigma) in dimensions 1, 2, 5, 10, 20, 50 and 100.
Edit `mu` and `Sigma` in `target.py:get_target_parameters` to change the target.
They default to a zero vector and identity covariance. Sigma must be symmetric
and positive definite. Density evaluation and exact initial draws share these
parameters. Each of the five methods uses 1,000 independent target pairs and
a 1,000,000-update cutoff, through separate stereographic, SCS, Mobius-DCS ball-walk and Gaussian-walk, and Euclidean experiment functions.

There is no VI or fitted-parameter loading. `map_parameters(d)` in
`run_experiment.py` defines the maps for five methods:

- stereographic: zero shift and isotropic preconditioning matrix sqrt(d) I;
- SCS11: latitude 1.1, zero shift and horizontal observer, isotropic matrix sqrt(d)/11 I;
- DCS: zero shift, beta=2, scalar R=sqrt(d+2), Mobius parameter delta=0;
- Euclidean: a scalar Gaussian random-walk step, without a transformation.
- DCS random walk: the same fixed DCS map, with Gaussian proposals in B(1).

The DCS choice is motivated by the paper's radial map with a Mobius
automorphism, x=R*RC_2(M_delta(s)) on normalized ball coordinates s in B(1):
https://arxiv.org/abs/2608.04284 (Section 2.2 and equation RCtotal).
For the standard normal and delta=0, putting t=||s||^2 gives the pull-back
potential, up to an additive constant,

    U(t) = R^2*t/[2*(1-t)] + (d+2)/2 * log(1-t).

Choosing R^2=d+2 makes U'(t)=(d+2)*t/[2*(1-t)^2] nonnegative and U''(t)
nonnegative, hence U(||s||^2) is convex. This is a Gaussian-specific fixed
choice derived here, not a prescribed optimal Gaussian radius from the paper.
It is asymptotically sqrt(d). delta=0 gives the identity Mobius transformation
for this centered symmetric target; it remains editable in map_parameters.
The sampler works on B(1): its step h corresponds to a radius R*h on B(R).

`run_experiment.py` has five editable step-size lists. For the original four
methods, tuning targets average Metropolis acceptance 0.23 using independent exact normal states and fresh
validation draws. Spherical settings with a higher large-step plateau keep a
reasonable large step. Five-pair fixed-length pilots additionally check the
method-specific experiment functions. These pilots do not run the full coupling experiment.

`run_all.py` runs the couplings and plots directly. `run_vi.py` is a
compatibility entry point that explains that no VI is needed. `plot_results.py`
plots the supplied normal results and checks the target metadata. The legacy
`fitted_parameters.json` files in this supplied folder describe Student-t
fits; the fixed-map normal experiment never reads them.

All data are read/saved in this experiment's own results/ directory. Map shifts
and scalar radii are fixed independently of the editable target mean/covariance.
Changing the target parameters therefore requires retuning the step sizes;
there is no automatic covariance preconditioning.

## DCS Gaussian random walk

`dcs_rw` calls `run_dcs_random_walk_experiment` with the same fixed `mu=0`,
`R=sqrt(d+2)`, and `delta=0` as the DCS ball walk. No VI is needed. Its step
size `h` is the Gaussian standard deviation in each native B(1) coordinate;
the corresponding standard deviation in B(R) is `R*h`. Proposals outside
B(1) are rejected.

`DCS_RANDOM_WALK_STEP_SIZES` contains these calibrated values in dimension order:

| Dimension | h | Independent pilot acceptance |
|---|---:|---:|
| 1 | 6.9126 | 22.80% |
| 2 | 1.65894 | 23.18% |
| 5 | 0.474046 | 22.95% |
| 10 | 0.196808 | 22.93% |
| 20 | 0.0887964 | 23.23% |
| 50 | 0.0343654 | 23.07% |
| 100 | 0.017 | 23.24% |

These steps target the average printed by experiments that stop at meeting:
for each chain, divide accepted proposals by the pair's iterations through
meeting (or the cutoff), then average across both chains and all pairs.
Independent validation used 100,000, 40,000, 20,000, 8,000, 3,000, 1,024, and
512 pairs in dimension order, with the production cutoff of 1,000,000.
Every validation pair met. A production run of 1,000 pairs will fluctuate
around these pilot estimates. Retune after changing the target, the DCS map,
or the stopping rule.

The existing four methods keep their scales, maps, and random streams.
DCS random walk uses the same initial pairs and a separate proposal stream
(index 5). Its plots and summaries appear under `dcs_rw`. Existing normal
results without this new method remain plottable. Existing saved results
have not been rerun or replaced.
