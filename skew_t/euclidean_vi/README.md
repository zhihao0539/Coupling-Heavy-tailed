# Standalone Gaussian-VI-preconditioned RWMH for skew-t

This comparison leaves the main `skew_t/run_experiment.py` runner and its
saved isotropic Euclidean results unchanged. The target remains the
100-dimensional skew-t with 2 degrees of freedom and skewness
`(100, -100, 0, ..., 0)`.

`../run_euclidean_vi.py` uses the existing generic VI and coupling functions.
It minimizes reverse KL with a full Gaussian reference, learning both the
location `mu` and the Cholesky factor `L=preconditioning_matrix`. The map is
`x=mu+Lz`, with covariance `Sigma=L L.T`. The proposal in original coordinates
has covariance `h^2 Sigma`.

VI starts from zero location and identity matrix, with seed 2026111801,
6000 Adam updates, batches of 2048, rates 0.01 then 0.002 after update 4500,
and an independent 50000-draw holdout. Holdout selection chooses between the
last iterate and the average of the last 500 unconstrained parameter vectors.

The scalar step is fixed at `h=0.40` after independent pilot calibration.
A 128-pair pilot using the full cutoff gave 22.66% acceptance; production
acceptance was 20.20%. `selected_step.json` records the pilot summary.
Acceptance is averaged over both chain rates and then over pairs, with each
pair stopped at meeting or cutoff.

Production reuses all 1000 pairs in `../new_initial_states.npz`, proposal seed
42 with offset 200000, and a 1000000-iteration cutoff. These starting pairs
come from uniform bright-cap points mapped through the saved SCS fit; they
are not target-distributed draws. All starts and seeds are shared with the
original Euclidean comparison.

| RWMH | Meetings | Mean among pairs that met | Acceptance |
|---|---:|---:|---:|
| Original isotropic, h=0.35 | 968/1000 | 50400.47 | 24.93% |
| Gaussian VI, h=0.40 | 946/1000 | 69438.41 | 20.20% |

Gaussian VI did not improve the observed meeting statistics in this comparison.
The mean among met pairs excludes censored trials. In `meeting_times.csv`,
`meeting_time=-1` identifies the 54 pairs that reached the cutoff.

From the repository root, rerun both VI and coupling with:

```sh
python skew_t/run_euclidean_vi.py
```

To reuse the supplied fit, call `run_euclidean(starts)` directly instead of
`main()`. It returns rows without overwriting other methods' results.
