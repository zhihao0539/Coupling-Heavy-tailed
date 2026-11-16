# NBA salary: stereographic and matrix DCS ball-walk coupling

This experiment compares VI-tuned stereographic and matrix DCS ball-walk couplings using the shared
[`generic_algorithms`](../../generic_algorithms/README.md) VI and coupling
runners. Replicates run sequentially.

The target in [`target.py`](target.py) uses standardized predictors and response,
no intercept, Student-t errors with 4 degrees of freedom, and independent standard
Student-t priors with `D+1` degrees of freedom on the coefficients. With the
supplied data, there are 81 coefficients and 82 state coordinates, so the
coefficient prior has 83 degrees of freedom. The scale prior is flat in
`u=log(sigma)`. The sampled state is `(beta, v)`, where `v=sinh(u)` and
`u=asinh(v)`. The target includes `log|du/dv|` to account for the change of
coordinates.

The stereographic map uses a fitted center `mu` and a lower-triangular
`L=preconditioning_matrix` with positive diagonal:

```text
x = mu + L*z_horizontal/(1-z_last),
a = solve(L, x-mu),
z = (2*a/(1+||a||^2), (||a||^2-1)/(1+||a||^2)).
```

Here `x=(beta,v)`. Its log Jacobian relative to sphere area is
`log det L - D*log(1-z_last)`. The implied scale matrix is `L L.T`.

The matrix DCS map sends uniform unit-ball reference points to Euclidean space:

```text
x = mu + L*s/(1 - ||s||**beta)**(1/beta),  ||s|| < 1.
```

For DCS, VI tunes `mu`, the full lower-triangular matrix `L`, and `beta` jointly.
It initializes `mu=0`, `L=sqrt(D)*I`, and `beta=2`; none of these parameters is
fixed. The reference distribution for this reverse-KL fit is uniform on the
unit ball. Stereographic VI uses the uniform sphere reference.

Both couplings use the same saved Euclidean initial pairs and iteration cutoff,
with separate proposal steps and random streams. The existing stereographic
step remains `STEREOGRAPHIC_H=0.1`. The DCS step is `DCS_H=0.049`, the ball-walk
radius in unit-ball coordinates, calibrated toward 23% acceptance with the
saved VI fit. Acceptance is averaged over the two chain rates and then over
pairs, with each pair stopping at meeting or the iteration cutoff. Both
out-of-ball proposals and Metropolis rejections count in this rate.

For the fitted exponent `beta=3.252823847717482`, 32 pilot pairs gave 22.78%
acceptance. A separate 16-pair check gave 22.77% (MCSE 0.14 percentage points),
including one pair that reached the 1,000,000-step cutoff without meeting.
A 10-pair calibration check using the production initial pairs and seeds gave
22.73% acceptance, with all 10 pairs meeting before its 1,000,000-step cutoff.
These pilots used the same stop-at-meeting rule; production now allows
2,000,000 steps per pair. The saved VI fits were retained.

## Files and usage

- [`target.py`](target.py): load `nba_salary.csv` and evaluate the target for sampling and VI.
- [`run_vi.py`](run_vi.py): `main()` calls `fit_dcs()` using `run_dcs_matrix_vi`. The existing `fit_stereographic()` function remains available for a separate refit.
- [`run_experiment.py`](run_experiment.py): `run_stereographic(starts)` and `run_dcs(starts)` call their shared method-specific experiment functions. `main()` combines, saves, and plots their results.
- [`fitted_parameters.json`](fitted_parameters.json): both fitted maps, including the DCS location, matrix, beta, and VI metadata. DCS refitting preserves the stereographic fit.
- [`initial_states.npz`](initial_states.npz): 1,000 saved initial pairs, shape `(1000, 2, 82)`.
- [`meeting_times.csv`](meeting_times.csv): saved coupling results. [`meeting_times.pdf`](meeting_times.pdf) and [`meeting_times.png`](meeting_times.png) plot these results with logarithmic meeting-time bins and a logarithmic survival x-axis above iteration 1 (linear near zero).
- [`nba_salary.csv`](nba_salary.csv): input dataset.

From the repository root, fit DCS first:

```sh
python student_t_regression/nba_salary/run_vi.py
```

This fits DCS only; the saved stereographic fit is retained. VI uses 4,000 Adam
updates, batch size 2,048, and a 20,000-sample holdout to choose between the last
iterate and the average of the last 500 iterates. The fitted exponent is saved
under `dcs.beta` and loaded by the coupling runner.

With the saved VI fit and calibrated step, run:

```sh
python student_t_regression/nba_salary/run_experiment.py
```

If DCS is refitted, recalibrate `DCS_H` for the new transformation before
running the comparison.

The current settings are `N_REP=1000`, `MAX_STEPS=2000000`, `SEED=42`,
`STEREOGRAPHIC_H=0.1`, `DCS_H=0.049`, and `PROGRESS_EVERY=10`. Each pair stops at meeting or the
iteration cutoff. The script writes `meeting_times.csv` with method labels
`stereographic` and `dcs`, and
plots both in `meeting_times.pdf` and `meeting_times.png`, replacing the saved results. The figure shows
meeting-time histograms for met pairs and survival curves for all pairs.
Edit the constants at the top of the runner to choose another run.

To redraw a saved stereographic/DCS comparison without simulation:

```sh
python - <<'PY'
import pandas as pd
from student_t_regression.nba_salary.run_experiment import OUTPUT, plot_comparison

plot_comparison(pd.read_csv(OUTPUT))
PY
```

To refit stereographic separately, if needed:

```sh
python -c "from student_t_regression.nba_salary.run_vi import fit_stereographic; fit_stereographic()"
```

Stereographic VI initializes `mu=0` and `L=sqrt(D)*I`, then fits both. Its
existing optimizer settings and random seed are retained. The DCS fit uses a
separate seed. Neither VI function changes the saved initial pairs.

The supplied initial pairs were generated by mapping 2,000 independent uniform
points on the full unit sphere in 83 dimensions through the saved stereographic
fit, using seed 42. They are preserved here as `initial_states.npz` under the key `initial_states`.

## Completed coupling results

Both methods have 1,000 saved trials, and all pairs met within the
2,000,000-step cutoff. Stereographic mean meeting time is 25.34. Matrix DCS
mean meeting time is 223,316.946, with maximum 1,657,016; 991 DCS pairs met
by 1,000,000 steps. Mean acceptance is 68.445% for stereographic and 22.791%
for DCS, averaged across stopped pairs.
