# AR(1) Gaussian: reverse-KL-fitted coupling comparison

The target is N(0,C), with C[i,j] = 0.9**abs(i-j), in dimensions
5, 10, 20, 50, and 100. Each trial starts from two independent exact target
draws. The same initial pairs are used for all three methods in each dimension.

## Transformation fitting

Both transformed methods minimize reverse KL, KL(q || pi), using the existing
generic VI functions. Target samples are not used in VI: stereographic uses
uniform sphere draws, and DCS uses uniform unit-ball draws.

| Method | VI parameters | Initial values |
|---|---|---|
| Stereographic | mu, full lower-triangular matrix L | mu=0, L=sqrt(d) I |
| Matrix DCS ball walk | mu, full lower-triangular matrix L, beta | mu=0, L=sqrt(d) I, beta=2 |
| Euclidean Gaussian RWM | None | Original coordinates |

All listed VI parameters are free. Stereographic uses 6,000 Adam updates;
DCS uses 16,000 updates to allow beta to converge. Both use batches of 2,048,
a 50,000-point independent holdout, and candidate selection between the final
iterate and the average of the last 500 iterates. Learning rates and seeds are
in `run_vi.py`.

For Gaussian targets, stereographic reverse KL is finite in these dimensions
(d>2). Matrix DCS requires beta>2. Its initial beta is above two, and the code
checks the entire optimization trace and fitted value before saving the DCS
fit. A failed check stops the run.

`run_vi.py` uses separate fitting functions and saves the two maps atomically
to `results/d_XXXX/fitted_parameters.json`. Use `python run_vi.py` to refit
both maps. Its settings, free parameters, and seeds are unchanged.
Existing loss traces and VI summaries
record the completed fitting run; the simplified runner saves only the fits.
The recorded KL estimates include the uniform reference's normalization
constant, which the generic optimization loss omits.

## Couplings and proposal steps

`run_experiment.py` contains one short wrapper per method around its matching
generic experiment runner. The comparison uses 100 pairs per dimension and a
1,000,000-update cutoff, stopping a pair at meeting or the cutoff.

Proposal steps are constants in `STEP_SIZES`. After fitting, DCS steps were
calibrated toward 23% acceptance averaged over chains and stopped pairs.
Out-of-ball proposals count as rejections. Stereographic remained above 23%
throughout the tested step range, so its step was chosen using the smallest
pilot mean meeting time. Calibration and independent validation records are
in `results/step_calibration.json`; tuning scripts are outside this folder.
Euclidean retains the calibrated steps for the same target and initialization.

## Run and output

With the scientific Python environment activated:

```sh
python run_vi.py
python run_experiment.py
python plot_results.py
```

Or use `python run_all.py` for all three stages. Changing the fitted maps can
require recalibrating the proposal steps.

All new output is in `results/`; the old output folders have been removed.
The dimension plot contains two panels: the mean meeting time where all
pairs met, and the mean among met pairs for every dimension and method.
It has no proposal-step panel. Error bars are one empirical Monte Carlo
standard error, using the number of met pairs in each group. When some pairs
reach the cutoff, the second panel describes only the met pairs and does not
estimate the uncensored mean. Summaries retain restricted means and proposal
steps as numerical records.
Per-dimension distribution and survival plots use logarithmic time axes.

The initial-state seed is 43, using SeedSequence([43,d]). Proposal streams use
SeedSequence([42+d,index,replicate]), with index 0 for stereographic, 3 for
DCS, and 4 for Euclidean. VI, calibration, validation, and production use
separate random streams. Execution details are in `results/execution_details.json`.

## Completed comparison

| d | Fitted DCS beta | Stereographic mean | DCS mean | Euclidean mean among met pairs | Euclidean met |
|---:|---:|---:|---:|---:|---:|
| 5 | 6.8363 | 4.26 | 14.10 | 101.75 | 100/100 |
| 10 | 8.3280 | 3.83 | 28.46 | 587.71 | 100/100 |
| 20 | 10.5315 | 4.76 | 85.99 | 2625.32 | 100/100 |
| 50 | 15.1503 | 3.60 | 253.08 | 44473.69 | 100/100 |
| 100 | 20.4558 | 4.22 | 693.93 | 587656.12 | 32/100 |

All stereographic and DCS pairs met. See `results/summary.csv` for full
acceptance rates, proposal steps, censoring counts, and Monte Carlo errors.
The Gaussian target permits an exact reverse-KL diagnostic after fitting;
`results/vi_diagnostics.csv` checks the VI fits against the best distributions
in their respective families. These analytical checks do not replace VI.
The long Euclidean run used a workspace-only C++ replay of the generic
NumPy kernel with the same random draws; the installed experiment uses the
public generic runner. Seeded replay checks are recorded in execution details.

## Unbiased Poisson asymptotic variance at d=50

`run_asymptotic_variance.py` compares v(P,h) for h(x)=x_1 using the
stored-signed-measure UPAVE estimator, with the existing fits and proposal steps.
No VI or meeting-time experiment is rerun. The lagged couplings start from
independent N(0,I_50) draws; the target remains the correlated N(0,C) Gaussian.

Each method uses 50 independent lag-one pilots. L=k is the ceiling of
their empirical 95th percentile, and ell=5L. Production uses 100 independent
replicates, two independent signed measures per replicate, and R=10 uniformly
selected atoms per measure for the Poisson terms. The reference y is zero
in the original Euclidean coordinates. Pilots and production use separate
random streams; states stay native and h is evaluated in original coordinates.

The runner uses the generic `run_upave` function, with R fixed at 10.
This was a provisional choice, not a result of inefficiency optimization.
The numerical results below record the earlier completed run; changing the
implementation does not regenerate those saved estimates.

| Method | Estimated v(P,h) | MCSE | L=k | ell |
|---|---:|---:|---:|---:|
| stereographic | 2.370620 | 0.162683 | 10 | 50 |
| dcs | 149.224198 | 10.191756 | 440 | 2200 |
| euclidean | 6920.815549 | 995.800169 | 83180 | 415900 |

MCSE is sample_std(v_hat)/sqrt(100), measuring uncertainty in the mean
UPAVE estimate. It is distinct from v(P,h). The target variance of x_1 is
one; v(P,h) additionally includes temporal dependence under each kernel.

The cutoff is 1,000,000 iterations per lagged trajectory or synchronous
Poisson coupling. An unfinished run raises an error instead of returning a
truncated estimate. All required runs completed; negative estimates are kept.
The recorded costs count marginal transitions, with a coupled step costing two.

Run `python run_asymptotic_variance.py` for all methods, or add
`--method stereographic`, `--method dcs`, or `--method euclidean` for one method.
Results, pilot records and settings are in `results/d_0050/asymptotic_variance/`.
Run `python plot_asymptotic_variance.py` to draw the three asymptotic-variance
bars with one-MCSE error bars on a logarithmic vertical axis. The plotting
script recomputes bar heights and errors from the saved estimator replicates.
