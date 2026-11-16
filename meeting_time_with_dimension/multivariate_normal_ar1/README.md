# AR(1) Gaussian coupling comparison

The target is N(0,C), with C[i,j] = 0.9**abs(i-j), in dimensions
5, 10, 20, 50, and 100. The comparison uses stereographic, matrix DCS ball
walk, and Euclidean Gaussian random-walk couplings.

## Initialization and production budget

Each method uses 100 pairs per dimension and a 1,000,000-step cutoff.
Both initial states are independent N(0,I_d) draws in Euclidean coordinates;
these are not stationary draws from the correlated target. All methods use
the same initial pairs. Every pair stops at meeting or the cutoff.
The saved coupling results use this initialization and explicitly record
`initialization=standard_normal`.

## VI and proposal parameters

The existing generic reverse-KL VI functions fit location mu and the full
lower-triangular preconditioning matrix for stereographic, and mu, the full
matrix, and beta for matrix DCS. No listed transformation field is fixed.
Reference draws are uniform on the sphere for stereographic and uniform in
the unit ball for DCS. Target draws are not required for fitting.

Stereographic uses 6,000 updates; matrix DCS uses 16,000. Both use batches
of 2,048, a 50,000-sample holdout, and averaging over the last 500 iterates.
The seeds and optimizer settings are explicit in `run_vi.py`. DCS checks
that beta remains above two throughout the fit and in the saved parameters.

Each dimension's maps are in `results/d_XXXX/fitted_parameters.json`.
`STEP_SIZES` in `run_experiment.py` specifies the actual proposal steps.
The saved fits and steps were retained when initialization changed from
target draws to N(0,I_d). No step-size recalibration was performed for that
initialization change. DCS calibration targeted 23% acceptance averaged over
chains and stopped pairs, counting out-of-ball proposals as rejections.

## Run

Reuse the supplied fits to simulate and redraw the comparison:

```sh
python run_experiment.py
python plot_results.py
```

`python run_vi.py` refits both maps. `python run_all.py` refits, simulates,
and plots in sequence. Refitting can require recalibrating proposal steps.
No tuning or test programs are included in this experiment folder.

## Completed results

| d | Fitted DCS beta | Stereographic mean | DCS mean | Euclidean mean among met pairs | Euclidean meetings |
|---:|---:|---:|---:|---:|---:|
| 5 | 6.8363 | 3.27 | 15.70 | 65.70 | 100/100 |
| 10 | 8.3280 | 2.94 | 25.89 | 325.83 | 100/100 |
| 20 | 10.5318 | 4.88 | 65.35 | 2264.61 | 100/100 |
| 50 | 15.0145 | 2.80 | 220.86 | 39441.49 | 100/100 |
| 100 | 17.4637 | 2.78 | 631.79 | 521871.41 | 29/100 |

All stereographic and DCS pairs met. The two dimension panels show the mean
where all pairs met and the mean among met pairs for every dimension/method.
There is no step-size panel. Error bars show one empirical Monte Carlo
standard error. When some pairs are censored, the met-pair mean does not
estimate the unrestricted mean. The raw CSVs and summaries preserve cutoff,
step size, acceptance, censoring, and initialization metadata.

The initial-state stream is SeedSequence([43,d]); proposal streams are
SeedSequence([42+d,index,replicate]) with indices 0, 3, and 4 for
stereographic, DCS, and Euclidean, respectively.
