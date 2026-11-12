# Meeting time versus dimension for Student-t(d+1)

In dimension d the target is a centered multivariate Student-t with d+1 degrees of
freedom and identity scale. The comparison uses d = 1, 2, 5, 10, 20, 50, 100.

| Coupling | Fixed geometry | VI fields |
|---|---|---|
| Stereographic | Location 0, radius √d (matrix √d I) | None |
| SCS11 | Location and horizontal observer 0, latitude 1.1 | Full preconditioning matrix |
| SCS19 | Location and horizontal observer 0, latitude 1.9 | Full preconditioning matrix |
| DCS Mobius ball walk | Location 0, proposals in the unit ball | Radius R, Mobius offset δ, and positive β, fitted jointly |
| Euclidean random walk | Isotropic Gaussian proposal | None |

DCS now uses `run_dcs_vi` with `tune=("radius", "delta", "beta")` and
`run_dcs_ball_experiment`. The map applies the Mobius automorphism to a point
in the unit ball, then the radial transformation with exponent β, and scales
by R. There is no DCS preconditioning matrix. VI keeps μ exactly zero and
initializes R=√(d+1), δ=0, and β=max(2, π√(d/6)). The generic parameterization
keeps R and β positive and ||δ||<1.

The remaining four couplings, their VI choices, initial draws, proposal seeds,
and step sizes are unchanged.

## Run

The Mobius DCS and SCS fits are saved, and the DCS steps have been calibrated
for these Mobius fits. To run the comparison:

```sh
python run_experiment.py
python plot_results.py
```

To refit SCS and DCS together, use `python run_vi.py`. To fit all configured
maps, simulate, and plot in sequence, use `python run_all.py`.

VI retains the existing 20,000-update budget, batch size 2,048, 20,000 holdout
samples, and averaging over the last 500 parameter vectors. Its learning rates
are 0.01 then 0.002 after update 10,000. The generic VI uses its existing slower
update rate for β. Fit metadata records the tuned fields and the fitted β.

Production uses 1,000 pairs of independent target draws per dimension and a
1,000,000-update cutoff. All five methods start from the same pairs. The short
`run_stereographic`, `run_scs11`, `run_scs19`, `run_dcs_ball`, and
`run_euclidean` functions call their matching `generic_algorithms` runners.

## Step sizes

`STEP_SIZES` in `run_experiment.py` contains an editable h for every method
and dimension. The non-DCS values retain their completed-experiment calibration.
The DCS values are calibrated for the current saved Mobius fits (including
each fitted β), with μ=0.

The target is 23% for the average of the two chain acceptance rates, averaged
across pairs stopped at meeting or the 1,000,000-update production cutoff.
Every attempted proposal counts in the denominator, including proposals
outside the ball. The validation pairs use fresh, independent target draws
and transition noise, separate from calibration and the production experiment.
For DCS ball walk, h is the proposal radius in the native unit ball.

Validation estimates below give one Monte Carlo standard error in percentage
points. All validation pairs met before the cutoff.

| d | DCS h | Validation acceptance | Pilot pairs | Validation pairs |
|---:|---:|---:|---:|---:|
| 1 | 11.4163883 | 23.32% ± 0.15 | 16384 | 32768 |
| 2 | 3.34218244 | 23.33% ± 0.21 | 8192 | 16384 |
| 5 | 1.53800871 | 23.54% ± 0.21 | 8192 | 16384 |
| 10 | 0.864056741 | 22.84% ± 0.18 | 4096 | 8192 |
| 20 | 0.539814447 | 23.25% ± 0.14 | 2048 | 4096 |
| 50 | 0.325021331 | 22.71% ± 0.09 | 1024 | 2048 |
| 100 | 0.225349985 | 22.66% ± 0.06 | 1024 | 2048 |

These are Monte Carlo estimates; the acceptance printed in a new run will vary.
If the DCS parameters are refitted or changed, recalibrate its steps for the
new map. Changing h alone does not require rerunning VI.

Existing `step_sizes.json` files describe the previous matrix-map calibration.
They are historical records and are not read by the experiment. The current
Mobius calibration scripts and detailed trials are kept outside this folder.
To change a step for a future run, edit `STEP_SIZES` directly. No tuning
programs have been added to this experiment folder.

## Results

All fitted maps, meeting times, summaries, and figures are under `results/`.
Each `d_XXXX/` folder contains that dimension's files. The combined
`meeting_times.csv` includes the actual h used for each row. New DCS rows also
record `dcs_map="mobius"`; the plotting stage labels this method as
"DCS Mobius ball walk".

The existing matrix-map results have been preserved. They must be regenerated
before plotting the new comparison. The experiment rejects old matrix DCS
fits before the main sampling stage, and the plotter rejects old DCS results
rather than relabeling them as Mobius results. Rerunning the experiment replaces
its CSVs; rerunning the plotter replaces the summaries and figures.

Meeting-time error bars are one Monte Carlo standard error across pairs.
Unrestricted mean meeting times are plotted where every pair met.
