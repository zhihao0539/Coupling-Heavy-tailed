# Multivariate Student-t coupling comparison (df=2)

The target is a multivariate Student-t with df=2 and identity scale in
dimensions 1, 2, 5, 10, 20, 50, and 100. The five couplings are stereographic,
SCS11, SCS19, DCS ball walk, and Euclidean random walk.

## VI parameters

| Map | VI tunes | Fixed parameters | Initial values |
|---|---|---|---|
| Stereographic | Full lower-triangular preconditioning matrix L | μ=0 | L=√d I |
| SCS11 | Full lower-triangular preconditioning matrix L | μ=0, horizontal observer=0, latitude=1.1 | L=(√d/11) I |
| SCS19 | Full lower-triangular preconditioning matrix L | μ=0, horizontal observer=0, latitude=1.9 | L=(9√d/19) I |
| DCS Mobius | Radius R and Mobius offset δ | μ=0, β=2 | R=1, δ=0 |
| Euclidean | No VI | — | — |

All fitted maps keep μ exactly zero throughout optimization. Stereographic
and SCS use `tune="preconditioning_matrix"`; SCS additionally fixes its
horizontal observer at zero. DCS uses `run_dcs_vi` with
`tune=("radius", "delta")`, keeping both μ=0 and β=2 fixed. Euclidean random
walk has no transformation-location parameter; its target is centered at zero.
The Mobius map has no preconditioning matrix.
The radius is positive and the Mobius offset stays inside the unit ball.

DCS ball walk uses the saved Mobius fit with β=2.

Each fit uses 20,000 updates, batches of 2,048, 20,000 holdout samples,
and averaging of the last 500 parameter vectors. Learning rates are 0.01,
then 0.002 after update 10,000. These settings and the seeds are unchanged.

## Run

The fixed-zero VI fits are saved and the step sizes are calibrated for them.
To run the comparison with these completed fits:

```sh
python run_experiment.py
python plot_results.py
```

To refit all maps, use `python run_vi.py`; recalibrate the steps after refitting.
Each dimension's parameter file is replaced once all fits finish successfully.
`python run_all.py` refits all maps, runs the couplings, and plots in sequence.

The optional `python run_vi.py --scs-dcs-only` shortcut can be used for later
SCS/DCS refits only when the saved stereographic μ is already exactly zero.
It rejects the older stereographic fits with learned μ and asks for a full
`python run_vi.py` run. This prevents preserving an incompatible fit.

The experiment checks all saved maps before starting sampling. It requires
μ=0 in every map, zero SCS horizontal observers, a Mobius DCS fit, and β=2.

## Coupling experiments and step sizes

The current production settings in `run_experiment.py` are **50 pairs per
dimension** and a **100,000-update cutoff**, stopping each pair at meeting
or the cutoff. All five methods start from the same independent target draws.
The existing proposal streams are preserved: indices 0, 1, 2, 3, and 4 for
stereographic, SCS11, SCS19, DCS ball, and Euclidean, respectively.

The five step-size lists in `run_experiment.py` use the saved fixed-zero VI
fits. The stereographic step at d=100 is manually set to 0.02; the other
steps retain their previous calibration values. Dimensions are listed in the
same order in every array.
The VI files, proposal streams, target initialization and current production
budget are unchanged by removing DCS Gaussian random walk. No tuning scripts
were added to this experiment folder.

The target is 23% mean acceptance across pairs stopped at meeting. When the
sphere-step search did not reach that target, a practical step was selected
from the tested meeting-time pilots. Those cases retain higher acceptance.
This is a numerical choice of steps; the table below reports the measured
acceptance rather than assuming every method can achieve 23%.

| d | Stereographic | SCS11 | SCS19 | DCS ball | Euclidean |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 2 | 11.473405 | 19.195641 |
| 2 | 1.4142136 | 1.4142136 | 1.4142136 | 3.4178425 | 5.6580131 |
| 5 | 0.4472136 | 0.89442719 | 0.89442719 | 1.5889588 | 2.12632 |
| 10 | 0.15811388 | 0.63245553 | 0.63245553 | 0.91377312 | 1.1888979 |
| 20 | 0.055901699 | 0.4472136 | 0.2236068 | 0.57663684 | 0.79 |
| 50 | 0.035355339 | 0.56568542 | 0.070710678 | 0.34911279 | 0.46191584 |
| 100 | 0.02 | 0.4 | 0.025 | 0.24329044 | 0.34930274 |

Earlier independent validation acceptance percentages (parentheses give one
Monte Carlo standard error in percentage points). The current stereographic
h=0.02 at d=100 has not been calibrated or validated; its entry is omitted:

| d | Stereographic | SCS11 | SCS19 | DCS ball | Euclidean |
|---:|---:|---:|---:|---:|---:|
| 1 | 88.68 (0.17) | 72.56 (0.28) | 83.58 (0.19) | 23.00 (0.15) | 23.30 (0.14) |
| 2 | 99.75 (0.03) | 69.27 (0.28) | 93.15 (0.13) | 22.97 (0.21) | 23.35 (0.18) |
| 5 | 86.93 (0.24) | 68.18 (0.37) | 90.67 (0.22) | 23.43 (0.30) | 23.00 (0.21) |
| 10 | 81.43 (0.48) | 68.85 (0.69) | 76.72 (0.67) | 23.03 (0.27) | 23.88 (0.24) |
| 20 | 81.93 (0.57) | 72.00 (0.94) | 66.97 (1.00) | 22.97 (0.20) | 22.74 (0.31) |
| 50 | 70.14 (1.59) | 73.69 (1.87) | 60.24 (1.78) | 22.78 (0.07) | 23.81 (0.65) |
| 100 | — | 81.96 (1.58) | 66.89 (1.41) | 22.78 (0.06) | 22.17 (0.88) |

Earlier validation used fresh target starts and transition noise with a
10,000,000-update cutoff, rather than the current production cutoff. The
sphere-step grid used 20,000-update pilots; DCS and Euclidean tuning used the
earlier 10,000,000-update cutoff. Detailed trials, seeds,
fit fingerprints and sample counts are retained in the external calibration
workspace. Orthogonal coordinates preserve the fitted full matrices during
calibration; the deterministic steps were checked against the generic samplers.

Earlier validation pairs reaching the 10,000,000-update cutoff:

- stereographic, d=10: 1/1024 pairs.
- stereographic, d=20: 1/512 pairs.
- stereographic, d=50: 2/128 pairs.
- stereographic, d=100 with the superseded h=0.4: 6/16 pairs.

The superseded d=100 stereographic h=0.4 was chosen because its 20,000-update
pilot acceptance was 23.05%, even though 68/128 pairs were censored and its
restricted mean was worse than that of smaller steps. The later 16-pair
validation was also highly uncertain (24.08% ± 4.98 percentage points). These
measurements do not describe the current h=0.02. All other cases without a
censored count above met before the earlier validation cutoff. Acceptance
averages depend on the step and the cutoff.
Changing the fitted maps or stopping rule requires recalibration. Changing
only the proposal step does not require rerunning VI.

DCS uses `run_dcs_ball_experiment` in B(1), with h as its proposal-ball radius.
Proposals outside B(1) count as rejections. If targeting acceptance near 23%,
use the average printed by these experiments: accepted proposals divided by iterations for
each chain through meeting or cutoff, then averaged across chains and pairs.

## Saved results and plots

Maps, CSVs, summaries, and plots use this experiment's own `results/` folder.
The revised experiment writes dimension, df, cutoff, actual h, and initialization
metadata in both per-dimension and combined CSVs. All rows record
`mu_fixed=True`. SCS rows additionally record
`scs_mu_fixed=True` and `scs_observer_fixed=True`; DCS rows record
`dcs_map="mobius"` and β. The figure legend identifies DCS ball walk as Mobius.

The completed fixed-zero VI fits are compatible with this comparison. No VI
rerun is needed. To obtain meeting-time results using the current steps, run
`python run_experiment.py`. The plotter detects older learned-location and
SCS/DCS configurations so they are not relabeled as the new comparison.
New runs replace the corresponding CSVs and plots.

The three figure panels remain:

1. Mean meeting time where every pair met.
2. Mean meeting time among all pairs that met in each method/dimension group,
   including groups with censored pairs.
3. Proposal step size versus dimension.

Error bars are one empirical Monte Carlo standard error. The middle panel uses
the standard deviation of observed meeting times divided by the square root
of the number of met pairs. A single meeting supplies a point without an
estimated error bar; groups without meetings are omitted. Summaries also retain
restricted means and censored counts. Per-dimension diagnostics include the
five configured methods. The plotter ignores saved rows for methods outside
this comparison, including previously run DCS Gaussian random walk trials.
Saved h values are retained, so earlier h=0.4 runs are not relabeled as h=0.02.
