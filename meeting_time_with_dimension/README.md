# Meeting time versus dimension: current settings

The current production settings below follow the experiment scripts.
Limits apply to each chain pair, which stops at meeting
or at its iteration cutoff. The same initial pair is shared across methods.

| Experiment | Dimensions | Pairs per method and dimension | Iteration cutoff |
|---|---|---:|---:|
| [Student-t with df=2](multivariate_student_t_2/) | 1, 2, 5, 10, 20, 50, 100 | 1,000 | 100,000 (10^5) |
| [Student-t with df=d+1](multivariate_student_t_d+1/) | 1, 2, 5, 10, 20, 50, 100 | 1,000 | 1,000,000 (10^6) |
| [AR(1) Gaussian](multivariate_normal_ar1/) | 5, 10, 20, 50, 100 | 100 | 1,000,000 (10^6) |

## Targets, methods, and initialization

The Student-t targets have identity scale. Their comparisons use stereographic,
SCS with latitudes 1.1 and 1.9, Mobius DCS ball walk, and Euclidean random walk.
The df=2 DCS map fixes beta=2; the df=d+1 DCS map learns beta jointly with
its radius and Mobius offset. Locations are fixed at zero, and SCS horizontal
observers are also fixed at zero. Further VI details are in each folder's README.

The AR(1) target is N(0,C), with C[i,j] = 0.9**abs(i-j). Its comparison uses
stereographic, matrix DCS ball walk, and Euclidean random walk. Reverse-KL VI
tunes mu and the full matrix for both transformed methods, and beta for DCS.

All three meeting-time comparisons initialize their chains independently
from the corresponding target distribution.

## AR(1) Gaussian UPAVE at d=50

The separate asymptotic-variance experiment uses h(x)=x_1 and starts its lagged
couplings from N(0,I_50). It runs 100 independent UPAVE replicates per method,
with 50 independent lag-one pilots. L=k is the ceiling of the pilots' 95th
meeting-time percentile, ell=5L, and R=10 uniform atoms are used per signed
measure. The Poisson reference is zero in original Euclidean coordinates for
all methods. The iteration cutoff for each required coupling is 1,000,000.

## Running the experiments

Run `python run_all.py` from an experiment folder to fit its configured maps,
run the couplings, and plot. The stages can also be run separately with
`run_vi.py`, `run_experiment.py`, and `plot_results.py`. For example, from the
repository root:

```sh
python meeting_time_with_dimension/multivariate_student_t_2/run_all.py
```

Proposal steps are supplied in each `run_experiment.py`. VI settings are in
`run_vi.py`; fitting new maps can require recalibrating the proposal steps.

## Reading saved results and calibration notes

`N_REP` and `MAX_STEPS` in each `run_experiment.py` control future coupling
runs. `MAX_ITERATIONS` controls the separate UPAVE cutoff. Saved CSV rows
record their actual stopping limits and proposal steps. The larger cutoffs
quoted in historical calibration notes describe those earlier pilots and
validation runs. They do not override the current production budgets above.

A meeting time of -1 denotes a pair that did not meet by its saved cutoff.
Means among met pairs are conditional on meeting before that cutoff.
They do not estimate the uncensored mean when some pairs have not met.
Error bars show one empirical Monte Carlo standard error. Saved fits and CSVs
from an earlier map configuration must be regenerated before plotting the
current comparison; their recorded settings are preserved.
