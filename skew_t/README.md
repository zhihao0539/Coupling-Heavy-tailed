# Skew-t coupling with fitted maps

The target is the 100-dimensional Azzalini-Capitanio skew-t distribution with
`NU=2`, identity scale, and skewness `(100, -100, 0, ..., 0)`. The experiment
uses the shared coupling, VI, and plotting functions in `generic_algorithms`.

## Run the three steps

From the repository root:

```sh
python skew_t/run_vi.py
python skew_t/generate_initial_states.py
python skew_t/run_experiment.py
```

1. `run_vi.py` fits stereographic, SCS, and DCS maps and saves
   `fitted_parameters.json`. Each method's initialization appears in
   the `run_stereographic_vi`, `run_scs_vi`, and `run_dcs_vi` calls;
   common optimizer settings are in `VI_SETTINGS`.
2. `generate_initial_states.py` loads the fitted SCS map, draws independent
   uniform bright-cap points, and projects them into Euclidean coordinates.
   It saves 1,000 pairs in `initial_states.npz`, under `initial_states`,
   with shape `(1000, 2, 100)`. It uses `INITIAL_SEED=43` and performs no VI
   or MCMC. Run this step again after changing the fit.
3. `run_experiment.py` loads all three fitted maps and the new initial pairs,
   calls `run_stereographic`, `run_euclidean`, `run_scs`, `run_dcs_ball`,
   and `run_dcs_random_walk`,
   then combines their result rows and plots them. Euclidean coupling uses
   the same Euclidean pairs. The output is `meeting_times.csv` and
   `meeting_times.pdf`.

## Parameters fitted by VI

| Method | Fitted parameters | Initial values | VI seed |
|---|---|---|---|
| Stereographic | Center `mu` and full lower-triangular `preconditioning_matrix` | `mu=0`, matrix `sqrt(D)*I` | 26092801 |
| SCS | Center `mu`, full lower-triangular `preconditioning_matrix`, and `horizontal_observer` | Center and observer zero, matrix `(sqrt(D)/11)*I` | 26093001 |
| DCS | Center `mu`, scalar `radius`, and Mobius offset `delta` | Center and offset zero, radius `sqrt(NU)` | 26092901 |

The diagonal entries of each fitted matrix are positive. The SCS latitude is
fixed at `1.1`; the DCS radial exponent is fixed at `beta=2`. DCS uses the
scalar-radius Mobius transformation, with no matrix preconditioning. DCS ball
walk and DCS Gaussian random walk load the same saved `dcs` VI map, including
its fitted location, radius, and Mobius offset; both use the common initial pairs.

Each fit uses 4,000 Adam updates and 2,048 fresh reference samples per update.
The learning rate is `0.01` for the first 3,000 updates and `0.002` afterwards.
An independent 20,000-point holdout chooses between the final iterate and the
average of the last 500 unconstrained parameter vectors. The saved `vi_info`
records the initial and candidate holdout losses. These objective values omit
constants and should not be compared directly between different map families.

## Initial distribution

In the code's zero-centered unit-sphere coordinates, the bright cap is

```text
B = {z on S^100 : z_last < LATITUDE - 1}.
```

The generator normalizes independent Gaussian vectors and rejects points outside
this cap. Accepted points are independent and uniform with respect to cap area.
It applies the fitted SCS map, including the fitted center, preconditioning
matrix, and observer, then arranges the Euclidean states into independent pairs.
Thus SCS starts uniformly on its bright cap. The same Euclidean pairs are used
by the other methods; their native coordinates generally have different laws.

The cap is fixed while `LATITUDE` stays fixed. Its Euclidean image changes with
the fitted map. The generated states are reference-distribution draws through
the SCS map; they are not asserted to be draws from the target skew-t density.

## Coupling settings and saved results

Proposal settings are the named constants `STEREOGRAPHIC_H`, `EUCLIDEAN_H`,
`SCS_H`, `DCS_H`, and `DCS_RANDOM_WALK_H`, chosen separately from VI. Each method has its own
application function in `run_experiment.py`:

| Method | Proposal step `h` |
|---|---:|
| Stereographic | 10.0 |
| Euclidean | 0.35 |
| SCS | 2.0 |
| DCS ball walk | 0.20 |
| DCS Gaussian random walk | 0.02 |

The current local runner uses `N_REP=1000`, `MAX_STEPS=1_000_000`, and `SEED=42`.
Change these constants to choose the run size and proposal steps. Every pair
stops at meeting or the iteration cutoff; SCS rejects dark-side proposals
directly. The runner performs no step-size adaptation.

The supplied `fitted_parameters.json`, `initial_states.npz`, `meeting_times.csv`,
and `meeting_times.pdf` are the current saved inputs and results. You can run
`python skew_t/run_experiment.py` directly with these inputs; it replaces the
saved CSV and PDF. Refitting is optional. When refitting, regenerate the initial
states before running the coupling comparison.

| Method | Meetings by 1,000,000 | Mean / restricted mean meeting time | Mean path acceptance |
|---|---:|---:|---:|
| SCS with matrix VI | 1,000/1,000 | 11.416 | 63.754% |
| DCS ball walk with scalar-radius VI | 1,000/1,000 | 1,268.872 | 24.381% |
| DCS Gaussian walk with scalar-radius VI | 1,000/1,000 | 1,328.853 | 24.090% |
| Euclidean | 968/1,000 | 80,787.652 | 24.927% |
| Stereographic with matrix VI | 367/1,000 | 651,503.660 | 14.200% |

For censored methods, the reported mean is the average of `min(tau, 1000000)`;
otherwise it is the ordinary mean. CSV meeting time `-1` denotes a censored
pair. Acceptance is the unweighted mean of the two per-chain rates across runs,
measured through meeting or censoring. It is not a stationary acceptance
estimate. The histogram includes meeting pairs only, while survival includes
all pairs.

To plot a new coupling run without running the samplers again:

```sh
python generic_algorithms/plot_results.py skew_t/meeting_times.csv --log-scale
```

## Run a method separately

Each application function loads only its required map and calls the corresponding
`run_*_experiment` function from `generic_algorithms`. Euclidean coupling needs
no fitted map. `main()` lists the five calls in comparison order; edit that list
to choose which methods to run. The shared initial pairs, method-specific seeds,
and step-size constants are visible at the top of the script.

For example, from the repository root:

```python
from skew_t.run_experiment import (
    load_initial_pairs, INITIAL_STATES_FILE, run_scs, run_dcs_random_walk,
)

starts = load_initial_pairs(INITIAL_STATES_FILE, n_pairs=10)
scs_rows = run_scs(starts)
gaussian_rows = run_dcs_random_walk(starts)
```

`run_dcs_random_walk` reuses the same fitted Mobius map and has its own seed
offset, 500,000. `main()` includes it with `DCS_RANDOM_WALK_H=0.02`. This is
the Gaussian standard deviation in unit-ball coordinates, sigma/R; the ball
walk keeps its own radius `DCS_H=0.2`. Its result label is `dcs_rw`, which the
shared plotter recognizes.
