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
   It saves 1,000 pairs in `new_initial_states.npz`, under `initial_states`,
   with shape `(1000, 2, 100)`. It uses `INITIAL_SEED=43` and performs no VI
   or MCMC. Run this step again after changing the fit.
3. `run_experiment.py` loads all three fitted maps and the new initial pairs,
   calls `run_stereographic`, `run_euclidean`, `run_scs`, `run_dcs_ball`,
   and `run_dcs_random_walk`,
   then combines their result rows and plots them. Euclidean coupling uses
   the same Euclidean pairs. The output is `new_meeting_times.csv` and
   `new_meeting_times.pdf`.

## Parameters fitted by VI

| Method | Fitted parameters | Initial values | VI seed |
|---|---|---|---|
| Stereographic | Center `mu` and full lower-triangular `preconditioning_matrix` | `mu=0`, matrix `sqrt(D)*I` | 26092801 |
| SCS | Center `mu`, full lower-triangular `preconditioning_matrix`, and `horizontal_observer` | Center and observer zero, matrix `(sqrt(D)/11)*I` | 26093001 |
| DCS | Center `mu`, scalar `radius`, and Mobius offset `delta` | Center and offset zero, radius `sqrt(NU)` | 26092901 |

The diagonal entries of each fitted matrix are positive. The SCS latitude is
fixed at `1.1`; the DCS radial exponent is fixed during each fit by `DCS_BETA` (default 2.0). DCS uses the
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
| Stereographic | 0.04 |
| Euclidean | 0.35 |
| SCS | 2.0 |
| DCS ball walk | 0.20 |
| DCS Gaussian random walk | 0.02 |

The current local runner uses `N_REP=1000`, `MAX_STEPS=1_000_000`, and `SEED=42`.
Change these constants to choose the run size and proposal steps. Every pair
stops at meeting or the iteration cutoff; SCS rejects dark-side proposals
directly. The runner performs no step-size adaptation.

The legacy `initial_states.npz`, `meeting_times.csv`, and `meeting_times.pdf`
files retained on GitHub are historical outputs and are not read by the
current runners. The supplied current comparison uses `new_initial_states.npz`,
`new_meeting_times.csv`, and `new_meeting_times.pdf`.

To plot a new coupling run without running the samplers again:

```sh
python generic_algorithms/plot_results.py skew_t/new_meeting_times.csv --log-scale
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
offset, 500,000. `main()` includes it with `DCS_RANDOM_WALK_H=0.3`. This is
the Gaussian standard deviation in unit-ball coordinates, sigma/R; the ball
walk keeps its own radius `DCS_H=0.2`. Its result label is `dcs_rw`, which the
shared plotter recognizes.


## DCS exponent

`DCS_BETA` in `run_vi.py` selects the fixed positive radial exponent (default
2.0). Both DCS proposals load and use that fitted exponent. Existing beta=2
fits still load without refitting. If you change `DCS_BETA`, refit DCS with
`run_dcs_vi(..., beta=DCS_BETA)` before sampling; the runner checks that the
saved exponent matches. Proposal steps are configured separately.

## Completed current comparison

The saved comparison contains 1,000 trials per method, each stopped at meeting
or 1,000,000 steps. Stereographic was rerun at h=0.04 using the saved fits,
initial pairs, and original proposal seeds; the other four methods retain
their completed results.

| Method | Meetings | Mean among met pairs | Restricted mean | Acceptance |
|---|---:|---:|---:|---:|
| stereographic | 782/1000 | 23059.910 | 236032.850 | 39.183% |
| euclidean | 968/1000 | 50400.467 | 80787.652 | 24.927% |
| scs | 1000/1000 | 11.416 | 11.416 | 63.754% |
| dcs | 1000/1000 | 1268.872 | 1268.872 | 24.381% |
| dcs_rw | 1000/1000 | 1328.853 | 1328.853 | 24.090% |

The restricted mean averages min(meeting time, cutoff) across all pairs.
Means among met pairs condition on meeting before the cutoff.
