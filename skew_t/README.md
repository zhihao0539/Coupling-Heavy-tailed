# Skew-t coupling comparison

The target is the 100-dimensional Azzalini-Capitanio skew-t distribution with
`NU=2`, identity scale, and skewness `(100, -100, 0, ..., 0)`. The experiment
uses the shared [coupling, VI, and plotting functions](../generic_algorithms/README.md).

## Files and usage

- [run_vi.py](run_vi.py): fit stereographic, SCS, and DCS maps using the shared VI runner.
- [generate_initial_states.py](generate_initial_states.py): generate common initial pairs through the fitted SCS bright cap.
- [run_experiment.py](run_experiment.py): load the fits and starts, run all four couplings, and plot the results.
- [target.py](target.py): NumPy and JAX target log densities.
- [fitted_parameters.json](fitted_parameters.json): the three fitted maps and VI metadata used for the saved run.
- [initial_states.npz](initial_states.npz): 1,000 Euclidean initial pairs under `initial_states`, shape `(1000, 2, 100)`.
- [meeting_times.csv](meeting_times.csv) and [meeting_times.pdf](meeting_times.pdf): the saved 1,000-repetition results and figure.

The supplied fits and starts are ready to use. From the repository root:

```sh
python -m pip install -r requirements.txt
python skew_t/run_experiment.py
```

This replaces `meeting_times.csv` and `meeting_times.pdf`. To refit the maps and
regenerate their initial pairs before running the couplings, use all three steps:

```sh
python skew_t/run_vi.py
python skew_t/generate_initial_states.py
python skew_t/run_experiment.py
```

To redraw the saved figure without simulation:

```sh
python generic_algorithms/plot_results.py skew_t/meeting_times.csv --log-scale --horizon 1000000
```

## VI settings

| Method | Fitted parameters | Initial values | VI seed |
|---|---|---|---|
| Stereographic | Center `mu` and full lower-triangular `preconditioning_matrix` | `mu=0`, matrix `sqrt(D)*I` | 26092801 |
| SCS | Center `mu`, full lower-triangular `preconditioning_matrix`, and `horizontal_observer` | Center and observer zero, matrix `(sqrt(D)/11)*I` | 26093001 |
| DCS | Center `mu`, scalar `radius`, and Mobius offset `delta` | Center and offset zero, radius `sqrt(NU)` | 26092901 |

Matrix diagonal entries stay positive. SCS latitude is fixed at `1.1`, and the
DCS radial exponent is fixed at `beta=2`. DCS uses the scalar-radius Mobius map.

Each fit uses 4,000 Adam updates and 2,048 fresh reference points per update.
The learning rate is `0.01` for the first 3,000 updates and `0.002` afterwards.
An independent 20,000-point holdout compares the last iterate with the average
of the last 500 unconstrained parameter vectors. All three saved fits select
the latter. The `vi_info` entries record the settings and losses. These losses
omit constants and should not be compared directly across map families.

## Uniform bright-cap initialization

In the code's zero-centered unit-sphere coordinates, the bright cap is

```text
B = {z on S^100 : z_last < LATITUDE - 1}.
```

The generator normalizes independent Gaussian vectors and rejects all points
with last coordinate at least `0.1`. Accepted points are independent and uniform
with respect to cap area. It applies the fitted SCS center, preconditioning
matrix, and observer, then arranges the Euclidean states into 1,000 independent
pairs. The initialization seed is `43`.

All four methods use these same Euclidean pairs. Their SCS representations are
uniform on the bright cap; the other methods' native coordinates generally have
different distributions. The initial Euclidean distribution changes when the
SCS fit changes, so regenerate `initial_states.npz` after refitting if uniform
initialization under the updated SCS map is intended.

## Coupling settings and results

The saved run uses `N_REP=1000`, `MAX_STEPS=1_000_000`, and `SEED=42`.
Proposal settings are fixed separately from VI:

| Method | Proposal step `h` | Seed offset |
|---|---:|---:|
| Stereographic | 10.0 | 100000 |
| Euclidean | 0.35 | 200000 |
| SCS | 2.0 | 300000 |
| DCS ball walk | 0.20 | 400000 |

The offsets are tied to method names. Each pair stops at exact meeting in native
coordinates or at the cutoff. SCS rejects dark-side proposals directly. There
is no proposal step-size adaptation.

| Method | Meetings by 1,000,000 | Mean / restricted mean meeting time | Mean path acceptance |
|---|---:|---:|---:|
| SCS with matrix VI | 1,000/1,000 | 11.416 | 63.754% |
| DCS ball walk with scalar-radius VI | 1,000/1,000 | 1,268.872 | 24.381% |
| Euclidean | 968/1,000 | 80,787.652 | 24.927% |
| Stereographic with matrix VI | 367/1,000 | 651,503.660 | 14.200% |

For censored methods, the reported mean is the average of `min(tau, 1000000)`;
for fully met methods, it is the ordinary mean. CSV meeting time `-1` denotes a
censored pair. Acceptance is the unweighted mean of both chains' per-run rates,
measured through meeting or censoring; it is not a stationary acceptance estimate.
The histogram includes only meeting pairs, while the survival curve includes all
pairs. These simultaneous, zero-lag meeting-time statistics do not alone provide
bounds on distance to stationarity.
