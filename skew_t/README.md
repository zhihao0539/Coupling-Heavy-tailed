# Skew-t coupling comparison

The experiment uses the shared [coupling, VI and plotting functions](../generic_algorithms/README.md). This folder supplies the target, the fixed settings and the saved inputs.

- [target.py](target.py): the 100-dimensional Azzalini–Capitanio skew-t log density with ν = 2 and α = (100, −100, 0, …, 0), in NumPy for sampling and in JAX for VI.
- [run_vi.py](run_vi.py): step 1. Sets the target and constants, then calls the shared `run_vi` to fit and save the SCS and DCS maps. It contains no application-specific fitting, saving, or loading functions.
- [run_experiment.py](run_experiment.py): step 2. Loads the saved maps, runs the four couplings with the shared runner and plots them. It does not run VI.
- `initial_states.npz`: the 1,000 common Euclidean initial pairs used by every method.
- `fitted_parameters.json`: the saved SCS and DCS maps. The committed file is the fit used for the paper results.
- `meeting_times.csv` and `meeting_times.pdf`: the saved 1,000-repetition results.

## Run

From the repository root:

```sh
python -m pip install -r requirements.txt
python skew_t/run_experiment.py   # use the committed VI fit
```

To refit the maps first, run `python skew_t/run_vi.py`, then run the experiment again.

1. **VI.** `run_vi.py` passes `METHODS`, `VI_SETTINGS`, `LATITUDE`, and the target to [`generic_algorithms/run_vi.py`](../generic_algorithms/run_vi.py). That shared runner fits SCS and DCS by reverse KL and writes them, with their VI details under `vi_info`, to `fitted_parameters.json`. Run it only when you change those settings; the committed file already holds the paper fit, so step 2 can run straight away. Rerunning step 1 replaces the committed fit, which Git can restore. If you remove a map from `METHODS`, only the other map is refitted and the removed map's saved fit is kept.
2. **Coupling.** `run_experiment.py` reads `fitted_parameters.json` with the shared `load_parameters` and needs no JAX. Each method starts from the same saved initial pairs and stops at meeting or at `MAX_STEPS`. Results are saved to `new_meeting_times.csv` and `new_meeting_times.pdf`, so the saved paper results are not overwritten. It checks only the maps requested in `STEP_SIZES`. It stops with a message if a required fit is missing, has invalid map parameters, or uses a different SCS latitude. The original committed fit has no `vi_info`; the application explicitly supplies its known SCS latitude of 1.1. A run with only Euclidean and stereographic methods does not require the fit file.

Start with a small `N_REP` and `MAX_STEPS`: at the one-million-step cutoff, the Euclidean and stereographic runs take many hours in plain Python.

To redraw the saved figure without rerunning anything:

```sh
python generic_algorithms/plot_results.py skew_t/meeting_times.csv --log-scale
```

## Settings

VI settings (top of `run_vi.py`):

| Setting | SCS | DCS |
|---|---|---|
| Fixed map setting | Latitude 1.1 | β = 2 |
| Initial radius | 10/11 | √2 |
| Initial location and offset | Zero | Zero |
| Seed | 26093001 | 26092901 |
| Fitted quantities | Location, radius, horizontal observer | Location, radius, Möbius offset |

Both fits use 4,000 Adam updates and 2,048 reference samples per update. The learning rate is 0.01 for 3,000 updates and then 0.002. A 20,000-sample holdout chooses between the final iterate and the average of the last 500 unconstrained parameter vectors. To fit only one map, remove the other entry from `METHODS`.

Shared optimizer settings are grouped in `VI_SETTINGS`; initial radii, seeds,
and scan sizes are in `METHODS`. The [shared VI runner guide](../generic_algorithms/README.md#run-vi-from-an-application)
explains these inputs, the output file, and how to reuse the same runner for
another target.

Coupling settings (top of `run_experiment.py`):

| Method | Map | Proposal step |
|---|---|---|
| Stereographic | Fixed: zero location, R = 10 | h = 14 |
| Euclidean | None | h = 0.26 |
| SCS | Fitted by VI | h = 0.08; dark-side proposals are rejected directly |
| DCS | Fitted by VI | Ball-walk radius γ/R = 0.14 |

VI tunes only the map parameters. Proposal steps are fixed separately; they were tuned toward 40% stationary acceptance. Each method keeps the random-number stream it used before this refactor (`seed_offset` 100,000 to 400,000, in table order). The offsets are tied to method names, so removing or reordering methods in `STEP_SIZES` does not change the remaining methods’ streams.
