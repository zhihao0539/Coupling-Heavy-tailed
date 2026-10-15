# Skew-t coupling comparison

The experiment uses the shared [coupling, VI and plotting functions](../generic_algorithms/README.md). This folder supplies the target, the fixed settings and the saved inputs.

- [target.py](target.py): the 100-dimensional Azzalini–Capitanio skew-t log density with ν = 2 and α = (100, −100, 0, …, 0), in NumPy for sampling and in JAX for VI.
- [run_vi.py](run_vi.py): step 1. Tunes the SCS and DCS maps with the shared `fit_vi` and saves them to `fitted_parameters.json`.
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

1. **VI.** `run_vi.py` fits SCS and DCS by reverse KL using the settings at its top, and writes them, with their VI details under `vi_info`, to `fitted_parameters.json`. Run it only when you change those settings; the committed file already holds the paper fit, so step 2 can run straight away. Rerunning step 1 replaces the committed fit, which Git can restore. If you remove a map from `METHODS`, only the other map is refitted and the removed map's saved fit is kept.
2. **Coupling.** `run_experiment.py` loads `fitted_parameters.json` and needs no JAX. Each method starts from the same saved initial pairs and stops at meeting or at `MAX_STEPS`. Results are saved to `new_meeting_times.csv` and `new_meeting_times.pdf`, so the saved paper results are not overwritten. It checks only the maps requested in `STEP_SIZES`. It stops with a message if a required fit is missing, has invalid map parameters, or uses a different SCS latitude. The original committed fit has no `vi_info`; its known SCS latitude is 1.1. A run with only Euclidean and stereographic methods does not require the fit file.

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

Coupling settings (top of `run_experiment.py`):

| Method | Map | Proposal step |
|---|---|---|
| Stereographic | Fixed: zero location, R = 10 | h = 14 |
| Euclidean | None | h = 0.26 |
| SCS | Fitted by VI | h = 0.08; dark-side proposals are rejected directly |
| DCS | Fitted by VI | Ball-walk radius γ/R = 0.14 |

VI tunes only the map parameters. Proposal steps are fixed separately; they were tuned toward 40% stationary acceptance. Each method keeps the random-number stream it used before this refactor (`seed_offset` 100,000 to 400,000, in table order). The offsets are tied to method names, so removing or reordering methods in `STEP_SIZES` does not change the remaining methods’ streams.

## Reproducibility

The committed parameters preserve the fit used for the saved results. A new VI run uses the documented seeds and settings, but floating-point arithmetic and numerical implementation changes can alter its optimization trajectory; it is not guaranteed to reproduce the committed values exactly. The NumPy and JAX targets use the same three-line formula. The small `stdtr_jax` helper evaluates the Student-t CDF through an equivalent symmetric incomplete-beta formula, with no near-zero cutoff or Taylor expansion. With the same fit, the shared sampler gives the same meeting times and acceptance rates as the earlier skew-t code on the same streams. The saved CSV comes from an older implementation with different random streams, so individual meeting times in a new run will differ. In the saved CSV, DCS is labelled `dcs` to match the shared code; the values are unchanged.

[Model and settings](../README.md)
