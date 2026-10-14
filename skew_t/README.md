# Skew-t coupling comparison

The experiment uses the shared [coupling, VI and plotting functions](../generic_algorithms/README.md). This folder supplies the target, the fixed settings and the saved inputs.

- [target.py](target.py): the 100-dimensional Azzalini–Capitanio skew-t log density with ν = 2 and α = (100, −100, 0, …, 0), in NumPy for sampling and in JAX for VI.
- [run_vi.py](run_vi.py): VI settings and `fit_parameters()`, which tunes the SCS and DCS maps with the shared `fit_vi`. Run it alone to fit without coupling.
- [run_experiment.py](run_experiment.py): fits the maps with `run_vi.py`, then runs the four couplings with the shared runner and plots them.
- `initial_states.npz`: the 1,000 common Euclidean initial pairs used by every method.
- `fitted_parameters.json`: the SCS and DCS fit used for the saved paper results.
- `meeting_times.csv` and `meeting_times.pdf`: the saved 1,000-repetition results.

## Run

From the repository root:

```sh
python -m pip install -r requirements.txt
python skew_t/run_experiment.py
```

The run has two steps:

1. **VI.** `fit_parameters()` fits SCS and DCS by reverse KL using the settings at the top of `run_vi.py`, and saves the fit to `new_fitted_parameters.json`.
2. **Coupling.** Each method starts from the same saved initial pairs and stops at meeting or at `MAX_STEPS`. SCS and DCS use the fit from step 1. The results are saved to `new_meeting_times.csv` and `new_meeting_times.pdf`.

The saved paper results are not overwritten. Start with a small `N_REP` and `MAX_STEPS`: at the one-million-step cutoff, the Euclidean and stereographic runs take many hours in plain Python.

To fit the maps without coupling, run `python skew_t/run_vi.py`. It writes the same `new_fitted_parameters.json`.

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

VI tunes only the map parameters. Proposal steps are fixed separately; they were tuned toward 40% stationary acceptance. Each method keeps the random-number stream it used before this refactor (`seed_offset` 100,000 to 400,000, in table order).

## Reproducibility

A new run refits VI. With the pinned library versions, the refit matched `fitted_parameters.json` to within 1e-14, taking about two minutes on a CPU; small differences are possible on other platforms or versions. With the same fit, the shared sampler gives the same meeting times and acceptance rates as the earlier skew-t code on the same streams. The saved CSV comes from an older implementation with different random streams, so individual meeting times in a new run will differ. In the saved CSV, DCS is labelled `dcs` to match the shared code; the values are unchanged.

[Model and settings](../README.md)
