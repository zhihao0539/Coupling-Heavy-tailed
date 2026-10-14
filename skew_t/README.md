# Skew-t coupling comparison

- [algorithms.py](algorithms.py): target density, transformations and coupling functions.
- [run_experiment.py](run_experiment.py): experiment settings, repetition loop, saving and plotting.
- [plot_results.py](plot_results.py): figure format and plotting saved results.
- [run_vi.py](run_vi.py): skew-t target and settings for tuning SCS and DCS through the shared VI functions.

Run `python plot_results.py` to reproduce the saved 1,000-repetition figure.

Run `python run_experiment.py` to simulate new results and plot them. Adjust `N_REP` and the step sizes at the top of that file. New runs write `new_meeting_times.csv` and `new_meeting_times.pdf`; the saved paper results stay unchanged. Start with a small `N_REP` to try the code.

`fitted_parameters.json` stores the parameters used for the saved coupling experiment. `initial_states.npz` stores the common initial pairs.

## Tune the transformation parameters

From the repository root:

```sh
python -m pip install -r generic_algorithms/requirements.txt
python skew_t/run_vi.py
```

`run_vi.py` calls [fit_vi](../generic_algorithms/vi.py) for both SCS and DCS. The shared [VI guide](../generic_algorithms/README.md#tune-parameters-with-vipy) explains its arguments and outputs. The application supplies the 100-dimensional skew-t log density with nu = 2 and alpha = (100, -100, 0, ...), plus these settings:

| Setting | SCS | DCS |
|---|---|---|
| Fixed map setting | Latitude 1.1 | beta = 2 |
| Initial radius | 10/11 | sqrt(2) |
| Initial location and offset | Zero | Zero |
| Seed | 26093001 | 26092901 |
| Fitted quantities | Location, radius, horizontal observer | Location, radius, Mobius offset |

Both fits use 4,000 Adam updates, 2,048 reference samples per update, learning rate 0.01 for 3,000 updates and then 0.002. A 20,000-sample holdout selects between the final iterate and the average of the last 500 unconstrained parameter vectors. These values can be changed at the top of `run_vi.py`; remove an entry from `METHODS` to fit only one method.

The script writes `scs_fit.json` and `dcs_fit.json` beside itself. Each contains a `parameters` object and a `fit` object with settings and holdout losses. To use a new fit for coupling, replace the matching `scs` or `dcs` entry in `fitted_parameters.json` with that file's `parameters` object. The script does not overwrite the saved parameters automatically. Small numerical differences across platforms or library versions are possible.

VI tunes the transformation parameters. Proposal steps remain separate in `run_experiment.py`: SCS h = 0.08 and DCS gamma/R = 0.14. SCS coupling rejects dark-side proposals directly, without stepping out.

[Saved figure](meeting_times.pdf) · [Model and settings](../README.md)
