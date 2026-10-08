# Skew-t coupling comparison

- [algorithms.py](algorithms.py): target density, transformations and coupling functions.
- [run_experiment.py](run_experiment.py): experiment settings, repetition loop, saving and plotting.
- [plot_results.py](plot_results.py): figure format and plotting saved results.
- [vi/](vi/): variational-inference functions and fitting scripts for SCS and DCS.

Run `python plot_results.py` to reproduce the saved 1,000-repetition figure.

Run `python run_experiment.py` to simulate new results and plot them. Adjust `N_REP` and the step sizes at the top of that file. New runs write `new_meeting_times.csv` and `new_meeting_times.pdf`; the saved paper results stay unchanged. Start with a small `N_REP` to try the code.

`fitted_parameters.json` stores the existing VI-fitted parameters. `initial_states.npz` stores the common initial pairs. Neither file needs editing.

[Saved figure](meeting_times.pdf) · [Model and settings](../README.md)
