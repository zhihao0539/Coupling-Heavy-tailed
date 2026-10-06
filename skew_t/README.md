# Skew-t coupling comparison

Open **[experiment.py](experiment.py)**. It contains the target, SCS/DCS maps, coupled proposals, sampling loop and plotting code.

- Leave `RUN_SIMULATION = False` to plot the saved 1,000-repetition [results](meeting_times.csv).
- Set it to `True` to simulate new results; adjust `N_REP` and `MAX_STEPS` at the top.
- Run `python experiment.py`. A new simulation writes `new_meeting_times.csv` and `new_meeting_times.pdf`.

`fitted_parameters.json` stores the existing VI-fitted parameters. `initial_states.npz` stores the common initial pairs. Neither file needs editing to run the experiment.

[Saved figure](meeting_times.pdf) · [Model and settings](../README.md)
