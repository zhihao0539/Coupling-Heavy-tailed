# Boston Housing: Student's t regression

The experiment now calls the shared [coupling algorithms](../../generic_algorithms/README.md). This folder supplies the target log density, data, and fixed experiment settings. The stereographic and Euclidean experiment functions handle simulation; the shared plotting function produces figures. No VI is used.

- [target.py](target.py): load and standardize Boston Housing data; evaluate the Student-t regression log density.
- [run_experiment.py](run_experiment.py): set the fixed parameters and initial states, then call the stereographic and Euclidean experiment functions. Its [parameter guide](../../generic_algorithms/README.md#run-a-fixed-parameter-experiment) explains each input and output.
- [Shared plot_results.py](../../generic_algorithms/plot_results.py): plot the saved meeting times; no separate plotting file is needed here.
- [BostonHousing.csv](BostonHousing.csv): input data.

From the repository root:

```sh
python -m pip install -r requirements.txt
python student_t_regression/boston_housing/run_experiment.py
python generic_algorithms/plot_results.py student_t_regression/boston_housing/meeting_times.csv
```

The full run uses 1,000 independent initial pairs and 20,000 updates for each of the two chains in each pair. Initial coordinates are independent N(0, 100). The model has 13 standardized predictors, no added intercept, Student-t errors with 4 degrees of freedom, and one log-scale coordinate. Its target log density in [target.py](target.py) is unchanged from the earlier experiment.

| Method | Fixed map and proposal parameters |
|---|---|
| Stereographic | Matrix L = sqrt(14) I; tangent Gaussian step h = 0.007 |
| Euclidean | Gaussian random-walk step h = 0.03 |

`run_experiment.py` supplies the model and initial pairs, then calls `run_stereographic_experiment` and `run_euclidean_experiment` with explicit parameters. Each performs its repetitions sequentially; no VI or worker setup is needed. Every pair continues for all 20,000 updates, including after meeting, so acceptance rates use the full run. The shared sampler records meeting at exact equality in proposal coordinates; the earlier regression script used a distance threshold of 1e-12. Fixed seed offsets preserve the previously used random streams.

The local functions are `generate_initial_pairs(dimension, n_pairs, seed)`,
`run_stereographic(starts, target)`, and `run_euclidean(starts, target)`.
`main()` loads the data, defines the regression target, and passes the same
starting pairs and target to both method functions before saving and plotting.
Change `STEREOGRAPHIC_H` or `EUCLIDEAN_H` at the top of the script to tune a
method's proposal; `N_REP`, `N_STEPS`, `SEED`, and `PROGRESS_EVERY` control the run.

The run writes [meeting_times.csv](meeting_times.csv) and calls the shared plotter to create [meeting_times.pdf](meeting_times.pdf). To redraw the figure without rerunning MCMC, use the last command above. The generic plotter takes a CSV path and saves a PDF beside it; see its [usage guide](../../generic_algorithms/README.md#plot-saved-results).

The saved 1,000-repetition results are:

| Method | Meetings | Mean meeting time | Mean acceptance |
|---|---:|---:|---:|
| Stereographic | 1,000/1,000 | 2,297.837 | 23.556% |
| Euclidean | 1,000/1,000 | 9,599.094 | 24.447% |
