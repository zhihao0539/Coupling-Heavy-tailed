# Coupling heavy-tailed distributions

The regression and skew-t experiments each have three small Python files:

- `algorithms.py`: target density, transformations and coupling functions.
- `run_experiment.py`: settings, simulation loop, saving and plotting.
- `plot_results.py`: the figure format and plotting saved results.

Install the four libraries and reproduce the saved 1,000-repetition figures:

```sh
python -m pip install -r requirements.txt
python student_t_regression/plot_results.py
python skew_t/plot_results.py
```

To simulate new results, edit the settings at the top of `run_experiment.py`, then run:

```sh
python student_t_regression/run_experiment.py
python skew_t/run_experiment.py
```

Each run saves `new_meeting_times.csv` and plots `new_meeting_times.pdf`. The saved paper results are preserved. Start with a small `N_REP` when trying the code; the plain Python skew-t run can be long at its one-million-iteration cutoff. Target constants such as degrees of freedom are at the top of `algorithms.py`. To change the plot style, edit `plot_results.py`.

## Spherical random walk — Section 5.1

[Julia code and run instructions](sphere_random_walk/README.md)

Three unchanged Julia files from `Coupling-on-Sphere`: the meeting-time simulation, its required `samplers.jl`, and the plotting script in `figures/`. Their original relative paths are preserved.

## Student's t regression — Section 5.2

[Run the experiment](student_t_regression/run_experiment.py) · [Figure](student_t_regression/meeting_times.pdf)

The supplied Boston Housing data are standardized column by column, with no added intercept. The target has 13 regression coefficients and log scale, Student's t errors with ν = 4, a flat coefficient prior and prior density proportional to 1/σ² on σ². Initial states are independent N(0,100 I₁₄). Stereographic coupling uses R = √14 and h = 0.007; Euclidean coupling uses h = 0.03. Each trial runs 20,000 iterations, including after meeting for the acceptance calculation.

| Method | Meetings | Mean meeting time | Acceptance |
|---|---:|---:|---:|
| Stereographic | 1,000/1,000 | 2,293.805 | 23.57% |
| Euclidean | 1,000/1,000 | 9,674.510 | 24.57% |

## Skew-t comparison — Section 5.3

[Run the experiment](skew_t/run_experiment.py) · [Figure](skew_t/meeting_times.pdf)

The target is the 100-dimensional Azzalini–Capitanio skew-t with ν = 2 and α = (100, −100, 0, …, 0). All four methods use the same saved Euclidean initial pairs in `initial_states.npz`. The file is needed to retain the paper comparison's exact starting pairs.

`fitted_parameters.json` contains the SCS and DCS transformation parameters already fitted by reverse-KL variational inference. Those fits used 4,000 Adam iterations and 2,048 samples per iteration, with the mean of the last 500 iterates selected against the last iterate on a 20,000-sample holdout. Proposal sizes were tuned separately toward 40% stationary acceptance. The experiment loads these fitted values; it does not refit them. SCS uses latitude 1.1 and h = 0.08, and rejects dark-side proposals directly. DCS uses β = 2 and normalized ball step γ/R = 0.14. Stereographic sampling uses R = 10, h = 14; Euclidean sampling uses h = 0.26.

| Method | Meetings by 1,000,000 | Mean / restricted mean |
|---|---:|---:|
| SCS with VI | 1,000/1,000 | 40.876 |
| DCS-Ball walk with VI | 1,000/1,000 | 1,691.961 |
| Euclidean | 987/1,000 | 55,963.372 |
| Stereographic | 558/1,000 | 458,504.412 |

For censored methods, the reported mean is the average of min(τ, cutoff). The histogram includes only meeting pairs; the survival curve includes all pairs. Skew-t acceptance is averaged over paths stopped at meeting or censoring, unlike regression acceptance over the full run. The survival panels are not claimed to bound distance to stationarity for these zero-lag, nonstationary starts.

The saved CSV observations are retained from the validated experiments. New runs use the same target and coupling formulas with new NumPy random streams, so individual meeting times will differ. Meeting times count updates starting from 1; `-1` in a CSV means the pair was censored. The previous implementation is recoverable in Git history; the original user files on disk were not edited.

The variational fits follow [Sub-Cauchy Sampling, Section 3](https://arxiv.org/abs/2601.11066) and [Diffeomorphic MCMC, Appendix D](https://arxiv.org/abs/2608.04284). The settings here use ν = 2, rather than the DCS paper's ν = 3 example.
