# Coupling heavy-tailed distributions

Both Python experiments use the shared coupling, VI and plotting functions in [`generic_algorithms/`](generic_algorithms/README.md). Each application keeps its target in `target.py`, and its fixed settings and run script, in its own folder: [`student_t_regression/`](student_t_regression/README.md) and [`skew_t/`](skew_t/README.md).

Use each application’s `run_experiment.py` for simulation and `generic_algorithms/plot_results.py` for figures.

Install the libraries and reproduce the saved 1,000-repetition figures:

```sh
python -m pip install -r requirements.txt
python generic_algorithms/plot_results.py student_t_regression/meeting_times.csv
python generic_algorithms/plot_results.py skew_t/meeting_times.csv --log-scale
```

To simulate new results, edit the settings at the top of `run_experiment.py`, then run:

```sh
python student_t_regression/run_experiment.py
python skew_t/run_experiment.py
```

The regression run saves `meeting_times.csv` and `meeting_times.pdf`. The skew-t run first fits its SCS and DCS maps by VI (settings in `skew_t/run_vi.py`, fit saved as `new_fitted_parameters.json`), then saves `new_meeting_times.csv` and `new_meeting_times.pdf`. Start with a small `N_REP` when trying the code; the plain Python skew-t run can be long at its one-million-iteration cutoff. Target constants are in each application’s `target.py`. Both figures come from `generic_algorithms/plot_results.py`; the skew-t figure uses its log-scale option.

## Spherical random walk — Section 5.1

[Julia code and run instructions](sphere_random_walk/README.md)

Three Julia files from `Coupling-on-Sphere`: the meeting-time simulation, its required `samplers.jl`, and `make_rw_meeting_times_fig.jl`. All three are in `sphere_random_walk/`; only the plotting script’s input paths were adjusted for this layout.

## Student's t regression — Section 5.2

[Run the experiment](student_t_regression/run_experiment.py) · [Figure](student_t_regression/meeting_times.pdf) · [Full regression details](student_t_regression/README.md)

The supplied Boston Housing data are standardized column by column, with no added intercept. The target has 13 regression coefficients and log scale, Student's t errors with ν = 4, a flat coefficient prior and prior density proportional to 1/σ² on σ². Initial states are independent N(0,100 I₁₄). Stereographic coupling uses R = √14 and h = 0.007; Euclidean coupling uses h = 0.03. Each trial runs 20,000 iterations, including after meeting for the acceptance calculation.

| Method | Meetings | Mean meeting time | Acceptance |
|---|---:|---:|---:|
| Stereographic | 1,000/1,000 | 2,297.837 | 23.556% |
| Euclidean | 1,000/1,000 | 9,599.094 | 24.447% |

These are the fixed-parameter results using the shared sampler. No variational inference is used in this regression experiment.

## Skew-t comparison — Section 5.3

[Run the experiment](skew_t/run_experiment.py) · [Figure](skew_t/meeting_times.pdf)

The target is the 100-dimensional Azzalini–Capitanio skew-t with ν = 2 and α = (100, −100, 0, …, 0). All four methods use the same saved Euclidean initial pairs in `initial_states.npz`. The file is needed to retain the paper comparison's exact starting pairs.

The SCS and DCS transformation parameters are fitted by reverse-KL variational inference, using 4,000 Adam iterations and 2,048 samples per iteration, with the mean of the last 500 iterates selected against the last iterate on a 20,000-sample holdout. `skew_t/run_experiment.py` repeats this fit before every coupling run; `fitted_parameters.json` keeps the fit used for the saved results. Proposal sizes were tuned separately toward 40% stationary acceptance. SCS uses latitude 1.1 and h = 0.08, and rejects dark-side proposals directly. DCS uses β = 2 and normalized ball step γ/R = 0.14. Stereographic sampling uses R = 10, h = 14; Euclidean sampling uses h = 0.26.

| Method | Meetings by 1,000,000 | Mean / restricted mean |
|---|---:|---:|
| SCS with VI | 1,000/1,000 | 40.876 |
| DCS-Ball walk with VI | 1,000/1,000 | 1,691.961 |
| Euclidean | 987/1,000 | 55,963.372 |
| Stereographic | 558/1,000 | 458,504.412 |

For censored methods, the reported mean is the average of min(τ, cutoff). The histogram includes only meeting pairs; the survival curve includes all pairs. Skew-t acceptance is averaged over paths stopped at meeting or censoring, unlike regression acceptance over the full run. The survival panels are not claimed to bound distance to stationarity for these zero-lag, nonstationary starts.

The saved skew-t CSV observations are retained from the validated experiments, which used different NumPy random streams, so individual skew-t meeting times in a new run will differ. A regression rerun reproduces its saved CSV exactly. Meeting times count updates starting from 1; `-1` in a CSV means the pair was censored. The previous implementation is recoverable in Git history; the original user files on disk were not edited.

The variational fits follow [Sub-Cauchy Sampling, Section 3](https://arxiv.org/abs/2601.11066) and [Diffeomorphic MCMC, Appendix D](https://arxiv.org/abs/2608.04284). The settings here use ν = 2, rather than the DCS paper's ν = 3 example.
