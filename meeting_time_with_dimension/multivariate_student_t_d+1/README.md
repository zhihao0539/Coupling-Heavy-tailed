# Multivariate Student-t coupling comparison (df=d+1)

The six methods are `stereographic`, `SCS11`, `SCS19`, `dcs`, `euclidean`, and `dcs_rw`.
`SCS11` and `SCS19` use the shared `scs` algorithm with latitudes 1.1 and 1.9.
They are fitted by separate calls in `run_vi.py` and follow
the same fitting, loading, coupling and plotting code.

Dimensions are 1, 2, 5, 10, 20, 50, 100. The target is multivariate Student-t
with df=d+1 and identity scale matrix. A shared chi-square scale for each
vector makes its coordinates dependent. Each method uses 1,000 pairs of independent target
draws and a 1,000,000-update cutoff. The pairs are shared across all methods.
DCS uses its preconditioning matrix without a Mobius offset.

- `run_vi.py`: supply each map's initial parameters to its named VI function
  once per dimension for all four VI maps. Running this script fits and saves
  the maps directly. Euclidean does not need a VI map.
- `run_experiment.py`: load the saved maps with the shared parameter loader,
  then call each method's experiment function explicitly. Each has
  one editable step-size list; existing random streams are preserved through
  the uniform `PROPOSAL_STREAMS` dictionary.
- `plot_results.py`: load all six saved methods through the same loop. The
  two panels show mean meeting time and step size, with subtitles and no
  overall title. Per-dimension diagnostics include all six methods together.
- `run_all.py`: fit maps, run all couplings, then plot. To use completed
  results without new simulations, run `plot_results.py` alone.

All six methods read and save data under this experiment's own `results/`
folder. `results/meeting_times.csv` will contain all six methods after the updated experiment runs, and
each `results/d_XXXX/` folder contains its combined coupling results and
`fitted_parameters.json`. The four VI maps are stored under `stereographic`,
`SCS11`, `SCS19`, and `dcs` in that same JSON file. Euclidean needs no VI map.
Only `SCS11` and `SCS19` have a `latitude` setting. The latitude column in CSVs
is blank for methods where it is not applicable. There is no `SAVED_RUNS`
mapping or dependency on results elsewhere in the Codex workspace.

The experiment calls the stereographic, SCS, matrix DCS, and Euclidean
functions directly. SCS11 and SCS19 are two calls to `run_scs_experiment`
with explicit latitudes and result labels.
The plotting helper accepts an optional method-style dictionary.

Error bars are one empirical Monte Carlo standard error. Unrestricted means
are omitted when any pair is censored; restricted means remain in the tables.
This code does not include the separate stereographic two-step-size study.

## DCS Gaussian random walk

`dcs_rw` uses `run_dcs_matrix_random_walk_experiment` and exactly the same
saved `dcs` center and preconditioning matrix as the ball walk. No extra VI
fit is needed. Its Gaussian proposal has coordinate standard deviation `h`
in the native unit ball B(1). A proposal outside B(1) is rejected.

`DCS_RANDOM_WALK_STEP_SIZES` in `run_experiment.py` contains the following
calibrated values, in the same order as `DIMENSIONS`:

| Dimension | h | Independent pilot acceptance |
|---|---:|---:|
| 1 | 9.19687 | 23.11% |
| 2 | 2.16524 | 23.04% |
| 5 | 0.5456 | 23.47% |
| 10 | 0.172029 | 22.90% |
| 20 | 0.0621627 | 22.98% |
| 50 | 0.0168939 | 22.87% |
| 100 | 0.00622 | 22.83% |

The tuning target is the average reported by these experiments: compute
accepted proposals divided by iterations separately for each chain through
meeting (or the cutoff), then average those rates across chains and pairs.
Independent validation used 40,000, 40,000, 20,000, 8,000, 3,000, 1,024, and
512 pairs respectively, with the saved matrix maps and the 1,000,000-update
cutoff. Every validation pair met before the cutoff. These are pilot estimates;
the 1,000-pair production results fluctuate. Retune if the fitted DCS maps or
stopping rule changes.

The original five methods retain their proposal scales, maps, and random
streams. DCS random walk has its own proposal stream (index 5) and uses the
same initial pairs. Plotting includes it once its results are available; older
saved files containing the original five methods can still be plotted.
Existing saved meeting-time results have not been rerun or replaced.
