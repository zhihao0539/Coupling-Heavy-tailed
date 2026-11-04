# Multivariate Student-t coupling comparison (df=2)

The six methods are `stereographic`, `SCS11`, `SCS19`, `dcs`, `euclidean`, and `dcs_rw`.
`SCS11` and `SCS19` use the shared `scs` algorithm with latitudes 1.1 and 1.9.
They are fitted by separate calls in `run_vi.py` and follow
the same fitting, loading, coupling and plotting code.

Dimensions are 1, 2, 5, 10, 20, 50, 100. The target is a multivariate
Student-t with df=2 and identity scale matrix. Each method uses 100 pairs of independent target
draws and a 10,000,000-update cutoff. The pairs are shared across all methods.
DCS uses its preconditioning matrix without a Mobius offset.

- `run_vi.py`: supply each map's initial parameters to its named VI function
  once per dimension for all four VI maps. Running this script fits and saves
  the maps directly. Euclidean does not need a VI map.
- `run_experiment.py`: load the saved maps with the shared parameter loader,
  then call each method's experiment function explicitly. Each has
  one editable step-size list; existing random streams are preserved through
  the uniform `PROPOSAL_STREAMS` dictionary.
- `plot_results.py`: load all six saved methods through the same loop. The
  three panels show the mean when all pairs met, the mean among pairs that
  met, and step size, with subtitles and no overall title. Per-dimension
  diagnostics include all six methods together.
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
The middle panel uses all met pairs in each method/dimension group, even when
other pairs in that group did not meet. Its error bars are the standard
deviation of those observed meeting times divided by the square root of
the number of met pairs. Groups with no meetings are omitted; a single
meeting supplies a point without an estimated error bar. These summaries
are saved as `mean_met_pairs` and `mean_met_pairs_mcse` in the summary CSVs.
This code does not include the separate stereographic two-step-size study.

## DCS Gaussian random walk

`dcs_rw` calls `run_dcs_matrix_random_walk_experiment` with the same saved
`dcs` location and preconditioning matrix as the DCS ball walk. No additional
VI fit is needed. Its step `h` is the Gaussian standard deviation in each
native B(1) coordinate. Proposals outside B(1) are rejected.

`DCS_RANDOM_WALK_STEP_SIZES` contains these calibrated values in dimension order:

| Dimension | h | Independent pilot acceptance |
|---|---:|---:|
| 1 | 9.19687 | 23.22% |
| 2 | 2.27511 | 23.06% |
| 5 | 0.656566 | 23.19% |
| 10 | 0.274625 | 23.24% |
| 20 | 0.126136 | 23.16% |
| 50 | 0.0489207 | 22.82% |
| 100 | 0.0241 | 23.11% |

The tuning target is the average printed when pairs stop at meeting:
divide accepted proposals by iterations separately for each chain through
meeting (or the cutoff), then average across both chains and all pairs.
Independent validation used 100,000, 100,000, 40,000, 12,000, 4,000, 1,024,
and 512 pairs respectively with the production cutoff of 1,000,000 updates.
Every validation pair met. The 1,000-pair production run will fluctuate
around these estimates. Retune after changing the target, fitted DCS maps,
or stopping rule.

The existing five methods retain their maps, scales, and random streams.
DCS random walk shares their initial pairs and has its own proposal stream
(index 5). Plots include it once its results are available, and older
five-method results remain plottable. The plotting metadata and labels use
the correct df=2. Existing saved results have not been rerun or replaced.

With the existing fits, run `run_experiment.py`, then `plot_results.py`.
`run_all.py` refits VI before simulation, so it is unnecessary for this addition.
