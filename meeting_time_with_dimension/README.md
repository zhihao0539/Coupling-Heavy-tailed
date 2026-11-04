# Meeting time versus dimension

These experiments compare coupling meeting times in dimensions
1, 2, 5, 10, 20, 50 and 100 using `generic_algorithms`.

| Folder | Target | Pairs per method and dimension | Update cutoff | Map parameters |
|---|---|---:|---:|---|
| `multivariate_normal` | Standard multivariate normal | 1,000 | 1,000,000 | Fixed |
| `multivariate_student_t_2` | Multivariate Student-t, df=2, identity scale | 100 | 10,000,000 | VI |
| `multivariate_student_t_d+1` | Multivariate Student-t, df=d+1, identity scale | 1,000 | 1,000,000 | VI |

Every method uses the same independent pair of target draws for each
replicate. Proposal random streams are separate for each method. For the
Student-t targets, one chi-square variable scales the whole vector; the
coordinates are not independent Student-t draws.

All comparisons include stereographic, SCS at latitude 1.1, Euclidean,
DCS ball walk and DCS Gaussian random walk. The Student-t experiments also
include SCS at latitude 1.9. Both DCS proposals share their map: a fixed
scalar-radius Mobius map for normal, or the saved matrix map for Student-t.

Install the repository's requirements, then run commands from the repository
root. For example, to plot the supplied Student-t df=2 results:

```bash
python meeting_time_with_dimension/multivariate_student_t_2/plot_results.py
```

To rerun with the supplied VI fits, run that folder's `run_experiment.py`
followed by `plot_results.py`. To refit all four Student-t maps first, run
`run_vi.py`, or use `run_all.py` to fit, simulate and plot. The normal
experiment uses fixed maps, so its `run_all.py` only simulates and plots.
Full runs may be slow, particularly for censored meeting times.

Set `N_REP`, `MAX_STEPS` and the method-specific step-size lists in
`run_experiment.py`. Student-t VI settings are in `run_vi.py`; normal map
settings are in `run_experiment.py`. Step sizes are used as supplied and
are not automatically tuned.

Each folder keeps its own `results/`: combined and per-dimension CSVs,
summary tables, per-dimension meeting-time plots and the comparison across
dimensions. Student-t folders also save their fitted maps. Running the
scripts replaces the corresponding outputs.

Meeting times of `-1` indicate pairs that did not meet before the cutoff.
Tables report the restricted mean, assigning the cutoff to those pairs.
The unrestricted mean is reported only if every pair met. The extra df=2
panel shows the mean among observed meetings; this conditional mean should
not be interpreted as the unrestricted mean when some pairs are censored.
Error bars are one empirical Monte Carlo standard error.
