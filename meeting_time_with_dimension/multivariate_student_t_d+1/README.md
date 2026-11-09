# Meeting time versus dimension for Student-t(d+1)

In dimension d the target is a centered multivariate Student-t with d+1 degrees of
freedom and identity scale. The comparison uses d = 1, 2, 5, 10, 20, 50, 100.

| Coupling | Fixed geometry | VI fields |
|---|---|---|
| Stereographic | Location 0, radius √d (matrix √d I) | None |
| SCS11 | Location and horizontal observer 0, latitude 1.1 | Full preconditioning matrix |
| SCS19 | Location and horizontal observer 0, latitude 1.9 | Full preconditioning matrix |
| DCS matrix ball walk | Location 0, proposals in the unit ball | Full preconditioning matrix and positive β, fitted jointly |
| Euclidean random walk | Isotropic Gaussian proposal | None |

The saved VI metadata confirms these choices in every dimension. DCS starts
from β = max(2, π√(d/6)) and L = √(d+1) I when refitted. Its matrix and β
fully specify the transformation once the center is fixed.

## Run

To use the saved fitted maps, run the coupling and plotting stages.

```sh
python run_experiment.py
python plot_results.py
```

To fit the maps first and then run the experiment, use `python run_all.py`.
For fitting alone, use `python run_vi.py`, or `python run_vi.py --dcs-only`
to fit only DCS.

Production uses 1,000 pairs of independent target draws per dimension and a
1,000,000-update cutoff. All five methods start from the same pairs. The short
`run_stereographic`, `run_scs11`, `run_scs19`, `run_dcs_ball`, and
`run_euclidean` functions call their matching `generic_algorithms` runners.

## Step sizes

`STEP_SIZES` in `run_experiment.py` contains an editable h for every method
and dimension. These are the exact calibrated values used in the saved
experiment. The calibration targeted acceptance averaged across chains and
pairs through meeting or cutoff. Its `step_sizes.json` records include the
pilot trials, selection reasons, and independent validation estimates.

The experiment reads its steps directly from `STEP_SIZES`. To change a step
for a future run, edit that dictionary.

## Results

All fitted maps, calibration records, meeting times, summaries, and figures
are directly under `results/`. Each `d_XXXX/` folder contains that
dimension's files. The combined `meeting_times.csv` includes the actual h
used for each row.

Meeting-time error bars are one Monte Carlo standard error across pairs.
Unrestricted mean meeting times are plotted where every pair met.
