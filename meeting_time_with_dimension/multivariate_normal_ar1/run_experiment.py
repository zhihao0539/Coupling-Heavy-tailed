"""Compare three AR(1)-Gaussian couplings using reverse-KL-fitted maps."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parents[1]))

from target import DIMENSIONS, make_log_density, sample_target, RHO
from generic_algorithms import (
    run_stereographic_experiment, run_dcs_matrix_ball_experiment,
    run_euclidean_experiment, load_parameters, combine_results, save_results,
)

RESULTS = FOLDER / "results"
N_REP = 100
MAX_STEPS = 1_000_000
SEED = 42
INITIAL_SEED = 43
PROGRESS_EVERY = 25

# Transformed-method steps are recalibrated after VI; tuning scripts stay outside this folder.
STEP_SIZES = {
    "stereographic": {
        5: 0.8944271909999159,
        10: 0.6324555320336759,
        20: 1.7888543819998317,
        50: 0.565685424949238,
        100: 0.4,
    },
    "dcs": {
        5: 1.490790564478997,
        10: 0.8516710764265888,
        20: 0.5437917089037533,
        50: 0.32007716108024103,
        100: 0.22293431187466978,
    },
    "euclidean": {
        5: 0.5414673072450846,
        10: 0.2897306905045301,
        20: 0.18987264159928455,
        50: 0.11407727605867513,
        100: 0.07846011787977335,
    },
}


def fitted_parameters(d, method):
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    parameters = load_parameters(path, d, [method])[method]
    if method == "dcs" and ("preconditioning_matrix" not in parameters or parameters["beta"] <= 2):
        raise ValueError("Refit matrix DCS with beta>2 using run_vi.py.")
    return parameters


def run_stereographic(d, starts):
    return run_stereographic_experiment(
        make_log_density(d), starts, STEP_SIZES["stereographic"][d], MAX_STEPS,
        **fitted_parameters(d, "stereographic"), seed=SEED+d, seed_index=0,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_dcs_ball(d, starts):
    return run_dcs_matrix_ball_experiment(
        make_log_density(d), starts, STEP_SIZES["dcs"][d], MAX_STEPS,
        **fitted_parameters(d, "dcs"), seed=SEED+d, seed_index=3,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_euclidean(d, starts):
    return run_euclidean_experiment(
        make_log_density(d), starts, STEP_SIZES["euclidean"][d], MAX_STEPS,
        seed=SEED+d, seed_index=4, stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def save_completed_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        temp = Path(tmp) / path.name
        save_results(rows, temp)
        temp.replace(path)


def run_dimension(d):
    rng = np.random.default_rng(np.random.SeedSequence([INITIAL_SEED, d]))
    starts = sample_target(d, (N_REP, 2), rng)
    rows = combine_results(run_stereographic(d, starts), run_dcs_ball(d, starts), run_euclidean(d, starts))
    rows = [dict(dimension=d, target="multivariate_normal_ar1", rho=RHO, cutoff=MAX_STEPS,
                 h=STEP_SIZES[row["method"]][d], initialization="target",
                 map_source="none" if row["method"] == "euclidean" else "reverse_kl",
                 dcs_map="matrix" if row["method"] == "dcs" else None, **row) for row in rows]
    save_completed_rows(rows, RESULTS/f"d_{d:04d}"/"meeting_times.csv")
    return rows


def main():
    rows = []
    for d in DIMENSIONS:
        rows.extend(run_dimension(d))
    save_completed_rows(rows, RESULTS/"meeting_times.csv")


if __name__ == "__main__":
    main()
