"""Run the five Student-t couplings using the step sizes from the completed experiment."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parents[1]))

from generic_algorithms import (
    load_parameters, run_stereographic_experiment, run_scs_experiment,
    run_dcs_matrix_ball_experiment, run_euclidean_experiment, combine_results, save_results,
)
from target import DIMENSIONS, make_log_density

RESULTS = FOLDER / "results"
MAX_STEPS = 1_000_000
N_REP = 1000
INITIAL_SEED = 43
PROPOSAL_SEED = 42
PROGRESS_EVERY = 25
LATITUDES = {"SCS11": 1.1, "SCS19": 1.9}

# h for each method and dimension, taken from the completed experiment.
STEP_SIZES = {
    "stereographic": {
        1: 2.0,
        2: 1.414213562373095,
        5: 0.8944271909999159,
        10: 0.6324555320336759,
        20: 0.8944271909999159,
        50: 0.565685424949238,
        100: 0.4,
    },
    "SCS11": {
        1: 2.0,
        2: 1.414213562373095,
        5: 0.8944271909999159,
        10: 0.6324555320336759,
        20: 0.4472135954999579,
        50: 0.1414213562373095,
        100: 0.07071067811865477,
    },
    "SCS19": {
        1: 2.0,
        2: 1.414213562373095,
        5: 0.8944271909999159,
        10: 0.6324555320336759,
        20: 0.8944271909999159,
        50: 0.565685424949238,
        100: 0.4,
    },
    "dcs": {
        1: 11.416388304026121,
        2: 3.3941125496954276,
        5: 1.517893276880822,
        10: 0.9025447423407345,
        20: 0.5366563145999494,
        50: 0.3250213312648542,
        100: 0.22008097036912108,
    },
    "euclidean": {
        1: 19.2,
        2: 5.234437116793236,
        5: 1.6552743557991434,
        10: 0.9025447423407345,
        20: 0.5852278608548841,
        50: 0.35443827502547975,
        100: 0.24,
    },
}


def initial_pairs(d, count, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, d]))
    x = rng.standard_normal((count, 2, d))
    return x / np.sqrt(rng.chisquare(d + 1, size=(count, 2, 1)) / (d + 1))


def proposal_seed(d):
    seed = PROPOSAL_SEED + d
    if seed == INITIAL_SEED:
        seed += 1_000_000
    return seed


def run_stereographic(d, starts):
    return run_stereographic_experiment(
        make_log_density(d), starts, STEP_SIZES["stereographic"][d], MAX_STEPS,
        mu=np.zeros(d), preconditioning_matrix=np.sqrt(d) * np.eye(d),
        seed=proposal_seed(d), seed_index=0,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_scs11(d, starts):
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    fitted = load_parameters(path, d, ["SCS11"], latitude=LATITUDES["SCS11"])["SCS11"]
    return run_scs_experiment(
        make_log_density(d), starts, STEP_SIZES["SCS11"][d], MAX_STEPS,
        **fitted, latitude=LATITUDES["SCS11"], label="SCS11",
        seed=proposal_seed(d), seed_index=1,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_scs19(d, starts):
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    fitted = load_parameters(path, d, ["SCS19"], latitude=LATITUDES["SCS19"])["SCS19"]
    return run_scs_experiment(
        make_log_density(d), starts, STEP_SIZES["SCS19"][d], MAX_STEPS,
        **fitted, latitude=LATITUDES["SCS19"], label="SCS19",
        seed=proposal_seed(d), seed_index=2,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_dcs_ball(d, starts):
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    fitted = load_parameters(path, d, ["dcs"])["dcs"]
    return run_dcs_matrix_ball_experiment(
        make_log_density(d), starts, STEP_SIZES["dcs"][d], MAX_STEPS, **fitted,
        seed=proposal_seed(d), seed_index=3,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_euclidean(d, starts):
    return run_euclidean_experiment(
        make_log_density(d), starts, STEP_SIZES["euclidean"][d], MAX_STEPS,
        seed=proposal_seed(d), seed_index=4,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def save_completed_rows(rows, path):
    with TemporaryDirectory(dir=path.parent) as tmp:
        temp = Path(tmp) / path.name
        save_results(rows, temp)
        temp.replace(path)


def run_dimension(d):
    starts = initial_pairs(d, N_REP, INITIAL_SEED)
    rows = combine_results(
        run_stereographic(d, starts),
        run_scs11(d, starts),
        run_scs19(d, starts),
        run_dcs_ball(d, starts),
        run_euclidean(d, starts),
    )
    rows = [dict(dimension=d, df=d+1, cutoff=MAX_STEPS, h=STEP_SIZES[row["method"]][d],
                 scs_latitude=LATITUDES.get(row["method"]), initialization="target", **row)
            for row in rows]
    save_completed_rows(rows, RESULTS / f"d_{d:04d}" / "meeting_times.csv")
    return rows


def main():
    rows = []
    for d in DIMENSIONS:
        rows.extend(run_dimension(d))
    save_completed_rows(rows, RESULTS / "meeting_times.csv")


if __name__ == "__main__":
    main()
