"""Run five explicit fixed-map normal couplings; no VI."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
REPOSITORY = FOLDER.parents[1]
sys.path.insert(0, str(REPOSITORY))

from target import DIMENSIONS, make_log_density, sample_target
from generic_algorithms import (
    run_stereographic_experiment, run_scs_experiment, run_dcs_ball_experiment,
    run_dcs_random_walk_experiment,
    run_euclidean_experiment, combine_results, save_results,
)

RESULTS = FOLDER / "results"
N_REP = 1000
MAX_STEPS = 1_000_000
SEED = 42
INITIAL_SEED = 43
PROGRESS_EVERY = 25

# One step per dimension: (1, 2, 5, 10, 20, 50, 100).
STEREOGRAPHIC_STEP_SIZES = [32, 11.3, 3.58, 2.53, 1.79, 1.13, 0.8]
SCS11_STEP_SIZES = [0.878, 0.402, 0.205, 0.133, 0.0931, 0.0577, 0.0411]
DCS_STEP_SIZES = [3.25, 1.54, 0.848, 0.561, 0.391, 0.243, 0.17]
EUCLIDEAN_STEP_SIZES = [5.39, 2.43, 1.22, 0.806, 0.553, 0.343, 0.244]
# Gaussian standard deviations in B(1), calibrated for mean acceptance through meeting.
DCS_RANDOM_WALK_STEP_SIZES = [6.9126, 1.65894, 0.474046, 0.196808, 0.0887964, 0.0343654, 0.017]
STEP_SIZES = {
    "stereographic": STEREOGRAPHIC_STEP_SIZES,
    "SCS11": SCS11_STEP_SIZES,
    "dcs": DCS_STEP_SIZES,
    "euclidean": EUCLIDEAN_STEP_SIZES,
    "dcs_rw": DCS_RANDOM_WALK_STEP_SIZES,
}
PROPOSAL_STREAMS = {
    "stereographic": (SEED, 0), "SCS11": (SEED, 1),
    "dcs": (SEED, 3), "euclidean": (SEED, 4),
    "dcs_rw": (SEED, 5),
}


def map_parameters(dimension):
    """Fixed isotropic maps; edit each method's explicit parameters here."""
    d = dimension
    return {
        "stereographic": {"mu": np.zeros(d), "preconditioning_matrix": np.sqrt(d) * np.eye(d)},
        "SCS11": {"mu": np.zeros(d), "preconditioning_matrix": np.sqrt(d)/11 * np.eye(d),
                  "horizontal_observer": np.zeros(d)},
        # R^2=d+2 gives a convex Gaussian pull-back potential for beta=2.
        # delta=0 is the identity Mobius map, appropriate for this symmetry.
        "dcs": {"mu": np.zeros(d), "radius": np.sqrt(d+2), "delta": np.zeros(d)},
    }


def run_dimension(index, d, starts, parameters, seeds):
    target = make_log_density(d)
    stereo, scs, dcs = (parameters[name] for name in ("stereographic", "SCS11", "dcs"))
    return combine_results(
        run_stereographic_experiment(
            target, starts, STEREOGRAPHIC_STEP_SIZES[index], MAX_STEPS,
            mu=stereo["mu"], preconditioning_matrix=stereo["preconditioning_matrix"],
            seed=seeds["stereographic"], seed_index=PROPOSAL_STREAMS["stereographic"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_scs_experiment(
            target, starts, SCS11_STEP_SIZES[index], MAX_STEPS, mu=scs["mu"],
            preconditioning_matrix=scs["preconditioning_matrix"],
            horizontal_observer=scs["horizontal_observer"], latitude=1.1, label="SCS11",
            seed=seeds["SCS11"], seed_index=PROPOSAL_STREAMS["SCS11"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_dcs_ball_experiment(
            target, starts, DCS_STEP_SIZES[index], MAX_STEPS,
            mu=dcs["mu"], radius=dcs["radius"], delta=dcs["delta"],
            seed=seeds["dcs"], seed_index=PROPOSAL_STREAMS["dcs"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_euclidean_experiment(
            target, starts, EUCLIDEAN_STEP_SIZES[index], MAX_STEPS,
            seed=seeds["euclidean"], seed_index=PROPOSAL_STREAMS["euclidean"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_dcs_random_walk_experiment(
            target, starts, DCS_RANDOM_WALK_STEP_SIZES[index], MAX_STEPS,
            mu=dcs["mu"], radius=dcs["radius"], delta=dcs["delta"],
            seed=seeds["dcs_rw"], seed_index=PROPOSAL_STREAMS["dcs_rw"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
    )


def main():
    if any(len(values) != len(DIMENSIONS) for values in STEP_SIZES.values()):
        raise ValueError("Each step-size list needs one entry per dimension.")
    all_rows = []
    for index, d in enumerate(DIMENSIONS):
        folder = RESULTS / f"d_{d:04d}"
        parameters = map_parameters(d)
        rng = np.random.default_rng(np.random.SeedSequence([INITIAL_SEED, d]))
        starts = sample_target(d, (N_REP, 2), rng)
        seeds = {name: base+d + (1_000_000 if base+d == INITIAL_SEED else 0)
                 for name, (base, _) in PROPOSAL_STREAMS.items()}
        rows = run_dimension(index, d, starts, parameters, seeds)
        dimension_rows = [dict(dimension=d, target="multivariate_normal", cutoff=MAX_STEPS,
                             h=STEP_SIZES[row["method"]][index],
                             map_radius={"stereographic": np.sqrt(d), "SCS11": np.sqrt(d)/11,
                                         "dcs": np.sqrt(d+2), "dcs_rw": np.sqrt(d+2)}.get(row["method"], np.nan),
                             scs_latitude=1.1 if row["method"] == "SCS11" else np.nan,
                             initialization="target", **row) for row in rows]
        save_results(dimension_rows, folder / "meeting_times.csv")
        all_rows.extend(dimension_rows)
        pd.DataFrame(all_rows).to_csv(RESULTS / "meeting_times.csv", index=False)


if __name__ == "__main__":
    main()
