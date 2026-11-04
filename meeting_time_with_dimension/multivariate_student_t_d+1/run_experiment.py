"""Step 2: run each of the six coupling experiments explicitly."""
import numpy as np
import pandas as pd

from run_vi import RESULTS
from target import DIMENSIONS, make_log_density
from generic_algorithms import (
    run_stereographic_experiment, run_scs_experiment, run_dcs_matrix_ball_experiment,
    run_dcs_matrix_random_walk_experiment,
    run_euclidean_experiment, load_parameters, combine_results, save_results,
)

N_REP = 1000
MAX_STEPS = 1_000_000
SEED = 42
INITIAL_SEED = 43
PROGRESS_EVERY = 25

# One step per dimension: (1, 2, 5, 10, 20, 50, 100).
STEREOGRAPHIC_STEP_SIZES = [8, 5.66, 3.58, 2.53, 1.79, 1.13, 0.8]
SCS11_STEP_SIZES = [16, 11.3, 3.58, 2.53, 0.488, 0.13, 0.0707]
SCS19_STEP_SIZES = [32, 5.66, 3.58, 2.53, 1.79, 1.13, 0.8]
DCS_STEP_SIZES = [4.36, 2.01, 0.976, 0.512, 0.278, 0.122, 0.0622]
EUCLIDEAN_STEP_SIZES = [7.96, 3.08, 1.38, 0.851, 0.571, 0.351, 0.241]
# Gaussian standard deviations in B(1), tuned for mean acceptance through meeting.
DCS_RANDOM_WALK_STEP_SIZES = [9.19687, 2.16524, 0.5456, 0.172029, 0.0621627, 0.0168939, 0.00622]
STEP_SIZES = {
    "stereographic": STEREOGRAPHIC_STEP_SIZES,
    "SCS11": SCS11_STEP_SIZES,
    "SCS19": SCS19_STEP_SIZES,
    "dcs": DCS_STEP_SIZES,
    "euclidean": EUCLIDEAN_STEP_SIZES,
    "dcs_rw": DCS_RANDOM_WALK_STEP_SIZES,
}
# Preserve each completed method's proposal stream independently of ordering.
PROPOSAL_STREAMS = {
    "stereographic": (SEED, 0), "SCS11": (SEED, 1), "SCS19": (SEED, 2),
    "dcs": (SEED, 3), "euclidean": (SEED, 4),
    "dcs_rw": (SEED, 5),
}


def run_dimension(index, d, starts, parameters, seeds):
    target = make_log_density(d)
    stereo, scs11, scs19, dcs = (parameters[name] for name in ("stereographic", "SCS11", "SCS19", "dcs"))
    return combine_results(
        run_stereographic_experiment(
            target, starts, STEREOGRAPHIC_STEP_SIZES[index], MAX_STEPS,
            mu=stereo["mu"], preconditioning_matrix=stereo["preconditioning_matrix"],
            seed=seeds["stereographic"], seed_index=PROPOSAL_STREAMS["stereographic"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_scs_experiment(
            target, starts, SCS11_STEP_SIZES[index], MAX_STEPS, mu=scs11["mu"],
            preconditioning_matrix=scs11["preconditioning_matrix"],
            horizontal_observer=scs11["horizontal_observer"], latitude=1.1, label="SCS11",
            seed=seeds["SCS11"], seed_index=PROPOSAL_STREAMS["SCS11"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_scs_experiment(
            target, starts, SCS19_STEP_SIZES[index], MAX_STEPS, mu=scs19["mu"],
            preconditioning_matrix=scs19["preconditioning_matrix"],
            horizontal_observer=scs19["horizontal_observer"], latitude=1.9, label="SCS19",
            seed=seeds["SCS19"], seed_index=PROPOSAL_STREAMS["SCS19"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_dcs_matrix_ball_experiment(
            target, starts, DCS_STEP_SIZES[index], MAX_STEPS,
            mu=dcs["mu"], preconditioning_matrix=dcs["preconditioning_matrix"],
            seed=seeds["dcs"], seed_index=PROPOSAL_STREAMS["dcs"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_euclidean_experiment(
            target, starts, EUCLIDEAN_STEP_SIZES[index], MAX_STEPS,
            seed=seeds["euclidean"], seed_index=PROPOSAL_STREAMS["euclidean"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_dcs_matrix_random_walk_experiment(
            target, starts, DCS_RANDOM_WALK_STEP_SIZES[index], MAX_STEPS,
            mu=dcs["mu"], preconditioning_matrix=dcs["preconditioning_matrix"],
            seed=seeds["dcs_rw"], seed_index=PROPOSAL_STREAMS["dcs_rw"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
    )


def main():
    if any(len(values) != len(DIMENSIONS) for values in STEP_SIZES.values()):
        raise ValueError("Each step-size list needs one entry per dimension.")
    all_rows = []
    for index, d in enumerate(DIMENSIONS):
        folder = RESULTS / f"d_{d:04d}"
        parameters = load_parameters(folder / "fitted_parameters.json", d, ['stereographic', 'dcs'])
        parameters.update(load_parameters(folder / "fitted_parameters.json", d, ['SCS11'], latitude=1.1))
        parameters.update(load_parameters(folder / "fitted_parameters.json", d, ['SCS19'], latitude=1.9))
        if "preconditioning_matrix" not in parameters["dcs"] or "delta" in parameters["dcs"]:
            raise ValueError("DCS requires a matrix map without a Mobius offset.")
        rng = np.random.default_rng(np.random.SeedSequence([INITIAL_SEED, d]))
        starts = rng.standard_normal(size=(N_REP, 2, d))
        # One shared chi-square scale per vector gives a multivariate Student-t.
        starts /= np.sqrt(rng.chisquare(d+1, size=(N_REP, 2, 1)) / (d+1))
        seeds = {name: base+d + (1_000_000 if base+d == INITIAL_SEED else 0)
                 for name, (base, _) in PROPOSAL_STREAMS.items()}
        rows = run_dimension(index, d, starts, parameters, seeds)
        dimension_rows = [dict(dimension=d, df=d+1, cutoff=MAX_STEPS,
                             h=STEP_SIZES[row["method"]][index],
                             scs_latitude={"SCS11": 1.1, "SCS19": 1.9}.get(row["method"], np.nan),
                             initialization="target", **row) for row in rows]
        save_results(dimension_rows, folder / "meeting_times.csv")
        all_rows.extend(dimension_rows)
        pd.DataFrame(all_rows).to_csv(RESULTS / "meeting_times.csv", index=False)


if __name__ == "__main__":
    main()
