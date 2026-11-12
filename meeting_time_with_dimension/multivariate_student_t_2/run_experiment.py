"""Step 2: run each of the five coupling experiments explicitly."""
import numpy as np
import pandas as pd

from run_vi import RESULTS, DCS_BETAS
from target import DIMENSIONS, make_log_density
from generic_algorithms import (
    run_stereographic_experiment, run_scs_experiment, run_dcs_ball_experiment,
    run_euclidean_experiment, load_parameters, combine_results, save_results,
)

N_REP = 1000
MAX_STEPS = 100000
SEED = 42
INITIAL_SEED = 43
PROGRESS_EVERY = 25

# One step per dimension: (1, 2, 5, 10, 20, 50, 100).
# Calibrated for the completed fixed-mu VI fits and Mobius DCS with beta=2.
# The target is 23% mean acceptance through meeting; where it was not reached,
# a practical step was selected from the tested meeting-time pilots. See README.md.
STEREOGRAPHIC_STEP_SIZES = [
    2.0,  
    1.414213562373095,  
    0.4472135954999579,  
    0.15811388300841897,
    0.05590169943749474,
    0.035355339059327376,
    0.02,
]
SCS11_STEP_SIZES = [
    2.0,
    1.414213562373095,
    0.8944271909999159,
    0.6324555320336759,
    0.4472135954999579,
    0.565685424949238,
    0.4,
]
SCS19_STEP_SIZES = [
    2.0, 
    1.414213562373095,
    0.8944271909999159,
    0.6324555320336759,
    0.22360679774997896,
    0.07071067811865475,
    0.025,
]
DCS_STEP_SIZES = [
    11.473405197452863,
    3.417842536783372,
    1.588958816539991,
    0.9137731178245505,
    0.5766368429877494,
    0.3491127935113013,
    0.2432904401869078,
]
EUCLIDEAN_STEP_SIZES = [
    19.19564075931999,
    5.658013140352347,
    2.126319991672692,
    1.18889788375733,
    0.79,
    0.461915841021642, 
    0.3493027358627649,
]
STEP_SIZES = {
    "stereographic": STEREOGRAPHIC_STEP_SIZES,
    "SCS11": SCS11_STEP_SIZES,
    "SCS19": SCS19_STEP_SIZES,
    "dcs": DCS_STEP_SIZES,
    "euclidean": EUCLIDEAN_STEP_SIZES,
}
# Preserve each completed method's proposal stream independently of ordering.
PROPOSAL_STREAMS = {
    "stereographic": (42, 0), "SCS11": (42, 1), "SCS19": (42, 2),
    "dcs": (42, 3), "euclidean": (42, 4),
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
        run_dcs_ball_experiment(
            target, starts, DCS_STEP_SIZES[index], MAX_STEPS,
            mu=dcs["mu"], radius=dcs["radius"], delta=dcs["delta"], beta=dcs["beta"],
            seed=seeds["dcs"], seed_index=PROPOSAL_STREAMS["dcs"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
        run_euclidean_experiment(
            target, starts, EUCLIDEAN_STEP_SIZES[index], MAX_STEPS,
            seed=seeds["euclidean"], seed_index=PROPOSAL_STREAMS["euclidean"][1],
            stop_at_meeting=True, progress_every=PROGRESS_EVERY),
    )


def load_fitted_parameters(d):
    path = RESULTS / f"d_{d:04d}" / "fitted_parameters.json"
    parameters = load_parameters(path, d, ["stereographic", "dcs"], beta=DCS_BETAS[d])
    parameters.update(load_parameters(path, d, ["SCS11"], latitude=1.1))
    parameters.update(load_parameters(path, d, ["SCS19"], latitude=1.9))
    for label, fitted in parameters.items():
        if not np.array_equal(fitted["mu"], np.zeros(d)):
            raise ValueError(
                f"{label} must have mu=0. Run python run_vi.py to refit all maps first.")
    for label in ("SCS11", "SCS19"):
        if not np.array_equal(parameters[label]["horizontal_observer"], np.zeros(d)):
            raise ValueError(
                f"{label} must have horizontal_observer=0. Run python run_vi.py first.")
    if "radius" not in parameters["dcs"] or "delta" not in parameters["dcs"]:
        raise ValueError(
            "DCS requires a Mobius map. Run python run_vi.py first.")
    return parameters


def main():
    if any(len(values) != len(DIMENSIONS) for values in STEP_SIZES.values()):
        raise ValueError("Each step-size list needs one entry per dimension.")
    # Check all saved maps before starting a potentially long coupling run.
    fitted_maps = {d: load_fitted_parameters(d) for d in DIMENSIONS}
    all_rows = []
    for index, d in enumerate(DIMENSIONS):
        folder = RESULTS / f"d_{d:04d}"
        parameters = fitted_maps[d]
        rng = np.random.default_rng(np.random.SeedSequence([INITIAL_SEED, d]))
        starts = rng.standard_normal(size=(N_REP, 2, d))
        # One shared chi-square scale per vector gives a multivariate Student-t.
        starts /= np.sqrt(rng.chisquare(2, size=(N_REP, 2, 1)) / 2)
        seeds = {name: base+d + (1_000_000 if base+d == INITIAL_SEED else 0)
                 for name, (base, _) in PROPOSAL_STREAMS.items()}
        rows = run_dimension(index, d, starts, parameters, seeds)
        rows = [dict(dimension=d, df=2, cutoff=MAX_STEPS,
                     h=STEP_SIZES[row["method"]][index],
                     scs_latitude={"SCS11": 1.1, "SCS19": 1.9}.get(row["method"], np.nan),
                     mu_fixed=True,
                     scs_mu_fixed=row["method"] in ("SCS11", "SCS19"),
                     scs_observer_fixed=row["method"] in ("SCS11", "SCS19"),
                     dcs_map="mobius" if row["method"] == "dcs" else None,
                     initialization="target", **row) for row in rows]
        save_results(rows, folder / "meeting_times.csv")
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(RESULTS / "meeting_times.csv", index=False)


if __name__ == "__main__":
    main()
