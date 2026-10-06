"""Set the parameters below, then run to simulate and plot."""
from pathlib import Path
import numpy as np
import pandas as pd
from algorithms import (load_parameters, to_sphere, scs_inverse,
                        dcs_inverse, coupled_sampler)
from plot_results import plot_results

# Experiment settings. Try a small N_REP for a quick run.
N_REP = 1000
MAX_STEPS = 1000000
SEED = 42
STEPS = {"stereographic": 14.0, "euclidean": 0.26, "scs": 0.08, "dcs_bw": 0.14}
FOLDER = Path(__file__).resolve().parent

def run_experiment(n_rep=N_REP, max_steps=MAX_STEPS):
    parameters = load_parameters()
    starts = np.load(FOLDER / "initial_states.npz")["initial_states"]
    if not 1 <= n_rep <= len(starts):
        raise ValueError("Choose between 1 and 1,000 saved initial pairs.")
    rows = []
    for rep, (x, y) in enumerate(starts[:n_rep]):
        for index, method in enumerate(STEPS):
            if method == "scs":
                a, b = scs_inverse(x, parameters["scs"]), scs_inverse(y, parameters["scs"])
            elif method == "dcs_bw":
                a, b = dcs_inverse(x, parameters["dcs"]), dcs_inverse(y, parameters["dcs"])
            elif method == "stereographic":
                a, b = to_sphere(x), to_sphere(y)
            else:
                a, b = x, y
            rng = np.random.default_rng(SEED + rep + (index + 1) * 100000)
            tau, iterations, acc1, acc2 = coupled_sampler(a, b, method, parameters, max_steps, rng, STEPS[method])
            rows.append(dict(method=method, replicate=rep, met=np.isfinite(tau),
                             meeting_time=tau if np.isfinite(tau) else -1,
                             iterations=iterations, acceptance_rate_chain1=acc1,
                             acceptance_rate_chain2=acc2, cutoff=max_steps))
        if (rep + 1) % 10 == 0:
            print(f"Finished {rep + 1}/{n_rep} repetitions", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_experiment()
    results.to_csv(FOLDER / "new_meeting_times.csv", index=False)
    plot_results(results, FOLDER / "new_meeting_times.pdf")
