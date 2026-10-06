"""Set the parameters below, then run to simulate and plot."""
from pathlib import Path
import numpy as np
import pandas as pd
from algorithms import (load_data, to_sphere, log_density,
                        sphere_log_density, coupled_sampler)
from plot_results import plot_results

# Experiment settings. Try a small N_REP for a quick run.
N_REP = 1000
N_STEPS = 20000
H_SP = 0.007
H_EU = 0.03
SEED = 42
FOLDER = Path(__file__).resolve().parent

def run_experiment(n_rep=N_REP, n_steps=N_STEPS):
    X, y = load_data()
    d = X.shape[1] + 1
    R = np.sqrt(d)
    starts = np.random.RandomState(SEED).normal(0, 10, (n_rep, 2, d))
    rows = []
    for rep, (x, y0) in enumerate(starts):
        for method, h in [("stereographic", H_SP), ("euclidean", H_EU)]:
            sphere = method == "stereographic"
            if sphere:
                a, b = to_sphere(x, R), to_sphere(y0, R)
                target = lambda z: sphere_log_density(z, X, y, R)
            else:
                a, b = x, y0
                target = lambda theta: log_density(theta, X, y)
            rng = np.random.default_rng(SEED + rep + (100000 if sphere else 200000))
            tau, acc1, acc2 = coupled_sampler(a, b, target, h, n_steps, rng, sphere, R)
            rows.append(dict(method=method, replicate=rep, met=np.isfinite(tau),
                             meeting_time=tau if np.isfinite(tau) else -1,
                             iterations=n_steps, acceptance_rate_chain1=acc1,
                             acceptance_rate_chain2=acc2))
        if (rep + 1) % 50 == 0:
            print(f"Finished {rep + 1}/{n_rep} repetitions", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_experiment()
    results.to_csv(FOLDER / "new_meeting_times.csv", index=False)
    plot_results(results, FOLDER / "new_meeting_times.pdf")
