"""Fixed-parameter regression experiment using the shared coupling functions."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
import sys
import time

# One numerical thread per worker avoids oversubscribing the CPU.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # Also allow: python student_t_regression/run_experiment.py
from generic_algorithms import coupled_sampler, default_parameters
from student_t_regression.target import load_data, log_density
from student_t_regression.plot_results import plot_results

N_REP = 1000
N_STEPS = 20000
H_SP = .007
H_EU = .03
SEED = 42
WORKERS = min(8, os.cpu_count() or 1)   # Set to 1 for a serial run.


def run_repetition(rep, initial_pair, X, y, n_steps):
    d = X.shape[1] + 1
    log_pdf = lambda theta: log_density(theta, X, y)
    rows = []
    for method, h, seed_offset in [('stereographic', H_SP, 100000),
                                    ('euclidean', H_EU, 200000)]:
        parameters = default_parameters(d, method, radius=np.sqrt(d))
        result = coupled_sampler(
            *initial_pair, log_pdf, h, n_steps,
            rng=np.random.default_rng(SEED + rep + seed_offset),
            method=method, parameters=parameters, stop_at_meeting=False,
        )
        tau = result['meeting_time']
        rows.append(dict(method=method, replicate=rep, met=np.isfinite(tau),
                         meeting_time=int(tau) if np.isfinite(tau) else -1,
                         iterations=result['iterations'],
                         acceptance_rate_chain1=result['acceptance_rate_chain1'],
                         acceptance_rate_chain2=result['acceptance_rate_chain2']))
    return rows


def run_experiment(n_rep=N_REP, n_steps=N_STEPS, workers=WORKERS, save_to=None):
    """Run independent repetitions. Worker count does not affect random seeds."""
    if any(not isinstance(n, int) or n < 1 for n in (n_rep, n_steps, workers)):
        raise ValueError('n_rep, n_steps and workers must be positive integers.')
    X, y = load_data()
    d = X.shape[1] + 1
    starts = np.random.RandomState(SEED).normal(0, 10, (n_rep, 2, d))
    arguments = (range(n_rep), starts, repeat(X), repeat(y), repeat(n_steps))
    rows, start = [], time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        repetitions = pool.map(run_repetition, *arguments) if pool else map(run_repetition, *arguments)
        for completed, pair_rows in enumerate(repetitions, 1):
            rows.extend(pair_rows)
            if completed % 25 == 0 or completed == n_rep:
                if save_to is not None:
                    pd.DataFrame(rows).to_csv(save_to, index=False)
                print(f'Finished {completed}/{n_rep} repetitions in {time.perf_counter() - start:.1f}s', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    output = FOLDER / 'new_meeting_times.csv'
    results = run_experiment(save_to=output)
    results.to_csv(output, index=False)
    plot_results(results, output.with_suffix('.pdf'))
