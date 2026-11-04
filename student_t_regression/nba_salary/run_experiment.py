"""Run NBA stereographic couplings sequentially with the saved VI fit.

Loads the supplied fitted map and saved initial pairs. Coupling settings
are below; run run_vi.py separately to refit the map.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent.parent))

from generic_algorithms import run_stereographic_experiment, load_parameters, save_results
from generic_algorithms.plot_results import plot_results
from student_t_regression.nba_salary.target import D, log_density
from student_t_regression.nba_salary.run_vi import PARAMETERS_FILE

N_REP = 1000
MAX_STEPS = 150
SEED = 42
PROGRESS_EVERY = 10
STEREOGRAPHIC_H = 0.1          # Tangent-Gaussian scale on the sphere.
INITIAL_STATES_FILE = FOLDER / 'initial_states.npz'
OUTPUT = FOLDER / 'meeting_times.csv'


def load_initial_pairs(path, n_pairs):
    """Load and check the Euclidean starting pairs used by the experiment."""
    with np.load(path, allow_pickle=False) as saved:
        starts = saved['initial_states']
    if starts.ndim != 3 or starts.shape[1:] != (2, D):
        raise ValueError(f'Saved initial pairs must have shape (n, 2, {D}).')
    if not isinstance(n_pairs, (int, np.integer)) or not 1 <= n_pairs <= len(starts):
        raise ValueError(f'n_pairs must be between 1 and {len(starts)} saved initial pairs.')
    starts = starts[:n_pairs]
    if not np.isfinite(starts).all():
        raise ValueError('Saved initial pairs must be finite.')
    return starts


def run_stereographic(starts):
    fitted = load_parameters(PARAMETERS_FILE, D, ['stereographic'])['stereographic']
    if 'preconditioning_matrix' not in fitted:
        raise ValueError('Run student_t_regression/nba_salary/run_vi.py to fit the full-matrix stereographic map first.')
    return run_stereographic_experiment(
        log_density, starts, STEREOGRAPHIC_H, MAX_STEPS,
        mu=fitted['mu'], preconditioning_matrix=fitted['preconditioning_matrix'],
        seed=SEED, seed_offset=100_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def main():
    starts = load_initial_pairs(INITIAL_STATES_FILE, N_REP)
    rows = run_stereographic(starts)
    save_results(rows, OUTPUT)
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'),
                 log_scale=False, horizon=MAX_STEPS)


if __name__ == '__main__':
    main()
