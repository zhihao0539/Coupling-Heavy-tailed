"""Compare VI-tuned stereographic and matrix DCS ball-walk couplings.

Both maps use the same saved initial pairs and iteration cutoff, with separate
proposal step sizes.
Loads the saved VI fits and uses each method's configured proposal step.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent.parent))

from generic_algorithms import (
    run_stereographic_experiment, run_dcs_matrix_ball_experiment,
    load_parameters, combine_results, save_results,
)
from generic_algorithms.plot_results import plot_results
from student_t_regression.nba_salary.target import D, log_density
from student_t_regression.nba_salary.run_vi import PARAMETERS_FILE

N_REP = 1000
MAX_STEPS = 2000000
SEED = 42
PROGRESS_EVERY = 10
STEREOGRAPHIC_H = 0.1          # VI-tuned map's tangent-Gaussian scale.
DCS_H = 0.049                 # Unit-ball radius; calibrated toward 23% acceptance.
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
    """Run the existing coupling with the saved VI-fitted location and matrix."""
    fitted = load_parameters(PARAMETERS_FILE, D, ['stereographic'])['stereographic']
    if 'preconditioning_matrix' not in fitted:
        raise ValueError('Run student_t_regression/nba_salary/run_vi.py to fit the full-matrix stereographic map first.')
    return run_stereographic_experiment(
        log_density, starts, STEREOGRAPHIC_H, MAX_STEPS,
        mu=fitted['mu'], preconditioning_matrix=fitted['preconditioning_matrix'],
        seed=SEED, seed_offset=100_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_dcs(starts):
    """Run matrix DCS ball walk with the saved fitted location, matrix, and beta."""
    fitted = load_parameters(PARAMETERS_FILE, D, ['dcs'])['dcs']
    return run_dcs_matrix_ball_experiment(
        log_density, starts, DCS_H, MAX_STEPS,
        mu=fitted['mu'], preconditioning_matrix=fitted['preconditioning_matrix'],
        beta=fitted['beta'], seed=SEED, seed_offset=400_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def plot_comparison(rows):
    """Save log-scale PDF/PNG plots, including pairs that reach the cutoff."""
    for suffix in ('.pdf', '.png'):
        plot_results(pd.DataFrame(rows), OUTPUT.with_suffix(suffix),
            log_scale=True, horizon=MAX_STEPS,
            styles={
                'stereographic': ('Stereographic', 'tab:blue'),
                'dcs': ('DCS Ball walk', 'tab:purple'),
            })


def main():
    starts = load_initial_pairs(INITIAL_STATES_FILE, N_REP)
    rows = combine_results(
        run_stereographic(starts),
        run_dcs(starts),
    )
    save_results(rows, OUTPUT)
    plot_comparison(rows)


if __name__ == '__main__':
    main()
