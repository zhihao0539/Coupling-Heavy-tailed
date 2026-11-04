"""Run the skew-t couplings using each method's explicit experiment function.

Fit the maps with run_vi.py and generate the shared pairs with
generate_initial_states.py before running this script.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import (
    run_stereographic_experiment, run_euclidean_experiment, run_scs_experiment,
    run_dcs_ball_experiment, run_dcs_random_walk_experiment,
    load_parameters, combine_results, save_results,
)
from generic_algorithms.plot_results import plot_results
from skew_t.target import D, log_density
from skew_t.run_vi import LATITUDE, PARAMETERS_FILE
from skew_t.generate_initial_states import INITIAL_STATES_FILE

N_REP = 1000
MAX_STEPS = 1_000_000
SEED = 42
PROGRESS_EVERY = 10
STEREOGRAPHIC_H = 10.0          # Tangent-Gaussian scale on the sphere.
EUCLIDEAN_H = 0.35             # Euclidean Gaussian standard deviation.
SCS_H = 2.0                    # Tangent-Gaussian scale on the bright cap.
DCS_H = 0.2                    # Ball-walk radius in unit-ball coordinates.
DCS_RANDOM_WALK_H = 0.02         # Gaussian standard deviation in unit-ball coordinates.
OUTPUT = FOLDER / 'meeting_times.csv'


def load_initial_pairs(path, n_pairs):
    """Load the common Euclidean starting pairs for every method."""
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
    return run_stereographic_experiment(
        log_density, starts, STEREOGRAPHIC_H, MAX_STEPS,
        mu=fitted['mu'], preconditioning_matrix=fitted['preconditioning_matrix'],
        seed=SEED, seed_offset=100_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_euclidean(starts):
    return run_euclidean_experiment(
        log_density, starts, EUCLIDEAN_H, MAX_STEPS,
        seed=SEED, seed_offset=200_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_scs(starts):
    fitted = load_parameters(PARAMETERS_FILE, D, ['scs'], latitude=LATITUDE)['scs']
    return run_scs_experiment(
        log_density, starts, SCS_H, MAX_STEPS,
        mu=fitted['mu'], preconditioning_matrix=fitted['preconditioning_matrix'],
        horizontal_observer=fitted['horizontal_observer'], latitude=LATITUDE,
        seed=SEED, seed_offset=300_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_dcs_ball(starts):
    fitted = load_parameters(PARAMETERS_FILE, D, ['dcs'])['dcs']
    return run_dcs_ball_experiment(
        log_density, starts, DCS_H, MAX_STEPS,
        mu=fitted['mu'], radius=fitted['radius'], delta=fitted['delta'],
        seed=SEED, seed_offset=400_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def run_dcs_random_walk(starts):
    """Run Gaussian DCS with the configured standard deviation DCS_RANDOM_WALK_H."""
    fitted = load_parameters(PARAMETERS_FILE, D, ['dcs'])['dcs']
    return run_dcs_random_walk_experiment(
        log_density, starts, DCS_RANDOM_WALK_H, MAX_STEPS,
        mu=fitted['mu'], radius=fitted['radius'], delta=fitted['delta'],
        seed=SEED, seed_offset=500_000,
        stop_at_meeting=True, progress_every=PROGRESS_EVERY)


def main():
    starts = load_initial_pairs(INITIAL_STATES_FILE, N_REP)
    rows = combine_results(
        run_stereographic(starts),
        run_euclidean(starts),
        run_scs(starts),
        run_dcs_ball(starts),
        run_dcs_random_walk(starts),
    )
    save_results(rows, OUTPUT)
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'),
                 log_scale=True, horizon=MAX_STEPS)


if __name__ == '__main__':
    main()
