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

from generic_algorithms import run_experiment, load_parameters
from student_t_regression.nba_salary.target import D, log_density
from student_t_regression.nba_salary.run_vi import PARAMETERS_FILE

N_REP = 1000
MAX_STEPS = 150
SEED = 42
STEP_SIZE = 0.1                # Tangent-Gaussian scale on the sphere.
INITIAL_STATES_FILE = FOLDER / 'initial_states.npz'
OUTPUT = FOLDER / 'meeting_times.csv'


def main():
    with np.load(INITIAL_STATES_FILE, allow_pickle=False) as saved:
        starts = saved['initial_states']
    if starts.ndim != 3 or starts.shape[1:] != (2, D):
        raise ValueError(f'Saved initial pairs must have shape (n, 2, {D}).')
    if not 1 <= N_REP <= len(starts):
        raise ValueError(f'N_REP must be between 1 and {len(starts)} saved initial pairs.')
    if not np.isfinite(starts[:N_REP]).all():
        raise ValueError('Saved initial pairs must be finite.')

    parameters = load_parameters(PARAMETERS_FILE, D, ['stereographic'])
    if 'preconditioning_matrix' not in parameters['stereographic']:
        raise ValueError('Run student_t_regression/nba_salary/run_vi.py to fit the full-matrix stereographic map first.')
    methods = {
        'stereographic': {'h': STEP_SIZE,
                         'parameters': parameters['stereographic'],
                         'seed_offset': 100_000},
    }
    rows = run_experiment(log_density, starts[:N_REP], methods, MAX_STEPS,
                          seed=SEED, stop_at_meeting=True, output_csv=OUTPUT,
                          progress_every=10)

    from generic_algorithms.plot_results import plot_results
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'),
                 log_scale=False, horizon=MAX_STEPS)


if __name__ == '__main__':
    main()
