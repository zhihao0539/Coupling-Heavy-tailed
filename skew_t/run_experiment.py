"""Skew-t comparison: fit the SCS and DCS maps by VI, then run the couplings.

VI settings are at the top of run_vi.py; coupling settings are below.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters, run_experiment
from generic_algorithms.plot_results import plot_results
from skew_t.target import D, log_density
from skew_t.run_vi import LATITUDE, fit_parameters, save_parameters

# Coupling settings. Try a small N_REP and MAX_STEPS first.
N_REP = 1000                    # Number of saved initial pairs to use (at most 1,000).
MAX_STEPS = 1_000_000           # Iteration cutoff; pairs unmet by then are censored.
SEED = 42
STEREOGRAPHIC_RADIUS = 10       # Fixed stereographic map: zero location, this radius.
STEP_SIZES = {                  # Proposal h for each method, in this order.
    'stereographic': 14.0,      # Tangent-Gaussian scale on the sphere.
    'euclidean': 0.26,          # Gaussian random-walk standard deviation.
    'scs': 0.08,                # Tangent-Gaussian scale on the sphere.
    'dcs': 0.14,                # Ball-walk radius in unit-ball coordinates (gamma/R).
}
OUTPUT = FOLDER / 'new_meeting_times.csv'


def main():
    starts = np.load(FOLDER / 'initial_states.npz')['initial_states']
    if not 1 <= N_REP <= len(starts):
        raise ValueError(f'N_REP must be between 1 and {len(starts)} saved initial pairs.')

    # 1. Tune the SCS and DCS maps by VI and keep the fit beside the results.
    parameters, info = fit_parameters()
    save_parameters(parameters, info, FOLDER / 'new_fitted_parameters.json')
    parameters['stereographic'] = default_parameters(D, 'stereographic', radius=STEREOGRAPHIC_RADIUS)

    # 2. Couple every method from the same saved initial pairs, stopping at meeting.
    methods = {}
    for index, (method, h) in enumerate(STEP_SIZES.items()):
        methods[method] = {'h': h, 'parameters': parameters.get(method),
                           'seed_offset': 100_000 * (index + 1)}  # Previous random streams.
        if method == 'scs':
            methods[method]['latitude'] = LATITUDE
    rows = run_experiment(log_density, starts[:N_REP], methods, MAX_STEPS, seed=SEED,
                          stop_at_meeting=True, output_csv=OUTPUT, progress_every=10)
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'), log_scale=True, horizon=MAX_STEPS)


if __name__ == '__main__':
    main()
