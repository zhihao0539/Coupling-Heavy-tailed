"""Step 3: run couplings with the fitted maps and SCS bright-cap starts.

Run run_vi.py and generate_initial_states.py first. Coupling settings are below.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import run_experiment, load_parameters
from generic_algorithms.plot_results import plot_results
from skew_t.target import D, log_density
from skew_t.run_vi import LATITUDE, PARAMETERS_FILE
from skew_t.generate_initial_states import INITIAL_STATES_FILE

# Coupling settings. Try a small N_REP and MAX_STEPS first.
N_REP = 1000                    # Number of saved initial pairs to use (at most 1,000).
MAX_STEPS = 1_000_000           # Iteration cutoff; pairs unmet by then are censored.
SEED = 42
STEP_SIZES = {                  # Proposal h for each method, in this order.
    'stereographic': 10.0,      # Tangent-Gaussian scale on the sphere.
    'euclidean': 0.35,          # Gaussian random-walk standard deviation.
    'scs': 2.0,               # Tangent-Gaussian scale on the sphere.
    'dcs': 0.20,                # Ball-walk radius in unit-ball coordinates (gamma/R).
}
SEED_OFFSETS = {'stereographic': 100_000, 'euclidean': 200_000,
                'scs': 300_000, 'dcs': 400_000}
OUTPUT = FOLDER / 'meeting_times.csv'


def main():
    with np.load(INITIAL_STATES_FILE, allow_pickle=False) as saved:
        starts = saved['initial_states']
    if starts.ndim != 3 or starts.shape[1:] != (2, D):
        raise ValueError(f'Saved initial pairs must have shape (n, 2, {D}).')
    if not 1 <= N_REP <= len(starts):
        raise ValueError(f'N_REP must be between 1 and {len(starts)} saved initial pairs.')

    # Every transformed method uses the corresponding saved VI fit.
    fitted_methods = [method for method in STEP_SIZES if method != 'euclidean']
    parameters = load_parameters(PARAMETERS_FILE, D, fitted_methods,
                                 latitude=LATITUDE) if fitted_methods else {}

    # Couple every method from the same saved initial pairs, stopping at meeting.
    methods = {}
    for method, h in STEP_SIZES.items():
        methods[method] = {'h': h, 'parameters': parameters.get(method),
                           'seed_offset': SEED_OFFSETS[method]}
        if method == 'scs':
            methods[method]['latitude'] = LATITUDE
    results = run_experiment(log_density, starts[:N_REP], methods, MAX_STEPS, seed=SEED,
                          stop_at_meeting=True, output_csv=OUTPUT)
    plot_results(results, OUTPUT.with_suffix('.pdf'), log_scale=True, horizon=MAX_STEPS)

    return results


if __name__ == '__main__':
    results_df = main()
