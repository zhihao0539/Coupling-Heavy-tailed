"""Step 2 of the skew-t experiment: run the couplings with the saved VI fit.

The committed fitted_parameters.json is ready to use. Run run_vi.py only
to refit the SCS and DCS maps. Coupling settings are below.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters, run_experiment, load_parameters
from generic_algorithms.plot_results import plot_results
from skew_t.target import D, log_density
from skew_t.run_vi import LATITUDE, PARAMETERS_FILE

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
SEED_OFFSETS = {'stereographic': 100_000, 'euclidean': 200_000,
                'scs': 300_000, 'dcs': 400_000}
OUTPUT = FOLDER / 'new_meeting_times.csv'


def main():
    starts = np.load(FOLDER / 'initial_states.npz')['initial_states']
    if not 1 <= N_REP <= len(starts):
        raise ValueError(f'N_REP must be between 1 and {len(starts)} saved initial pairs.')

    # SCS and DCS maps fitted by run_vi.py; the stereographic map is fixed.
    fitted_methods = [method for method in STEP_SIZES if method in ('scs', 'dcs')]
    parameters = load_parameters(PARAMETERS_FILE, D, fitted_methods, latitude=LATITUDE,
                                 legacy_scs_latitude=1.1) if fitted_methods else {}
    parameters['stereographic'] = default_parameters(D, 'stereographic', radius=STEREOGRAPHIC_RADIUS)

    # Couple every method from the same saved initial pairs, stopping at meeting.
    methods = {}
    for method, h in STEP_SIZES.items():
        methods[method] = {'h': h, 'parameters': parameters.get(method),
                           'seed_offset': SEED_OFFSETS[method]}
        if method == 'scs':
            methods[method]['latitude'] = LATITUDE
    rows = run_experiment(log_density, starts[:N_REP], methods, MAX_STEPS, seed=SEED,
                          stop_at_meeting=True, output_csv=OUTPUT, progress_every=10)
    plot_results(pd.DataFrame(rows), OUTPUT.with_suffix('.pdf'), log_scale=True, horizon=MAX_STEPS)


if __name__ == '__main__':
    main()
