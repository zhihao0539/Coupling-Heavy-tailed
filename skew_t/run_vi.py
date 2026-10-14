"""Tune the skew-t SCS and DCS maps with the shared VI function.

run_experiment.py calls fit_parameters() with these settings before coupling.
Run this file alone to fit the maps without running the couplings.
"""
import json
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters
from skew_t.target import D, NU, log_density_jax

# VI settings shared by both map families.
LATITUDE = 1.1                  # Fixed SCS cap latitude; sampling uses the same value.
STEPS = 4000                    # Adam updates.
BATCH_SIZE = 2048               # Fresh reference samples per update.
HOLDOUT_SIZE = 20000            # Samples comparing the last iterate with the average.
AVERAGE_LAST = 500              # Final unconstrained iterates averaged for one candidate.
LEARNING_RATES = (0.01, 0.002)  # Before and after CHANGE_AFTER updates.
CHANGE_AFTER = 3000
# Per-family seed, initial radius (zero location and offset), and JAX scan
# chunk; DCS keeps its original single-scan layout. Remove an entry to skip it.
METHODS = {
    'scs': {'seed': 26093001, 'radius': 10 / 11, 'chunk_size': 250},
    'dcs': {'seed': 26092901, 'radius': np.sqrt(NU), 'chunk_size': STEPS},
}
OUTPUT = FOLDER / 'new_fitted_parameters.json'


def fit_parameters():
    """Fit every map in METHODS. Return {method: parameters}, {method: VI info}."""
    from generic_algorithms.vi import fit_vi  # Loads JAX in 64-bit mode.

    parameters, info = {}, {}
    for method, settings in METHODS.items():
        print(f'Fitting the {method.upper()} map by VI', flush=True)
        parameters[method], info[method], _ = fit_vi(
            log_density_jax, D, method,
            initial_parameters=default_parameters(D, method, radius=settings['radius']),
            latitude=LATITUDE, seed=settings['seed'], steps=STEPS,
            batch_size=BATCH_SIZE, holdout_size=HOLDOUT_SIZE,
            average_last=AVERAGE_LAST, learning_rates=LEARNING_RATES,
            change_after=CHANGE_AFTER, chunk_size=settings['chunk_size'],
        )
        info[method].update(df=NU, initial_radius=float(settings['radius']))
        print(f"  radius {parameters[method]['radius']:.6f}, chosen {info[method]['chosen_candidate']}, "
              f"holdout losses {info[method]['candidate_holdout_losses']}", flush=True)
    return parameters, info


def save_parameters(parameters, info, path=OUTPUT):
    """Save fitted maps in the layout of fitted_parameters.json, plus VI details."""
    record = {method: {name: float(value) if name == 'radius' else np.asarray(value).tolist()
                       for name, value in values.items()}
              for method, values in parameters.items()}
    record['vi_info'] = info
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Saved {path.name}', flush=True)


if __name__ == '__main__':
    save_parameters(*fit_parameters())
