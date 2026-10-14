"""Step 1 of the skew-t experiment: tune the SCS and DCS maps by VI and save them.

Run this once; run_experiment.py then loads the saved maps from
PARAMETERS_FILE, so the couplings never refit VI.
"""
import json
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import default_parameters
from generic_algorithms.transformations import validate_parameters
from skew_t.target import D, NU, log_density_jax

PARAMETERS_FILE = FOLDER / 'fitted_parameters.json'  # Written here, read by run_experiment.py.

# VI settings shared by both map families.
LATITUDE = 1.1                  # Fixed SCS cap latitude; sampling uses the same value.
STEPS = 4000                    # Adam updates.
BATCH_SIZE = 2048               # Fresh reference samples per update.
HOLDOUT_SIZE = 20000            # Samples comparing the last iterate with the average.
AVERAGE_LAST = 500              # Final unconstrained iterates averaged for one candidate.
LEARNING_RATES = (0.01, 0.002)  # Before and after CHANGE_AFTER updates.
CHANGE_AFTER = 3000
# Per-family seed, initial radius (zero location and offset), and JAX scan
# chunk; DCS keeps its original single-scan layout. Remove an entry to refit
# only the other map; the saved fit of the removed map is kept.
METHODS = {
    'scs': {'seed': 26093001, 'radius': 10 / 11, 'chunk_size': 250},
    'dcs': {'seed': 26092901, 'radius': np.sqrt(NU), 'chunk_size': STEPS},
}


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


def save_parameters(parameters, info, path=PARAMETERS_FILE):
    """Write the fitted maps (and their VI details) into path.

    Maps already in the file but not refitted are kept unchanged.
    """
    path = Path(path)
    record = json.loads(path.read_text()) if path.exists() else {}
    vi_info = record.pop('vi_info', {})
    for method, values in parameters.items():
        record[method] = {name: float(value) if name == 'radius' else np.asarray(value).tolist()
                          for name, value in values.items()}
        vi_info[method] = info[method]
    record['vi_info'] = vi_info
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Saved the {', '.join(m.upper() for m in parameters)} fit to {path.name}", flush=True)


def load_parameters(path=PARAMETERS_FILE, methods=('scs', 'dcs')):
    """Read and validate the requested maps; the original paper fit used latitude 1.1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; run skew_t/run_vi.py first.')
    record = json.loads(path.read_text())
    parameters = {}
    for method in methods:
        if method not in record:
            raise ValueError(f'{path.name} has no {method.upper()} fit; run skew_t/run_vi.py first.')
        info = record.get('vi_info', {}).get(method, {})
        saved_latitude = info.get('latitude', 1.1)  # Original committed fit has no vi_info.
        if method == 'scs' and not np.isclose(saved_latitude, LATITUDE, rtol=0, atol=1e-12):
            raise ValueError(f'The saved SCS map was fitted with latitude {saved_latitude}, '
                             f'but LATITUDE is {LATITUDE}; rerun skew_t/run_vi.py.')
        parameters[method] = {name: float(value) if name == 'radius' else np.asarray(value, dtype=float)
                              for name, value in record[method].items()}
        validate_parameters(parameters[method], D, method, LATITUDE)
    return parameters


if __name__ == '__main__':
    save_parameters(*fit_parameters())
