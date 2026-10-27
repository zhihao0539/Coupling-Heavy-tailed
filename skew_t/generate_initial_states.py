"""Step 2: project uniform bright-cap points through the fitted SCS map."""
from pathlib import Path
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import from_native, load_parameters
from skew_t.target import D
from skew_t.run_vi import LATITUDE, PARAMETERS_FILE

N_INITIAL_PAIRS = 1000
INITIAL_SEED = 43
INITIAL_STATES_FILE = FOLDER / 'initial_states.npz'


def generate_initial_pairs(parameters, n_pairs=N_INITIAL_PAIRS, seed=INITIAL_SEED):
    """Independent uniform cap draws, mapped to Euclidean states with fitted SCS."""
    if not isinstance(n_pairs, (int, np.integer)) or n_pairs < 1:
        raise ValueError('n_pairs must be a positive integer.')
    rng = np.random.default_rng(seed)
    points = np.empty((2 * n_pairs, D + 1))
    filled = 0
    while filled < len(points):
        sphere = rng.normal(size=(2 * (len(points) - filled), D + 1))
        sphere /= np.linalg.norm(sphere, axis=-1, keepdims=True)
        bright = sphere[sphere[:, -1] < LATITUDE - 1]
        count = min(len(bright), len(points) - filled)
        points[filled:filled + count] = bright[:count]
        filled += count
    starts = from_native(points, 'scs', parameters, latitude=LATITUDE)
    if not np.isfinite(starts).all():
        raise FloatingPointError('The fitted SCS map produced non-finite initial states.')
    return starts.reshape(n_pairs, 2, D)


def main():
    parameters = load_parameters(PARAMETERS_FILE, D, ['scs'], latitude=LATITUDE)['scs']
    if 'preconditioning_matrix' not in parameters:
        raise ValueError('Run skew_t/run_vi.py to fit the matrix SCS map first.')
    starts = generate_initial_pairs(parameters)
    np.savez_compressed(INITIAL_STATES_FILE, initial_states=starts)
    print(f'Saved {len(starts)} initial pairs to {INITIAL_STATES_FILE}')


if __name__ == '__main__':
    main()
