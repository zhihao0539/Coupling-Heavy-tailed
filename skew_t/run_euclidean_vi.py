"""Standalone skew-t Gaussian VI and preconditioned RWMH comparison."""
from pathlib import Path
import json
import sys

import numpy as np

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

from generic_algorithms import run_euclidean_vi, run_euclidean_experiment, load_parameters, save_results
from generic_algorithms.transformations import affine_inverse
from skew_t.target import D, NU, ALPHA, log_density, log_density_jax

RESULTS = FOLDER / 'euclidean_vi'
PARAMETERS_FILE = RESULTS / 'fitted_parameters.json'
INITIAL_STATES_FILE = FOLDER / 'new_initial_states.npz'
N_REP = 1000
MAX_STEPS = 1_000_000
H = 0.40  # Gaussian standard deviation in VI-whitened coordinates.
SEED = 42
VI_SEED = 2026111801


def fit_euclidean():
    """Fit the location and full Cholesky factor using a Gaussian reference."""
    return run_euclidean_vi(
        log_density_jax, D, mu=np.zeros(D), preconditioning_matrix=np.eye(D),
        tune=('mu', 'preconditioning_matrix'), seed=VI_SEED,
        steps=6000, batch_size=2048, learning_rates=(.01, .002), change_after=4500,
        average_last=500, holdout_size=50000, chunk_size=250,
        output_json=PARAMETERS_FILE,
        target_info=dict(target='skew_t', dimension=D, df=NU, alpha=ALPHA,
                         objective='reverse_kl', initial_mu='zero', initial_matrix='identity'))


def summarize(rows):
    """Keep censored pairs separate from the mean among pairs that met."""
    times = np.array([row['meeting_time'] for row in rows])
    met = times >= 0
    rates = [(row['acceptance_rate_chain1'] + row['acceptance_rate_chain2']) / 2
             for row in rows]
    return dict(h=H, cutoff=MAX_STEPS, n_pairs=len(rows), n_met=int(met.sum()),
                n_censored=int((~met).sum()), mean_acceptance=float(np.mean(rates)),
                mean_meeting_time=float(times.mean()) if met.all() else None,
                mean_meeting_time_met=float(times[met].mean()) if met.any() else None)


def run_euclidean(starts):
    """Couple the common Euclidean pairs after whitening by the fitted map."""
    parameters = load_parameters(PARAMETERS_FILE, D, ['euclidean'])['euclidean']
    mu, scale = parameters['mu'], parameters['preconditioning_matrix']
    log_det = np.log(np.diag(scale)).sum()

    def whitened_log_density(z):
        return log_density(mu + scale @ z) + log_det

    rows = run_euclidean_experiment(
        whitened_log_density, affine_inverse(starts, parameters), H, MAX_STEPS,
        seed=SEED, seed_offset=200_000, stop_at_meeting=True, progress_every=10)
    return [dict(row, h=H, cutoff=MAX_STEPS, map_source='gaussian_reverse_kl') for row in rows]


def main():
    fit_euclidean()
    with np.load(INITIAL_STATES_FILE, allow_pickle=False) as saved:
        starts = saved['initial_states'][:N_REP]
    rows = run_euclidean(starts)
    save_results(rows, RESULTS / 'meeting_times.csv')
    summary = summarize(rows)
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
