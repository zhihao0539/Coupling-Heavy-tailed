"""Fit stereographic and matrix DCS maps by reverse KL, including DCS beta."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_experiment import RESULTS
from generic_algorithms import run_stereographic_vi, run_dcs_matrix_vi, save_parameters
from target import DIMENSIONS, make_log_density, target_info

SEED = 20261112
VI_SETTINGS = dict(
    steps=6000, batch_size=2048, holdout_size=50000,
    average_last=500, learning_rates=(.01, .002), change_after=4500,
    chunk_size=250,
)
DCS_VI_SETTINGS = dict(VI_SETTINGS, steps=16000, change_after=12000)


def fit_stereographic(d, seed, path, settings):
    run_stereographic_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=np.sqrt(d)*np.eye(d),
        tune=("mu", "preconditioning_matrix"), seed=seed,
        output_json=path, target_info=target_info(d), **settings)


def fit_dcs(d, seed, path, settings):
    initial_beta = 4
    parameters, info, history = run_dcs_matrix_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=np.sqrt(d)*np.eye(d), beta=initial_beta,
        tune=("mu", "preconditioning_matrix", "beta"), seed=seed,
        target_info=target_info(d), **settings)
    minimum_beta = min(initial_beta, float(np.min(history["beta"])), parameters["beta"])
    if minimum_beta <= 2:
        raise ValueError("Gaussian reverse KL needs beta>2; retry with a larger initial beta.")
    info["minimum_training_beta"] = minimum_beta
    save_parameters({"dcs": parameters}, {"dcs": info}, path)


def fit_dimension(d, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fitted_parameters.json"
    seed = SEED + 1000*d
    with TemporaryDirectory(dir=folder) as tmp:
        new_fit = Path(tmp) / path.name
        fit_stereographic(d, seed+1, new_fit, VI_SETTINGS)
        fit_dcs(d, seed+2, new_fit, DCS_VI_SETTINGS)
        new_fit.replace(path)
    print(f"d={d}: saved VI maps to {path}", flush=True)


def main():
    for d in DIMENSIONS:
        fit_dimension(d, RESULTS/f"d_{d:04d}")


if __name__ == "__main__":
    main()
