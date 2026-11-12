"""Fit the two SCS matrices and jointly fit DCS Mobius radius, delta and beta, with mu=0."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_experiment import LATITUDES, RESULTS
from generic_algorithms import run_scs_vi, run_dcs_vi
from target import DIMENSIONS, make_log_density, target_info

SEED = 20261027
VI_SETTINGS = dict(
    steps=20000, batch_size=2048, holdout_size=20000,
    average_last=500, learning_rates=(0.01, 0.002),
    change_after=10000, chunk_size=250,
)


def fit_scs(d, label, latitude, initial_scale, seed, path, settings):
    run_scs_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        preconditioning_matrix=initial_scale * np.eye(d),
        horizontal_observer=np.zeros(d), latitude=latitude, tune="preconditioning_matrix",
        label=label, seed=seed, output_json=path, target_info=target_info(d), **settings)


def fit_dcs(d, seed, path, settings):
    run_dcs_vi(
        make_log_density(d, jax=True), d, mu=np.zeros(d),
        radius=np.sqrt(d + 1), delta=np.zeros(d),
        beta=max(2., np.pi * np.sqrt(d / 6)), tune=("radius", "delta", "beta"),
        seed=seed, output_json=path, target_info=target_info(d), **settings)


def fit_dimension(d, folder, settings):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fitted_parameters.json"
    seed = SEED + 1000*d
    with TemporaryDirectory(dir=folder) as tmp:
        new_fit = Path(tmp) / path.name
        fit_scs(d, "SCS11", LATITUDES["SCS11"], np.sqrt(d)/11, seed+1, new_fit, settings)
        fit_scs(d, "SCS19", LATITUDES["SCS19"], 9*np.sqrt(d)/19, seed+2, new_fit, settings)
        fit_dcs(d, seed+3, new_fit, settings)
        new_fit.replace(path)
    print(f"d={d}: saved VI maps to {path}", flush=True)


def main():
    for d in DIMENSIONS:
        fit_dimension(d, RESULTS / f"d_{d:04d}", VI_SETTINGS)


if __name__ == "__main__":
    main()
