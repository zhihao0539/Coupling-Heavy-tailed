"""Small native-coordinate wrappers around the existing generic samplers.

Each factory returns single_kernel, coupled_kernel, to_native, to_euclidean.
Keep states native throughout UPAVE. Transform only to initialize and evaluate h.
"""
import numpy as np

from generic_algorithms.algorithms import (
    _coupled_metropolis_step, _gaussian_proposal, _sphere_proposal, _ball_proposal,
    _checked_log_density, _matrix_parameters,
    _stereographic_log_density, _scs_log_density,
    _dcs_log_density, _dcs_matrix_log_density,
)
from generic_algorithms.transformations import (
    validate_stereographic_parameters, validate_scs_parameters,
    validate_dcs_parameters, validate_dcs_matrix_parameters,
    stereographic_inverse, stereographic_map, scs_inverse, scs_map,
    dcs_mobius_inverse, dcs_mobius_map, dcs_matrix_inverse, dcs_matrix_map,
)


def _metropolis_kernels(target, propose, step_size, to_native, to_euclidean):
    """Reuse the same proposal coupling and shared MH uniform as algorithms.py."""
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive.")
    def coupled_kernel(x, y, rng):
        log_x = target(x)
        log_y = log_x if np.array_equal(x, y) else target(y)
        if not np.isfinite([log_x, log_y]).all():
            raise ValueError("Both states must have finite native log density.")
        result = _coupled_metropolis_step(x, y, log_x, log_y, target, step_size, rng, propose)
        return result[0], result[1]
    def single_kernel(x, rng):
        return coupled_kernel(x, x, rng)[0]
    return dict(single_kernel=single_kernel, coupled_kernel=coupled_kernel,
                to_native=to_native, to_euclidean=to_euclidean)


def euclidean_kernels(log_pdf, step_size):
    def target(x):
        return _checked_log_density(log_pdf(x))
    return _metropolis_kernels(target, _gaussian_proposal, step_size, np.copy, np.copy)


def stereographic_kernels(log_pdf, step_size, *, mu, preconditioning_matrix):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    validate_stereographic_parameters(parameters, len(parameters['mu']))
    def target(z):
        return _stereographic_log_density(z, log_pdf, parameters)
    def to_native(x):
        return stereographic_inverse(x, parameters)
    def to_euclidean(z):
        return stereographic_map(z, parameters)[0]
    return _metropolis_kernels(target, _sphere_proposal, step_size, to_native, to_euclidean)


def scs_kernels(log_pdf, step_size, *, mu, preconditioning_matrix,
                horizontal_observer, latitude=1.1):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['horizontal_observer'] = np.asarray(horizontal_observer, dtype=float)
    validate_scs_parameters(parameters, len(parameters['mu']), latitude)
    def target(z):
        return _scs_log_density(z, log_pdf, parameters, latitude)
    def to_native(x):
        return scs_inverse(x, parameters, latitude)
    def to_euclidean(z):
        return scs_map(z, parameters, latitude)[0]
    return _metropolis_kernels(target, _sphere_proposal, step_size, to_native, to_euclidean)


def dcs_mobius_ball_kernels(log_pdf, step_size, *, mu, radius, delta, beta=2.):
    parameters = dict(mu=np.asarray(mu, dtype=float), radius=float(radius),
                      delta=np.asarray(delta, dtype=float), beta=float(beta))
    validate_dcs_parameters(parameters, len(parameters['mu']))
    def target(z):
        return _dcs_log_density(z, log_pdf, parameters)
    def to_native(x):
        return dcs_mobius_inverse(x, parameters)
    def to_euclidean(z):
        return dcs_mobius_map(z, parameters)[0]
    return _metropolis_kernels(target, _ball_proposal, step_size, to_native, to_euclidean)


def dcs_matrix_ball_kernels(log_pdf, step_size, *, mu, preconditioning_matrix, beta=2.):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['beta'] = float(beta)
    validate_dcs_matrix_parameters(parameters, len(parameters['mu']))
    def target(z):
        return _dcs_matrix_log_density(z, log_pdf, parameters)
    def to_native(x):
        return dcs_matrix_inverse(x, parameters)
    def to_euclidean(z):
        return dcs_matrix_map(z, parameters)[0]
    return _metropolis_kernels(target, _ball_proposal, step_size, to_native, to_euclidean)
