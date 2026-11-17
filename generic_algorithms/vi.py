"""Explicit reverse-KL fits with Gaussian, sphere, cap, or ball references."""
import math

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.special import betainc

from .transformations import (
    validate_euclidean_parameters, affine_map,
    validate_stereographic_parameters, validate_scs_parameters,
    validate_dcs_parameters, validate_dcs_matrix_parameters, validate_dcs_beta,
    stereographic_map, scs_map, dcs_mobius_map, dcs_matrix_map,
)


_BETA_RATE_MULTIPLIER = 0.1


def fit_euclidean_vi(log_pdf, dimension, *, mu, preconditioning_matrix, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected affine fields using a standard Gaussian reference."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    validate_euclidean_parameters(initial, dimension)

    def sample(key, n):
        return jax.random.normal(key, (n, dimension), dtype=jnp.float64)

    parameters, info, history = _fit_map(
        log_pdf, dimension, initial, _matrix_coordinates(dimension), tune,
        affine_map, sample, seed=seed, steps=steps, batch_size=batch_size,
        learning_rates=learning_rates, change_after=change_after,
        average_last=average_last, holdout_size=holdout_size, chunk_size=chunk_size)
    validate_euclidean_parameters(parameters, dimension)
    info.update(method='euclidean', variational_family='full_covariance_gaussian',
                reference_distribution='standard_gaussian',
                scale_parameterization='full_cholesky')
    return parameters, info, history


def fit_stereographic_vi(log_pdf, dimension, *, mu, preconditioning_matrix, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit the selected location and/or matrix fields using uniform sphere points."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    validate_stereographic_parameters(initial, dimension)
    coordinates = _matrix_coordinates(dimension)

    def sample(key, n):
        return sample_uniform_sphere(key, n, dimension)

    parameters, info, history = _fit_map(
        log_pdf, dimension, initial, coordinates, tune, stereographic_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_stereographic_parameters(parameters, dimension)
    info.update(method='stereographic', scale_parameterization='full_cholesky')
    return parameters, info, history


def fit_scs_vi(log_pdf, dimension, *, mu, preconditioning_matrix,
        horizontal_observer, latitude=1.1, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected SCS fields using uniform points on the cap at the given latitude."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    initial['horizontal_observer'] = np.asarray(horizontal_observer, dtype=float)
    validate_scs_parameters(initial, dimension, latitude)
    coordinates = _matrix_coordinates(dimension)
    coordinates['horizontal_observer'] = _offset_coordinates(math.sqrt(latitude * (2 - latitude)))

    def sample(key, n):
        return sample_uniform_scs_cap(key, n, dimension, latitude)

    def transform(points, parameters, xp):
        return scs_map(points, parameters, latitude, xp=xp)

    parameters, info, history = _fit_map(
        log_pdf, dimension, initial, coordinates, tune, transform, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_scs_parameters(parameters, dimension, latitude)
    info.update(method='scs', scale_parameterization='full_cholesky', latitude=latitude)
    return parameters, info, history


def fit_dcs_vi(log_pdf, dimension, *, mu, radius, delta, beta=2.0, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected Mobius DCS fields. Include 'beta' in tune to learn the exponent."""
    initial = dict(mu=np.asarray(mu, dtype=float), radius=radius,
                   delta=np.asarray(delta, dtype=float), beta=beta)
    validate_dcs_parameters(initial, dimension)
    coordinates = dict(mu=(np.asarray, jnp.asarray), radius=(_pack_positive, _unpack_positive),
                       delta=_offset_coordinates(1.0), beta=(_pack_positive, _unpack_positive))

    def sample(key, n):
        return sample_uniform_ball(key, n, dimension)

    parameters, info, history = _fit_map(
        log_pdf, dimension, initial, coordinates, tune, dcs_mobius_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_dcs_parameters(parameters, dimension)
    info.update(method='dcs', initial_beta=float(beta), beta=parameters['beta'])
    return parameters, info, history


def fit_dcs_matrix_vi(log_pdf, dimension, *, mu, preconditioning_matrix, beta=2.0, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected matrix DCS fields. Include 'beta' in tune to learn the exponent."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    initial['beta'] = beta
    validate_dcs_matrix_parameters(initial, dimension)
    coordinates = _matrix_coordinates(dimension)
    coordinates['beta'] = (_pack_positive, _unpack_positive)

    def sample(key, n):
        return sample_uniform_ball(key, n, dimension)

    parameters, info, history = _fit_map(
        log_pdf, dimension, initial, coordinates, tune, dcs_matrix_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_dcs_matrix_parameters(parameters, dimension)
    info.update(method='dcs', initial_beta=float(beta), beta=parameters['beta'],
                scale_parameterization='full_cholesky', ball_automorphism='appendix_b_affine')
    return parameters, info, history


def _matrix_coordinates(dimension):
    def unpack(values):
        return _unpack_cholesky(values, dimension)

    return dict(mu=(np.asarray, jnp.asarray), preconditioning_matrix=(_pack_cholesky, unpack))


def _offset_coordinates(limit):
    def pack(value):
        return _pack_offset(value, limit)

    def unpack(values):
        return _unpack_offset(values, limit)

    return pack, unpack


def _pack_positive(value):
    return np.array([math.log(value)])


def _unpack_positive(values):
    return jnp.exp(values[0])


def _free_coordinates(initial, coordinates, tune):
    """Pack selected fields in canonical order and close over exact physical values for the rest."""
    if tune is None:
        tune = [name for name in coordinates if name != 'beta']
    elif isinstance(tune, str):
        tune = [tune]
    else:
        tune = list(tune)
    unknown = set(tune).difference(coordinates)
    if unknown or len(set(tune)) != len(tune):
        raise ValueError(f'tune must contain unique parameter names from {tuple(coordinates)}; got {tune}.')

    slices = {}
    parts = []
    start = 0
    for name, (pack, _) in coordinates.items():
        if name in tune:
            values = pack(initial[name])
            stop = start + len(values)
            slices[name] = (start, stop)
            parts.append(values)
            start = stop

    frozen = {name: jnp.asarray(value, dtype=jnp.float64)
              for name, value in initial.items() if name not in slices}

    def unpack(theta):
        parameters = dict(frozen)
        for name, (start, stop) in slices.items():
            _, decode = coordinates[name]
            parameters[name] = decode(theta[start:stop])
        return parameters

    theta0 = jnp.asarray(np.concatenate(parts) if parts else np.empty(0), dtype=jnp.float64)
    return theta0, unpack, slices


def _fit_map(log_pdf, dimension, initial, coordinates, tune, transform, sample, **settings):
    theta0, unpack, slices = _free_coordinates(initial, coordinates, tune)

    def objective(theta, samples):
        x, log_jacobian = transform(samples, unpack(theta), xp=jnp)
        return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)

    # Use the slower beta update from the Student-t joint-fit experiment.
    rates = 1.
    beta_columns = slice(*slices['beta']) if 'beta' in slices else None
    if beta_columns is not None:
        rates = jnp.ones_like(theta0).at[beta_columns].set(_BETA_RATE_MULTIPLIER)
    theta, info, history = _train_vi(theta0, dimension, objective, sample, rate_multipliers=rates, **settings)
    info.update(tuned_parameters=list(slices), fixed_parameters=[name for name in initial if name not in slices],
                theta_slices=slices)
    if beta_columns is not None:
        _, decode_beta = coordinates['beta']
        history['beta'] = np.asarray(jax.vmap(decode_beta)(history['theta'][:, beta_columns]))
        info['beta_learning_rate_multiplier'] = _BETA_RATE_MULTIPLIER
    return _numpy_parameters(unpack(jnp.asarray(theta))), info, history


def _matrix_parameters(mu, preconditioning_matrix):
    return dict(mu=np.asarray(mu, dtype=float),
                preconditioning_matrix=np.asarray(preconditioning_matrix, dtype=float))


def _pack_cholesky(scale):
    lower = np.tril_indices(len(scale), -1)
    return np.concatenate((np.log(np.diag(scale)), scale[lower]))


def _unpack_cholesky(values, dimension):
    scale = jnp.diag(jnp.exp(values[:dimension]))
    return scale.at[np.tril_indices(dimension, -1)].set(values[dimension:])


def _pack_matrix(parameters):
    return np.concatenate((parameters['mu'], _pack_cholesky(parameters['preconditioning_matrix'])))


def _unpack_matrix(theta, dimension):
    end = dimension + dimension * (dimension + 1) // 2
    return dict(mu=theta[:dimension], preconditioning_matrix=_unpack_cholesky(theta[dimension:end], dimension))


def _pack_offset(offset, limit):
    offset = np.asarray(offset)
    norm = np.linalg.norm(offset)
    return offset * (np.arctanh(norm / limit) / norm) if norm else np.zeros(len(offset))


def _unpack_offset(v, limit):
    norm = jnp.sqrt(jnp.sum(v * v) + 1e-24)
    return limit * (jnp.tanh(norm) / norm) * v


def pack_stereographic_parameters(mu, preconditioning_matrix):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    validate_stereographic_parameters(parameters, len(parameters['mu']))
    return jnp.asarray(_pack_matrix(parameters))


def unpack_stereographic_parameters(theta, dimension):
    return _unpack_matrix(theta, dimension)


def pack_scs_parameters(mu, preconditioning_matrix, horizontal_observer, latitude=1.1):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['horizontal_observer'] = np.asarray(horizontal_observer)
    validate_scs_parameters(parameters, len(parameters['mu']), latitude)
    offset = _pack_offset(parameters['horizontal_observer'], math.sqrt(latitude * (2 - latitude)))
    return jnp.asarray(np.concatenate([_pack_matrix(parameters), offset]))


def unpack_scs_parameters(theta, dimension, latitude=1.1):
    parameters = _unpack_matrix(theta, dimension)
    end = 2 * dimension + len(np.tril_indices(dimension, -1)[0])
    parameters['horizontal_observer'] = _unpack_offset(theta[end:], math.sqrt(latitude * (2 - latitude)))
    return parameters


def pack_dcs_parameters(mu, radius, delta, *, beta=2.0):
    validate_dcs_beta(beta)
    parameters = dict(mu=np.asarray(mu, dtype=float), radius=float(radius), delta=np.asarray(delta), beta=float(beta))
    validate_dcs_parameters(parameters, len(parameters['mu']))
    parts = [parameters['mu'], _pack_positive(radius), _pack_offset(parameters['delta'], 1.)]
    return jnp.asarray(np.concatenate(parts))


def unpack_dcs_parameters(theta, dimension, *, beta=2.0):
    return dict(mu=theta[:dimension], radius=jnp.exp(theta[dimension]),
                delta=_unpack_offset(theta[dimension + 1:], 1.), beta=beta)


def pack_dcs_matrix_parameters(mu, preconditioning_matrix, *, beta=2.0):
    validate_dcs_beta(beta)
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['beta'] = float(beta)
    validate_dcs_matrix_parameters(parameters, len(parameters['mu']))
    return jnp.asarray(_pack_matrix(parameters))


def unpack_dcs_matrix_parameters(theta, dimension, *, beta=2.0):
    parameters = _unpack_matrix(theta, dimension)
    parameters['beta'] = beta
    return parameters


def sample_uniform_ball(key, n, dimension):
    """Uniform reference points inside the DCS unit ball."""
    _validate_reference_counts(n, dimension)
    direction_key, radius_key = jax.random.split(key)
    g = jax.random.normal(direction_key, (n, dimension), dtype=jnp.float64)
    radii = jax.random.uniform(radius_key, (n, 1), dtype=jnp.float64)**(1 / dimension)
    return g / jnp.linalg.norm(g, axis=-1, keepdims=True) * radii


def sample_uniform_sphere(key, n, dimension):
    """Uniform reference points on the stereographic sphere."""
    _validate_reference_counts(n, dimension)
    g = jax.random.normal(key, (n, dimension + 1), dtype=jnp.float64)
    return g / jnp.linalg.norm(g, axis=-1, keepdims=True)


def sample_uniform_scs_cap(key, n, dimension, latitude=1.1):
    """Uniform bright-cap points, using rejection sampling on the sphere."""
    _validate_reference_counts(n, dimension)
    if not 0 < latitude < 2:
        raise ValueError('SCS latitude must lie strictly between 0 and 2.')
    # Size the rejection batch for the cap's probability. At the experiment's
    # latitude and batch size this is the original 2*n draw.
    probability = float(betainc(dimension / 2, dimension / 2, latitude / 2))
    if probability <= 0:
        raise ValueError('This cap is too small for reference rejection sampling.')
    count = max(2 * n, math.ceil((n + 8 * math.sqrt(n) + 32) / probability))
    if count * (dimension + 1) > 50_000_000:
        raise ValueError('Reference batch is too large; increase latitude or reduce batch_size/holdout_size.')

    def draw(state):
        key, subkey = jax.random.split(state[0])
        g = jax.random.normal(subkey, (count, dimension + 1), dtype=jnp.float64)
        z = g / jnp.linalg.norm(g, axis=1, keepdims=True)
        valid = z[:, -1] < latitude - 1
        indices = jnp.nonzero(valid, size=n, fill_value=0)[0]
        return key, z[indices], jnp.sum(valid, dtype=jnp.int32)

    initial = (key, jnp.zeros((n, dimension + 1)), jnp.int32(0))
    return jax.lax.while_loop(lambda state: state[2] < n, draw, initial)[1]


def stereographic_reverse_kl(theta, samples, log_pdf, dimension):
    parameters = unpack_stereographic_parameters(theta, dimension)
    x, log_jacobian = stereographic_map(samples, parameters, xp=jnp)
    return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)


def scs_reverse_kl(theta, samples, log_pdf, dimension, latitude=1.1):
    parameters = unpack_scs_parameters(theta, dimension, latitude)
    x, log_jacobian = scs_map(samples, parameters, latitude, xp=jnp)
    return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)


def dcs_reverse_kl(theta, samples, log_pdf, dimension, *, beta=2.0):
    parameters = unpack_dcs_parameters(theta, dimension, beta=beta)
    x, log_jacobian = dcs_mobius_map(samples, parameters, xp=jnp)
    return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)


def dcs_matrix_reverse_kl(theta, samples, log_pdf, dimension, *, beta=2.0):
    parameters = unpack_dcs_matrix_parameters(theta, dimension, beta=beta)
    x, log_jacobian = dcs_matrix_map(samples, parameters, xp=jnp)
    return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)


def adam_update(theta, gradient, m, v, step, learning_rate):
    """One Adam update; step starts at 1. beta1=.9, beta2=.999, epsilon=1e-8."""
    m = .9 * m + .1 * gradient
    v = .999 * v + .001 * gradient * gradient
    direction = (m / (1 - .9**step)) / (jnp.sqrt(v / (1 - .999**step)) + 1e-8)
    return theta - learning_rate * direction, m, v


def _numpy_parameters(parameters):
    return {k: float(v) if k in ('radius', 'beta') else np.asarray(v) for k, v in parameters.items()}


def _train_vi(theta0, dimension, objective, sample, *, seed, steps, batch_size,
              learning_rates, change_after, average_last, holdout_size, chunk_size, rate_multipliers=1.):
    """Run Adam and holdout selection; each caller supplies its own numerical objective."""
    _validate_vi_settings(dimension, steps, batch_size, average_last,
                          holdout_size, chunk_size, learning_rates, change_after)
    history = _optimize(theta0, objective, sample, seed=seed, steps=steps,
                        batch_size=batch_size, learning_rates=learning_rates,
                        change_after=change_after, chunk_size=chunk_size, rate_multipliers=rate_multipliers)
    theta, selection = _select_candidate(theta0, history, objective, sample,
                                         seed, average_last, holdout_size)
    info = dict(dimension=dimension, seed=seed, steps=steps,
                batch_size=batch_size, learning_rates=list(learning_rates),
                change_after=change_after, **selection)
    return theta, info, history


def _validate_reference_counts(n, dimension):
    if any(not isinstance(value, (int, np.integer)) or value < 1 for value in (n, dimension)):
        raise ValueError('Reference dimension and sample count must be positive integers.')


def _validate_vi_settings(dimension, steps, batch_size, average_last,
                          holdout_size, chunk_size, learning_rates, change_after):
    counts = (dimension, steps, batch_size, average_last, holdout_size, chunk_size)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ValueError('Dimensions, iteration counts and sample counts must be positive integers.')
    if len(learning_rates) != 2 or not np.isfinite(learning_rates).all() or min(learning_rates) <= 0:
        raise ValueError('Provide two finite positive learning rates.')
    if not isinstance(change_after, (int, np.integer)) or change_after < 0:
        raise ValueError('change_after must be a nonnegative integer.')


def _optimize(theta0, objective, sample, *, seed, steps, batch_size,
              learning_rates, change_after, chunk_size, rate_multipliers=1.):
    """Run Adam in compiled chunks and collect finite losses and iterates."""
    value_and_grad = jax.value_and_grad(objective)

    def update(state, k):
        theta, m, v, key = state
        key, subkey = jax.random.split(key)
        value, gradient = value_and_grad(theta, sample(subkey, batch_size))
        rate = learning_rates[0] * jnp.where(k < change_after, 1., learning_rates[1] / learning_rates[0])
        theta, m, v = adam_update(theta, gradient, m, v, k + 1, rate * rate_multipliers)
        return (theta, m, v, key), (value, theta)

    run_chunk = jax.jit(lambda state, ks: jax.lax.scan(update, state, ks))
    state = (theta0, jnp.zeros_like(theta0), jnp.zeros_like(theta0), jax.random.key(seed))
    losses, thetas = [], []
    for start in range(0, steps, chunk_size):
        state, (values, parameters) = run_chunk(state, jnp.arange(start, min(start + chunk_size, steps)))
        values, parameters = np.asarray(values), np.asarray(parameters)
        if not np.isfinite(values).all() or not np.isfinite(parameters).all():
            raise FloatingPointError('Non-finite VI loss/parameters. Check target support, gradients and learning rate.')
        losses.append(values)
        thetas.append(parameters)
    return {'loss': np.concatenate(losses), 'theta': np.concatenate(thetas)}


def _select_candidate(theta0, history, objective, sample, seed, average_last, holdout_size):
    """Compare the final iterate and its trailing average on an independent holdout."""
    count = min(average_last, len(history['theta']))
    candidates = [history['theta'][-1], history['theta'][-count:].mean(axis=0)]
    holdout = sample(jax.random.key(seed + 500), holdout_size)
    evaluate = jax.jit(objective)
    candidate_losses = [float(evaluate(jnp.asarray(theta), holdout)) for theta in candidates]
    if not np.isfinite(candidate_losses).all():
        raise FloatingPointError('Non-finite VI holdout objective.')
    chosen = int(np.argmin(candidate_losses))
    info = dict(average_last=count, holdout_size=holdout_size, holdout_seed=seed + 500,
                initial_holdout_loss=float(evaluate(theta0, holdout)),
                candidate_holdout_losses=candidate_losses,
                chosen_candidate=['last', f'last_{count}_parameter_average'][chosen])
    return candidates[chosen], info
