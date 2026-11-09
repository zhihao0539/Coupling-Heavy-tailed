"""Explicit reverse-KL fits with uniform sphere, cap, or ball references (64-bit JAX)."""
import math

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.special import betainc

from .transformations import (
    validate_stereographic_parameters, validate_scs_parameters,
    validate_dcs_parameters, validate_dcs_matrix_parameters,
    stereographic_map, scs_map, dcs_mobius_map, dcs_matrix_map,
)


def fit_stereographic_vi(log_pdf, dimension, *, mu, preconditioning_matrix, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected location/matrix fields using uniform sphere reference points."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    validate_stereographic_parameters(initial, dimension)
    theta0 = pack_stereographic_parameters(mu, preconditioning_matrix)

    def sample(key, n):
        return sample_uniform_sphere(key, n, dimension)

    def unpack(theta):
        return unpack_stereographic_parameters(theta, dimension)

    parameters, info, history = _fit_selected_map(
        log_pdf, dimension, initial, theta0, _matrix_field_sizes(dimension),
        tune, unpack, stereographic_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_stereographic_parameters(parameters, dimension)
    info = dict(method='stereographic', **info)
    info.update({'scale_parameterization': 'full_cholesky'})
    return parameters, info, history


def fit_scs_vi(log_pdf, dimension, *, mu, preconditioning_matrix,
        horizontal_observer, latitude=1.1, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected SCS fields using a uniform cap at the supplied latitude."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    initial['horizontal_observer'] = np.asarray(horizontal_observer)
    validate_scs_parameters(initial, dimension, latitude)
    theta0 = pack_scs_parameters(mu, preconditioning_matrix, horizontal_observer, latitude)

    def sample(key, n):
        return sample_uniform_scs_cap(key, n, dimension, latitude)

    def unpack(theta):
        return unpack_scs_parameters(theta, dimension, latitude)

    def transform(points, parameters, xp):
        return scs_map(points, parameters, latitude, xp=xp)

    sizes = dict(_matrix_field_sizes(dimension), horizontal_observer=dimension)
    parameters, info, history = _fit_selected_map(
        log_pdf, dimension, initial, theta0, sizes, tune, unpack, transform, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_scs_parameters(parameters, dimension, latitude)
    info = dict(method='scs', **info)
    info.update({'scale_parameterization': 'full_cholesky', 'latitude': latitude})
    return parameters, info, history


def fit_dcs_vi(log_pdf, dimension, *, mu, radius, delta, beta=2.0, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected Mobius DCS fields, including beta when explicitly selected."""
    initial = dict(mu=np.asarray(mu), radius=radius, delta=np.asarray(delta), beta=beta)
    validate_dcs_parameters(initial, dimension)
    theta0 = jnp.concatenate((pack_dcs_parameters(mu, radius, delta, beta=beta),
                              jnp.array([math.log(beta)])))
    if tune is None:
        tune = ("mu", "radius", "delta")

    def sample(key, n):
        return sample_uniform_ball(key, n, dimension)

    def unpack(theta):
        return unpack_dcs_parameters(theta[:-1], dimension, beta=jnp.exp(theta[-1]))

    sizes = dict(mu=dimension, radius=1, delta=dimension, beta=1)
    parameters, info, history = _fit_selected_map(
        log_pdf, dimension, initial, theta0, sizes, tune, unpack, dcs_mobius_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_dcs_parameters(parameters, dimension)
    info = dict(method='dcs', **info)
    info.update(initial_beta=float(beta), beta=parameters['beta'])
    return parameters, info, history


def fit_dcs_matrix_vi(log_pdf, dimension, *, mu, preconditioning_matrix, beta=2.0, tune=None,
        seed=0, steps=4000, batch_size=2048,
        learning_rates=(.01, .002), change_after=3000,
        average_last=500, holdout_size=20000, chunk_size=250):
    """Fit selected matrix DCS fields, including beta when explicitly selected."""
    initial = _matrix_parameters(mu, preconditioning_matrix)
    initial['beta'] = beta
    validate_dcs_matrix_parameters(initial, dimension)
    theta0 = jnp.concatenate((pack_dcs_matrix_parameters(mu, preconditioning_matrix, beta=beta),
                              jnp.array([math.log(beta)])))
    if tune is None:
        tune = ("mu", "preconditioning_matrix")

    def sample(key, n):
        return sample_uniform_ball(key, n, dimension)

    def unpack(theta):
        return unpack_dcs_matrix_parameters(theta[:-1], dimension, beta=jnp.exp(theta[-1]))

    sizes = dict(_matrix_field_sizes(dimension), beta=1)
    parameters, info, history = _fit_selected_map(
        log_pdf, dimension, initial, theta0, sizes,
        tune, unpack, dcs_matrix_map, sample,
        seed=seed, steps=steps, batch_size=batch_size, learning_rates=learning_rates,
        change_after=change_after, average_last=average_last,
        holdout_size=holdout_size, chunk_size=chunk_size)
    validate_dcs_matrix_parameters(parameters, dimension)
    info = dict(method='dcs', **info)
    info.update(initial_beta=float(beta), beta=parameters['beta'],
                scale_parameterization='full_cholesky', ball_automorphism='appendix_b_affine')
    return parameters, info, history


def _matrix_field_sizes(dimension):
    return dict(mu=dimension, preconditioning_matrix=dimension * (dimension + 1) // 2)


def _selected_coordinates(theta0, initial, sizes, tune, unpack):
    """Optimize a subset of the existing packed vector and retain exact fixed fields."""
    if tune is None:
        names = list(sizes)
    elif isinstance(tune, str):
        names = [tune]
    else:
        names = list(tune)
    if any(not isinstance(name, str) for name in names):
        raise ValueError('tune must contain parameter names.')
    if len(set(names)) != len(names) or set(names).difference(sizes):
        raise ValueError(f'tune must contain unique names from {tuple(sizes)}; got {names}.')
    indices, slices = [], {}
    full_start = free_start = 0
    for name, size in sizes.items():
        if name in names:
            indices.extend(range(full_start, full_start + size))
            slices[name] = (free_start, free_start + size)
            free_start += size
        full_start += size
    indices = jnp.asarray(indices, dtype=jnp.int32)
    frozen = {name: jnp.asarray(value, dtype=jnp.float64)
              for name, value in initial.items() if name not in slices}

    def decode(theta):
        parameters = unpack(theta0.at[indices].set(theta))
        parameters.update(frozen)
        return parameters

    return theta0[indices], decode, slices


def _fit_selected_map(log_pdf, dimension, initial, theta0, sizes, tune, unpack, transform, sample, **settings):
    theta0, decode, slices = _selected_coordinates(theta0, initial, sizes, tune, unpack)

    def objective(theta, samples):
        x, log_jacobian = transform(samples, decode(theta), xp=jnp)
        return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)

    rates = 1.
    if 'beta' in slices:
        start, stop = slices['beta']
        rates = jnp.ones_like(theta0).at[start:stop].set(.1)
    theta, info, history = _train_vi(theta0, dimension, objective, sample, rate_multipliers=rates, **settings)
    info.update(tuned_parameters=list(slices),
                fixed_parameters=[name for name in initial if name not in slices],
                theta_slices=slices)
    if 'beta' in slices:
        history['beta'] = np.exp(history['theta'][:, start])
        info['beta_learning_rate_multiplier'] = .1
    return _numpy_parameters(decode(jnp.asarray(theta))), info, history


def _matrix_parameters(mu, preconditioning_matrix):
    return dict(mu=np.asarray(mu, dtype=float),
                preconditioning_matrix=np.asarray(preconditioning_matrix, dtype=float))


def _pack_matrix(parameters):
    scale = parameters['preconditioning_matrix']
    parts = [parameters['mu'], np.log(np.diag(scale)),
             scale[np.tril_indices(len(parameters['mu']), -1)]]
    return np.concatenate(parts)


def _unpack_matrix(theta, dimension):
    lower = np.tril_indices(dimension, -1)
    start = 2 * dimension
    end = start + len(lower[0])
    scale = jnp.diag(jnp.exp(theta[dimension:start]))
    return {'mu': theta[:dimension], 'preconditioning_matrix': scale.at[lower].set(theta[start:end])}


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
    parameters = dict(mu=np.asarray(mu, dtype=float), radius=float(radius), delta=np.asarray(delta), beta=beta)
    validate_dcs_parameters(parameters, len(parameters['mu']))
    parts = [parameters['mu'], np.array([math.log(radius)]), _pack_offset(parameters['delta'], 1.)]
    return jnp.asarray(np.concatenate(parts))


def unpack_dcs_parameters(theta, dimension, *, beta=2.0):
    return dict(mu=theta[:dimension], radius=jnp.exp(theta[dimension]),
                delta=_unpack_offset(theta[dimension + 1:], 1.), beta=beta)


def pack_dcs_matrix_parameters(mu, preconditioning_matrix, *, beta=2.0):
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['beta'] = beta
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
