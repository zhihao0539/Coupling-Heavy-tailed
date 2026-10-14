"""Reusable reverse-KL fitting with Monte Carlo gradients and Adam (64-bit JAX)."""
import math
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.special import betainc
from .transformations import default_parameters, validate_parameters, map_to_euclidean

VI_METHODS = ('stereographic', 'scs', 'dcs')


def unpack_parameters(theta, dimension, method, latitude=1.1):
    """Unconstrained variables -> shift, positive radius, admissible offset."""
    parameters = {'mu': theta[:dimension], 'radius': jnp.exp(theta[dimension])}
    if method in ('scs', 'dcs'):
        v = theta[dimension + 1:]
        norm = jnp.sqrt(jnp.sum(v * v) + 1e-24)
        limit = math.sqrt(latitude * (2 - latitude)) if method == 'scs' else 1.
        name = 'horizontal_observer' if method == 'scs' else 'delta'
        parameters[name] = limit * (jnp.tanh(norm) / norm) * v
    return parameters


def pack_parameters(parameters, method, latitude=1.1):
    dimension = len(parameters['mu'])
    validate_parameters(parameters, dimension, method, latitude)
    parts = [np.asarray(parameters['mu'], dtype=float), np.array([math.log(parameters['radius'])])]
    if method in ('scs', 'dcs'):
        offset = np.asarray(parameters['horizontal_observer' if method == 'scs' else 'delta'])
        limit = math.sqrt(latitude * (2 - latitude)) if method == 'scs' else 1.
        norm = np.linalg.norm(offset)
        parts.append(offset * (np.arctanh(norm / limit) / norm) if norm else np.zeros(dimension))
    return jnp.asarray(np.concatenate(parts))


def sample_reference(key, n, dimension, method, latitude=1.1):
    """Uniform sphere, bright cap, or ball; the reference does not depend on theta."""
    if method == 'dcs':
        direction_key, radius_key = jax.random.split(key)
        g = jax.random.normal(direction_key, (n, dimension), dtype=jnp.float64)
        radii = jax.random.uniform(radius_key, (n, 1), dtype=jnp.float64)**(1 / dimension)
        return g / jnp.linalg.norm(g, axis=-1, keepdims=True) * radii
    if method == 'stereographic':
        g = jax.random.normal(key, (n, dimension + 1), dtype=jnp.float64)
        return g / jnp.linalg.norm(g, axis=-1, keepdims=True)
    if method != 'scs' or not 0 < latitude < 2:
        raise ValueError('Choose a VI method and, for SCS, 0 < latitude < 2.')
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


def reverse_kl(theta, samples, log_pdf, dimension, method, latitude=1.1):
    """KL(q_theta || target) up to constants; log_pdf takes ONE vector."""
    parameters = unpack_parameters(theta, dimension, method, latitude)
    x, log_jacobian = map_to_euclidean(samples, method, parameters, latitude, xp=jnp)
    return -jnp.mean(jax.vmap(log_pdf)(x) + log_jacobian)


def adam_update(theta, gradient, m, v, step, learning_rate):
    """One Adam update; step starts at 1. beta1=.9, beta2=.999, epsilon=1e-8."""
    m = .9 * m + .1 * gradient
    v = .999 * v + .001 * gradient * gradient
    direction = (m / (1 - .9**step)) / (jnp.sqrt(v / (1 - .999**step)) + 1e-8)
    return theta - learning_rate * direction, m, v


def fit_vi(log_pdf, dimension, method='scs', *, initial_parameters=None,
           latitude=1.1, seed=0, steps=4000, batch_size=2048,
           learning_rates=(.01, .002), change_after=3000,
           average_last=500, holdout_size=20000, chunk_size=250):
    """Fit map parameters for any differentiable, scalar JAX log density.

    Returns (parameters, info, history). Parameters can be passed directly to
    coupled_sampler. History contains loss and unconstrained theta per update.
    Fresh reference samples are used for every gradient. An independent
    holdout selects between the last iterate and the last-window average.
    The target's reverse KL must be finite for the chosen variational family.
    """
    if method not in VI_METHODS:
        raise ValueError(f'VI supports {VI_METHODS}.')
    counts = (dimension, steps, batch_size, average_last, holdout_size, chunk_size)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ValueError('Dimensions, iteration counts and sample counts must be positive integers.')
    if len(learning_rates) != 2 or not np.isfinite(learning_rates).all() or min(learning_rates) <= 0:
        raise ValueError('Provide two finite positive learning rates.')
    if not isinstance(change_after, (int, np.integer)) or change_after < 0:
        raise ValueError('change_after must be a nonnegative integer.')
    if initial_parameters is None:
        initial_parameters = default_parameters(dimension, method)
    validate_parameters(initial_parameters, dimension, method, latitude)
    theta0 = pack_parameters(initial_parameters, method, latitude)
    objective = lambda theta, z: reverse_kl(theta, z, log_pdf, dimension, method, latitude)
    value_and_grad = jax.value_and_grad(objective)
    sample = lambda key, n: sample_reference(key, n, dimension, method, latitude)

    def update(state, k):
        theta, m, v, key = state
        key, subkey = jax.random.split(key)
        value, gradient = value_and_grad(theta, sample(subkey, batch_size))
        rate = learning_rates[0] * jnp.where(k < change_after, 1., learning_rates[1] / learning_rates[0])
        theta, m, v = adam_update(theta, gradient, m, v, k + 1, rate)
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
    history = {'loss': np.concatenate(losses), 'theta': np.concatenate(thetas)}
    count = min(average_last, steps)
    candidates = [history['theta'][-1], history['theta'][-count:].mean(axis=0)]
    holdout = sample(jax.random.key(seed + 500), holdout_size)
    evaluate = jax.jit(objective)
    candidate_losses = [float(evaluate(jnp.asarray(theta), holdout)) for theta in candidates]
    if not np.isfinite(candidate_losses).all():
        raise FloatingPointError('Non-finite VI holdout objective.')
    chosen = int(np.argmin(candidate_losses))
    parameters = unpack_parameters(jnp.asarray(candidates[chosen]), dimension, method, latitude)
    parameters = {k: float(v) if k == 'radius' else np.asarray(v) for k, v in parameters.items()}
    validate_parameters(parameters, dimension, method, latitude)
    info = dict(method=method, dimension=dimension, seed=seed, steps=steps,
                batch_size=batch_size, learning_rates=list(learning_rates),
                change_after=change_after, average_last=count, holdout_size=holdout_size,
                holdout_seed=seed + 500, initial_holdout_loss=float(evaluate(theta0, holdout)),
                candidate_holdout_losses=candidate_losses,
                chosen_candidate=['last', f'last_{count}_parameter_average'][chosen])
    if method == 'scs':
        info['latitude'] = latitude
    elif method == 'dcs':
        info['beta'] = 2
    return parameters, info, history

