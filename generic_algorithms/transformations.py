"""Target-independent maps. Pass xp=jax.numpy to use the same formulas in VI."""
import numpy as np

METHODS = ('euclidean', 'stereographic', 'scs', 'dcs')


def default_parameters(dimension, method, radius=1.0):
    if method not in METHODS or dimension < 1:
        raise ValueError('Choose a supported method and a positive dimension.')
    parameters = {'mu': np.zeros(dimension), 'radius': float(radius)}
    if method == 'scs':
        parameters['horizontal_observer'] = np.zeros(dimension)
    elif method == 'dcs':
        parameters['delta'] = np.zeros(dimension)
    return parameters


def validate_parameters(parameters, dimension, method, latitude=1.1):
    """Validate once, outside the sampling/optimization loops."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if method == 'euclidean':
        return
    mu = np.asarray(parameters['mu'])
    if mu.shape != (dimension,) or not np.isfinite(mu).all():
        raise ValueError('mu must be a finite vector of the target dimension.')
    if not np.isfinite(parameters['radius']) or parameters['radius'] <= 0:
        raise ValueError('radius must be finite and positive.')
    if method == 'scs' and not 0 < latitude < 2:
        raise ValueError('SCS latitude must lie strictly between 0 and 2.')
    if method in ('scs', 'dcs'):
        name = 'horizontal_observer' if method == 'scs' else 'delta'
        offset = np.asarray(parameters[name])
        limit = np.sqrt(latitude * (2 - latitude)) if method == 'scs' else 1.
        if offset.shape != (dimension,) or not np.isfinite(offset).all() or np.linalg.norm(offset) >= limit:
            raise ValueError(f'{name} must have dimension {dimension} and norm below {limit}.')


def stereographic_map(z, parameters, xp=np):
    """Sphere -> R^d, returning the log Jacobian relative to sphere area."""
    mu, radius = parameters['mu'], parameters['radius']
    d = mu.shape[-1]
    gap = 1 - z[..., -1]
    return mu + radius * z[..., :-1] / gap[..., None], d * (xp.log(radius) - xp.log(gap))


def stereographic_inverse(x, parameters, xp=np):
    u = (x - parameters['mu']) / parameters['radius']
    norm2 = xp.sum(u * u, axis=-1, keepdims=True)
    return xp.concatenate((2 * u / (1 + norm2), (norm2 - 1) / (1 + norm2)), axis=-1)


def scs_map(z, parameters, latitude=1.1, xp=np):
    """Bright cap -> R^d. The sphere is centered at zero."""
    mu, radius, observer = parameters['mu'], parameters['radius'], parameters['horizontal_observer']
    d = mu.shape[-1]
    height = z[..., -1]
    gap = latitude - 1 - height
    x = mu + radius * (latitude * z[..., :-1] - (1 + height[..., None]) * observer) / gap[..., None]
    cosine = 1 - (latitude - 1) * height - xp.sum(z[..., :-1] * observer, axis=-1)
    log_jacobian = d * xp.log(radius * latitude) + xp.log(cosine) - (d + 1) * xp.log(gap)
    return x, log_jacobian


def scs_inverse(x, parameters, latitude=1.1, xp=np):
    mu, radius, observer = parameters['mu'], parameters['radius'], parameters['horizontal_observer']
    b = latitude - 1
    a = (x - mu) / radius - observer
    aa = xp.sum(a * a, axis=-1) + latitude**2
    half_b = xp.sum(a * observer, axis=-1) - latitude * b
    cc = xp.sum(observer * observer) + b**2 - 1
    root = xp.sqrt(half_b**2 - aa * cc)
    # Both denominators stay nonzero, including the unused branch under JAX.
    m = xp.where(half_b > 0, -cc / (root + xp.maximum(half_b, 0)),
                 (-half_b + root) / aa)
    return xp.concatenate((m[..., None] * a + observer, (b - latitude * m)[..., None]), axis=-1)


def mobius(s, delta, xp=np):
    norm2 = xp.sum(s * s, axis=-1)
    dot = xp.sum(s * delta, axis=-1)
    delta2 = xp.sum(delta * delta)
    denominator = 1 + 2 * dot + delta2 * norm2
    return ((1 - delta2) * s + (1 + 2 * dot + norm2)[..., None] * delta) / denominator[..., None]


def dcs_map(s, parameters, xp=np):
    """Unit ball -> R^d, using the beta=2 DCS family from the experiment."""
    mu, radius, delta = parameters['mu'], parameters['radius'], parameters['delta']
    d = mu.shape[-1]
    norm2, delta2 = xp.sum(s * s, axis=-1), xp.sum(delta * delta)
    denominator = 1 + 2 * xp.sum(s * delta, axis=-1) + delta2 * norm2
    gap = (1 - delta2) * (1 - norm2) / denominator
    x = mu + radius * mobius(s, delta, xp) / xp.sqrt(gap[..., None])
    log_jacobian = d * xp.log(radius) + d * xp.log1p(-delta2) - d * xp.log(denominator) - (1 + d / 2) * xp.log(gap)
    return x, log_jacobian


def dcs_inverse(x, parameters, xp=np):
    u = (x - parameters['mu']) / parameters['radius']
    return mobius(u / xp.sqrt(1 + xp.sum(u * u, axis=-1, keepdims=True)), -parameters['delta'], xp)


def map_to_euclidean(z, method, parameters, latitude=1.1, xp=np):
    if method == 'stereographic':
        return stereographic_map(z, parameters, xp)
    if method == 'scs':
        return scs_map(z, parameters, latitude, xp)
    if method == 'dcs':
        return dcs_map(z, parameters, xp)
    if method == 'euclidean':
        return z, xp.zeros(z.shape[:-1])
    raise ValueError(f'Unknown method: {method}')


def to_native(x, method, parameters=None, latitude=1.1):
    """Convert an application state to the state used by its proposal."""
    x = np.asarray(x, dtype=float)
    if method == 'stereographic':
        return stereographic_inverse(x, parameters)
    if method == 'scs':
        return scs_inverse(x, parameters, latitude)
    if method == 'dcs':
        return dcs_inverse(x, parameters)
    if method == 'euclidean':
        return x.copy()
    raise ValueError(f'Unknown method: {method}')


def from_native(z, method, parameters=None, latitude=1.1):
    return map_to_euclidean(z, method, parameters, latitude)[0]


def native_log_pdf(z, log_pdf, method, parameters=None, latitude=1.1):
    """Density in proposal coordinates, including the transformation Jacobian."""
    if method == 'scs' and z[-1] >= latitude - 1:
        return -np.inf                         # Reject dark side, without stepping out.
    if method == 'dcs' and z @ z >= 1:
        return -np.inf                         # Reject outside the unit ball.
    if method == 'stereographic' and z[-1] >= 1:
        return -np.inf
    x, log_jacobian = map_to_euclidean(z, method, parameters, latitude)
    value = float(log_pdf(x) + log_jacobian)
    if np.isnan(value) or np.isposinf(value):
        raise FloatingPointError('log_pdf + log Jacobian must be finite or -inf.')
    return value

