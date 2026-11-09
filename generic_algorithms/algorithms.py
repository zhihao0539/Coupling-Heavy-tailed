"""Method-specific coupled samplers and maximal-reflection proposals."""

import numpy as np

from .transformations import (
    validate_stereographic_parameters, validate_scs_parameters,
    validate_dcs_parameters, validate_dcs_matrix_parameters, validate_dcs_beta,
    stereographic_map, stereographic_inverse, scs_map, scs_inverse,
    dcs_mobius_map, dcs_mobius_inverse, dcs_matrix_map, dcs_matrix_inverse,
)


def coupled_euclidean(x, y, log_pdf, h, n_steps, rng, *,
        stop_at_meeting=True, store_chains=False):
    """Euclidean Gaussian random walk; h is the coordinate standard deviation."""
    target = lambda z: _checked_log_density(log_pdf(z))
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=np.copy, forward=np.copy, propose=_gaussian_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_stereographic(x, y, log_pdf, h, n_steps, rng, *,
        mu, preconditioning_matrix,
        stop_at_meeting=True, store_chains=False):
    """Stereographic coupling with location mu and lower-triangular scale matrix."""
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    validate_stereographic_parameters(parameters, np.size(x))
    def inverse(z):
        return stereographic_inverse(z, parameters)
    forward = lambda z: stereographic_map(z, parameters)[0]
    target = lambda z: _stereographic_log_density(z, log_pdf, parameters)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_sphere_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_scs(x, y, log_pdf, h, n_steps, rng, *,
        mu, preconditioning_matrix, horizontal_observer, latitude=1.1,
        stop_at_meeting=True, store_chains=False):
    """SCS coupling with an explicit cap latitude and horizontal observer."""
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['horizontal_observer'] = np.asarray(horizontal_observer, dtype=float)
    validate_scs_parameters(parameters, np.size(x), latitude)
    def inverse(z):
        return scs_inverse(z, parameters, latitude)
    forward = lambda z: scs_map(z, parameters, latitude)[0]
    target = lambda z: _scs_log_density(z, log_pdf, parameters, latitude)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_sphere_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_dcs_ball(x, y, log_pdf, h, n_steps, rng, *,
        mu, radius, delta, beta=2.0,
        stop_at_meeting=True, store_chains=False):
    """DCS ball walk with scalar radius and Mobius offset delta; h = gamma/R."""
    validate_dcs_beta(beta)
    parameters = dict(mu=np.asarray(mu, dtype=float), radius=float(radius),
                      delta=np.asarray(delta, dtype=float), beta=float(beta))
    validate_dcs_parameters(parameters, np.size(x))
    def inverse(z):
        return dcs_mobius_inverse(z, parameters)
    forward = lambda z: dcs_mobius_map(z, parameters)[0]
    target = lambda z: _dcs_log_density(z, log_pdf, parameters)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_ball_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_dcs_random_walk(x, y, log_pdf, h, n_steps, rng, *,
        mu, radius, delta, beta=2.0,
        stop_at_meeting=True, store_chains=False):
    """DCS Gaussian walk with scalar radius and Mobius offset; h = sigma/R."""
    validate_dcs_beta(beta)
    parameters = dict(mu=np.asarray(mu, dtype=float), radius=float(radius),
                      delta=np.asarray(delta, dtype=float), beta=float(beta))
    validate_dcs_parameters(parameters, np.size(x))
    def inverse(z):
        return dcs_mobius_inverse(z, parameters)
    forward = lambda z: dcs_mobius_map(z, parameters)[0]
    target = lambda z: _dcs_log_density(z, log_pdf, parameters)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_gaussian_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_dcs_matrix_ball(x, y, log_pdf, h, n_steps, rng, *,
        mu, preconditioning_matrix, beta=2.0,
        stop_at_meeting=True, store_chains=False):
    """DCS ball walk with a matrix radial map; h is the native ball-step radius."""
    validate_dcs_beta(beta)
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['beta'] = float(beta)
    validate_dcs_matrix_parameters(parameters, np.size(x))
    def inverse(z):
        return dcs_matrix_inverse(z, parameters)
    forward = lambda z: dcs_matrix_map(z, parameters)[0]
    target = lambda z: _dcs_matrix_log_density(z, log_pdf, parameters)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_ball_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def coupled_dcs_matrix_random_walk(x, y, log_pdf, h, n_steps, rng, *,
        mu, preconditioning_matrix, beta=2.0,
        stop_at_meeting=True, store_chains=False):
    """DCS Gaussian walk with a matrix radial map; h is the native standard deviation."""
    validate_dcs_beta(beta)
    parameters = _matrix_parameters(mu, preconditioning_matrix)
    parameters['beta'] = float(beta)
    validate_dcs_matrix_parameters(parameters, np.size(x))
    def inverse(z):
        return dcs_matrix_inverse(z, parameters)
    forward = lambda z: dcs_matrix_map(z, parameters)[0]
    target = lambda z: _dcs_matrix_log_density(z, log_pdf, parameters)
    return _run_coupling(x, y, target, h, n_steps, rng,
        inverse=inverse, forward=forward, propose=_gaussian_proposal,
        stop_at_meeting=stop_at_meeting, store_chains=store_chains)


def sphere_log_proposal(z, p, h):
    """Proposal log density up to a constant; z and p must be unit vectors."""
    c = min(float(z @ p), 1.)
    if c <= 0:
        return -np.inf
    return -len(z) * np.log(c) - (1 - c) * (1 + c) / (2 * h**2 * c**2)


def coupled_gaussian_proposals(a, b, noise, uniform, h):
    """Maximal-reflection coupling of Gaussian random-walk proposals."""
    move = h * noise
    p = a + move
    delta = (a - b) / h
    log_ratio = -noise @ delta - .5 * (delta @ delta)
    common = uniform == 0 or np.log(uniform) <= min(0., log_ratio)
    if np.array_equal(a, b) or common:
        return p, p.copy()
    direction = (a - b) / np.linalg.norm(a - b)
    return p, b + move - 2 * direction * (direction @ move)


def coupled_sphere_proposals(a, b, noise, uniform, h):
    """Maximal-reflection coupling on the sphere, used by SCS and stereographic."""
    tangent = noise - a * (a @ noise) / (a @ a)
    p = a + h * tangent
    p /= np.linalg.norm(p)
    log_ratio = sphere_log_proposal(b, p, h) - sphere_log_proposal(a, p, h)
    common = np.isfinite(log_ratio) and (uniform == 0 or np.log(uniform) <= min(0., log_ratio))
    if np.array_equal(a, b) or common:
        return p, p.copy()
    direction = (a - b) / np.linalg.norm(a - b)
    q = p - 2 * direction * (direction @ p)
    q /= np.linalg.norm(q)
    return p, q


def coupled_ball_proposals(a, b, noise, radius_uniform, h):
    """Maximal-reflection coupling of uniform-ball proposals, used by DCS."""
    if radius_uniform is None:
        raise ValueError('A ball proposal needs radius_uniform.')
    move = h * noise / np.linalg.norm(noise) * radius_uniform**(1 / len(a))
    p = a + move
    common = np.sum((p - b)**2) <= h**2
    if np.array_equal(a, b) or common:
        return p, p.copy()
    direction = (a - b) / np.linalg.norm(a - b)
    return p, b + move - 2 * direction * (direction @ move)


def _matrix_parameters(mu, preconditioning_matrix):
    return dict(mu=np.asarray(mu, dtype=float),
                preconditioning_matrix=np.asarray(preconditioning_matrix, dtype=float))


def _checked_log_density(value):
    value = float(value)
    if np.isnan(value) or np.isposinf(value):
        raise FloatingPointError('log_pdf + log Jacobian must be finite or -inf.')
    return value


def _stereographic_log_density(z, log_pdf, parameters):
    if z[-1] >= 1:
        return -np.inf
    x, log_jacobian = stereographic_map(z, parameters)
    return _checked_log_density(log_pdf(x) + log_jacobian)


def _scs_log_density(z, log_pdf, parameters, latitude):
    if z[-1] >= latitude - 1:
        return -np.inf
    x, log_jacobian = scs_map(z, parameters, latitude)
    return _checked_log_density(log_pdf(x) + log_jacobian)


def _dcs_log_density(z, log_pdf, parameters):
    if z @ z >= 1:
        return -np.inf
    x, log_jacobian = dcs_mobius_map(z, parameters)
    return _checked_log_density(log_pdf(x) + log_jacobian)


def _dcs_matrix_log_density(z, log_pdf, parameters):
    if z @ z >= 1:
        return -np.inf
    x, log_jacobian = dcs_matrix_map(z, parameters)
    return _checked_log_density(log_pdf(x) + log_jacobian)


def _gaussian_proposal(a, b, h, rng):
    noise, uniform = rng.normal(size=len(a)), rng.random()
    rng.random()  # Retain the old radius draw to preserve seeded results.
    return coupled_gaussian_proposals(a, b, noise, uniform, h)


def _sphere_proposal(a, b, h, rng):
    noise, uniform = rng.normal(size=len(a)), rng.random()
    rng.random()  # Retain the old radius draw to preserve seeded results.
    return coupled_sphere_proposals(a, b, noise, uniform, h)


def _ball_proposal(a, b, h, rng):
    noise = rng.normal(size=len(a))
    rng.random()  # Retain the old common-proposal uniform draw.
    return coupled_ball_proposals(a, b, noise, rng.random(), h)


def _validate_chain_inputs(x, y, h, n_steps):
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1 or not np.isfinite(h) or h <= 0:
        raise ValueError('n_steps must be a positive integer and h finite and positive.')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.size == 0 or y.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValueError('Initial states must be finite vectors of equal positive dimension.')
    return x, y


def _run_coupling(x, y, target, h, n_steps, rng, *, inverse, forward, propose,
                  stop_at_meeting, store_chains):
    """Shared bookkeeping only; the caller supplies its maps, density and proposal."""
    x, y = _validate_chain_inputs(x, y, h, n_steps)
    a, b = inverse(x), inverse(y)
    log_a, log_b = target(a), target(b)
    if not np.isfinite([log_a, log_b]).all():
        raise ValueError('Both initial states must have finite transformed log density.')
    accepted_a = accepted_b = iterations = 0
    meeting_time = 0 if np.array_equal(a, b) else np.inf
    chain_a, chain_b = ([x.copy()], [y.copy()]) if store_chains else (None, None)
    for iteration in range(1, n_steps + 1):
        if stop_at_meeting and np.isfinite(meeting_time):
            break
        a, b, log_a, log_b, accept_a, accept_b = _coupled_metropolis_step(
            a, b, log_a, log_b, target, h, rng, propose)
        accepted_a += accept_a
        accepted_b += accept_b
        iterations = iteration
        if not np.isfinite(meeting_time) and np.array_equal(a, b):
            meeting_time = iteration
        if store_chains:
            chain_a.append(np.array(forward(a)))
            chain_b.append(np.array(forward(b)))
    result = dict(meeting_time=meeting_time, iterations=iterations,
                  acceptance_rate_chain1=accepted_a / iterations if iterations else np.nan,
                  acceptance_rate_chain2=accepted_b / iterations if iterations else np.nan,
                  final_x=forward(a), final_y=forward(b))
    if store_chains:
        result.update(chain1=np.asarray(chain_a), chain2=np.asarray(chain_b))
    return result


def _coupled_metropolis_step(a, b, log_a, log_b, target, h, rng, propose):
    """One proposal and one shared acceptance uniform, with no method dispatch."""
    p, q = propose(a, b, h, rng)
    log_p = target(p)
    log_q = log_p if np.array_equal(p, q) else target(q)
    u = rng.random()
    log_u = np.log(u) if u > 0 else -np.inf
    accept_a = np.isfinite(log_p) and log_u <= min(0., log_p - log_a)
    accept_b = np.isfinite(log_q) and log_u <= min(0., log_q - log_b)
    if accept_a:
        a, log_a = p, log_p
    if accept_b:
        b, log_b = q, log_q
    return a, b, log_a, log_b, int(accept_a), int(accept_b)


def sample_chain(sampler, x, log_pdf, h, n_steps, rng, **map_parameters):
    """Run one chain using an explicit sampler, e.g. sample_chain(coupled_scs, ...)."""
    result = sampler(x, x, log_pdf, h, n_steps, rng, **map_parameters,
                     stop_at_meeting=False, store_chains=True)
    return result['chain1'], result['acceptance_rate_chain1']
