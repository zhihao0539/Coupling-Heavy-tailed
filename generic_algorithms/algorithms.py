"""Maximal-reflection proposals and a shared Metropolis coupling loop."""
import numpy as np
from .transformations import (METHODS, default_parameters, validate_parameters,
                              to_native, from_native, native_log_pdf)


def sphere_log_proposal(z, p, h):
    """Proposal log density up to a constant; z and p must be unit vectors."""
    c = min(float(z @ p), 1.)
    if c <= 0:
        return -np.inf
    return -len(z) * np.log(c) - (1 - c) * (1 + c) / (2 * h**2 * c**2)


def coupled_proposals(a, b, noise, uniform, h, proposal='gaussian', radius_uniform=None):
    """Couple Gaussian, projected-Gaussian sphere, or uniform-ball proposals.

    noise is a standard normal vector; uniform and radius_uniform are U[0,1).
    Inputs are native states. Return exactly equal arrays on a common proposal.
    """
    if proposal == 'sphere':
        tangent = noise - a * (a @ noise) / (a @ a)
        p = a + h * tangent
        p /= np.linalg.norm(p)
        log_ratio = sphere_log_proposal(b, p, h) - sphere_log_proposal(a, p, h)
        common = np.isfinite(log_ratio) and (uniform == 0 or np.log(uniform) <= min(0., log_ratio))
    elif proposal == 'gaussian':
        move = h * noise
        p = a + move
        delta = (a - b) / h
        log_ratio = -noise @ delta - .5 * (delta @ delta)
        common = uniform == 0 or np.log(uniform) <= min(0., log_ratio)
    elif proposal == 'ball':
        if radius_uniform is None:
            raise ValueError('A ball proposal needs radius_uniform.')
        move = h * noise / np.linalg.norm(noise) * radius_uniform**(1 / len(a))
        p = a + move
        common = np.sum((p - b)**2) <= h**2
    else:
        raise ValueError('proposal must be gaussian, sphere, or ball.')
    if np.array_equal(a, b) or common:
        return p, p.copy()
    direction = (a - b) / np.linalg.norm(a - b)
    if proposal == 'sphere':
        q = p - 2 * direction * (direction @ p)
        q /= np.linalg.norm(q)
    else:
        q = b + move - 2 * direction * (direction @ move)
    return p, q


def coupled_sampler(x, y, log_pdf, h, n_steps, rng, method='euclidean',
                    parameters=None, latitude=1.1, stop_at_meeting=True,
                    store_chains=False):
    """Run two coupled chains, supplying only a target and numerical settings.

    x and y are Euclidean initial states. The optional chains include the
    initial states. Meeting means exact equality;
    identical initial states have meeting time 0. Unmet pairs return np.inf.
    Acceptance is measured over the updates actually performed.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1 or not np.isfinite(h) or h <= 0:
        raise ValueError('n_steps must be a positive integer and h finite and positive.')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.size == 0 or y.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValueError('Initial states must be finite vectors of equal positive dimension.')
    if parameters is None:
        parameters = default_parameters(len(x), method)
    else:
        parameters = {k: np.asarray(v, dtype=float) if k != 'radius' else float(v) for k, v in parameters.items()}
    validate_parameters(parameters, len(x), method, latitude)
    a, b = to_native(x, method, parameters, latitude), to_native(y, method, parameters, latitude)
    target = lambda z: native_log_pdf(z, log_pdf, method, parameters, latitude)
    log_a, log_b = target(a), target(b)
    if not np.isfinite([log_a, log_b]).all():
        raise ValueError('Both initial states must have finite transformed log density.')
    proposal = 'sphere' if method in ('stereographic', 'scs') else 'ball' if method == 'dcs' else 'gaussian'
    accepted_a = accepted_b = iterations = 0
    meeting_time = 0 if np.array_equal(a, b) else np.inf
    chain_a, chain_b = ([x.copy()], [y.copy()]) if store_chains else (None, None)
    for iteration in range(1, n_steps + 1):
        if stop_at_meeting and np.isfinite(meeting_time):
            break
        p, q = coupled_proposals(a, b, rng.normal(size=len(a)), rng.random(), h,
                                 proposal, rng.random())
        log_p = target(p)
        log_q = log_p if np.array_equal(p, q) else target(q)
        u = rng.random()                         # Same MH uniform for both chains.
        log_u = np.log(u) if u > 0 else -np.inf
        if np.isfinite(log_p) and log_u <= min(0., log_p - log_a):
            a, log_a = p, log_p
            accepted_a += 1
        if np.isfinite(log_q) and log_u <= min(0., log_q - log_b):
            b, log_b = q, log_q
            accepted_b += 1
        iterations = iteration
        if not np.isfinite(meeting_time) and np.array_equal(a, b):
            meeting_time = iteration
        if store_chains:
            chain_a.append(np.array(from_native(a, method, parameters, latitude)))
            chain_b.append(np.array(from_native(b, method, parameters, latitude)))
    result = dict(meeting_time=meeting_time, iterations=iterations,
                  acceptance_rate_chain1=accepted_a / iterations if iterations else np.nan,
                  acceptance_rate_chain2=accepted_b / iterations if iterations else np.nan,
                  final_x=from_native(a, method, parameters, latitude),
                  final_y=from_native(b, method, parameters, latitude))
    if store_chains:
        result.update(chain1=np.asarray(chain_a), chain2=np.asarray(chain_b))
    return result


def sample_chain(x, log_pdf, h, n_steps, rng, method='euclidean', parameters=None, latitude=1.1):
    """Single-chain convenience wrapper using the same transition implementation."""
    result = coupled_sampler(x, x, log_pdf, h, n_steps, rng, method, parameters,
                             latitude, stop_at_meeting=False, store_chains=True)
    return result['chain1'], result['acceptance_rate_chain1']
