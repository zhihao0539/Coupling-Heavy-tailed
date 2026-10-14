"""Numerical checks: python -m unittest generic_algorithms.test_algorithms -v."""
import unittest
import numpy as np
from scipy.stats import ks_2samp, norm
from .algorithms import coupled_proposals, coupled_sampler
from .transformations import (default_parameters, to_native, from_native,
                              map_to_euclidean, native_log_pdf)


class CouplingChecks(unittest.TestCase):
    def test_round_trips_and_support(self):
        rng = np.random.default_rng(1)
        for d in [1, 3, 14, 100]:
            x = rng.normal(size=(20, d)) * 2
            for method in ['stereographic', 'scs', 'dcs']:
                p = default_parameters(d, method, radius=1.7)
                p['mu'] = rng.normal(size=d) * .2
                if method != 'stereographic':
                    offset = rng.normal(size=d)
                    p['horizontal_observer' if method == 'scs' else 'delta'] = .2 * offset / np.linalg.norm(offset)
                z = to_native(x, method, p)
                np.testing.assert_allclose(from_native(z, method, p), x, rtol=2e-11, atol=2e-11)
                if method == 'dcs':
                    self.assertTrue(np.all(np.sum(z * z, axis=1) < 1))
                else:
                    np.testing.assert_allclose(np.sum(z * z, axis=1), 1., atol=1e-12)
                    if method == 'scs':
                        self.assertTrue(np.all(z[:, -1] < .1))

    def test_jacobians_by_finite_differences(self):
        rng = np.random.default_rng(2)
        for method in ['stereographic', 'scs', 'dcs']:
            p = default_parameters(3, method, radius=1.3)
            p['mu'] = np.array([.2, -.4, .1])
            if method != 'stereographic':
                p['horizontal_observer' if method == 'scs' else 'delta'] = np.array([.1, -.15, .05])
            z = to_native(rng.normal(size=3), method, p)
            basis = np.eye(3) if method == 'dcs' else np.linalg.svd(z[None, :], full_matrices=True)[2][1:].T
            jac = []
            for tangent in basis.T:
                plus, minus = z + 1e-6 * tangent, z - 1e-6 * tangent
                if method != 'dcs':
                    plus, minus = plus / np.linalg.norm(plus), minus / np.linalg.norm(minus)
                jac.append((from_native(plus, method, p) - from_native(minus, method, p)) / 2e-6)
            numeric = np.linalg.slogdet(np.asarray(jac).T)[1]
            analytic = map_to_euclidean(z, method, p)[1]
            self.assertAlmostEqual(numeric, analytic, places=7)

    def test_proposal_marginals(self):
        rng = np.random.default_rng(3)
        n, h = 12000, .6
        for proposal in ['gaussian', 'ball', 'sphere']:
            a, b = np.array([-.3, 0., 0.]), np.array([.3, 0., 0.])
            if proposal == 'sphere':
                a = np.array([0., 0., 1.])
                b = np.array([.6, 0., .8])
            pairs = np.asarray([coupled_proposals(a, b, rng.normal(size=3), rng.random(), h,
                                                  proposal, rng.random()) for _ in range(n)])
            for index, center in enumerate([a, b]):
                noise = rng.normal(size=(n, 3))
                if proposal == 'sphere':
                    reference = center + h * (noise - (noise @ center)[:, None] * center)
                    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
                    self.assertTrue(np.all(pairs[:, index] @ center > 0))
                elif proposal == 'ball':
                    reference = center + h * noise / np.linalg.norm(noise, axis=1, keepdims=True) * rng.random((n, 1))**(1 / 3)
                    self.assertTrue(np.all(np.linalg.norm(pairs[:, index] - center, axis=1) <= h * (1 + 1e-12)))
                else:
                    reference = center + h * noise
                for coordinate in range(3):
                    self.assertGreater(ks_2samp(pairs[:, index, coordinate], reference[:, coordinate]).pvalue, 1e-5)
            if proposal == 'gaussian':
                expected = 2 * norm.cdf(-np.linalg.norm(a - b) / (2 * h))
                observed = np.mean(np.all(pairs[:, 0] == pairs[:, 1], axis=1))
                self.assertLess(abs(observed - expected), 6 * np.sqrt(expected * (1 - expected) / n))

    def test_boundary_rejection_and_invalid_targets(self):
        def should_not_be_called(x):
            raise AssertionError('Target evaluated outside the map domain')
        self.assertEqual(native_log_pdf(np.array([0., 0., 1.]), should_not_be_called, 'scs'), -np.inf)
        self.assertEqual(native_log_pdf(np.array([2., 0.]), should_not_be_called, 'dcs'), -np.inf)
        with self.assertRaises(FloatingPointError):
            coupled_sampler(np.zeros(2), np.ones(2), lambda x: np.nan, .1, 2, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            coupled_sampler(np.zeros(2), np.ones(2), lambda x: -np.inf, .1, 2, np.random.default_rng(1))

    def test_meeting_is_exact_and_sticky(self):
        target = lambda x: -2.5 * np.log1p(np.sum(x * x) / 2)
        for method in ['euclidean', 'stereographic', 'scs', 'dcs']:
            x = np.zeros(3)
            result = coupled_sampler(x, x, target, .15, 100, np.random.default_rng(4),
                                     method, stop_at_meeting=False, store_chains=True)
            self.assertEqual(result['meeting_time'], 0)
            self.assertEqual(result['iterations'], 100)
            np.testing.assert_array_equal(result['chain1'], result['chain2'])
        # A different initial pair also remains identical after its first meeting.
        result = coupled_sampler(np.zeros(3), np.ones(3) * .1, target, .5, 300,
                                 np.random.default_rng(6), stop_at_meeting=False, store_chains=True)
        self.assertTrue(np.isfinite(result['meeting_time']))
        k = int(result['meeting_time'])
        np.testing.assert_array_equal(result['chain1'][k:], result['chain2'][k:])


class VIChecks(unittest.TestCase):
    def test_known_exact_reference_families(self):
        import jax
        import jax.numpy as jnp
        from .vi import pack_parameters, unpack_parameters, sample_reference
        for method, df, latitude in [('scs', 1., 1.), ('dcs', 2., 1.1), ('stereographic', 3., 1.1)]:
            p = default_parameters(3, method, radius=np.sqrt(df))
            theta = pack_parameters(p, method, latitude)
            z = sample_reference(jax.random.key(7), 1000, 3, method, latitude)
            x, jac = map_to_euclidean(z, method, unpack_parameters(theta, 3, method, latitude), latitude, xp=jnp)
            log_target = -(3 + df) / 2 * jnp.log1p(jnp.sum(x * x, axis=1) / df)
            # With these exact Student-t families, transformed density is constant.
            self.assertLess(float(jnp.std(log_target + jac)), 1e-10)

    def test_reusable_vi_and_gradients(self):
        import jax
        import jax.numpy as jnp
        from .vi import pack_parameters, sample_reference, reverse_kl, fit_vi
        for method, d in [('scs', 2), ('dcs', 5), ('stereographic', 3)]:
            df = 2. if method != 'stereographic' else 3.
            target = lambda x: -(d + df) / 2 * jnp.log1p(jnp.sum(x * x) / df)
            initial = default_parameters(d, method, radius=.6)
            initial['mu'] = np.full(d, .5)
            theta = pack_parameters(initial, method)
            z = sample_reference(jax.random.key(11), 256, d, method)
            objective = lambda t: reverse_kl(t, z, target, d, method)
            direction = jnp.arange(1., len(theta) + 1)
            direction /= jnp.linalg.norm(direction)
            analytic = float(jax.grad(objective)(theta) @ direction)
            numeric = float((objective(theta + 1e-5 * direction) - objective(theta - 1e-5 * direction)) / 2e-5)
            self.assertAlmostEqual(analytic, numeric, places=6)
            p, info, history = fit_vi(target, d, method, initial_parameters=initial,
                                     steps=160, batch_size=256, holdout_size=2000,
                                     change_after=120, average_last=40, seed=12)
            self.assertLess(min(info['candidate_holdout_losses']), info['initial_holdout_loss'] - .1)
            self.assertEqual(history['theta'].shape[0], 160)
            # The fit can be used directly by the coupling function.
            result = coupled_sampler(np.zeros(d), np.ones(d), target, .1, 10,
                                     np.random.default_rng(2), method, p)
            self.assertTrue(np.isfinite(result['final_x']).all())


if __name__ == '__main__':
    unittest.main()
