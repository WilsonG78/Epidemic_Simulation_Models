import unittest
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation import compare_sir, create_simulation, sir_final_susceptible, sir_ode


class SIRValidationTests(unittest.TestCase):
    def test_recovery_only_matches_exact_solution(self):
        counts = sir_ode([80, 20, 0], beta=0, gamma=0.2, steps=30)
        np.testing.assert_allclose(
            counts[:, 1], 20 * np.exp(-0.2 * np.arange(31)), atol=1e-8
        )
        np.testing.assert_allclose(counts.sum(axis=1), 100, atol=1e-10)
        np.testing.assert_allclose(counts[:, 0], 80)
        np.testing.assert_allclose(
            sir_ode([90, 0, 10], beta=2, gamma=0.1, steps=10),
            np.tile([90, 0, 10], (11, 1)),
        )

    def test_final_size_peak_invariant_and_convergence(self):
        initial = [990, 10, 0]
        coarse = sir_ode(initial, beta=0.4, gamma=0.1, steps=300, substeps=20)
        fine = sir_ode(initial, beta=0.4, gamma=0.1, steps=300, substeps=40)
        np.testing.assert_allclose(coarse, fine, atol=1e-5)
        final = sir_final_susceptible(initial, beta=0.4, gamma=0.1)
        self.assertAlmostEqual(final, fine[-1, 0], places=5)
        s = fine[:, 0] / 1000
        r = fine[:, 2] / 1000
        np.testing.assert_allclose(np.log(s / 0.99) + 4 * r, 0, atol=1e-8)
        peak = 1000 * (1 - 0.25 + 0.25 * np.log(0.25 / 0.99))
        self.assertLess(abs(fine[:, 1].max() - peak), 0.5)
        self.assertIsNone(sir_final_susceptible(initial, beta=0.4, gamma=0))
        self.assertEqual(sir_final_susceptible([990, 0, 10], beta=0.4, gamma=0.1), 990)

    def test_ensemble_preserves_input_and_sir_properties(self):
        model = create_simulation((8, 9), initial_infected=3, seed=8)
        original = deepcopy(model)
        report = compare_sir(model, steps=10, repeats=3, seed=10)
        other = compare_sir(model, steps=10, repeats=3, seed=10)
        np.testing.assert_array_equal(report.samples, other.samples)
        np.testing.assert_array_equal(model.grid, original.grid)
        self.assertEqual(
            model.rng.bit_generator.state, original.rng.bit_generator.state
        )
        np.testing.assert_array_equal(report.samples.sum(axis=2), 72)
        self.assertTrue(report.summary["spatial_S_nonincreasing"])
        self.assertTrue(report.summary["spatial_R_nondecreasing"])
        np.testing.assert_allclose(report.mean_field_counts.sum(axis=1), 72)
        fig, _ = report.plot()
        plt.close(fig)
        for options in ({"repeats": 0}, {"steps": -1}, {"beta": -1}):
            with self.assertRaises(ValueError):
                compare_sir(model, **options)
