"""Check the continuous SIR model against invariants and an exact solution."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulation.core.SIR import SIR


class SIRTests(unittest.TestCase):
    def test_recovery_only_matches_exact_solution(self):
        model = SIR((0.8, 0.2, 0.0), beta=0, gamma=0.1)
        solution = model.solve((0, 40), points=101)
        infected = 0.2 * np.exp(-0.1 * solution.t)
        np.testing.assert_allclose(solution.y[0], 0.8, atol=1e-12)
        np.testing.assert_allclose(solution.y[1], infected, atol=1e-8)
        np.testing.assert_allclose(solution.y[2], 0.2 - infected, atol=1e-8)
        self.assertEqual((model.S, model.I, model.R), (0.8, 0.2, 0.0))

    def test_conservation_monotonicity_and_plot(self):
        model = SIR((0.99, 0.01, 0), beta=0.3, gamma=0.1)
        solution = model.solve()
        np.testing.assert_allclose(solution.y.sum(axis=0), 1, atol=1e-12)
        self.assertTrue(np.all(np.diff(solution.y[0]) <= 1e-10))
        self.assertTrue(np.all(np.diff(solution.y[2]) >= -1e-10))
        self.assertGreater(solution.y[1].max(), 0.01)
        figure, ax = model.plot(solution)
        self.assertEqual(len(ax.lines), 3)
        for line, values in zip(ax.lines, solution.y):
            np.testing.assert_array_equal(line.get_xdata(), solution.t)
            np.testing.assert_array_equal(line.get_ydata(), values)
        plt.close(figure)

    def test_rejects_invalid_inputs(self):
        for population in ((1, 1, 0), (-0.1, 0.1, 1), (0.9, 0.1), (np.nan, 0, 1)):
            with self.subTest(population=population), self.assertRaises(ValueError):
                SIR(population, beta=0.3, gamma=0.1)
        for beta, gamma in ((-1, 0.1), (0.3, -1), (np.inf, 0.1)):
            with self.subTest(beta=beta, gamma=gamma), self.assertRaises(ValueError):
                SIR((0.99, 0.01, 0), beta=beta, gamma=gamma)
        model = SIR((0.99, 0.01, 0), beta=0.3, gamma=0.1)
        with self.assertRaises(ValueError):
            model.solve((10, 0))
        with self.assertRaises(ValueError):
            model.solve(points=1)


if __name__ == "__main__":
    unittest.main()
