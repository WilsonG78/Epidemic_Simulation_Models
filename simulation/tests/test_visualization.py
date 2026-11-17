import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation import create_simulation, run


class VisualizationTests(unittest.TestCase):
    def test_reproducible_history_and_conservation(self):
        a = run(create_simulation((8, 10), initial_infected=4, seed=5), 12)
        b = run(create_simulation((8, 10), initial_infected=4, seed=5), 12)
        np.testing.assert_array_equal(a.grids, b.grids)
        np.testing.assert_array_equal(a.counts.sum(axis=1), 80)
        self.assertEqual(a.grids.shape, (13, 8, 10))
        self.assertEqual(a.counts[0].tolist(), [76, 4, 0])
        self.assertFalse(a.grids.flags.writeable)

    def test_replay_does_not_mutate_simulation(self):
        simulation = create_simulation((4, 4), seed=1)
        result = run(simulation, 3)
        final_grid = simulation.grid.copy()
        animation = result.animate()
        for frame in (3, 0, 2, 0):
            animation._func(frame)
        np.testing.assert_array_equal(simulation.grid, final_grid)
        animation._draw_was_started = True
        plt.close(animation._fig)

    def test_zero_steps_and_validation(self):
        result = run(create_simulation((2, 3), initial_infected=0), 0)
        figure, _ = result.plot()
        plt.close(figure)
        self.assertEqual(result.counts.tolist(), [[6, 0, 0]])
        with self.assertRaises(ValueError):
            run(create_simulation(), -1)
        with self.assertRaises(ValueError):
            create_simulation((2, 2), initial_infected=5)
        with self.assertRaises(IndexError):
            result.plot(1)


if __name__ == "__main__":
    unittest.main()
