import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backend_bases import MouseButton

from simulation import SimulationExplorer, create_simulation


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.explorer = SimulationExplorer(
            create_simulation((5, 7), initial_infected=0, seed=42), steps=3
        )
        self.addCleanup(self.explorer.close)

    def test_click_pause_limit_and_extend(self):
        e = self.explorer
        event = SimpleNamespace(
            inaxes=e.axes[0], xdata=4, ydata=2, button=MouseButton.LEFT
        )
        e._click(event)
        self.assertEqual(e.result.grids[0, 2, 4], 2)
        e.start()
        e._tick()
        self.assertEqual(e.current_step, 1)
        e.pause()
        e._tick()
        self.assertEqual(e.current_step, 1)
        e.step()
        self.assertFalse(e.running)
        e.start()
        e._tick()
        self.assertEqual(e.current_step, 3)
        self.assertFalse(e.running)
        e.step()
        self.assertEqual(e.current_step, 3)
        e.set_duration(4)
        e.start()
        e._tick()
        self.assertEqual(e.current_step, 4)

    def test_reset_reproduces_selected_outbreak(self):
        e = self.explorer
        e.set_infected(2, 3)
        for _ in range(3):
            e.step()
        first = e.result
        with self.assertRaises(RuntimeError):
            e.set_infected(0, 0)
        e.reset()
        self.assertEqual(e.result.counts[0].tolist(), [34, 1, 0])
        for _ in range(3):
            e.step()
        np.testing.assert_array_equal(first.grids, e.result.grids)
        e.reset()
        e.clear()
        self.assertEqual(e.result.counts[0].tolist(), [35, 0, 0])

    def test_speed_invalid_duration_and_close(self):
        e = self.explorer
        e.speed_slider.set_val(500)
        self.assertEqual(e.timer.interval, 500)
        e._set_duration("bad")
        self.assertEqual(e.max_steps, 3)
        e.set_duration(0)
        e.start()
        self.assertFalse(e.running)
        e.close()
        e.set_duration(10)
        e.start()
        e.step()
        self.assertEqual(e.current_step, 0)
