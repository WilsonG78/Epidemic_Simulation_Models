import unittest
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import numpy as np

from simulation import SimulationExplorer, benchmark, create_simulation, run, run_timed


class TimingTests(unittest.TestCase):
    def test_benchmark_preserves_model_and_rng(self):
        model = create_simulation((5, 7), seed=42)
        grid = model.grid.copy()
        rng = deepcopy(model.rng.bit_generator.state)
        result = benchmark(model, steps=5, repeats=2)
        np.testing.assert_array_equal(grid, model.grid)
        self.assertEqual(rng, model.rng.bit_generator.state)
        self.assertEqual(len(result.samples_seconds), 2)
        self.assertGreater(result.milliseconds_per_step, 0)
        self.assertGreater(result.steps_per_second, 0)

    def test_timed_run_matches_run(self):
        for steps in (0, 8):
            model = create_simulation((6, 9), seed=10)
            expected = run(deepcopy(model), steps)
            report = run_timed(model, steps)
            np.testing.assert_array_equal(expected.grids, report.result.grids)
            np.testing.assert_array_equal(expected.counts, report.result.counts)
            self.assertGreaterEqual(
                report.total_seconds, report.compute_seconds + report.recording_seconds
            )
            if steps == 0:
                self.assertEqual(report.milliseconds_per_step, 0)
        for kwargs in ({"steps": 0}, {"repeats": -1}):
            with self.assertRaises(ValueError):
                benchmark(model, **kwargs)

    def test_frame_batch_records_every_step_and_respects_limit(self):
        model = create_simulation((5, 7), seed=42)
        expected = run(deepcopy(model), 7)
        e = SimulationExplorer(model, steps=7, steps_per_frame=4)
        self.addCleanup(e.close)
        e.start()
        e._tick()
        self.assertEqual(e.current_step, 4)
        e.pause()
        e._tick()
        self.assertEqual(e.current_step, 4)
        e.step()
        self.assertEqual(e.current_step, 5)
        e.start()
        e._tick()
        self.assertEqual(e.current_step, 7)
        self.assertFalse(e.running)
        np.testing.assert_array_equal(e.result.grids, expected.grids)
        e.reset()
        self.assertEqual(e.current_step, 0)
