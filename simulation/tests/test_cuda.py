"""Run without hardware: NUMBA_ENABLE_CUDASIM=1 uv run python -m unittest discover -s tests -p test_cuda.py."""

import os
import unittest
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import numpy as np

from simulation import (
    SimulationExplorer,
    benchmark,
    compare_sir,
    create_simulation,
    run,
    run_timed,
)

try:
    from numba import cuda
    from numba.cuda.random import (
        create_xoroshiro128p_states,
        xoroshiro128p_uniform_float64,
    )
    from simulation.kernels.gpu_numba import update

    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        raise
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:

    @cuda.jit
    def draw(states, values):
        i = cuda.grid(1)
        if i < values.size:
            values[i] = xoroshiro128p_uniform_float64(states, i)


@unittest.skipUnless(CUDA_AVAILABLE, "Requires CUDA or NUMBA_ENABLE_CUDASIM=1")
class CudaTests(unittest.TestCase):
    def model(self, **kwargs):
        return create_simulation((3, 5), backend="cuda", seed=42, **kwargs)

    def test_endpoints_match_cpu_and_keep_synchronous_periodic_update(self):
        for p, q in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
            config = {"infection_probability": p, "recover_probability": q}
            gpu = self.model(**config)
            cpu = create_simulation((3, 5), seed=42, **config)
            np.testing.assert_array_equal(run(gpu, 3).grids, run(cpu, 3).grids)

    def test_kernel_against_known_random_draws(self):
        grid = np.array([[2, 1, 3], [1, 1, 2]], dtype=np.uint8)
        p, q = 0.25, 0.7
        states = create_xoroshiro128p_states(grid.size, seed=77)
        # Draw from a copy with the same GPU generator, independently from the
        # update kernel. Compare actual cells, including wrapped/duplicate edges.
        copied = cuda.to_device(states.copy_to_host())
        values = cuda.device_array(grid.size, dtype=np.float64)
        draw[1, 32](copied, values)
        uniforms = values.copy_to_host()
        expected = grid.copy()
        for row in range(2):
            for col in range(3):
                u = uniforms[row * 3 + col]
                if grid[row, col] == 1:
                    k = sum(
                        grid[(row + dr) % 2, (col + dc) % 3] == 2
                        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1))
                    )
                    expected[row, col] = 2 if u < 1 - (1 - p) ** k else 1
                elif grid[row, col] == 2:
                    expected[row, col] = 3 if u < q else 2
        source = cuda.to_device(grid)
        target = cuda.to_device(np.full(grid.shape, 255, dtype=np.uint8))
        update[(1, 1), (16, 16)](source, target, p, q, states)
        np.testing.assert_array_equal(target.copy_to_host(), expected)

    def test_batch_clone_and_reset_reproducibility(self):
        model = self.model(infection_probability=0.35, recover_probability=0.2)
        e = SimulationExplorer(model, steps=5)
        self.addCleanup(e.close)
        first = run(model, 3)
        # Reset from an already advanced GPU model must restore device RNG,
        # not merely reseed it from the host generator.
        advanced_editor = SimulationExplorer(model, steps=8)
        self.addCleanup(advanced_editor.close)
        continued = deepcopy(model)
        checkpoint = model.get_random_state()
        model.advance(2)
        continued.simulation_step()
        continued.simulation_step()
        np.testing.assert_array_equal(model.grid, continued.grid)
        expected_continuation = model.grid.copy()
        advanced_editor.reset()
        model.advance(2)
        np.testing.assert_array_equal(model.grid, expected_continuation)
        saved_grid = advanced_editor._initial_grid.copy()
        model.grid = saved_grid
        model.set_random_state(checkpoint)
        model.advance(2)
        np.testing.assert_array_equal(model.grid, expected_continuation)
        e.reset()
        np.testing.assert_array_equal(first.grids, run(model, 3).grids)
        # A manually edited host grid is uploaded before the next batch.
        model.grid[:] = 3
        model.advance(2)
        self.assertTrue((model.grid == 3).all())

    def test_existing_apis_and_nonmutating_benchmark(self):
        model = self.model()
        before = model.grid.copy()
        state = deepcopy(model.rng.bit_generator.state)
        report = benchmark(model, steps=2, repeats=1)
        self.assertGreater(report.median_seconds, 0)
        np.testing.assert_array_equal(before, model.grid)
        self.assertEqual(state, model.rng.bit_generator.state)
        self.assertIsNone(model._cuda_runtime)
        timed = run_timed(model, 2)
        self.assertEqual(timed.result.grids.shape, (3, 3, 5))
        result = compare_sir(model, steps=2, repeats=2)
        np.testing.assert_array_equal(result.samples.sum(axis=2), 15)


if __name__ == "__main__":
    unittest.main()
