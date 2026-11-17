import unittest
from copy import deepcopy

import numpy as np
from simulation.kernels.cpu_numba import step_grid

from simulation import State, create_simulation, run


class NumbaTests(unittest.TestCase):
    def test_exact_python_equivalence(self):
        for shape in ((1, 1), (1, 7), (2, 2), (8, 11)):
            for infection, recovery in ((0.0, 0.0), (1.0, 1.0), (0.25, 0.05)):
                with self.subTest(shape=shape, p=infection, r=recovery):
                    config = {
                        "grid_shape": shape,
                        "initial_infected": 1,
                        "seed": 42,
                        "infection_probability": infection,
                        "recover_probability": recovery,
                    }
                    cpu = create_simulation(**config, backend="python")
                    jit = create_simulation(**config)
                    np.testing.assert_array_equal(
                        run(cpu, 20).grids, run(jit, 20).grids
                    )
                    self.assertEqual(
                        cpu.rng.bit_generator.state, jit.rng.bit_generator.state
                    )
        self.assertTrue(step_grid.nopython_signatures)

    def test_initial_state_reset_and_buffer_independence(self):
        model = create_simulation((6, 7), seed=9)
        initial, rng_state = model.grid.copy(), deepcopy(model.rng.bit_generator.state)
        first = run(model, 10)
        model.grid = initial.copy()
        model.next_step_grid = initial.copy()
        model.rng.bit_generator.state = rng_state
        np.testing.assert_array_equal(first.grids, run(model, 10).grids)
        self.assertFalse(np.shares_memory(model.grid, model.next_step_grid))
        self.assertEqual(model.grid.dtype, np.uint8)
        self.assertEqual(int(State.INFECTIOUS), 2)

    def test_backend_validation(self):
        with self.assertRaises(ValueError):
            create_simulation(backend="unknown")
