from collections.abc import Callable
from copy import deepcopy

import numpy as np
from simulation.kernels.cpu_numba import step_grid
from simulation.utils.state import State


class Simulation:
    grid: np.ndarray
    next_step_grid: np.ndarray
    width: int
    height: int

    def __init__(
        self,
        grid_shape: tuple[int, int],
        infectious_distribution: Callable[[np.ndarray, np.ndarray], np.ndarray],
        infection_probability: float,
        recover_probability: float,
        *,
        backend: str = "numba",
    ):
        for name, probability in (
            ("infection_probability", infection_probability),
            ("recover_probability", recover_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        if backend not in ("numba", "python", "cuda"):
            raise ValueError("backend must be 'numba', 'python' or 'cuda'")
        if backend == "cuda":
            try:
                from simulation.kernels.cuda_runtime import require_cuda
            except Exception as error:
                raise RuntimeError(
                    "Install the cuda extra and a compatible NVIDIA driver/toolkit"
                ) from error
            require_cuda()
        self.backend = backend
        self._cuda_runtime = None
        self._cuda_snapshot = None
        initial = np.fromfunction(infectious_distribution, grid_shape, dtype=int)
        if (
            initial.ndim != 2
            or 0 in initial.shape
            or not np.isin(initial, [1, 2, 3]).all()
        ):
            raise ValueError(
                "initial grid must be nonempty, 2D and contain only states 1, 2, 3"
            )
        self.grid = np.array(initial, dtype=np.uint8, order="C")
        self.height, self.width = self.grid.shape
        self.infection_probability = infection_probability
        self.recover_probability = recover_probability
        self.rng = np.random.default_rng()
        self.next_step_grid = self.grid.copy()

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name != "_cuda_runtime":
                setattr(clone, name, deepcopy(value, memo))
        clone._cuda_runtime = None
        if self._cuda_runtime is not None:
            clone._cuda_snapshot = self._cuda_runtime.snapshot()
        return clone

    def reset_accelerator(self):
        self._cuda_runtime = None
        self._cuda_snapshot = None

    def prepare_accelerator(self):
        if self.backend != "cuda":
            return
        from simulation.kernels.cuda_runtime import CudaRuntime, rng_signature

        if (
            self._cuda_runtime is not None
            and self._cuda_runtime.anchor != rng_signature(self.rng)
        ):
            self.reset_accelerator()
        if self._cuda_runtime is None:
            self._cuda_runtime = CudaRuntime(self.grid, self.rng, self._cuda_snapshot)
            self._cuda_snapshot = None

    def get_random_state(self):
        device = self._cuda_snapshot
        if self._cuda_runtime is not None:
            device = self._cuda_runtime.snapshot()
        return {
            "host": deepcopy(self.rng.bit_generator.state),
            "cuda": deepcopy(device),
        }

    def set_random_state(self, state):
        self.rng.bit_generator.state = deepcopy(state["host"])
        self.reset_accelerator()
        self._cuda_snapshot = deepcopy(state["cuda"])

    def advance(self, steps: int = 1):
        """Advance without history; CUDA transfers once per batch and synchronizes."""
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if not steps:
            return
        if self.backend != "cuda":
            for _ in range(steps):
                self.simulation_step()
            return
        self.prepare_accelerator()
        self.grid = self._cuda_runtime.advance(
            self.grid, self.infection_probability, self.recover_probability, steps
        )
        self.next_step_grid = self.grid.copy()

    def simulation_step(self):
        if self.backend == "cuda":
            self.advance(1)
            return
        if self.backend == "numba":
            # A distinct output buffer keeps all reads in the previous time step.
            step_grid(
                self.grid,
                self.next_step_grid,
                self.infection_probability,
                self.recover_probability,
                self.rng,
            )
            self.grid, self.next_step_grid = self.next_step_grid, self.grid
            return
        self.next_step_grid = self.grid.copy()
        for row in range(self.height):
            for col in range(self.width):
                self.update(row, col)
        self.grid = self.next_step_grid

    def update(self, row: int, col: int) -> None:
        current_state = self.grid[row, col]
        neighbors = ((-1, 0), (1, 0), (0, 1), (0, -1))

        if current_state == State.INFECTIOUS:
            for row_offset, col_offset in neighbors:
                neighbor_row = (row + row_offset) % self.height
                neighbor_col = (col + col_offset) % self.width
                if (
                    self.grid[neighbor_row, neighbor_col] == State.SUSCEPTIBLE
                    and self.rng.random() < self.infection_probability
                ):
                    self.next_step_grid[neighbor_row, neighbor_col] = State.INFECTIOUS
            if self.rng.random() < self.recover_probability:
                self.next_step_grid[row, col] = State.RECOVERED
