"""Optional CUDA resources; imported only when the CUDA backend is selected."""

import pickle
from copy import deepcopy

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from simulation.kernels.gpu_numba import update


def require_cuda():
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the cuda extra and a compatible NVIDIA "
            "driver/toolkit, or select backend='numba'. For logic tests only, "
            "set NUMBA_ENABLE_CUDASIM=1 before importing numba."
        )


def rng_signature(rng):
    return pickle.dumps(rng.bit_generator.state)


class CudaRuntime:
    def __init__(self, grid, rng, snapshot=None):
        require_cuda()
        self.current = cuda.device_array(grid.shape, dtype=np.uint8)
        self.output = cuda.device_array_like(self.current)
        if snapshot is not None and snapshot["anchor"] == rng_signature(rng):
            self.states = cuda.to_device(snapshot["states"])
            self.anchor = snapshot["anchor"]
        else:
            seed = int(rng.integers(0, 2**63))
            self.states = create_xoroshiro128p_states(grid.size, seed=seed)
            self.anchor = rng_signature(rng)
        self.threads = (16, 16)
        self.blocks = tuple((size + 15) // 16 for size in grid.shape)

    def snapshot(self):
        cuda.synchronize()
        return {"states": self.states.copy_to_host(), "anchor": deepcopy(self.anchor)}

    def advance(self, grid, infection_probability, recover_probability, steps):
        # Upload once per batch so direct host edits and the existing editor work.
        self.current.copy_to_device(np.ascontiguousarray(grid))
        for _ in range(steps):
            update[self.blocks, self.threads](
                self.current,
                self.output,
                infection_probability,
                recover_probability,
                self.states,
            )
            self.current, self.output = self.output, self.current
        # Explicit synchronization makes wall-clock API measure completed work.
        cuda.synchronize()
        return self.current.copy_to_host()
