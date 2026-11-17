"""CUDA gather update: each thread writes exactly one output cell."""

from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64


@cuda.jit
def update(grid, next_grid, infection_probability, recover_probability, rng_states):
    row, col = cuda.grid(2)
    height, width = grid.shape
    if row >= height or col >= width:
        return
    index = row * width + col
    state = grid[row, col]
    next_state = state
    if state == 1:
        no_infection = 1.0
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            if grid[(row + dr) % height, (col + dc) % width] == 2:
                no_infection *= 1.0 - infection_probability
        if xoroshiro128p_uniform_float64(rng_states, index) < 1.0 - no_infection:
            next_state = 2
    elif (
        state == 2
        and xoroshiro128p_uniform_float64(rng_states, index) < recover_probability
    ):
        next_state = 3
    next_grid[row, col] = next_state
