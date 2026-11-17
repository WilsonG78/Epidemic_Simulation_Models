"""CPU kernels. Keep model iteration and RNG draw order identical to Python."""

from numba import njit


@njit(cache=True)
def step_grid(grid, next_grid, infection_probability, recover_probability, rng):
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            next_grid[row, col] = grid[row, col]
    for row in range(height):
        for col in range(width):
            if grid[row, col] == 2:
                for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                    nr, nc = (row + dr) % height, (col + dc) % width
                    if grid[nr, nc] == 1 and rng.random() < infection_probability:
                        next_grid[nr, nc] = 2
                if rng.random() < recover_probability:
                    next_grid[row, col] = 3
