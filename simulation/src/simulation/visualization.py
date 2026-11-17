"""Notebook-friendly SIR experiments, plots and replayable animations."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap

from simulation.core.simulation import Simulation
from simulation.utils.state import State

COLORS = ("#4C9F70", "#E45756", "#4C78A8")
LABELS = ("Susceptible", "Infected", "Recovered")


def create_simulation(
    grid_shape: tuple[int, int] = (50, 50),
    infection_probability: float = 0.25,
    recover_probability: float = 0.05,
    initial_infected: int = 1,
    seed: int | None = None,
    backend: str = "numba",
) -> Simulation:
    """Create a simulation with randomly placed infections and a reproducible seed."""
    if len(grid_shape) != 2 or any(
        not isinstance(size, (int, np.integer)) or size <= 0 for size in grid_shape
    ):
        raise ValueError("grid_shape must contain two positive integers")
    population = int(np.prod(grid_shape))
    if (
        not isinstance(initial_infected, (int, np.integer))
        or not 0 <= initial_infected <= population
    ):
        raise ValueError(
            "initial_infected must be an integer between 0 and population size"
        )
    rng = np.random.default_rng(seed)
    initial = np.full(grid_shape, State.SUSCEPTIBLE, dtype=np.uint8)
    initial.flat[rng.choice(population, size=initial_infected, replace=False)] = (
        State.INFECTIOUS
    )
    simulation = Simulation(
        grid_shape,
        lambda rows, cols: initial.copy(),
        infection_probability,
        recover_probability,
        backend=backend,
    )
    simulation.rng = rng
    return simulation


@dataclass(frozen=True)
class SimulationResult:
    """Recorded numeric grids (1=S, 2=I, 3=R) and counts, including step zero."""

    grids: np.ndarray
    counts: np.ndarray

    @property
    def steps(self) -> np.ndarray:
        return np.arange(len(self.grids))

    def plot(self, step: int = -1, *, figsize: tuple[float, float] = (12, 5)):
        """Return (figure, axes) for a recorded step and the full SIR curves."""
        index = self._index(step)
        figure, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        grid_axis, count_axis = axes
        image = grid_axis.imshow(
            self.grids[index],
            cmap=ListedColormap(COLORS),
            norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3),
            interpolation="nearest",
        )
        grid_axis.set(title=f"Population state — step {index}", xticks=[], yticks=[])
        figure.colorbar(image, ax=grid_axis, ticks=[1, 2, 3]).ax.set_yticklabels(LABELS)
        for column, (color, label) in enumerate(zip(COLORS, LABELS)):
            count_axis.plot(
                self.steps, self.counts[:, column], color=color, label=label
            )
        count_axis.axvline(index, color="#555555", linestyle="--", alpha=0.6)
        count_axis.set(
            title="Population counts over time",
            xlabel="Simulation step",
            ylabel="Number of people",
            xlim=(0, max(1, len(self.grids) - 1)),
            ylim=(0, self.grids[0].size),
        )
        count_axis.grid(alpha=0.25)
        count_axis.legend()
        return figure, axes

    def _index(self, step: int) -> int:
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        if not -len(self.grids) <= step < len(self.grids):
            raise IndexError("step is outside the recorded simulation")
        return step % len(self.grids)

    def animate(
        self, *, interval: int = 120, figsize: tuple[float, float] = (12, 5)
    ) -> FuncAnimation:
        """Return a replayable animation; rendering never advances the simulation."""
        if interval <= 0:
            raise ValueError("interval must be positive")
        figure, axes = self.plot(0, figsize=figsize)
        grid_axis, count_axis = axes
        image = grid_axis.images[0]
        lines = count_axis.lines[:3]
        cursor = count_axis.lines[3]

        def update(frame):
            image.set_data(self.grids[frame])
            grid_axis.set_title(f"Population state — step {frame}")
            for column, line in enumerate(lines):
                line.set_data(self.steps[: frame + 1], self.counts[: frame + 1, column])
            cursor.set_xdata([frame, frame])
            return image, cursor, *lines

        return FuncAnimation(
            figure,
            update,
            frames=len(self.grids),
            interval=interval,
            blit=False,
            repeat=False,
        )

    def display(self, *, interval: int = 120, figsize: tuple[float, float] = (12, 5)):
        """Return IPython HTML with play/pause, speed and frame slider controls."""
        from IPython.display import HTML

        animation = self.animate(interval=interval, figsize=figsize)
        try:
            return HTML(animation.to_jshtml())
        finally:
            plt.close(animation._fig)


def run(simulation: Simulation, steps: int = 100) -> SimulationResult:
    """Advance an existing simulation and record independent snapshots.

    This mutates the supplied simulation. Step zero is its state at call time.
    """
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    grids = np.empty((steps + 1, *simulation.grid.shape), dtype=np.uint8)
    for step in range(steps + 1):
        if step:
            simulation.simulation_step()
        for state in State:
            grids[step][simulation.grid == state] = state.value
    counts = np.stack(
        [(grids == state.value).sum(axis=(1, 2)) for state in State], axis=1
    )
    grids.setflags(write=False)
    counts.setflags(write=False)
    return SimulationResult(grids, counts)
