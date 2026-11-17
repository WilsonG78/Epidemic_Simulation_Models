"""Wall-clock measurements that exclude plotting and isolate JIT warmup."""

from copy import deepcopy
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

from simulation.core.simulation import Simulation
from simulation.utils.state import State
from simulation.visualization import SimulationResult


def _positive_integer(value, name, *, zero=False):
    if not isinstance(value, (int, np.integer)) or value < (0 if zero else 1):
        raise ValueError(
            f"{name} must be {'non-negative' if zero else 'positive'} integer"
        )


def _warmup(simulation):
    # Compile on a copy: neither the grid nor RNG of the experiment advances.
    clone = deepcopy(simulation)
    start = perf_counter()
    clone.simulation_step()
    return perf_counter() - start


@dataclass(frozen=True)
class BenchmarkResult:
    steps: int
    samples_seconds: tuple[float, ...]
    warmup_seconds: float

    @property
    def median_seconds(self):
        return median(self.samples_seconds)

    @property
    def milliseconds_per_step(self):
        return self.median_seconds * 1000 / self.steps

    @property
    def steps_per_second(self):
        return self.steps / self.median_seconds


@dataclass(frozen=True)
class TimedRun:
    result: SimulationResult
    compute_seconds: float
    recording_seconds: float
    total_seconds: float
    warmup_seconds: float

    @property
    def milliseconds_per_step(self):
        steps = len(self.result.grids) - 1
        return self.compute_seconds * 1000 / steps if steps else 0.0


def benchmark(
    simulation: Simulation, *, steps: int = 100, repeats: int = 5
) -> BenchmarkResult:
    """Measure completed model steps without history, plotting or JIT warmup.

    CUDA includes one upload/download per repeat and explicit synchronization.
    Device allocation and RNG initialization are performed outside the samples.

    Every repeat starts from an independent copy of the supplied grid and RNG.
    The supplied simulation is never advanced. Warmup time is reported
    separately and includes a single step (plus compilation if needed).
    """
    _positive_integer(steps, "steps")
    _positive_integer(repeats, "repeats")
    warmup = _warmup(simulation)
    samples = []
    for _ in range(repeats):
        model = deepcopy(simulation)
        model.prepare_accelerator()
        start = perf_counter()
        model.advance(steps)
        samples.append(perf_counter() - start)
    return BenchmarkResult(steps, tuple(samples), warmup)


def run_timed(simulation: Simulation, steps: int = 100) -> TimedRun:
    """Advance a model and record history, measuring computation and storage.

    Like run(), this mutates the supplied simulation. Warmup uses a copy.
    total_seconds excludes warmup but includes allocation, recording and result
    construction. No plots are created.
    Per-step timers add overhead: use benchmark() for throughput comparisons.
    """
    _positive_integer(steps, "steps", zero=True)
    warmup = _warmup(simulation) if steps else 0.0
    start = perf_counter()
    grids = np.empty((steps + 1, *simulation.grid.shape), dtype=np.uint8)
    counts = np.empty((steps + 1, 3), dtype=np.int64)
    compute = recording = 0.0
    for index in range(steps + 1):
        if index:
            mark = perf_counter()
            simulation.simulation_step()
            compute += perf_counter() - mark
        mark = perf_counter()
        grids[index] = simulation.grid
        counts[index] = [np.count_nonzero(simulation.grid == state) for state in State]
        recording += perf_counter() - mark
    grids.setflags(write=False)
    counts.setflags(write=False)
    result = SimulationResult(grids, counts)
    total = perf_counter() - start
    return TimedRun(result, compute, recording, total, warmup)
