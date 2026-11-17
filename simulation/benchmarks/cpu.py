"""Run with: uv run python benchmarks/cpu.py (no plotting or recording)."""

from statistics import median
from time import perf_counter

from simulation import create_simulation

# Compile outside the measured region.
create_simulation((2, 2), seed=42).simulation_step()
for size in (40, 100, 200):
    timings = {}
    for backend in ("python", "numba"):
        samples = []
        for _ in range(3):
            model = create_simulation(
                (size, size),
                initial_infected=size * size // 5,
                seed=42,
                backend=backend,
            )
            start = perf_counter()
            for _ in range(20):
                model.simulation_step()
            samples.append((perf_counter() - start) / 20)
        timings[backend] = median(samples)
    print(
        f"{size}x{size}: Python {timings['python'] * 1000:.3f} ms/step; "
        f"Numba {timings['numba'] * 1000:.3f} ms/step; "
        f"speedup {timings['python'] / timings['numba']:.1f}x"
    )
