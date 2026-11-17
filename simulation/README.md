# SIR simulation

## Notebook

Set up the environment from the `simulation` directory:

```bash
uv sync --extra notebook
```

Open `notebooks/playground.ipynb` in Jupyter or VS Code and select the
`simulation/.venv/bin/python` kernel. The notebook includes an animation with a
time slider, a plot of a selected step, and a parameter comparison.

```python
from simulation import create_simulation, run

model = create_simulation(
    grid_shape=(40, 40),
    infection_probability=0.25,
    recover_probability=0.05,
    initial_infected=4,
    seed=42,
)
result = run(model, steps=100)
result.display(interval=100)  # last expression in the cell
```

- `result.plot(step=25)` returns a Matplotlib `(figure, axes)` pair.
- `result.animate(interval=100)` returns a `FuncAnimation`, for example to save to a file.
- `result.display()` returns HTML with playback, a slider, and speed controls;
  it works with the default inline backend without ipywidgets.
- `result.steps`, `result.counts`, and `result.grids` expose the recorded data.
  The counts columns are susceptible, infected, and recovered. Grid values are 1, 2, 3.
- `run(model, steps)` changes the model and records the initial state and every step.
  Animation playback uses the recording and does not change the model.
- `seed` controls random initial infections and subsequent transitions.

You can also pass a custom `Simulation` object with your own initial distribution
to `run`. The model uses four neighbors and periodic boundaries. The recording
stores the full history in memory, and the HTML contains every frame. Start with
a 40×40 grid and 100 steps; plots are more suitable for analyzing long runs.

## Terminal

```bash
uv run simulation
```

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

## Live controls and initial infection selection

Run the following in a notebook:

```python
%matplotlib widget
from simulation import explore

explorer = explore(grid_shape=(40, 40), steps=200, interval=120, seed=42)
```

Click grid cells before starting: left-click toggles an infection, and right-click
removes it. Disable zoom/pan when selecting initial infections. The panel includes
Start/Pause, a single Step, Reset, Clear infections, a step limit (confirmed with
Enter), and a delay slider in milliseconds. A larger delay slows the simulation.
The limit counts model steps, not wall-clock time. You can increase it and resume.

Reset restores the selected initial infections and random generator state; cells
can then be edited again. `explorer.result` retrieves the recorded history without
advancing the model. `explorer.close()` closes the panel and timer.

The API also provides `start()`, `pause()`, `step()`, `reset()`,
`set_duration(steps)`, and `set_infected(row, col, infected=True)`.
You can supply a custom model: `explore(model, steps=300)`.
The notebook requires ipympl (included in the `notebook` extra) and a running kernel.
Restart the kernel after installing dependencies. The inline backend does not
support interactive clicks.

## CPU acceleration with Numba

By default, `create_simulation()` and `explore()` use `backend="numba"`.
The step computation is compiled with `@njit(cache=True)`. The first step may take
longer because of compilation; subsequent steps use machine code. Numba is a
standard project dependency (`uv sync --extra notebook`). Restart the notebook
kernel after updating.

For comparison, use `create_simulation(..., backend="python")` or
`explore(..., backend="python")`. Both variants preserve the same random draw order,
transition rules, periodic boundaries, and reads from the previous step.
`seed` and reset behave as before. Acceleration applies to CPU computation;
it does not speed up Matplotlib rendering.

The model grid stores `np.uint8` values (1, 2, 3). `State` is an `IntEnum`, so
`State.INFECTIOUS`, `.value`, and iteration over `State` still work. Grid elements
are numbers: use `int(grid[row, col])` instead of `grid[row, col].value`.
History recording and initial infection selection support this format.

## Timing and less frequent rendering

```python
from simulation import benchmark, create_simulation, run_timed

model = create_simulation((100, 100), initial_infected=100, seed=42)
report = benchmark(model, steps=500, repeats=5)
print(report.median_seconds, report.milliseconds_per_step, report.steps_per_second)
print(report.warmup_seconds)  # one step and any compilation, outside the samples

measured = run_timed(model, steps=500)
print(measured.compute_seconds, measured.recording_seconds, measured.total_seconds)
result = measured.result
# Separately, call result.plot() or result.display().
```

`benchmark` leaves the model and RNG unchanged; every sample starts from the same
copy. It measures the step loop, excluding model copying, frame recording,
compilation, and plotting. `run_timed` changes the model, records every step, and
separates computation time from recording time. Total time also includes allocation
and result preparation, but excludes warmup. Per-step clocks add a small overhead;
use `benchmark` for performance comparisons. Stop active panels before measuring.

```python
explorer = explore(steps=1000, steps_per_frame=10, interval=100)
```

The panel performs up to 10 steps per refresh, recording every step in the history.
The Step button still advances exactly one step. `interval` is the delay between
frames, rather than individual steps. The default `steps_per_frame=1` preserves the
previous behavior. Large batches can delay the response to Pause, so start with
5–10 steps. Recording full grids still costs memory and time; use `benchmark` to
measure computation separately.

Less frequent drawing reduces presentation overhead but does not eliminate it.
For the most repeatable measurements, compute a run without an open panel and
replay the result afterward. `draw_idle()` schedules rendering; timing that call
does not capture all work in the notebook frontend and is not a rendering benchmark.

## SIR comparison and analytical checks

The `notebooks/validation.ipynb` notebook presents equations, plots, and interpretation.

```python
from simulation import create_simulation, compare_sir

model = create_simulation((40, 40), initial_infected=20,
                          infection_probability=0.06, recover_probability=0.05, seed=42)
comparison = compare_sir(model, steps=200, repeats=20, seed=123)
print(comparison.summary)
figure, axes = comparison.plot()
```

`compare_sir` leaves the model and RNG unchanged. It computes independent runs from
the same initial state, storing only S, I, R counts. The result compares the grid
ensemble mean with a numerical SIR ODE solution (RK4) and a discrete
independent-neighbor approximation. The band shows the 10th–90th percentiles of
runs, not a confidence interval.

For population fractions: s'=-beta*s*i, i'=beta*s*i-gamma*i, r'=gamma*i.
By default, beta=4*p and gamma=q, with one time unit per step. This is a first-order
approximation, particularly inaccurate for large probabilities. You can override
`beta` and `gamma` for the ODE only. The discrete mean-field model uses infection
incidence s*[1-(1-p*i)^4] and recoveries q*i. Both references ignore spatial
correlations; neither is an exact solution of the grid process.

`summary` reports fraction RMSE, infection peaks, the analytical ODE peak, the error
in the analytical invariant log(s/s0)+(beta/gamma)*(r-r0), population conservation,
monotonicity of S and R, and the analytical final epidemic size for the ODE. The
last point of a finite simulation need not be the end of the epidemic. A large RMSE
alone does not indicate an implementation error: SIR assumes homogeneous mixing,
whereas the grid uses local contacts. No automatic acceptance threshold is imposed
on agreement between these models.

`sir_ode(initial, beta=..., gamma=..., steps=..., substeps=20)` returns population
counts at successive integer times; increase substeps to check convergence.
`sir_final_susceptible` solves the analytical final-size equation by bisection.
For gamma=0 it returns None because the formula does not apply.

Sources: [SIR equations](https://sites.math.duke.edu/education/ccp/materials/diffcalc/sir/sir2.html),
[epidemic final size](https://kingaa.github.io/short-course/parest/odes.html),
[grid model and mean field](https://arxiv.org/abs/nlin/0207048).

## CUDA backend

Kernels are in `simulation/kernels/cpu_numba.py` and `gpu_numba.py`.
The CUDA version requires an NVIDIA GPU and a CUDA 12-compatible driver. The
optional extra installs `numba-cuda[cu12]` with toolkit libraries (a potentially
large download):

```bash
uv sync --extra notebook --extra cuda
```

Restart the notebook kernel after installation. Without the CUDA extra, the `python`
and `numba` backends still work; CUDA is imported only when that backend is selected.
If no GPU is available, a clear error is raised without automatically falling back
to the CPU.

```python
from simulation import create_simulation, run_timed, benchmark

model = create_simulation((100, 100), initial_infected=20, seed=42, backend="cuda")
report = run_timed(model, steps=100)
report.result.display()  # visualization after computation

# Computation without history; data remains on the GPU between steps:
model.advance(100)
print(benchmark(model, steps=100, repeats=3).milliseconds_per_step)
```

`explore(..., backend="cuda")`, `run`, `run_timed`, and `compare_sir` support CUDA.
Initial infection selection and Matplotlib remain on the CPU. `simulation_step()`
preserves compatibility with the old API: it uploads the grid and downloads the
result after every step so direct edits to `model.grid` work. `advance(n)` uploads
the grid once per batch, launches n kernels, and downloads the final grid. It does
not record intermediate frames. Full history in `run` and the panel still requires
a transfer every step, even with `steps_per_frame>1`; that parameter reduces drawing,
not history recording. Selecting CUDA therefore does not guarantee faster small runs.

Each thread writes only its own new state. A susceptible cell with k infected
contacts becomes infected with probability `1-(1-p)**k`; an infected cell recovers
with probability q. Every cell receives an output value, including susceptible
cells with no infected neighbors and recovered cells. Boundaries are periodic and
all reads come from the previous step. For grid dimensions of 1 or 2, repeated
contacts are counted as in the existing CPU model.

CUDA uses separate xoroshiro128+ states for each cell. The same seed provides
reproducibility within the backend, but not identical CPU and GPU frames. Copying
a model preserves the GPU RNG state; resetting the panel restores that state too,
even if the panel was opened after advancing the simulation. For custom checkpoints,
use `model.get_random_state()` and `model.set_random_state(state)`; save and restore
the grid separately.

`benchmark` measures completed computation with GPU synchronization and one
round-trip transfer per sample. Buffer and RNG initialization take place outside
the samples. With CUDA, `run_timed.compute_seconds` also includes transfers and
synchronization after every step; it is not pure kernel time. Do not use simulator
results to assess GPU performance.

Test logic without a GPU (set the variable before importing Numba):

```bash
NUMBA_ENABLE_CUDASIM=1 uv run --extra cuda python -m unittest discover -s tests -p test_cuda.py -v
```

The simulator does not replace compilation and execution checks on a physical GPU.
Normal test runs skip CUDA tests when no GPU is available.
Sources: [kernels](https://nvidia.github.io/numba-cuda/user/kernels.html),
[GPU random numbers](https://nvidia.github.io/numba-cuda/user/random.html),
[installation](https://nvidia.github.io/numba-cuda/user/installation.html).

## Google Colab and VS Code

`notebooks/colab_gpu.ipynb` contains a current source snapshot. Installation uses
`%pip` in the main Colab kernel; restart the kernel after installation. Subsequent
cells use ordinary Python: `model`, `result`, and benchmark results remain available
between cells. There is no separate venv or `run_in_env`.

1. Upload the current notebook to Colab and select a GPU runtime.
2. Run the source setup and `%pip install` cells.
3. Restart the kernel and continue from diagnostics without reinstalling.
4. Run computation, then generate animations and plots in subsequent cells.

The notebook pins NumPy 2.4.6 and Numba 0.65.1. This resolves reported conflicts:
numba-cuda 0.30.4 uses `np.row_stack`, removed in NumPy 2.5, while PyTensor 2.38.3
requires Numba <=0.65.1. PyTensor is not removed. The project's Numba requirement
was broadened to >=0.65.1; local environments can still use a newer version.
Installation changes packages in the Colab session, so additional dependencies may
need separate adjustments. Use `%pip check` to check them.

Requires Python >=3.12 and an NVIDIA GPU with a compatible driver. The initial CUDA
installation may be large. The notebook tests actual kernel execution; a physical
Colab GPU is not available for local verification. Repeat installation after deleting
the runtime, and save results before deleting it.

After changing sources, update the snapshot:

```bash
python tools/build_colab_notebook.py
```

The generator embeds only `src/**/*.py`, `pyproject.toml`, and the README, excluding
the venv. In VS Code, install the official
[Google Colab extension](https://marketplace.visualstudio.com/items?itemName=google.colab),
open the notebook, select Select Kernel → Colab → New Colab Server, then choose a GPU.
Source: [Google instructions](https://github.com/googlecolab/colab-vscode).

## Continuous SIR model (part A)

`simulation/core/SIR.py` solves the homogeneous differential equations using
SciPy's adaptive RK45 solver. Initial values are fractions satisfying S + I + R = 1.
Transmission and recovery rates remain fixed throughout each run.

```python
import matplotlib.pyplot as plt
from simulation.core.SIR import SIR

model = SIR(population=(0.99, 0.01, 0.0), beta=0.3, gamma=0.1)
solution = model.solve(t_span=(0, 160), points=1000)
S, I, R = solution.y
figure, ax = model.plot(solution)
plt.show()
```

For a standalone example, run `uv run python -m simulation.core.SIR` from the
`simulation` directory. To study parameter uncertainty, draw valid initial
fractions or nonnegative rates before constructing each model. Output times are
evenly spaced; the solver chooses its own internal time steps. The last recovered
fraction is R(t_end), not automatically R(infinity). A duration threshold must be
defined separately to measure epidemic duration.

### Parameter-distribution experiments

Open [notebooks/sir_distributions.ipynb](notebooks/sir_distributions.ipynb) for
Monte Carlo experiments with the continuous SIR model. It compares positive
truncated Gaussian transmission and recovery rates at several parent means and
standard deviations, independent joint uncertainty, and uniform/lognormal
alternatives. Every run keeps its sampled rates constant.

The notebook includes parameter histograms, ensemble trajectories, final-size
and extinction-time distributions, fixed-time cumulative infections, and summary
statistics with deciles. Gaussian parameters describe the parent distribution;
truncation changes its moments. Unresolved epidemic durations are reported as
right-censored rather than treated as completed runs. Edit the configuration
cell and run all cells to repeat the experiments.
