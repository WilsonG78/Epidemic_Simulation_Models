"""Live Matplotlib controls, usable with ``%matplotlib widget`` in notebooks."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.widgets import Button, Slider, TextBox

from simulation.core.simulation import Simulation
from simulation.utils.state import State
from simulation.visualization import COLORS, LABELS, SimulationResult, create_simulation


class SimulationExplorer:
    """Control a live model and edit its initial infections by clicking cells.

    Left click toggles an infection; right click clears it. Editing is allowed
    only at step zero while paused. Reset restores the selected initial grid
    and random generator state. Time is measured in model steps; interval is
    the wall-clock delay in milliseconds between frames. steps_per_frame
    controls how many model steps are computed per frame (all are recorded).
    """

    def __init__(
        self,
        simulation: Simulation,
        *,
        steps: int = 200,
        interval: int = 120,
        steps_per_frame: int = 1,
    ):
        self._validate_steps(steps)
        if not isinstance(steps_per_frame, (int, np.integer)) or steps_per_frame < 1:
            raise ValueError("steps_per_frame must be a positive integer")
        self.steps_per_frame = steps_per_frame
        if not 10 <= interval <= 2000:
            raise ValueError("interval must be between 10 and 2000 milliseconds")
        self.simulation = simulation
        self.max_steps = steps
        self.running = False
        self.closed = False
        self._initial_grid = simulation.grid.copy()
        self._initial_random_state = simulation.get_random_state()
        self._grids = []
        self._counts = []
        self._record()
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.figure.subplots_adjust(bottom=0.30, top=0.88, wspace=0.3)
        grid_axis, count_axis = self.axes
        self.image = grid_axis.imshow(
            self._grids[0],
            cmap=ListedColormap(COLORS),
            norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3),
            interpolation="nearest",
        )
        grid_axis.set(xticks=[], yticks=[])
        self.lines = [
            count_axis.plot([], [], color=c, label=l)[0] for c, l in zip(COLORS, LABELS)
        ]
        count_axis.set(
            xlabel="Simulation step",
            ylabel="Number of people",
            ylim=(0, simulation.grid.size),
        )
        count_axis.legend()
        count_axis.grid(alpha=0.25)
        self.status = self.figure.text(0.08, 0.21, "", fontsize=10)
        self.play_button = Button(
            self.figure.add_axes((0.08, 0.12, 0.13, 0.05)), "Start / Pause"
        )
        self.step_button = Button(self.figure.add_axes((0.23, 0.12, 0.1, 0.05)), "Step")
        self.reset_button = Button(
            self.figure.add_axes((0.35, 0.12, 0.1, 0.05)), "Reset"
        )
        self.clear_button = Button(
            self.figure.add_axes((0.47, 0.12, 0.14, 0.05)), "Clear infections"
        )
        self.duration_box = TextBox(
            self.figure.add_axes((0.78, 0.12, 0.13, 0.05)),
            "Step limit ",
            initial=str(steps),
        )
        self.speed_slider = Slider(
            self.figure.add_axes((0.25, 0.04, 0.60, 0.03)),
            "Delay [ms]",
            10,
            2000,
            valinit=interval,
            valstep=10,
        )
        self.timer = self.figure.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._tick)
        self.play_button.on_clicked(
            lambda event: self.pause() if self.running else self.start()
        )
        self.step_button.on_clicked(lambda event: self.step())
        self.reset_button.on_clicked(lambda event: self.reset())
        self.clear_button.on_clicked(lambda event: self.clear())
        self.duration_box.on_submit(self._set_duration)
        self.speed_slider.on_changed(
            lambda value: setattr(self.timer, "interval", int(value))
        )
        self._click_id = self.figure.canvas.mpl_connect(
            "button_press_event", self._click
        )
        self._close_id = self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw()

    @staticmethod
    def _validate_steps(steps):
        if not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a non-negative integer")

    @property
    def current_step(self):
        return len(self._grids) - 1

    @property
    def result(self) -> SimulationResult:
        """Return an independent snapshot of the history recorded so far."""
        grids, counts = np.array(self._grids), np.array(self._counts)
        grids.setflags(write=False)
        counts.setflags(write=False)
        return SimulationResult(grids, counts)

    def _record(self):
        self._grids.append(self.simulation.grid.copy())
        self._counts.append(
            [int(np.count_nonzero(self.simulation.grid == state)) for state in State]
        )

    def _draw(self):
        self.image.set_data(self._grids[-1])
        self.axes[0].set_title(
            f"Population — step {self.current_step}/{self.max_steps}"
        )
        self.axes[1].set_xlim(0, max(1, self.max_steps, self.current_step))
        times = np.arange(len(self._counts))
        counts = np.asarray(self._counts)
        for index, line in enumerate(self.lines):
            line.set_data(times, counts[:, index])
        if self.current_step == 0 and not self.running:
            message = "Click cells to select initial infections, then press Start."
        elif self.current_step >= self.max_steps:
            message = "Step limit reached. Increase the limit or press Reset."
        else:
            message = (
                "Simulation running."
                if self.running
                else "Paused. Start resumes; Reset enables editing initial infections."
            )
        self.status.set_text(message)
        self.figure.canvas.draw_idle()

    def start(self):
        if self.closed or self.current_step >= self.max_steps:
            return
        self.running = True
        self.timer.start()
        self._draw()

    def pause(self):
        self.running = False
        self.timer.stop()
        self._draw()

    def step(self):
        """Pause and advance exactly one model step, up to the chosen limit."""
        self.running = False
        self.timer.stop()
        self._advance()

    def _advance(self, count=1):
        if self.closed or self.current_step >= self.max_steps:
            return
        for _ in range(min(count, self.max_steps - self.current_step)):
            self.simulation.simulation_step()
            self._record()
        if self.current_step >= self.max_steps:
            self.running = False
            self.timer.stop()
        self._draw()

    def _tick(self):
        if self.running and not self.closed:
            self._advance(self.steps_per_frame)

    def reset(self):
        self.pause()
        self.simulation.grid = self._initial_grid.copy()
        self.simulation.next_step_grid = self.simulation.grid.copy()
        self.simulation.set_random_state(self._initial_random_state)
        self._grids.clear()
        self._counts.clear()
        self._record()
        self._draw()

    def set_infected(self, row: int, col: int, infected: bool = True):
        """Edit a cell at step zero while paused; coordinates are (row, column)."""
        if self.current_step or self.running:
            raise RuntimeError(
                "Reset and pause the simulation before editing infections"
            )
        height, width = self.simulation.grid.shape
        if not (0 <= row < height and 0 <= col < width):
            raise IndexError("cell is outside the grid")
        self.simulation.grid[row, col] = (
            State.INFECTIOUS if infected else State.SUSCEPTIBLE
        )
        self._initial_grid = self.simulation.grid.copy()
        self.simulation.next_step_grid = self.simulation.grid.copy()
        self._grids.clear()
        self._counts.clear()
        self._record()
        self._draw()

    def clear(self):
        """Remove initial infections. Reset first if the simulation has advanced."""
        if self.current_step or self.running:
            self.status.set_text("Press Reset before editing initial infections.")
            self.figure.canvas.draw_idle()
            return
        self.simulation.grid[:] = State.SUSCEPTIBLE
        self._initial_grid = self.simulation.grid.copy()
        self.reset()

    def _click(self, event):
        if (
            event.inaxes is not self.axes[0]
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if self.running or self.current_step:
            return
        if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
            return
        row, col = int(np.floor(event.ydata + 0.5)), int(np.floor(event.xdata + 0.5))
        height, width = self.simulation.grid.shape
        if 0 <= row < height and 0 <= col < width:
            infected = (
                event.button == MouseButton.LEFT
                and self.simulation.grid[row, col] != State.INFECTIOUS
            )
            self.set_infected(row, col, infected)

    def set_duration(self, steps: int):
        self._validate_steps(steps)
        self.max_steps = steps
        if self.current_step >= steps:
            self.pause()
        self._draw()

    def _set_duration(self, text):
        try:
            self.set_duration(int(text))
        except ValueError:
            self.status.set_text("The limit must be a non-negative integer.")
            self.figure.canvas.draw_idle()

    def _on_close(self, event):
        self.running = False
        self.closed = True
        self.timer.stop()

    def close(self):
        self._on_close(None)
        plt.close(self.figure)


def explore(
    simulation: Simulation | None = None,
    *,
    steps: int = 200,
    interval: int = 120,
    steps_per_frame: int = 1,
    **parameters,
) -> SimulationExplorer:
    """Open a paused explorer. With no model, start with zero infections.

    Additional parameters are passed to create_simulation. Keep the returned
    controller alive. In notebooks, run ``%matplotlib widget`` first.
    """
    if simulation is not None and parameters:
        raise ValueError("Pass either a simulation or creation parameters")
    if simulation is None:
        simulation = create_simulation(**{"initial_infected": 0, **parameters})
    backend = plt.get_backend().lower()
    if "inline" in backend or backend == "agg":
        raise RuntimeError(
            "Interactive backend required. In a notebook run %matplotlib widget first (install ipympl)."
        )
    return SimulationExplorer(
        simulation, steps=steps, interval=interval, steps_per_frame=steps_per_frame
    )
