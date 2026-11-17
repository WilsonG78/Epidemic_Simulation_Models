"""SIR reference equations and exploratory comparisons for the spatial model.

The ODE is a mean-field reference, not an exact solution of the lattice model.
Default beta=4*p and gamma=q are first-order, one-step approximations.
"""

from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from simulation.core.simulation import Simulation
from simulation.utils.state import State
from simulation.visualization import COLORS, LABELS


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def sir_ode(initial, *, beta: float, gamma: float, steps: int, substeps: int = 20):
    """Integrate fraction-based SIR using RK4, returning counts at integer times.

    initial=(S0, I0, R0) are counts. beta and gamma are rates per model step.
    Increase substeps to check convergence. No clipping or renormalizing is used.
    """
    _integer(steps, "steps")
    _integer(substeps, "substeps", 1)
    initial = np.asarray(initial, dtype=float)
    if (
        initial.shape != (3,)
        or not np.isfinite(initial).all()
        or (initial < 0).any()
        or initial.sum() <= 0
    ):
        raise ValueError(
            "initial must contain three finite nonnegative counts with positive sum"
        )
    if not np.isfinite([beta, gamma]).all() or beta < 0 or gamma < 0:
        raise ValueError("beta and gamma must be finite nonnegative rates")
    n = initial.sum()
    y = initial / n
    trajectory = np.empty((steps + 1, 3))
    trajectory[0] = initial
    subdivisions = max(substeps, int(np.ceil(20 * max(beta, gamma))))
    h = 1 / subdivisions

    def rhs(state):
        incidence = beta * state[0] * state[1]
        recovery = gamma * state[1]
        return np.array([-incidence, incidence - recovery, recovery])

    for step in range(1, steps + 1):
        for _ in range(subdivisions):
            k1 = rhs(y)
            k2 = rhs(y + h * k1 / 2)
            k3 = rhs(y + h * k2 / 2)
            k4 = rhs(y + h * k3)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        trajectory[step] = y * n
    return trajectory


def sir_final_susceptible(initial, *, beta: float, gamma: float) -> float | None:
    """Analytic SIR final-size relation, solved by bisection; None if gamma=0.

    s_inf = s0 * exp[-(beta/gamma)*(s0+i0-s_inf)].
    Returns a count, not a fraction. Assumes a closed homogeneous SIR model.
    """
    initial = np.asarray(initial, dtype=float)
    # Share validation with the ODE solver.
    sir_ode(initial, beta=beta, gamma=gamma, steps=0)
    n = initial.sum()
    s0, i0, _ = initial / n
    if gamma == 0:
        return None
    if beta == 0 or i0 == 0 or s0 == 0:
        return float(initial[0])
    lo, hi = 0.0, s0
    for _ in range(100):
        mid = (lo + hi) / 2
        residual = s0 * np.exp(-(beta / gamma) * (s0 + i0 - mid)) - mid
        if residual > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * n / 2


@dataclass(frozen=True)
class SIRComparison:
    samples: np.ndarray  # (replicates, time, S/I/R), counts only, no grid history
    ode_counts: np.ndarray
    mean_field_counts: np.ndarray
    beta: float
    gamma: float
    final_susceptible: float | None

    @property
    def mean_counts(self):
        return self.samples.mean(axis=0)

    @property
    def summary(self):
        mean = self.mean_counts
        n = self.samples[0, 0].sum()
        ode_peak = int(self.ode_counts[:, 1].argmax())
        spatial_peak = int(mean[:, 1].argmax())
        s0, i0, _ = self.ode_counts[0] / n
        peak_fraction = i0
        if i0 > 0 and self.beta > 0 and self.beta * s0 > self.gamma:
            threshold = self.gamma / self.beta
            peak_fraction = (
                s0 + i0
                if threshold == 0
                else s0 + i0 - threshold + threshold * np.log(threshold / s0)
            )
        invariant_error = None
        if self.gamma > 0 and s0 > 0 and (self.ode_counts[:, 0] > 0).all():
            fraction = self.ode_counts / n
            invariant = np.log(fraction[:, 0] / s0) + self.beta / self.gamma * (
                fraction[:, 2] - fraction[0, 2]
            )
            invariant_error = float(np.abs(invariant).max())
        return {
            "replicates": len(self.samples),
            "beta": self.beta,
            "gamma": self.gamma,
            "ode_R0": self.beta / self.gamma if self.gamma else None,
            "rmse_fraction_S_I_R": np.sqrt(
                np.mean(((mean - self.ode_counts) / n) ** 2, axis=0)
            ).tolist(),
            "ode_analytic_peak_infected": float(peak_fraction * n),
            "ode_max_analytic_invariant_error": invariant_error,
            "ode_peak_step": ode_peak,
            "spatial_mean_peak_step": spatial_peak,
            "ode_peak_infected": float(self.ode_counts[ode_peak, 1]),
            "spatial_mean_peak_infected": float(mean[spatial_peak, 1]),
            "ode_final_susceptible_infinite_time": self.final_susceptible,
            "spatial_mean_susceptible_at_end": float(mean[-1, 0]),
            "spatial_mean_infected_at_end": float(mean[-1, 1]),
            "spatial_max_population_error": float(
                np.abs(self.samples.sum(axis=2) - n).max()
            ),
            "ode_max_population_error": float(
                np.abs(self.ode_counts.sum(axis=1) - n).max()
            ),
            "spatial_S_nonincreasing": bool(
                (np.diff(self.samples[:, :, 0], axis=1) <= 0).all()
            ),
            "spatial_R_nondecreasing": bool(
                (np.diff(self.samples[:, :, 2], axis=1) >= 0).all()
            ),
        }

    def plot(self):
        """Compare ensemble mean and empirical 10–90% band to SIR references."""
        times = np.arange(len(self.ode_counts))
        mean = self.mean_counts
        low, high = np.quantile(self.samples, [0.1, 0.9], axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for i, (color, label) in enumerate(zip(COLORS, LABELS)):
            axes[0].plot(times, mean[:, i], color=color, label=f"{label}: grid")
            axes[0].fill_between(times, low[:, i], high[:, i], color=color, alpha=0.15)
            axes[0].plot(
                times,
                self.ode_counts[:, i],
                color=color,
                linestyle="--",
                label=f"{label}: ODE",
            )
        axes[1].plot(times, mean[:, 1], color=COLORS[1], label="Grid: mean")
        axes[1].fill_between(
            times,
            low[:, 1],
            high[:, 1],
            color=COLORS[1],
            alpha=0.15,
            label="Runs: 10–90 percentiles",
        )
        axes[1].plot(
            times, self.ode_counts[:, 1], "k--", label="SIR: differential equations"
        )
        axes[1].plot(
            times,
            self.mean_field_counts[:, 1],
            color="#8054a1",
            linestyle=":",
            label="Discrete mean-field model",
        )
        axes[0].set_title("S, I, R — grid and homogeneous mixing")
        axes[1].set_title("Infected — comparison of model assumptions")
        for ax in axes:
            ax.set(xlabel="Time [model steps]", ylabel="Number of people")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{len(self.samples)} runs | β={self.beta:g}, γ={self.gamma:g} | band is not a confidence interval"
        )
        return fig, axes


def compare_sir(
    simulation: Simulation,
    *,
    steps: int = 200,
    repeats: int = 20,
    seed: int | None = 42,
    beta: float | None = None,
    gamma: float | None = None,
    substeps: int = 20,
) -> SIRComparison:
    """Compare lattice replicates with a numerical SIR ODE and discrete mean field.

    Does not mutate simulation. Each replicate starts at the same current grid,
    with independent RNG streams derived from seed. beta=4*p and gamma=q are
    first-order approximations, not an exact conversion of probabilities to
    continuous rates. Supplied rates override the ODE only. One time unit is
    one simulation step. Discrepancies measure model differences, not correctness.
    """
    _integer(steps, "steps")
    _integer(repeats, "repeats", 1)
    p, q = simulation.infection_probability, simulation.recover_probability
    beta = 4 * p if beta is None else beta
    gamma = q if gamma is None else gamma
    initial = np.array([np.count_nonzero(simulation.grid == state) for state in State])
    ode = sir_ode(initial, beta=beta, gamma=gamma, steps=steps, substeps=substeps)
    samples = np.empty((repeats, steps + 1, 3), dtype=np.int64)
    for index, stream in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        model = deepcopy(simulation)
        model.rng = np.random.default_rng(stream)
        samples[index, 0] = initial
        for step in range(1, steps + 1):
            model.simulation_step()
            samples[index, step] = [
                np.count_nonzero(model.grid == state) for state in State
            ]
    # Independent-neighbor closure of the discrete update (not an exact lattice expectation).
    mean_field = np.empty_like(ode)
    n = initial.sum()
    mean_field[0] = initial
    for step in range(1, steps + 1):
        s, i, r = mean_field[step - 1] / n
        incidence = s * (1 - (1 - p * i) ** 4)
        recovery = q * i
        mean_field[step] = n * np.array(
            [s - incidence, i + incidence - recovery, r + recovery]
        )
    for array in (samples, ode, mean_field):
        array.setflags(write=False)
    return SIRComparison(
        samples,
        ode,
        mean_field,
        beta,
        gamma,
        sir_final_susceptible(initial, beta=beta, gamma=gamma),
    )
