"""Continuous, homogeneous SIR model with population fractions summing to one."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class SIR:
    """Solve SIR equations with fixed transmission and recovery rates.

    ``population`` contains initial fractions (S, I, R), not individual counts.
    For part A, draw random parameters before creating each model.
    """

    def __init__(
        self,
        population: tuple[float, float, float],
        beta: float,
        gamma: float,
    ):
        initial = np.asarray(population, dtype=float)
        if (
            initial.shape != (3,)
            or not np.isfinite(initial).all()
            or np.any(initial < 0)
            or np.any(initial > 1)
            or not np.isclose(initial.sum(), 1.0, rtol=0, atol=1e-12)
        ):
            raise ValueError("population must contain fractions S, I, R summing to 1")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        self.S, self.I, self.R = initial
        self.beta = float(beta)
        self.gamma = float(gamma)

    def derivatives(self, t, y):
        """Return dS/dt, dI/dt, dR/dt for the supplied state."""
        S, I, _R = y
        infections = self.beta * S * I
        recoveries = self.gamma * I
        return [-infections, infections - recoveries, recoveries]

    def solve(
        self,
        t_span: tuple[float, float] = (0.0, 160.0),
        *,
        points: int = 1000,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ):
        """Return a SciPy solution without changing the initial population.

        RK45 chooses its own internal steps. ``points`` controls only the number
        of evenly spaced output times. The final R value is R(t_end), not an
        automatic estimate of R(infinity).
        """
        times = np.asarray(t_span, dtype=float)
        if times.shape != (2,) or not np.isfinite(times).all() or times[1] <= times[0]:
            raise ValueError("t_span must contain two finite, increasing times")
        if not isinstance(points, (int, np.integer)) or points < 2:
            raise ValueError("points must be an integer >= 2")
        if not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be finite and positive")
        if not np.isfinite(atol) or atol <= 0:
            raise ValueError("atol must be finite and positive")
        solution = solve_ivp(
            self.derivatives,
            t_span=times,
            y0=[self.S, self.I, self.R],
            t_eval=np.linspace(*times, points),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"SIR integration failed: {solution.message}")
        return solution

    def plot(self, solution=None, *, ax=None):
        """Plot a solution and return (figure, axes); call plt.show() to display.

        Without a supplied solution, solve over the default time interval.
        """
        if solution is None:
            solution = self.solve()
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for values, label, color in zip(
            solution.y,
            ("Susceptible", "Infectious", "Recovered"),
            ("#44aa77", "#ee5555", "#4477aa"),
        ):
            ax.plot(solution.t, values, label=label, color=color)
        ax.set(
            xlabel="Time",
            ylabel="Population fraction",
            title="Homogeneous SIR model",
            ylim=(0, 1),
        )
        ax.grid(alpha=0.25)
        ax.legend()
        return ax.figure, ax


if __name__ == "__main__":
    model = SIR(population=(0.99, 0.01, 0.0), beta=0.3, gamma=0.1)
    model.plot()
    plt.show()
