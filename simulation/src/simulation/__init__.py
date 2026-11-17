"""Spatial SIR simulation and notebook visualization API."""

from simulation.core.simulation import Simulation
from simulation.interactive import SimulationExplorer, explore
from simulation.timing import BenchmarkResult, TimedRun, benchmark, run_timed
from simulation.utils.state import State
from simulation.validation import (
    SIRComparison,
    compare_sir,
    sir_final_susceptible,
    sir_ode,
)
from simulation.visualization import SimulationResult, create_simulation, run

__all__ = [
    "BenchmarkResult",
    "SIRComparison",
    "Simulation",
    "SimulationExplorer",
    "SimulationResult",
    "State",
    "TimedRun",
    "benchmark",
    "compare_sir",
    "create_simulation",
    "explore",
    "run",
    "run_timed",
    "sir_final_susceptible",
    "sir_ode",
]
