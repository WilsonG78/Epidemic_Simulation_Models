import matplotlib.pyplot as plt

from simulation.visualization import create_simulation, run


def main() -> None:
    result = run(create_simulation(), steps=100)
    _animation = result.animate()
    plt.show()
