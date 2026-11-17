from enum import IntEnum


class State(IntEnum):
    SUSCEPTIBLE = 1
    INFECTIOUS = 2
    RECOVERED = 3
