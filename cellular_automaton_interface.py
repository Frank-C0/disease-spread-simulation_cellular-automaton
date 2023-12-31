import numpy as np


class CellularAutomaton:
    def __init__(self, size, initial_state=None):
        self.SIZE = size
        if initial_state is None:
            self.grid = initial_state = np.random.choice(
                (False, True), (self.SIZE, self.SIZE)
            ).astype(np.int32)

    def update():
        raise NotImplementedError()
