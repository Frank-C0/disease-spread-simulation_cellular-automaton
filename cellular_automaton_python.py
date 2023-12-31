from cellular_automaton_interface import CellularAutomaton
import numpy as np


class CellularAutomatonPython(CellularAutomaton):
    def __init__(self, size=100, rule=None, initial_state=None):
        super().__init__(size, initial_state)
        self.rule = rule
        self.grid = np.random.choice((False, True), (self.SIZE, self.SIZE)).astype(
            np.int32
        )

    def update(self):
        new_grid = np.zeros_like(self.grid)

        for y in range(self.SIZE):
            for x in range(self.SIZE):
                new_grid[x, y] = self.rule(x, y)

        self.grid = new_grid
        return self.grid
