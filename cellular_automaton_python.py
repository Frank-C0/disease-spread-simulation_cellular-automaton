from cellular_automaton_interface import CellularAutomaton
import numpy as np


class CellularAutomatonPython(CellularAutomaton):
    def __init__(self, size, rule):
        self.SIZE = size
        self.rule = rule
        self.grid = np.random.choice((False, True), (self.SIZE, self.SIZE)).astype(
            np.int32
        )
        super().__init__(size)

    def update(self):
        new_grid = np.zeros_like(self.grid)

        for y in range(self.SIZE):
            for x in range(self.SIZE):
                new_grid[x, y] = self.rule(x, y)

        self.grid = new_grid
        return self.grid
