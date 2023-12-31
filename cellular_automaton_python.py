from cellular_automaton_interface import CellularAutomaton
import numpy as np


class CellularAutomatonPython(CellularAutomaton):
    def __init__(self, size=100, num_states=2, rule=None, initial_state=None):
        super().__init__(size, num_states, initial_state)
        self.rule = rule
        self.grid = np.random.choice(
            range(num_states), (self.SIZE, self.SIZE)
        ).astype(np.int32)

    def update(self):
        new_grid = np.zeros_like(self.grid)

        for y in range(self.SIZE):
            for x in range(self.SIZE):
                new_grid[x, y] = self.rule(x, y)

        self.grid = new_grid
        return self.grid
