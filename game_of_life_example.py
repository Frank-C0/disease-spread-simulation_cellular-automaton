from cellular_automaton_gif import CellularAutomatonGif
from cellular_automaton_opencL import CellularAutomatonOpenCL
from cellular_automaton_python import CellularAutomatonPython

# Example: Game of Life Rule Kernel
game_of_life_kernel = """
    __kernel void gol(__global int *grid, __global int *out_grid, const unsigned int SIZE) {
        int x = get_global_id(0);
        int y = get_global_id(1);

        int n = SIZE;
        int total = (
            grid[y * n + (x - 1) % n] + grid[y * n + (x + 1) % n] +
            grid[((y - 1) % n) * n + x] + grid[((y + 1) % n) * n + x] +
            grid[((y - 1) % n) * n + (x - 1) % n] + grid[((y - 1) % n) * n + (x + 1) % n] +
            grid[((y + 1) % n) * n + (x - 1) % n] + grid[((y + 1) % n) * n + (x + 1) % n]
        );

        if (grid[y * n + x]) {
            out_grid[y * n + x] = (total == 2) || (total == 3);
        } else {
            out_grid[y * n + x] = (total == 3);
        }
    }

"""


class GameOfLifeAutomaton(CellularAutomatonPython):
    def __init__(self, size):
        super().__init__(size, rule=self.game_of_life_rule)

    def game_of_life_rule(self, x, y):
        n = self.SIZE
        total = (
            self.grid[(x - 1) % n, (y - 1) % n] +
            self.grid[x, (y - 1) % n] +
            self.grid[(x + 1) % n, (y - 1) % n] +
            self.grid[(x - 1) % n, y] +
            self.grid[(x + 1) % n, y] +
            self.grid[(x - 1) % n, (y + 1) % n] +
            self.grid[x, (y + 1) % n] +
            self.grid[(x + 1) % n, (y + 1) % n]
        )

        if self.grid[x, y]:
            return 1 if total == 2 or total == 3 else 0
        else:
            return 1 if total == 3 else 0


if __name__ == "__main__":
    game_of_life = GameOfLifeAutomaton(100)
    game = CellularAutomatonGif(
        max_frames=100,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=game_of_life,
        filename_gif="python_gif.gif"
    )
    game.generate_frames()
    game.combine_gifs()

    game_of_life = CellularAutomatonOpenCL(size=100, rule_kernel=game_of_life_kernel)
    game = CellularAutomatonGif(
        max_frames=100,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=game_of_life,
        filename_gif="opencl_gif.gif"
    )
    game.generate_frames()
    game.combine_gifs()
