from cellular_automaton_interface import CellularAutomaton
from cellular_automaton_gif import CellularAutomatonGif
from cellular_automaton_opencL import CellularAutomatonOpenCL
from cellular_automaton_python import CellularAutomatonPython


class GameOfLifeAutomatonOpenCL(CellularAutomatonOpenCL):
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
    def __init__(self, size=100, initial_state=None):
        super().__init__(size, rule_kernel=GameOfLifeAutomatonOpenCL.game_of_life_kernel, initial_state=initial_state)

    

class GameOfLifeAutomatonPython(CellularAutomatonPython):
    def __init__(self, size=100, initial_state=None):
        super().__init__(size, rule=self.game_of_life_rule, initial_state=initial_state)

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
    # game_of_life = GameOfLifeAutomatonPython(100)
    # game = CellularAutomatonGif(
    #     max_frames=100,
    #     save_interval=10,
    #     output_folder="output_gifs/",
    #     automaton=game_of_life,
    #     filename_gif="python_gif.gif"
    # )
    # game.generate_frames()
    # game.combine_gifs()

    # game_of_life = GameOfLifeAutomatonOpenCL(size=100)
    game_of_life = GameOfLifeAutomatonOpenCL(initial_state=CellularAutomaton.load_image('R:\Labs\FC-Lab1\TIF\input.bmp'))
    gif_generator = CellularAutomatonGif(
        max_frames=100,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=game_of_life,
        filename_gif="opencl_gif.gif"
    )
    gif_generator.generate_frames()
    gif_generator.combine_gifs()
