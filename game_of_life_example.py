from cellular_automaton_interface import CellularAutomaton
from cellular_automaton_gif import CellularAutomatonGif
from cellular_automaton_opencL import CellularAutomatonOpenCL
from cellular_automaton_python import CellularAutomatonPython


class GameOfLifeAutomatonOpenCL(CellularAutomatonOpenCL):
    game_of_life_kernel = """
        __kernel void gol(__global int *grid, __global int *out_grid, const unsigned int SIZE, const unsigned int NUM_STATES) {
            int x = get_global_id(0);
            int y = get_global_id(1);

            int n = SIZE;
            int total = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    total += grid[((y + i) % n) * n + (x + j) % n];
                }
            }

            total -= grid[y * n + x];

            if (grid[y * n + x] > 0) {
                // Any live cell with two or three live neighbors survives.
                out_grid[y * n + x] = (total == 2 || total == 3) ? grid[y * n + x] : 0;
            } else {
                // Any dead cell with three live neighbors becomes a live cell.
                out_grid[y * n + x] = (total == 3) ? 1 : 0;
            }
        }
    """

    def __init__(self, size=100, num_states=2, initial_state=None):
        super().__init__(
            size, num_states, rule_kernel=GameOfLifeAutomatonOpenCL.game_of_life_kernel, initial_state=initial_state
        )
    

class GameOfLifeAutomatonPython(CellularAutomatonPython):
    def __init__(self, size=100, num_states=2, initial_state=None):
        super().__init__(
            size, num_states, rule=self.game_of_life_rule, initial_state=initial_state
        )

    def game_of_life_rule(self, x, y):
        n = self.SIZE
        total = (
            self.grid[(x - 1) % n, (y - 1) % n]
            + self.grid[x, (y - 1) % n]
            + self.grid[(x + 1) % n, (y - 1) % n]
            + self.grid[(x - 1) % n, y]
            + self.grid[(x + 1) % n, y]
            + self.grid[(x - 1) % n, (y + 1) % n]
            + self.grid[x, (y + 1) % n]
            + self.grid[(x + 1) % n, (y + 1) % n]
        )

        if self.grid[x, y] > 0:
            return (
                self.grid[x, y]
                if total == 2 or total == 3
                else 0
            )
        else:
            return (
                self.grid[x, y]
                if total == 3
                else 0
            )



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
    game_of_life = GameOfLifeAutomatonOpenCL(num_states=2, initial_state=CellularAutomaton.load_image("R:\\Labs\\FC-Lab1\\TIF\\input.bmp"))
    gif_generator = CellularAutomatonGif(
        max_frames=100,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=game_of_life,
        filename_gif="opencl_gif.gif"
    )
    gif_generator.generate_frames()
    gif_generator.combine_gifs()
