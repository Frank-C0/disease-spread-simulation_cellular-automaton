import math
import pyopencl as cl
import numpy as np
import time
import imageio
from PIL import Image
import os


class PythonRuleAutomaton:
    def __init__(self, size, rule):
        self.SIZE = size
        self.rule = rule
        self.grid = np.random.choice((False, True), (self.SIZE, self.SIZE)).astype(np.int32)

    def update(self):
        new_grid = np.zeros_like(self.grid)

        for y in range(self.SIZE):
            for x in range(self.SIZE):
                new_grid[x, y] = self.rule(x, y)

        self.grid = new_grid
        return self.grid


class CellularAutomatonOpenCL:
    def __init__(self, size, rule_kernel, initial_state=None):
        self.SIZE = size
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        self.kernel_code = rule_kernel

        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        if initial_state is None:
            initial_state = np.random.choice(
                (False, True), (self.SIZE, self.SIZE)
            ).astype(np.int32)

        self.GRID1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=initial_state,
        )
        self.GRID2 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.GRID1.size)

        self.ACTIVE_GRID = 1


    def update(self):
        t_start = time.time()

        if self.ACTIVE_GRID == 1:
            self.prg.gol(self.queue, (self.SIZE, self.SIZE), None, self.GRID1, self.GRID2, np.uint32(self.SIZE))
            self.ACTIVE_GRID = 2
            img_data = np.empty((self.SIZE, self.SIZE), dtype=np.int32)
            cl.enqueue_copy(self.queue, img_data, self.GRID1)
        else:
            self.prg.gol(self.queue, (self.SIZE, self.SIZE), None, self.GRID2, self.GRID1, np.uint32(self.SIZE))
            self.ACTIVE_GRID = 1
            img_data = np.empty((self.SIZE, self.SIZE), dtype=np.int32)
            cl.enqueue_copy(self.queue, img_data, self.GRID2)

        self.queue.finish()

        print(f"\tcomputed in {(time.time() - t_start) * 1000:.2f}ms ({(1/ (time.time() - t_start)):.2f} fps)", flush=False)
        print(f"total gpu time: {(time.time() - t_start) * 1000:.2f}ms ({(1/ (time.time() - t_start)):.2f} fps)\n")

        # img = Image.fromarray(img_data.astype('uint8') * 255)
        # img = img.convert('L')

        return img_data


class GifSaver:
    def __init__(self, output_folder):
        self.OUTPUT_FOLDER = output_folder

        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

        self.frames = []

    def div_or_inf(self, a, b):
        return a / b if b else math.inf

    def save_frames(self, frames, filename):
        imageio.mimsave(filename, frames, fps=10, palettesize=2)
        print(f"Saved {len(frames)} frames to {filename}")

    def combine_gifs(self, save_interval, max_frames):
        all_gifs = [
            f"{self.OUTPUT_FOLDER}automaton_animation_bw_{i}.gif"
            for i in range(save_interval, max_frames, save_interval)
        ]

        final_filename = f"{self.OUTPUT_FOLDER}automaton_animation_bw_final.gif"
        if os.path.exists(final_filename):
            all_gifs.append(final_filename)

        combined_filename = "combined_gif.gif"
        print(all_gifs)
        with imageio.get_writer(combined_filename, fps=10, palettesize=2) as writer:
            for gif_file in all_gifs:
                frames_to_add = imageio.mimread(gif_file)

                for frame in frames_to_add:
                    writer.append_data(frame)

        print(f"Combined GIF saved to {combined_filename}")


class CellularAutomatonGif:
    def __init__(self, max_frames, save_interval, output_folder, automaton):
        self.max_frames = max_frames
        self.save_interval = save_interval
        self.output_folder = output_folder
        self.automaton = automaton
        self.gif_saver = GifSaver(output_folder)

    def generate_frames(self):
        for frame in range(self.max_frames):
            data = self.automaton.update()
            img = Image.fromarray(data.astype("uint8") * 255)
            img = img.convert("L")
            self.gif_saver.frames.append(np.array(img))

            if frame % self.save_interval == 0 and frame > 0:
                filename = f"{self.output_folder}automaton_animation_bw_{frame}.gif"
                self.gif_saver.save_frames(self.gif_saver.frames, filename)
                self.gif_saver.frames = []

        if self.gif_saver.frames:
            final_filename = f"{self.output_folder}automaton_animation_bw_final.gif"
            self.gif_saver.save_frames(self.gif_saver.frames, final_filename)

    def combine_gifs(self):
        self.gif_saver.combine_gifs(self.save_interval, self.max_frames)


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


class GameOfLifeAutomaton(PythonRuleAutomaton):
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
    # game_of_life = CellularAutomatonOpenCL(size=100, rule_kernel=game_of_life_kernel)
    game_of_life = GameOfLifeAutomaton(100)
    game = CellularAutomatonGif(
        max_frames=100,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=game_of_life,
    )
    game.generate_frames()
    game.combine_gifs()
