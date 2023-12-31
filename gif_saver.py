import math
import imageio
from PIL import Image
import os


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

    def combine_gifs(
        self, save_interval, max_frames, combined_filename="combined_gif.gif"
    ):
        all_gifs = [
            f"{self.OUTPUT_FOLDER}automaton_animation_bw_{i}.gif"
            for i in range(save_interval, max_frames, save_interval)
        ]

        final_filename = f"{self.OUTPUT_FOLDER}automaton_animation_bw_final.gif"
        if os.path.exists(final_filename):
            all_gifs.append(final_filename)

        print(all_gifs)
        with imageio.get_writer(combined_filename, fps=10, palettesize=2) as writer:
            for gif_file in all_gifs:
                frames_to_add = imageio.mimread(gif_file)

                for frame in frames_to_add:
                    writer.append_data(frame)

        print(f"Combined GIF saved to {combined_filename}")
