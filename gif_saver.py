import math
import imageio
from PIL import Image
import os
import numpy as np


class GifSaver:
    def __init__(self, output_folder, frame_rate=10):
        self.OUTPUT_FOLDER = output_folder
        self.FRAME_RATE = frame_rate

        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)

        self.frames = []

    def div_or_inf(self, a, b):
        return a / b if b else math.inf

    def save_frames(self, frames, filename, num_colors=256):
        # Convertir la lista de frames a un arreglo de imágenes
        image_array = np.array(frames)

        # Redimensionar la imagen a 8 bits por canal RGB
        image_array = (image_array * 255).astype(np.uint8)

        # Guardar el GIF
        imageio.mimsave(
            filename, image_array, fps=self.FRAME_RATE, quantizer="nq", palettesize=num_colors
        )
        print(f"Saved {len(frames)} frames to {filename}")

    def combine_gifs(
        self, save_interval, max_frames, combined_filename="combined_gif.gif", num_colors=256
    ):
        all_gifs = [
            f"{self.OUTPUT_FOLDER}automaton_animation_color_{i}.gif"
            for i in range(save_interval, max_frames, save_interval)
        ]

        final_filename = f"{self.OUTPUT_FOLDER}automaton_animation_color_final.gif"
        if os.path.exists(final_filename):
            all_gifs.append(final_filename)

        with imageio.get_writer(
            combined_filename, fps=self.FRAME_RATE, quantizer="nq", palettesize=num_colors
        ) as writer:
            for gif_file in all_gifs:
                frames_to_add = imageio.mimread(gif_file)

                # Convertir la lista de frames a un arreglo de imágenes
                image_array = np.array(frames_to_add)

                # Redimensionar la imagen a 8 bits por canal RGB
                image_array = (image_array * 255).astype(np.uint8)

                # Agregar los frames al nuevo GIF
                for frame in image_array:
                    writer.append_data(frame)

        print(f"Combined GIF saved to {combined_filename}")
