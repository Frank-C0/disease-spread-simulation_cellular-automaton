import numpy as np
from PIL import Image
from gif_saver import GifSaver


class CellularAutomatonGif:
    def __init__(
        self,
        automaton,
        max_frames,
        save_interval,
        output_folder,
        filename_gif="combined_gif.gif",
    ):
        self.max_frames = max_frames
        self.save_interval = save_interval
        self.output_folder = output_folder
        self.automaton = automaton
        self.gif_saver = GifSaver(output_folder)
        self.filename_gif = filename_gif

    def generate_frames(self):
        for frame in range(self.max_frames):
            self.automaton.update()
            img = Image.fromarray(self.automaton.grid.astype("uint8") * 255)
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
        self.gif_saver.combine_gifs(
            self.save_interval, self.max_frames, self.filename_gif
        )
