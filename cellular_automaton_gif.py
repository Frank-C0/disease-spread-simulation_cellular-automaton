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
        frame_rate=10,
        colors=None,
        num_states=2,
    ):
        # Parámetros de la animación
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self.save_interval = save_interval
        self.output_folder = output_folder
        self.automaton = automaton
        self.num_states = num_states
        self.colors = colors or self.generate_default_colors(num_states)
        self.gif_saver = GifSaver(output_folder, frame_rate=self.frame_rate)
        self.filename_gif = filename_gif

        # Inicializar la primera imagen en la lista de frames del GIF
        img = self.generate_image_from_grid(self.automaton.grid)
        self.gif_saver.frames.append(np.array(img))

    def generate_default_colors(self, num_states):
        # Generar colores predeterminados si no se proporcionan
        return [(np.random.randint(256), np.random.randint(256), np.random.randint(256)) for _ in range(num_states)]

    def generate_frames(self):
        # Generar frames para la animación
        for frame in range(self.max_frames):
            self.automaton.update()
            img = self.generate_image_from_grid(self.automaton.grid)

            # Añadir el frame actual a la lista de frames del GIF
            self.gif_saver.frames.append(np.array(img))

            # Guardar el GIF parcial cada cierto intervalo
            if frame % self.save_interval == 0 and frame > 0:
                filename = f"{self.output_folder}automaton_animation_color_{frame}.gif"
                self.gif_saver.save_frames(self.gif_saver.frames, filename)
                self.gif_saver.frames = []

        # Guardar el último conjunto de frames si queda alguno
        if self.gif_saver.frames:
            final_filename = f"{self.output_folder}automaton_animation_color_final.gif"
            self.gif_saver.save_frames(self.gif_saver.frames, final_filename)

    def generate_image_from_grid(self, grid):
        # Convertir la cuadrícula a una imagen usando colores
        height, width = grid.shape
        img_data = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(self.num_states):
            img_data[grid == i] = self.colors[i]

        img = Image.fromarray(img_data, 'RGB')
        return img

    def combine_gifs(self):
        # Combinar los GIFs parciales en uno solo
        self.gif_saver.combine_gifs(
            self.save_interval, self.max_frames, self.filename_gif
        )
