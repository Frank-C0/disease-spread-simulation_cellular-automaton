import numpy as np
from PIL import Image


class CellularAutomaton:
    def __init__(self, size=100, initial_state=None):
        if initial_state is None:
            self.SIZE = size
            self.grid = np.random.choice(
                (False, True), (self.SIZE, self.SIZE)
            ).astype(np.int32)
        else:
            self.SIZE = initial_state.shape[0]  # Obtener el tamaño del estado inicial
            self.grid = initial_state
        print(initial_state)

    def update():
        raise NotImplementedError()

    def load_image(image_path, umbral=128):
    # Abrir la imagen
        img = Image.open(image_path).convert("L")  # Convertir a escala de grises

        # Obtener los datos de píxeles como un array de NumPy
        pixel_data = np.array(img)

        # Normalizar los valores de píxeles a 0 o 1
        grid = (pixel_data > umbral).astype(np.int32)

        return grid
