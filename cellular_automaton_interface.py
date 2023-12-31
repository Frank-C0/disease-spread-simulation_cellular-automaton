import numpy as np
from PIL import Image


class CellularAutomaton:
    def __init__(self, size=100, num_states=2, initial_state=None):
        if initial_state is None:
            self.SIZE = size
            self.NUM_STATES = num_states
            self.grid = np.random.choice(
                range(num_states), (self.SIZE, self.SIZE)
            ).astype(np.int32)
        else:
            self.SIZE = initial_state.shape[0]
            self.NUM_STATES = num_states
            self.grid = initial_state

    def update(self):
        raise NotImplementedError()

    @staticmethod
    def load_image(image_path, num_states=2, umbral=128):
        img = Image.open(image_path).convert("L")
        pixel_data = np.array(img)
        grid = np.digitize(pixel_data, bins=np.linspace(0, 255, num_states), right=True) - 1
        return grid
