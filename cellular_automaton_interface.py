import numpy as np
from PIL import Image

class CellularAutomaton:
    def __init__(self, size=100, num_states=2, initial_state=None):
        # Verificar las categorías presentes en la cuadrícula inicial
        # unique_categories = np.unique(initial_state)
        # print("Categorías presentes en la cuadrícula:", unique_categories)
        # print(initial_state)
        
        # Inicializar la cuadrícula según la presencia o ausencia de un estado inicial
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
        # Método abstracto para actualizar la cuadrícula, debe ser implementado por clases derivadas
        raise NotImplementedError()

    @staticmethod
    def load_image(image_path, num_states=2):
        # Cargar una imagen y convertirla a una cuadrícula con estados discretos
        img = Image.open(image_path).convert("L")
        pixel_data = np.array(img)

        # Convertir los valores de píxeles a categorías discretas según el número de estados
        grid = np.digitize(pixel_data, bins=np.linspace(0, 255, num_states), right=True) 

        print(np.linspace(0, 255, num_states))

        return grid.astype(np.int32)
