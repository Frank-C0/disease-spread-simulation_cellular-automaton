from cellular_automaton_interface import CellularAutomaton
from stochastic_memorized_cellular_automaton_opencl import (
    StochasticCellularAutomatonOpenCLMemory,
)
from cellular_automaton_gif import CellularAutomatonGif
import numpy as np


class IllAutomate(StochasticCellularAutomatonOpenCLMemory):
    def __init__(self, size=100, initial_state=None):
        super().__init__(
            size,
            4,
            rule_kernel=IllAutomate.kernel,
            initial_state=initial_state,
        )

    kernel = """
        #define R0 2.4
        #define START_ILL 0.001
        #define ILL_DURATION 14
        #define PC ((R0/ILL_DURATION)/8)

        typedef struct {
            uchar timesIll;
            uchar state[4];
        } Cell;

        Cell copyCell(Cell original) {
            Cell copy;
            copy.timesIll = original.timesIll;
            for (int i = 0; i < 4; ++i) {
                copy.state[i] = original.state[i];
            }
            return copy;
        }
        unsigned int calculateUniqueValue(int gid_x, int gid_y, int SIZE) {
            return gid_x + gid_y * SIZE;
        }

        unsigned int calculateRandom(unsigned int seed, unsigned int unique_value) {
            return (1103515245 * (seed ^ unique_value) + 12345) % 0xFFFFFFFF;
        }

        float convertToFloat(unsigned int random) {
            return convert_float(random) / 0xFFFFFFFF;
        }

        __kernel void automate(
            __global int *grid,
            __global Cell *memory,
            __global int *out_grid,
            __global Cell *out_memory,
            const unsigned int SIZE,
            const unsigned int NUM_STATES,
            const unsigned int NUM_BOOLS,
            const unsigned int seed
        ) {
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);
            int index = gid_y * SIZE + gid_x;

            int isIll = 0;
            int isInmune = 2;
            int wasIll = 1;

            Cell current_memory = memory[index];
            Cell new_memory = copyCell(current_memory);

            // Accede al estado actual de la celda
            int current_state = grid[index];

            // Calcula el valor único para cada celda basado en la posición
            unsigned int unique_value = calculateUniqueValue(gid_x, gid_y, SIZE);

            // Calcula el número aleatorio único para cada celda
            unsigned int random = calculateRandom(seed, unique_value);

            // Convierte el número aleatorio en un valor de punto flotante entre 0 y 1
            float random_float = convertToFloat(random);

            // Implementa las reglas de procesamiento
            if (!new_memory.state[isIll] && !new_memory.state[wasIll] && !new_memory.state[isInmune]) {
                
                 int surrounding = 0;

                // Recorre las celdas circundantes
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        // Evita el índice actual
                        if (i == 0 && j == 0) continue;

                        // Calcula las coordenadas de la celda vecina
                        int neighbor_x = gid_x + i;
                        int neighbor_y = gid_y + j;

                        // Verifica los límites para asegurarse de no salirse de la cuadrícula
                        if (neighbor_x >= 0 && neighbor_x < SIZE && neighbor_y >= 0 && neighbor_y < SIZE) {
                            // Obtiene el índice de la celda vecina
                            int neighbor_index = neighbor_y * SIZE + neighbor_x;

                            // Incrementa el contador si la celda vecina está enferma
                            surrounding += memory[neighbor_index].state[isIll];
                        }
                    }
                }

                if (random_float < 1 - pow(1 - PC, surrounding)) {
                    new_memory.state[isIll] = 1;
                    new_memory.timesIll = ILL_DURATION;

                }
            } else if (new_memory.state[isIll]) {
                new_memory.timesIll--;
                if (new_memory.timesIll == 0) {
                    new_memory.state[isIll] = 0;
                    new_memory.state[wasIll] = 1;
                }
            }

            int new_state = 0;
            if (new_memory.state[isIll]) {
                new_state = 1;  // Estado para células enfermas
            } else if (new_memory.state[wasIll]) {
                new_state = 2;  // Estado para células que estuvieron enfermas
            } else if (new_memory.state[isInmune]) {
                new_state = 3;  // Estado para células inmunes
            } else {
                new_state = 0;  // Estado para células saludables
            }

            // Actualiza el estado y la memoria
            out_grid[index] = new_state;
            out_memory[index] = new_memory;
        }

    """


if __name__ == "__main__":
    automaton = IllAutomate(
        initial_state=CellularAutomaton.load_image(
            "R:\\Labs\\FC-Lab1\\TIF\\initial_desired.bmp", num_states=4
        ),
    )
    # automaton = IllAutomate(
    #     initial_state=np.array(
    #         [
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    #             [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0],
    #             [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         ]
    #     )
    # )

    gif_generator = CellularAutomatonGif(
        max_frames=500,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=automaton,
        filename_gif="opencl_memory_desired_gif.gif",
        frame_rate=5.0,
        num_states=4,
        colors=[
            (0, 255, 0), 
            (255, 0, 0), 
            (0, 255, 255), 
            (0, 0, 255)],
    )
    gif_generator.generate_frames()
    gif_generator.combine_gifs()

    
