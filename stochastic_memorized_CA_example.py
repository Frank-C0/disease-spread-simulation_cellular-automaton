from cellular_automaton_interface import CellularAutomaton
from stochastic_memorized_cellular_automaton_opencl import StochasticCellularAutomatonOpenCLMemory
from cellular_automaton_gif import CellularAutomatonGif

class RandomRulesCA(StochasticCellularAutomatonOpenCLMemory):
    def __init__(self, size=100, initial_state=None):
        super().__init__(
            size,
            2, 
            rule_kernel=RandomRulesCA.kernel,
            initial_state=initial_state,
        )

    kernel = """
        typedef struct {
            uchar value;
            uchar val;
        } Cell;

        // Función para calcular el número aleatorio único para cada celda
        unsigned int calculateUniqueValue(int gid_x, int gid_y, int SIZE) {
            return gid_x + gid_y * SIZE;
        }

        // Función para calcular el número aleatorio a partir de la semilla y el valor único
        unsigned int calculateRandom(unsigned int seed, unsigned int unique_value) {
            return (1103515245 * (seed ^ unique_value) + 12345) % 0xFFFFFFFF;
        }

        // Función para convertir el número aleatorio en un valor de punto flotante entre 0 y 1
        float convertToFloat(unsigned int random) {
            return convert_float(random) / 0xFFFFFFFF;
        }

        // Función principal del kernel
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

            // Accede al estado actual y la memoria asociada
            int current_state = grid[index];
            Cell current_memory = memory[index];

            // Calcula el valor único para cada celda basado en la posición
            unsigned int unique_value = calculateUniqueValue(gid_x, gid_y, SIZE);

            // Calcula el número aleatorio único para cada celda
            unsigned int random = calculateRandom(seed, unique_value);

            // Convierte el número aleatorio en un valor de punto flotante entre 0 y 1
            float random_float = convertToFloat(random);

            // Define la probabilidad de cambio de estado
            float change_probability = 0.1 + current_memory.value * 0.05;

            // Cambia de estado con la probabilidad calculada
            int new_state = current_state;
            if (random_float < change_probability) {
                new_state = 1 - current_state;  // Cambia de 0 a 1 o de 1 a 0
                current_memory.value = 0;  // Reinicia el contador de duración del estado
            } else {
                current_memory.value += 1;  // Incrementa el contador de duración del estado
            }

            // Actualiza el estado y la memoria
            out_grid[index] = new_state;
            out_memory[index] = current_memory;
        }

    """
