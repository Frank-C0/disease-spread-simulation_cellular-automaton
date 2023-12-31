from cellular_automaton_interface import CellularAutomaton
import pyopencl as cl
import numpy as np
import time

# Número de booleanos por celda
NUM_BOOLS = 4


class StochasticCellularAutomatonOpenCLMemory(CellularAutomaton):
    def __init__(self, size=100, num_states=2, rule_kernel=None, initial_state=None, initializer_func=None):
        super().__init__(size, num_states, initial_state)
        print(self.SIZE)

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        self.kernel_code = rule_kernel

        self.prg = cl.Program(self.ctx, self.kernel_code).build()
        

        # Matrices para almacenar estados y memoria de cada celda
        # Utiliza un struct para almacenar el entero y los booleanos
        dtype = np.dtype([("value", np.uint8), ("state", np.uint8, NUM_BOOLS)])

        self.GRID1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.grid,
        )

        if initializer_func is None:
            initial_memory = np.zeros_like(self.grid, dtype=dtype)
        else:
            initial_memory = initializer_func(self.grid, np.zeros_like(self.grid, dtype=dtype))
            
        
        self.MEMORY1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=initial_memory,
        )


        self.GRID2 = cl.Buffer(
            self.ctx, 
            cl.mem_flags.READ_WRITE , 
            size=self.GRID1.size)
        
        self.MEMORY2 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE,
            size=self.MEMORY1.size,
        )

        self.ACTIVE_GRID = 1

        print(self.GRID1.size)
        print(self.GRID2.size)

    def update(self):
        # Genera un nuevo random seed para cada llamada a la función
        random_seed = np.random.randint(0, 2**31 - 1)

        if self.ACTIVE_GRID == 1:
            self.prg.automate(
                self.queue,
                (self.SIZE, self.SIZE),
                None,
                self.GRID1,
                self.MEMORY1,
                self.GRID2,
                self.MEMORY2,
                np.uint32(self.SIZE),
                np.uint32(self.NUM_STATES),
                np.uint32(NUM_BOOLS),
                np.uint32(random_seed),
            )
            self.ACTIVE_GRID = 2
            cl.enqueue_copy(self.queue, self.grid, self.GRID1)
            # cl.enqueue_copy(self.queue, self.memory, self.MEMORY1)
        else:
            self.prg.automate(
                self.queue,
                (self.SIZE, self.SIZE),
                None,
                self.GRID2,
                self.MEMORY2,
                self.GRID1,
                self.MEMORY1,
                np.uint32(self.SIZE),
                np.uint32(self.NUM_STATES),
                np.uint32(NUM_BOOLS),
                np.uint32(random_seed),
            )
            self.ACTIVE_GRID = 1
            cl.enqueue_copy(self.queue, self.grid, self.GRID2)
            # cl.enqueue_copy(self.queue, self.memory, self.MEMORY2)
        self.queue.finish()
