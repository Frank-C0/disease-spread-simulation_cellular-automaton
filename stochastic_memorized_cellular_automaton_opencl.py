from cellular_automaton_interface import CellularAutomaton
import pyopencl as cl
import numpy as np
import time

# Number of booleans per cell
NUM_BOOLS = 4

class StochasticCellularAutomatonOpenCLMemory(CellularAutomaton):
    def __init__(self, size=100, num_states=2, rule_kernel=None, initial_state=None, initializer_func=None):
        """
        Initializes the StochasticCellularAutomatonOpenCLMemory object.

        Parameters:
        - size: Size of the square grid.
        - num_states: Number of states for each cell.
        - rule_kernel: OpenCL kernel code for the cellular automaton rule.
        - initial_state: Initial state of the grid (optional).
        """
        super().__init__(size, num_states, initial_state)
        print(self.SIZE)

        # Initialize OpenCL platform, device, context, and command queue
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        self.kernel_code = rule_kernel

        # Build OpenCL program from the provided kernel code
        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        # Define a structured data type for memory with an integer and boolean array
        dtype = np.dtype([("value", np.uint8), ("state", np.uint8, NUM_BOOLS)])

        # Create OpenCL buffers for the grid and memory
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

        # Create additional buffers for the second grid and memory
        self.GRID2 = cl.Buffer(
            self.ctx, 
            cl.mem_flags.READ_WRITE, 
            size=self.GRID1.size
        )
        
        self.MEMORY2 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE,
            size=self.MEMORY1.size,
        )

        # Set the active grid to GRID1 initially
        self.ACTIVE_GRID = 1

        print(self.GRID1.size)
        print(self.GRID2.size)

    def update(self):
        # Generate a new random seed for each function call
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
        self.queue.finish()
