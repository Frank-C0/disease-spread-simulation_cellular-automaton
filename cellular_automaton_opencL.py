from cellular_automaton_interface import CellularAutomaton
import pyopencl as cl
import numpy as np
import time

class CellularAutomatonOpenCL(CellularAutomaton):
    def __init__(self, size=100, num_states=2, rule_kernel=None, initial_state=None):
        """
        Initializes the CellularAutomatonOpenCL object.

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

        # Build OpenCL program from the provided kernel code
        self.kernel_code = rule_kernel
        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        # Create OpenCL buffers for the two grids
        self.GRID1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.grid,
        )
        self.GRID2 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.GRID1.size)

        # Set the active grid to GRID1 initially
        self.ACTIVE_GRID = 1

        print(self.GRID1.size)
        print(self.GRID2.size)

    def update(self):
        """
        Updates the cellular automaton grid using the specified OpenCL kernel.

        The update is performed by executing the OpenCL kernel on the active grid
        and copying the result to the opposite grid.

        Note: This implementation assumes a specific kernel function named 'update'.
        """
        # t_start = time.time()
        if self.ACTIVE_GRID == 1:
            self.prg.update(
                self.queue,
                (self.SIZE, self.SIZE),
                None,
                self.GRID1,
                self.GRID2,
                np.uint32(self.SIZE),
                np.uint32(self.NUM_STATES),
            )
            self.ACTIVE_GRID = 2
            cl.enqueue_copy(self.queue, self.grid, self.GRID1)
        else:
            self.prg.update(
                self.queue,
                (self.SIZE, self.SIZE),
                None,
                self.GRID2,
                self.GRID1,
                np.uint32(self.SIZE),
                np.uint32(self.NUM_STATES),
            )
            self.ACTIVE_GRID = 1
            cl.enqueue_copy(self.queue, self.grid, self.GRID2)
        self.queue.finish()
        # print(f"\tcomputed in {(time.time() - t_start) * 1000:.2f}ms ({(1/ (time.time() - t_start)):.2f} fps)", flush=False,)
