from cellular_automaton_interface import CellularAutomaton
import pyopencl as cl
import numpy as np
import time


class CellularAutomatonOpenCL(CellularAutomaton):
    def __init__(self, size=100, num_states=2, rule_kernel=None, initial_state=None):
        super().__init__(size, num_states, initial_state)
        print(self.SIZE)

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        self.kernel_code = rule_kernel

        self.prg = cl.Program(self.ctx, self.kernel_code).build()

        self.GRID1 = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.grid,
            
        )
        self.GRID2 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.GRID1.size)

        self.ACTIVE_GRID = 1

        print(self.GRID1.size)
        print(self.GRID2.size)

    def update(self):
        
        if self.ACTIVE_GRID == 1:
            self.prg.gol(
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
            self.prg.gol(
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
        # unique_categories = np.unique(self.grid)
        # print("Categorías presentes en la cuadrícula:", unique_categories)
        # print(self.grid)
        # print(f"\tcomputed in {(time.time() - t_start) * 1000:.2f}ms ({(1/ (time.time() - t_start)):.2f} fps)", flush=False,)
