import math
import pyopencl as cl
import numpy as np
import time
import imageio
from PIL import Image
import os

SIZE = 1000
MAX_FRAMES = 100
SAVE_INTERVAL = 10
OUTPUT_FOLDER = 'output_gifs/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Configurar el entorno de OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Definir el kernel de OpenCL
kernel_code = """
    __kernel void gol(__global int *grid, __global int *out_grid, const unsigned int SIZE) {
        int x = get_global_id(0);
        int y = get_global_id(1);

        int n = SIZE;
        int total = (
            grid[y * n + (x - 1) % n] + grid[y * n + (x + 1) % n] +
            grid[((y - 1) % n) * n + x] + grid[((y + 1) % n) * n + x] +
            grid[((y - 1) % n) * n + (x - 1) % n] + grid[((y - 1) % n) * n + (x + 1) % n] +
            grid[((y + 1) % n) * n + (x - 1) % n] + grid[((y + 1) % n) * n + (x + 1) % n]
        );

        if (grid[y * n + x]) {
            out_grid[y * n + x] = (total == 2) || (total == 3);
        } else {
            out_grid[y * n + x] = (total == 3);
        }
    }
"""

# Compilar el kernel
prg = cl.Program(ctx, kernel_code).build()

# Crear los buffers de OpenCL
GRID1 = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.random.choice((False, True), (SIZE, SIZE)).astype(np.int32))
GRID2 = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=GRID1.size)

ACTIVE_GRID = 1

def div_or_inf(a, b):
    return a / b if b else math.inf

def save_frames(frames, filename):
    imageio.mimsave(filename, frames, fps=10, palettesize=2)
    print(f"Saved {len(frames)} frames to {filename}")

def update(frame, save_interval=SAVE_INTERVAL):
    """
    Update the game of life simulation for a given frame.

    Parameters:
    - frame: The frame number.
    - save_interval: The interval at which to save frames.

    Returns:
    - img: The updated image.
    """
    global ACTIVE_GRID

    t_start = time.time()

    if ACTIVE_GRID == 1:
        prg.gol(queue, (SIZE, SIZE), None, GRID1, GRID2, np.uint32(SIZE))
        ACTIVE_GRID = 2
    else:
        prg.gol(queue, (SIZE, SIZE), None, GRID2, GRID1, np.uint32(SIZE))
        ACTIVE_GRID = 1

    queue.finish()

    print(f"\tcomputed in {(time.time() - t_start) * 1000:.2f}ms ({div_or_inf(1, (time.time() - t_start)):.2f} fps)", flush=False)
    print(f"total gpu time: {(time.time() - t_start) * 1000:.2f}ms ({div_or_inf(1, (time.time() - t_start)):.2f} fps)\n")

    # Convertir a imagen de escala de grises
    data = np.empty((SIZE, SIZE), dtype=np.int32)
    cl.enqueue_copy(queue, data, GRID1)

    img = Image.fromarray(data.astype('uint8') * 255)
    img = img.convert('L')

    return img



# Crear una lista para almacenar los fotogramas
frames = []

# Generar fotogramas y almacenarlos
for frame in range(MAX_FRAMES):
    img = update(frame)
    frames.append(np.array(img))  # Convertir de nuevo a un array de NumPy para imageio

    if frame % SAVE_INTERVAL == 0 and frame > 0:
        filename = f'{OUTPUT_FOLDER}gol_animation_bw_{frame}.gif'
        save_frames(frames, filename)
        frames = []  # Limpiar la lista de frames para liberar memoria

# Juntar todos los GIFs en uno solo
all_gifs = [f'{OUTPUT_FOLDER}gol_animation_bw_{i}.gif' for i in range(SAVE_INTERVAL, MAX_FRAMES, SAVE_INTERVAL)]

# Guardar los fotogramas restantes como un GIF
if frames:
  final_filename = f'{OUTPUT_FOLDER}gol_animation_bw_final.gif'
  save_frames(frames, final_filename)
  all_gifs.append(final_filename)

print(str(all_gifs))
# Juntar todos los GIFs en uno solo
combined_filename = 'combined_gif.gif'

with imageio.get_writer(combined_filename, fps=10, palettesize=2) as writer:
    for filename in all_gifs:
        # Utilizar imageio.mimread para leer cada GIF uno por uno
        frames_to_add = imageio.mimread(filename)
        # Agregar cada frame al nuevo GIF
        for frame in frames_to_add:
            writer.append_data(frame)

print(f'Combined GIF saved to {combined_filename}')

