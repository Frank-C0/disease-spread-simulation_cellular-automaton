import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_f
from game_of_life_example import GameOfLifeAutomatonOpenCL, GameOfLifeAutomatonPython
import numpy as np

class CellularAutomatePygame:
    def __init__(self, automaton, initial_screen_size):
        self.automaton = automaton
        self.size = automaton.SIZE
        self.screen_size = initial_screen_size
        self.cell_size = self.screen_size // self.size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)
        pygame.display.set_caption('Cellular Automaton')

        self.fullscreen = False
        self.cell_changed = np.zeros((self.size, self.size), dtype=bool)
        self.white_surface = pygame.Surface((self.cell_size, self.cell_size))
        self.white_surface.fill((255, 255, 255))
        self.black_surface = pygame.Surface((self.cell_size, self.cell_size))
        self.black_surface.fill((0, 0, 0))

        self.offset_x = 0
        self.offset_y = 0

    def draw_grid(self, data):
        self.screen.fill((255, 255, 255))  # Llenar la pantalla con blanco

        cell_size_x = max(1, self.screen_size // self.size)
        cell_size_y = max(1, self.screen_size // self.size)

        for y in range(self.size):
            for x in range(self.size):
                pixel_x = (x * cell_size_x) - int(self.offset_x * cell_size_x)
                pixel_y = (y * cell_size_y) - int(self.offset_y * cell_size_y)

                if 0 <= pixel_x < self.screen_size and 0 <= pixel_y < self.screen_size:
                    if data[y, x] or self.cell_changed[y, x]:
                        surface = self.black_surface if data[y, x] else self.white_surface
                        scaled_surface = pygame.transform.scale(surface, (cell_size_x, cell_size_y))
                        rect = scaled_surface.get_rect(topleft=(pixel_x, pixel_y))
                        self.screen.blit(scaled_surface, rect.topleft)

        self.cell_changed[:] = False

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_f:
                        self.toggle_fullscreen()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_size = min(event.w, event.h)
                    self.cell_size = self.screen_size // self.size
                    self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)

            self.automaton.update()

            self.draw_grid(self.automaton.grid)

            pygame.display.flip()

            clock.tick(10)  # Ajusta la tasa de frames según sea necesario

        pygame.quit()

if __name__ == "__main__":
    # Choose either GameOfLifeAutomatonPython or GameOfLifeAutomatonOpenCL
    game_of_life_automaton = GameOfLifeAutomatonOpenCL(size=600)
    
    game_pygame = CellularAutomatePygame(automaton=game_of_life_automaton, initial_screen_size=600)
    game_pygame.run()
