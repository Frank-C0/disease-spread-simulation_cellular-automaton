import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_f
from cellular_automaton_interface import CellularAutomaton
from TIF.example_3_states_cellular_automaton import GameOfLifeAutomatonOpenCL
import numpy as np

class CellularAutomatePygame:
    def __init__(self, automaton, initial_screen_size, colors=None):
        self.automaton = automaton
        self.size = automaton.SIZE
        self.screen_size = initial_screen_size
        self.cell_size = self.screen_size // self.size
        self.colors = colors or [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)
        pygame.display.set_caption('Cellular Automaton')
        self.clock = pygame.time.Clock()
        
        self.fullscreen = False
        


    def draw_grid(self, data):
        self.screen.fill((255, 255, 255)) 

        color_matrix = np.array([self.colors[state] for state in data.flat], dtype=np.uint8).reshape(data.shape + (3,))
        scaled_color_matrix = pygame.surfarray.make_surface(np.transpose(color_matrix, axes=(1, 0, 2)))
        scaled_color_matrix = pygame.transform.scale(scaled_color_matrix, (self.screen_size, self.screen_size))

        self.screen.blit(scaled_color_matrix, scaled_color_matrix.get_rect())
        pygame.display.flip()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)

    def run(self):
        running = True

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
            self.clock.tick(5)  # Ajusta la tasa de frames según sea necesario

        pygame.quit()

if __name__ == "__main__":
    game_of_life_automaton = GameOfLifeAutomatonOpenCL(
        initial_state=CellularAutomaton.load_image("R:\\Labs\\FC-Lab1\\TIF\\input_3states.bmp", num_states=3),
    )
    
    game_pygame = CellularAutomatePygame(automaton=game_of_life_automaton, initial_screen_size=600)
    game_pygame.run()
