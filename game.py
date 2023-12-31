import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_f
from TIF.game_of_life_example import GameOfLifeAutomatonPython

class CellularAutomatePygame:
    def __init__(self, automaton, initial_screen_size):
        self.automaton = automaton
        self.size = automaton.SIZE
        self.screen_size = initial_screen_size
        self.cell_size = self.screen_size // self.size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size), pygame.RESIZABLE)
        pygame.display.set_caption('Cellular Automaton')

        self.fullscreen = False

    def draw_grid(self, data):
        cell_size = self.screen_size // self.size

        for y in range(self.size):
            for x in range(self.size):
                color = (0, 0, 0) if data[y, x] else (255, 255, 255)
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)

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

            data = self.automaton.update()

            self.screen.fill((255, 255, 255))  # Fill the screen with white
            self.draw_grid(data)
            pygame.display.flip()

            clock.tick(10)  # Set the frames per second

        pygame.quit()

if __name__ == "__main__":
    # Create an instance of GameOfLifeCalculator or any other cellular automaton
    # game_of_life_calculator = GameOfLifeAutomaton(size=50)
    game_of_life = CellularAutomatonOpenCL(size=100, rule_kernel=game_of_life_kernel)
    
    # Use the GameOfLifePygame class with the chosen automaton
    game_pygame = CellularAutomatePygame(automaton=game_of_life_calculator, initial_screen_size=500)
    game_pygame.run()
