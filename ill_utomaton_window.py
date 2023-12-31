from cellular_automaton_interface import CellularAutomaton
from disease_spread_cellular_automaton import IllAutomaton
from pygame_GUI import CellularAutomatePygameGUI


if __name__ == "__main__":
    game_of_life_automaton = IllAutomaton(
        initial_state=CellularAutomaton.load_image(
            ".\initial_states\\initial_desired_centra.bmp", num_states=4
        ),
        R0=2.8,
        ILL_DURATION=16
    )
    
    game_pygame = CellularAutomatePygameGUI(
        automaton=game_of_life_automaton, 
        initial_screen_size=600,
        colors=[
            (0, 255, 0), 
            (255, 0, 0), 
            (0, 255, 255), 
            (0, 0, 255)
        ]
        )
    game_pygame.run()
