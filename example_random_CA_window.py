from cellular_automaton_interface import CellularAutomaton
from stochastic_memorized_CA_example import RandomRulesCA
from pygame_GUI import CellularAutomatePygameGUI


if __name__ == "__main__":
    game_of_life_automaton = RandomRulesCA(
        initial_state=CellularAutomaton.load_image(
            "R:\\Labs\\FC-Lab1\\TIF\\initial_desired_centra.bmp", num_states=2
        )
    )
    game_pygame = CellularAutomatePygameGUI(
        automaton=game_of_life_automaton, 
        initial_screen_size=600,
        colors=[
            (0,0,0),
            (255,255,255)
        ])
    game_pygame.run()