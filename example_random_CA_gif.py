
from cellular_automaton_gif import CellularAutomatonGif
from cellular_automaton_interface import CellularAutomaton
from stochastic_memorized_CA_example import RandomRulesCA


if __name__ == "__main__":
    automaton = RandomRulesCA(
        initial_state=CellularAutomaton.load_image(
            "R:\\Labs\\FC-Lab1\\TIF\\input_3states.bmp", num_states=2
        ),
    )
    gif_generator = CellularAutomatonGif(
        max_frames=50,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=automaton,
        filename_gif="opencl_memory_gif.gif",
        frame_rate=10.0,
        num_states=2,
        colors=[
            (0,255,0),
            (0,0,255),
            (255,0,0)
        ]
    )
    gif_generator.generate_frames()
    gif_generator.combine_gifs()