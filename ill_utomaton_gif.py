from cellular_automaton_gif import CellularAutomatonGif
from cellular_automaton_interface import CellularAutomaton
from disease_spread_cellular_automaton import IllAutomaton

R0 = 3.8
ILL_DURATION = 16
FRAMES=2000
if __name__ == "__main__":
    automaton = IllAutomaton(
        initial_state=CellularAutomaton.load_image(
            "initial_states\initial_desired_centra.bmp", num_states=2
        ),
        R0=R0,
        ILL_DURATION=ILL_DURATION,
    )

    gif_generator = CellularAutomatonGif(
        max_frames=500,
        save_interval=10,
        output_folder="output_gifs/",
        automaton=automaton,
        filename_gif=f"IllAutomaton_out_gif_R0_{R0}__ID_{ILL_DURATION}_f{FRAMES}.gif",
        frame_rate=10.0,
        num_states=4,
        colors=[(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)],
    )
    gif_generator.generate_frames()
    gif_generator.combine_gifs()
