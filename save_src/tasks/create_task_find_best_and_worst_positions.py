import os

from datasets import Dataset
from data_processing import moves_to_FENs
from data_processing import evaluate_positions
from data_processing.get_random_data_samples import get_random_data_samples


TASK_ID = "BEST_N_WORST_POSITIONS"

PROMPT_FIND_WORST_AND_BEST_POSITIONS = """Given a full set of chess moves, write the past worst and best positions from each side."""

PROMPT_LABEL = """
White
worst: {}
best: {}

Black
worst: {}
best: {}
"""


def generate_prompt_find_worst_and_best_position(data_point, path_to_stockfish):

    FENs = moves_to_FENs(data_point["Moves"])
    evaluations = evaluate_positions(
        fens=FENs,
        depth=12,
    )

    # WHITE POSITIONS
    white_positions = [i for i in range(0, len(data_point["Moves"]), 2)]

    idx_best_position = evaluations.index(max([evaluations[i] for i in white_positions]))
    best_position_from_white = data_point["Moves"][idx_best_position]

    idx_worst_position = evaluations.index(min([evaluations[i] for i in white_positions]))
    worst_position_from_white = data_point["Moves"][idx_worst_position]

    # BLACK POSITIONS
    black_positions = [i for i in range(1, len(data_point["Moves"]), 2)]

    # NOTE: here we inverte min/max because a cp < 0 means an advantage for black
    idx_best_position = evaluations.index(min([evaluations[i] for i in black_positions]))
    best_position_from_black = data_point["Moves"][idx_best_position]

    idx_worst_position = evaluations.index(max([evaluations[i] for i in black_positions]))
    worst_position_from_black = data_point["Moves"][idx_worst_position]

    result = {
        "task": PROMPT_FIND_WORST_AND_BEST_POSITIONS,
        "input": {
            "moves": data_point["Moves"],
        },
        "expected_output": {
            "white's worst position": worst_position_from_white,
            "white's best position": best_position_from_white,
            "black's worst position": worst_position_from_black,
            "black's best position": best_position_from_black,
        },
    }

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    path_to_stockfish = ""

    os.makedirs(
        name=path_to_output_dataset if not "." in path_to_output_dataset else os.path.split(path_to_output_dataset)[0],
        exist_ok=True,
    )

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_worst_and_best_position,
        fn_kwargs={
            "path_to_stockfish": path_to_stockfish,
        }, num_proc=os.cpu_count() - 1,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)
