from random import random
from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs
from data_processing import evaluate_move


PROMPT_FIND_WORST_AND_BEST_MOVE = """<s>[INST]
Given a full set of chess moves, write the worst and best moves from each side (white and black).

Follow the following output format:
White
worst: [insert white's worst move]
best: [insert white's best move]

Black
worst: [insert black's worst move]
best: [insert black's best move]
[/INST]

[INPUTS]
Moves: {}
[/INPUTS]

[OUTPUTS]"""

PROMPT_LABEL = """
White
worst: {}
best: {}

Black
worst: {}
best: {}
"""


def generate_prompt_find_worst_and_best_move(data_point, tokenizer, return_tensors=None):

    FENs = moves_to_FENs(data_point["Moves"])
    evaluations = [evaluate_move(fen, depth=8) for fen in FENs]

    # WHITE MOVES
    white_moves = [i for i in range(0, len(data_point["Moves"]), 2)]

    idx_best_move = evaluations.index(max([evaluations[i] for i in white_moves]))
    best_move_from_white = data_point["Moves"][idx_best_move]

    idx_worst_move = evaluations.index(min([evaluations[i] for i in white_moves]))
    worst_move_from_white = data_point["Moves"][idx_worst_move]

    # BLACK MOVES
    black_moves = [i for i in range(1, len(data_point["Moves"]), 2)]

    idx_best_move = evaluations.index(max([evaluations[i] for i in black_moves]))
    best_move_from_black = data_point["Moves"][idx_best_move]

    idx_worst_move = evaluations.index(min([evaluations[i] for i in black_moves]))
    worst_move_from_black = data_point["Moves"][idx_worst_move]

    full_prompt = PROMPT_FIND_WORST_AND_BEST_MOVE.format(data_point["Moves"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(
        tokenizer,
        PROMPT_LABEL.format(
            worst_move_from_white,
            best_move_from_white,
            worst_move_from_black,
            best_move_from_black,
        ) + "</s>"
    )["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_worst_and_best_move,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
        num_proc=8,
    )

    dataset.save_to_disk(path_to_output_dataset)
