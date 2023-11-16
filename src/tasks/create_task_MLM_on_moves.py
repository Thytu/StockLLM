from random import random
from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs


TASK_ID = "MLM_ON_MOVES"

PROMPT_MLM_ON_MOVES = """<s>[INST]
Given an incomplit set of chess moves and some informations regarding this game, write the missing chess moves.

Missing chess moves are indicated with a "?" mark. Write ONLY the missing moves, not the provided ones.
Output Format: A comma-separated list of the missing chess moves.
[/INST]

[INPUTS]
Moves: {}
FEN: {}
Score: {}
[/INPUTS]

[OUTPUTS]
Missing moves:"""


def __remove_random_moves(moves, prob_to_remove: float = 0.1):
    remaining_moves = []
    removed_moves = []

    for m in moves:
        if random() <= prob_to_remove:
            removed_moves.append(m)
            remaining_moves.append("?")
        else:
            remaining_moves.append(m)

    return remaining_moves, removed_moves


def generate_prompt_MLM_on_moves(data_point, tokenizer, return_tensors=None):

    final_FEN = moves_to_FENs(data_point["Moves"])[-1]

    moves, removed_moves = __remove_random_moves(
        moves=data_point["Moves"],
        prob_to_remove=0.025,
    )

    if len(removed_moves) == 0:
        removed_moves = moves[-1]
        moves[-1] = "?"

    full_prompt = PROMPT_MLM_ON_MOVES.format(
        moves,
        final_FEN,
        data_point["Result"],
    ).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves).replace("'", "") + "</s>")["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_MLM_on_moves,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
        num_proc=8,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset.save_to_disk(path_to_output_dataset)
