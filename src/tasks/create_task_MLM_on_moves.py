from random import random
from datasets import load_from_disk, Dataset
from src.model import tokenize, get_tokenizer


PATH_TO_INPUT_DATASET = "outputs/samples"
PATH_TO_OUTPUT_DATASET = "outputs/tasks/MLM"


PROMPT_MLM_ON_MOVES = """<s>[INST] Given an incomplit set of chess moves and the game's score, write the missing chess moves.

Missing chess moves are indicated with a "?" mark. Write ONLY the missing moves, not the provided ones.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: A comma-separated list of the missing chess moves.

Moves: {}
Score: {} [/INST]

Output:"""


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


def generate_prompt_MLM_on_moves(tokenizer, data_point, return_tensors=None):
    moves, removed_moves = __remove_random_moves(
        moves=data_point["Moves"],
        prob_to_remove=0.025,
    )

    if len(removed_moves) == 0:
        removed_moves = moves[-1]
        moves[-1] = "?"

    full_prompt = PROMPT_MLM_ON_MOVES.format(moves, data_point["Result"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves))["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_input_dataset: str,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    if number_of_samples < 0:
        raise RuntimeError(f"{number_of_samples=} must be positive.")

    dataset = load_from_disk(path_to_input_dataset).shuffle()
    dataset = dataset.select(range(number_of_samples))
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        generate_prompt_MLM_on_moves,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
    )

    dataset.save_to_disk(path_to_output_dataset)


if __name__ == "__main__":
    import dvc.api

    params = dvc.api.params_show()

    main(
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["MLM_on_moves"],
        path_to_input_dataset=PATH_TO_INPUT_DATASET,
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
