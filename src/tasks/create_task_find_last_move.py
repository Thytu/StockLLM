from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples


PATH_TO_OUTPUT_DATASET = "outputs/tasks/findLastMove"

PROMPT_FIND_LAST_MOVE = """<s>[INST]
Given an incomplit set of chess moves and the game's final score, write the last missing chess move.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: The missing chess move
[/INST]

[INPUTS]
Moves: {}
Score: {}
[/INPUTS]

[OUTPUTS]
"""


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def generate_prompt_find_last_move(data_point, tokenizer, return_tensors=None):
    moves, removed_moves = data_point["Moves"][:-1], data_point["Moves"][-1]
    moves.append("?")

    full_prompt = PROMPT_FIND_LAST_MOVE.format(moves, data_point["Result"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves) + "</s>")["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_last_move,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
        num_proc=8,
    )

    dataset.save_to_disk(path_to_output_dataset)


if __name__ == "__main__":
    import dvc.api

    params = dvc.api.params_show()

    main(
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["find_last_move"],
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
