from datasets import load_from_disk, Dataset
from src.model import tokenize, get_tokenizer


PATH_TO_INPUT_DATASET = "outputs/samples"
PATH_TO_OUTPUT_DATASET = "outputs/tasks/findLastMove"

PROMPT_FIND_LAST_MOVE = """<s>[INST] Given an incomplit set of chess moves and the game's final score, write the last missing chess move.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: The missing chess move

Moves: {}
Score: {} [/INST]

Output:"""


def generate_prompt_find_last_move(tokenizer, data_point, return_tensors=None):
    moves, removed_moves = data_point["Moves"][:-1], data_point["Moves"][-1]

    full_prompt = PROMPT_FIND_LAST_MOVE.format(moves, data_point["Result"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, str(removed_moves))["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_input_dataset: str,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    if number_of_samples > 1:
        raise RuntimeError(f"{number_of_samples=} must be positive.")

    dataset = load_from_disk(path_to_input_dataset).shuffle()
    dataset = dataset.select(range(number_of_samples))
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        generate_prompt_find_last_move,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
    )

    dataset.save_to_disk(path_to_output_dataset)


if __name__ == "__main__":
    import dvc.api

    params = dvc.api.params_show()

    main(
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["find_last_move"],
        path_to_input_dataset=PATH_TO_INPUT_DATASET,
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
