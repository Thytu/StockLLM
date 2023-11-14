from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples


PATH_TO_OUTPUT_DATASET = "outputs/tasks/findScore"


PROMPT_FIND_SCORE = """<s>[INST]
Given a full set of chess moves, announce the final score of the game.

Input Format: A comma-separated list of chess moves.
Output Format: "1-0" if White wins, "0-1" if Black wins, and "1/2-1/2" in case of a draw.
[/INST]

[INPUTS]
Moves: {}
[/INPUTS]

[OUTPUTS]
"""


def generate_prompt_find_score(data_point, tokenizer, return_tensors=None):
    full_prompt = PROMPT_FIND_SCORE.format(data_point["Moves"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(tokenizer, data_point["Result"] + "</s>")["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_score,
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
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["find_score"],
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
