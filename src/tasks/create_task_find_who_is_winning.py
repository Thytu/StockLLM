from random import randint
from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs, evaluate_move


TASK_ID = "FIND_ADVANTAGED_PLAYER"

PROMPT_FIND_WHO_IS_WINNING = """<s>[INST]
Given some set of chess moves, write who is more advantaged (white or black)
[/INST].

[INPUTS]
Moves: {}
[/INPUTS]

[OUTPUTS]
"""


def generate_prompt_find_who_is_winning(data_point, tokenizer, return_tensors=None):

    FENs = moves_to_FENs(data_point["Moves"])
    idx_to_cut_at = randint(
        min(5, len(FENs)),
        len(FENs),
    )

    data_point["Moves"] = data_point["Moves"][:idx_to_cut_at]
    FENs = FENs[:idx_to_cut_at]
    evaluation = evaluate_move(FENs[-1], depth=8)

    full_prompt = PROMPT_FIND_WHO_IS_WINNING.format(data_point["Moves"]).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(
        tokenizer,
        "White" + "</s>" if evaluation >= 0 else "Black" + "</s>"
    )["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_who_is_winning,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
        num_proc=8,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset.save_to_disk(path_to_output_dataset)
