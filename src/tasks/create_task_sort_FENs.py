from random import sample
from datasets import Dataset
from model import tokenize, get_tokenizer
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs


PROMPT_SORT_FENS = """<s>[INST]
Given a list of partial FENs, sort them from the ealier in the game to the latest in the game.

Output Format: Sorted list of FENs
[/INST]

[INPUTS]
FENs: {}
[/INPUTS]

[OUTPUTS]
Sorted FENs:"""


def generate_prompt_sort_FENs(data_point, tokenizer, return_tensors=None):

    FENs = moves_to_FENs(data_point["Moves"])
    sampled_FENs = sample([(i, fen) for i, fen in enumerate(FENs)], min(5, len(FENs)))

    # FENs with only the board i.e
    # before: r2q1rk1/ppp2ppp/3bbn2/3p4/8/1B1P4/PPP2PPP/RNB1QRK1 w - - 5 11
    # after : r2q1rk1/ppp2ppp/3bbn2/3p4/8/1B1P4/PPP2PPP/RNB1QRK1
    full_prompt = PROMPT_SORT_FENS.format(
        [elem[1].split(' ')[0] for elem in sampled_FENs],
    ).replace("'", "")

    result = tokenize(tokenizer, full_prompt, return_tensors=return_tensors)

    sampled_FENs.sort(key=lambda x : x[0])
    labels = [elem[1].split(' ')[0] for elem in sampled_FENs]
    result["labels"] = tokenize(tokenizer, str(labels).replace("'", "") + "</s>")["input_ids"]

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    _tokenizer = get_tokenizer()

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_sort_FENs,
        fn_kwargs={
            "tokenizer": _tokenizer,
        },
    )

    dataset.save_to_disk(path_to_output_dataset)
