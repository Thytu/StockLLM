import os
from random import sample
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs


TASK_ID = "SORT_FENS"

PROMPT_SORT_FENS = """<s>[INST]
Given a list of partial FENs, sort them from the ealier in the game to the latest in the game.

Output Format: Sorted list of FENs
[/INST]

[INPUTS]
FENs: {}
[/INPUTS]

[OUTPUTS]
Sorted FENs:"""


def generate_prompt_sort_FENs(data_point):

    FENs = moves_to_FENs(data_point["Moves"])
    sampled_FENs = sample([(i, fen) for i, fen in enumerate(FENs)], min(5, len(FENs)))

    # FENs with only the board i.e
    # before: r2q1rk1/ppp2ppp/3bbn2/3p4/8/1B1P4/PPP2PPP/RNB1QRK1 w - - 5 11
    # after : r2q1rk1/ppp2ppp/3bbn2/3p4/8/1B1P4/PPP2PPP/RNB1QRK1
    sampled_FENs.sort(key=lambda x : x[0])
    labels = [elem[1].split(' ')[0] for elem in sampled_FENs]

    result = {
        "task": PROMPT_SORT_FENS,
        "input": {
            "FENs": [elem[1].split(' ')[0] for elem in sampled_FENs],
        },
        "expected_output": {
            "sorted FENs": str(labels).replace("'", ""),
        },
    }

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    os.makedirs(
        name=path_to_output_dataset if not "." in path_to_output_dataset else os.path.split(path_to_output_dataset)[0],
        exist_ok=True,
    )

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_sort_FENs,
        num_proc=os.cpu_count(),
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)
