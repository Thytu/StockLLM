import os

from random import randint
from utils.retry import retry
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs, evaluate_positions


TASK_ID = "FIND_ADVANTAGED_PLAYER"

PROMPT_FIND_WHO_IS_WINNING = """Given some set of chess moves, write the player who is currently the most advantaged (white or black)"""

@retry
def generate_prompt_find_who_is_winning(data_points):

    samples = []

    for idx in range(len(data_points["Moves"])):
        FENs = moves_to_FENs(data_points["Moves"][idx])

        idx_to_cut_at = randint(
            min(5, len(FENs)),
            len(FENs),
        )

        data_points["Moves"][idx] = data_points["Moves"][idx][:idx_to_cut_at]

        FENs = FENs[:idx_to_cut_at]

        samples.append({
            "moves": data_points["Moves"][idx],
            "FENs": FENs,
        })

    evaluations = evaluate_positions(
        fens=[_sample["FENs"][-1] for _sample in samples],
        depth=12,
    )

    result = {
        "task": [PROMPT_FIND_WHO_IS_WINNING] * len(samples),
        "input": [{
            "moves": _sample["moves"],
        } for _sample in samples],
        "expected_output": [{
            "Most advantaged": "White" if _evaluation >= 0 else "Black"
        } for _evaluation in evaluations],
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
        generate_prompt_find_who_is_winning,
        batched=True,
        batch_size=20,
        num_proc=os.cpu_count() - 1,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)
