import os
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples


TASK_ID = "FIND_LAST_MOVE"

PATH_TO_OUTPUT_DATASET = "outputs/tasks/findLastMove"

PROMPT_FIND_LAST_MOVE = """Given an incomplit set of chess moves and the game's final score, write the last missing chess move.

Input Format: A comma-separated list of chess moves followed by the game score.
Output Format: The missing chess move"""


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def generate_prompt_find_last_move(data_point):
    moves, removed_moves = data_point["Moves"][:-1], data_point["Moves"][-1]
    moves.append("?")

    result = {
        "task": PROMPT_FIND_LAST_MOVE,
        "input": {
            "moves": moves,
            "result": data_point["Result"],
        },
        "expected_output": {
            "missing move": str(removed_moves)
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
        generate_prompt_find_last_move,
        num_proc=os.cpu_count(),
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)


if __name__ == "__main__":
    import dvc.api

    params = dvc.api.params_show()

    main(
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["find_last_move"],
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
