import os
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples


TASK_ID = "FIND_FINAL_SCORE"

PATH_TO_OUTPUT_DATASET = "outputs/tasks/findScore"

PROMPT_FIND_SCORE = """Given a full set of chess moves, announce the final score of the game.

Input Format: A comma-separated list of chess moves.
Output Format: "1-0" if White wins, "0-1" if Black wins, and "1/2-1/2" in case of a draw."""


def generate_prompt_find_score(data_point):
    result = {
        "task": PROMPT_FIND_SCORE,
        "input": {
            "moves": data_point["Moves"],
        },
        "expected_output": data_point["Result"],
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
        generate_prompt_find_score,
        num_proc=os.cpu_count(),
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)


if __name__ == "__main__":
    import dvc.api

    params = dvc.api.params_show()

    main(
        fraction_of_data_to_generate_from=params["data"]["quantity_per_task"]["find_score"],
        path_to_output_dataset=PATH_TO_OUTPUT_DATASET,
    )
