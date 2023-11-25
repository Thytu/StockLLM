import os

from random import randint
from utils.retry import retry
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs, evaluate_position


TASK_ID = "FIND_ADVANTAGED_PLAYER"

PROMPT_FIND_WHO_IS_WINNING = """Given some set of chess moves, write who is more advantaged (white or black)"""

@retry
def generate_prompt_find_who_is_winning(data_point):

    FENs = moves_to_FENs(data_point["Moves"])

    idx_to_cut_at = randint(
        min(5, len(FENs)),
        len(FENs),
    )

    data_point["Moves"] = data_point["Moves"][:idx_to_cut_at]

    FENs = FENs[:idx_to_cut_at]

    evaluation = evaluate_position(
        fen=FENs[-1],
        depth=12,
    )

    result = {
        "task": PROMPT_FIND_WHO_IS_WINNING,
        "input": {
            "moves": data_point["Moves"],
        },
        "expected_output": {
            "Most advantaged": "White" if evaluation >= 0 else "Black"
        },
    }

    return result


def main(
    number_of_samples: float,
    path_to_output_dataset: str,
) -> Dataset:

    # path_to_stockfish = os.getenv("PATH_TO_STOCKFISH")

    # if not path_to_stockfish:
    #     raise RuntimeError("You must set PATH_TO_STOCKFISH env var.")

    os.makedirs(
        name=path_to_output_dataset if not "." in path_to_output_dataset else os.path.split(path_to_output_dataset)[0],
        exist_ok=True,
    )

    # engine = SimpleEngine.popen_uci(path_to_stockfish)

    dataset = get_random_data_samples(number_of_samples=number_of_samples)

    dataset = dataset.map(
        generate_prompt_find_who_is_winning,
        fn_kwargs={
            # "engine": engine,
        }, num_proc=os.cpu_count() - 1,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)
