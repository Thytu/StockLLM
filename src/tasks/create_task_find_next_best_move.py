import os

from random import randint
from utils.retry import retry
from datasets import Dataset
from data_processing.get_random_data_samples import get_random_data_samples
from data_processing import moves_to_FENs, get_next_best_moves


TASK_ID = "FIND_NEXT_BEST_MOVE"

PROMPT_FIND_BEST_NEXT_MOVE = """Given some set of chess moves, write the best possible move"""

# @retry
def generate_prompt_find_next_best_move(data_point):

    CHESS_MATE = "chess mate"

    samples = []

    for idx in range(len(data_point["Moves"])):
        FENs = moves_to_FENs(data_point["Moves"][idx])

        idx_to_cut_at = randint(
            min(5, len(FENs) - 1),
            len(FENs) - 1,
        )

        data_point["Moves"][idx] = data_point["Moves"][idx][:idx_to_cut_at]
        last_FEN = FENs[idx_to_cut_at]

        samples.append({
            "moves": data_point["Moves"][idx],
            "fen": last_FEN,
        })

    best_next_moves = get_next_best_moves(
        fens=[s["fen"] for s in samples],
        depth=12
    )

    return {
        "task": [PROMPT_FIND_BEST_NEXT_MOVE] * len(samples),
        "input": [{
            "moves": _sample["moves"],
        } for _sample in samples],
        "expected_output": [{
            "next best move": _best_next_move if _best_next_move is not None else CHESS_MATE
        } for _best_next_move in best_next_moves]
    }


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
        generate_prompt_find_next_best_move,
        batched=True,
        batch_size=20,
        num_proc=os.cpu_count() - 1,
    )
    dataset = dataset.add_column("KIND", [TASK_ID] * len(dataset))

    dataset = dataset.to_pandas()
    dataset.to_parquet(path_to_output_dataset)
