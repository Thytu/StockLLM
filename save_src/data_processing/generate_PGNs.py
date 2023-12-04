from typing import List
from datasets import Dataset


def generate_pgn(moves: List[str]) -> str:
    pairs_of_move = [[]]

    for m in moves:
        if len(pairs_of_move[-1]) < 2:
            pairs_of_move[-1].append(m)
        else:
            pairs_of_move.append([m])

    pgn_notation = ""

    for nb in range(len(pairs_of_move)):
        pgn_notation += f"{nb + 1}. {pairs_of_move[nb][0]} {pairs_of_move[nb][1] if len(pairs_of_move[nb]) > 1 else ''}"

    return pgn_notation


def _add_pgn_notation(example):
    example["PGN"] = generate_pgn(example["Moves"])

    return example


def main(dataset: Dataset) -> Dataset:

    dataset.cleanup_cache_files()

    dataset = dataset.map(
        _add_pgn_notation,
        num_proc=24,
    )

    return dataset


if __name__ == "__main__":
    import os

    from datasets import load_from_disk


    OUTPUT_FOLDER = "outputs/PGNs"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset = load_from_disk("outputs/samples")

    dataset: Dataset = main(dataset=dataset)
    dataset.save_to_disk(OUTPUT_FOLDER)
