import io
import chess.pgn

from typing import List
from datasets import Dataset


def pgn_to_fen_list(pgn: str) -> List[str]:
    chess_game = chess.pgn.read_game(pgn)

    FENs = []

    while chess_game.next():

        chess_game = chess_game.next()

        FENs.append(chess_game.board().fen())

    return FENs


def _add_fen_list(example):

    example["FENs"] = pgn_to_fen_list(io.StringIO(example["PGN"]))

    return example


def main(dataset: Dataset, num_proc: int = 0):
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        _add_fen_list,
        num_proc=num_proc,
    )

    return dataset


if __name__ == "__main__":
    import os

    from datasets import load_from_disk
    from multiprocessing import cpu_count

    OUTPUT_FOLDER = "outputs/FENs"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset = load_from_disk("outputs/PGNs")

    dataset = main(
        dataset=dataset,
        num_proc=int(os.getenv("NUM_PROC", cpu_count())),
    )

    dataset.save_to_disk(OUTPUT_FOLDER)
