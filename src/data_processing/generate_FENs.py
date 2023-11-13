# import requests
import io
import chess.pgn

from typing import List
from datasets import Dataset


# moves = [ "d2d4", "f7f5", "g2g3", "g7g6", "f1g2", "f8g7", "g1f3", "d7d6", "c2c3", "e7e6", "a2a4", "g8f6", "d1c2", "d8e7", "b1d2", "e6e5", "d4e5", "d6e5", "e2e4", "b8c6", "e1g1", "f5e4", "d2e4", "c8f5", "f3d2", "e8c8", "b2b4", "g7h6", "f1e1", "h6d2", "c1d2", "f6e4", "g2e4", "e7e6", "d2g5", "d8d6", "a1d1", "d6d1", "e1d1", "h7h6", "g5e3", "a7a5", "c2b1", "h6h5", "b4b5", "c6e7", "e3g5", "h8e8", "h2h4", "e6c4", "d1e1", "f5e4", "e1e4", "c4e6", "g5f4", "e6f5", "f4e5", "e7d5", "b1e1", "d5b6", "f2f4", "b6d7", "e1e2", "b7b6", "e4e3", "e8e7", "e3e4", "d7c5", "e4d4", "e7d7", "g1g2", "c8d8", "g2h2", "d8c8", "e2g2", "c8b8", "g2a2", "b8a7", "a2g2", "a7b8", "g2e2", "b8c8", "e2f3", "c8b8", "f3d1", "b8c8", "d1e2", "c8b8", "e2d1", "b8b7", "d4d7", "c5d7", "e5d4", "d7c5", "h2g2", "f5d5", "g2g1", "d5f5", "d4c5", "f5c5", "d1d4", "c5f5", "d4d2", "f5b1", "g1f2", "b1b3", "d2d4", "b3c2", "f2e3", "b7c8", "d4h8", "c8b7", "h8d4", "b7b8", "d4d8", "b8b7", "d8d5", "b7b8", "d5g8", "b8b7", "g8c4", "b7b8", "c4g8", "b8b7", "g8d5", "b7b8", "d5d8", "b8b7", "d8d4", "b7b8", "d4d8", "b8b7", "d8d3", "c2a4", "d3g6", "a4b5", "g6e4", "b7a7", "f4f5", "a5a4", "f5f6", "a4a3", "f6f7", "b5c5", "e3e2", "c5b5", "e2e3", "b5c5", "e3d3", "c5b5", "d3e3", "b5c5", "e3d3", "c5b5", "e4c4", "b5c4", "d3c4", "a3a2", "f7f8q", "a2a1q", "f8f3", "a1b1", "f3h5", "b1e4", "c4b3", "e4b1", "b3a3", "b1c1", "a3b3", "c1b1", "b3c4", "b1e4", "c4b3", "e4b1", "b3a4", "b1a2", "a4b4", "a2b1", "b4a4", "b1c2", "a4b4", "c2b1", "b4a4", "b1a2", "a4b4", "a2b1", "b4a4", "b1c2", "a4b4", "c2b1", "b4c4", "b1e4", "c4b3", "e4b1", "b3c4", "b1e4", "c4b3", "e4b1" ]


# def _eval_move(fen):
#     resp = requests.get(f"https://stockfish.online/api/stockfish.php?fen={fen}&depth=5&mode=eval").json()

#     if not resp["success"]:
#         raise RuntimeError("An error occured")

#     print(resp["data"])

#     return float(resp["data"].replace("Total evaluation: ", "").replace(" (white side)", ""))

# pgn = [[]]
# for m in moves:
#     if len(pgn[-1]) < 2:
#         pgn[-1].append(m)
#     else:
#         pgn.append([m])

# pgn_notation = ""
# for nb in range(len(pgn)):
#     pgn_notation += f"{nb + 1}. {pgn[nb][0]} {pgn[nb][1]}"

# print(pgn_notation)
# with open("anderssen_kieseritzky_1851.pgn", "w+") as f:
#     f.write(pgn_notation)


# pgn = open("anderssen_kieseritzky_1851.pgn")

# mygame=chess.pgn.read_game(pgn)


# resp = requests.get(f"https://stockfish.online/api/stockfish.php?fen={mygame.board().fen()}&depth=5&mode=eval").json()
# move_eval = _eval_move(mygame.board().fen())
# print(mygame.board().fen(), move_eval)

# if not resp["success"]:
#     raise RuntimeError("An error occured")

# while mygame.next():
#     mygame=mygame.next()
#     #print(mygame.board().fen())
#     #resp = requests.get(f"https://stockfish.online/api/stockfish.php?fen={mygame.board().fen()}&depth=5&mode=eval").json()
#     #if not resp["success"]:
#     #    raise RuntimeError("An error occured")
#     #move_eval = float(resp["data"].replace("Total evaluation: ", "").split(" (")[0])

#     move_eval = _eval_move(mygame.board().fen())

#     print(mygame.board().fen(), move_eval)


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
