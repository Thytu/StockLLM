import io
import chess.pgn

from typing import List
from datasets import Dataset
from data_processing.generate_PGNs import generate_pgn


def _pgn_to_fen_list(pgn: str) -> List[str]:
    chess_game = chess.pgn.read_game(io.StringIO(pgn))

    FENs = []

    while chess_game.next():

        chess_game = chess_game.next()

        FENs.append(chess_game.board().fen())

    return FENs


def moves_to_FENs(moves) -> List[str]:
    pgn = generate_pgn(moves)

    return _pgn_to_fen_list(pgn=pgn)
