import os

from stockfish import Stockfish
from fastapi import FastAPI, Body
from typing import List, Dict, Union


PATH_TO_STOCKFISH = os.getenv("PATH_TO_STOCKFISH")

if not PATH_TO_STOCKFISH:
    raise RuntimeError("You must set PATH_TO_STOCKFISH env var.")


app = FastAPI()


def _evaluate_move(stockfish, fen: str) -> float:

    CHESS_MATE_CP = 100_000

    stockfish.set_fen_position(fen)

    evaluation =  stockfish.get_evaluation()

    if evaluation["type"] == "cp":
        return evaluation["value"]

    else:
        return CHESS_MATE_CP / evaluation["value"] if evaluation["value"] > 0 else CHESS_MATE_CP


def evaluate_moves(fens: List[str], depth: int = 12) -> List[float]:

    stockfish = Stockfish(
        path=PATH_TO_STOCKFISH,
        depth=depth,
        turn_perspective=False,
        parameters={
            "Threads": 1,
        }
    )
    stockfish.resume_full_strength()

    scores = [_evaluate_move(stockfish=stockfish, fen=fen) for fen in fens]

    return scores


def _find_next_best_move(stockfish, fen: str) -> float:

    stockfish.set_fen_position(fen)

    return stockfish.get_best_move()


def find_next_best_moves(fens: List[str], depth: int = 12) -> List[str]:

    stockfish = Stockfish(
        path=PATH_TO_STOCKFISH,
        depth=depth,
        turn_perspective=False,
        parameters={
            "Threads": 1,
        }
    )
    stockfish.resume_full_strength()

    moves = [_find_next_best_move(stockfish=stockfish, fen=fen) for fen in fens]

    return moves


@app.post("/stockfish/evaluate/")
def route_evaluate_moves(fen: str = Body(...), depth: int = Body(12)) -> Dict[str, float]:

    result = {
        "score": evaluate_moves(fens=[fen], depth=depth)[0]
    }

    return result


@app.post("/stockfish/evaluates/")
def route_evaluate_moves(fens: List[str] = Body(...), depth: int = Body(12)) -> Dict[str, List[float]]:

    result = {
        "scores": evaluate_moves(fens=fens, depth=depth)
    }

    return result


@app.post("/stockfish/best-move/")
def route_find_next_best_move(fen: str = Body(...), depth: int = Body(12)) -> Dict[str, str]:

    result = {
        "move": find_next_best_moves(fens=[fen], depth=depth)[0]
    }

    return result

@app.post("/stockfish/best-moves/")
def route_find_next_best_move(fens: List[str] = Body(...), depth: int = Body(12)) -> Dict[str, List[Union[str, None]]]:

    result = {
        "moves": find_next_best_moves(fens=fens, depth=depth)
    }

    return result