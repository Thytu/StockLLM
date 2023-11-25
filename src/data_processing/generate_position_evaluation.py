import httpx

from typing import List
from utils.retry import retry


STOCKFISH_ONLINE_ENDPOINT = "http://localhost:8000/stockfish/"


@retry(n_tries=3)
def evaluate_position(
    fen: str,
    depth: int = 12,
) -> float:

    response = httpx.post(
        url=STOCKFISH_ONLINE_ENDPOINT + "evaluate/",
        json={
            "fen": fen,
            "depth": depth,
        }, timeout = 300
    )

    if response.status_code != 200:
        raise RuntimeError(f"An error occured: {response.content}")

    return response.json()["score"]


def evaluate_positions(fens: List[str], depth: int):
    response = httpx.post(
        url=STOCKFISH_ONLINE_ENDPOINT + "evaluates/",
        json={
            "fens": fens,
            "depth": depth,
        }, timeout = 300
    )

    if response.status_code != 200:
        raise RuntimeError(f"An error occured: {response.content}")

    return response.json()["scores"]


def get_next_best_move(fen: str, depth: int):
    response = httpx.post(
        url=STOCKFISH_ONLINE_ENDPOINT + "best-move/",
        json={
            "fen": fen,
            "depth": depth,
        }, timeout = 300
    )

    if response.status_code != 200:
        raise RuntimeError(f"An error occured: {response.content}")

    return response.json()["move"]


def get_next_best_moves(fens: List[str], depth: int):
    response = httpx.post(
        url=STOCKFISH_ONLINE_ENDPOINT + "best-moves/",
        json={
            "fens": fens,
            "depth": depth,
        }, timeout = 300
    )

    if response.status_code != 200:
        raise RuntimeError(f"An error occured: {response.content}")

    return response.json()["moves"]