import httpx
import asyncio

from utils import retry
from datasets import Dataset
from time import sleep


STOCKFISH_ONLINE_ENDPOINT = "https://stockfish.online/api/stockfish.php"


@retry.retry(n_tries=3)
def evaluate_move(
# async def evaluate_move(
    fen: str,
    depth: int = 5,
) -> float:

    # client = httpx.AsyncClient()

    # await asyncio.sleep(1)

    response = httpx.get(
    # response = await client.get(
        url=STOCKFISH_ONLINE_ENDPOINT,
        params={
            "fen": fen,
            "depth": depth,
            "mode": "eval",
        }, timeout = 300
    )

    if response.status_code != 200:
        raise RuntimeError(f"An error occured: {response.content}")

    try:
        body = response.json()
    except Exception as e:
        print(f"Tried with {fen=}, received {response.content}")
        raise e

    # print(response.json())

    if not body["success"]:
        raise RuntimeError(f"An error occured: {response.content}")

    return float(body["data"].replace("Total evaluation: ", "").replace(" (white side)", ""))


def _add_evaluations(example):

    # NOTE: The API is too unstable to allow parallelization over request
    # keeping the code commented for the moment

    # coroutines = [evaluate_move(fen) for fen in example["FENs"]]

    # loop = asyncio.get_event_loop()

    # example["evaluations"] = loop.run_until_complete(asyncio.gather(*coroutines))

    example["evaluations"] = [evaluate_move(fen) for fen in example["FENs"]]

    return example


def main(dataset: Dataset, num_proc: int = 0):
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        _add_evaluations,
        num_proc=num_proc,
    )

    return dataset


if __name__ == "__main__":
    import os

    from datasets import load_from_disk
    from multiprocessing import cpu_count

    NB_SHARDS = 100
    NB_SHARDS_TO_ANNOTATE = 3
    OUTPUT_FOLDER = "outputs/moves_evalutation"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset = load_from_disk("outputs/FENs")

    for current_shard_idx in range(NB_SHARDS):

        if current_shard_idx >= NB_SHARDS_TO_ANNOTATE:
            break

        shard = dataset.shard(index=current_shard_idx, num_shards=NB_SHARDS)

        if os.path.exists(os.path.join(OUTPUT_FOLDER, str(current_shard_idx))):
            print(f"Shard {current_shard_idx} already exists, skipping...")
            continue

        shard = main(
            dataset=shard,
            num_proc=int(os.getenv("NUM_PROC", cpu_count())),
        )

        shard.save_to_disk(os.path.join(OUTPUT_FOLDER, str(current_shard_idx)))
