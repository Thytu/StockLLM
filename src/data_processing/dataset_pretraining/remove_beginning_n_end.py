import os

from tqdm import tqdm
from typing import List


def main(
    path_to_literature: List[str],
    num_lines_to_remove_beginning: int,
    num_lines_to_remove_end: int,
    path_to_output: str,
) -> None:
    

    pbar = tqdm([os.path.join(path_to_literature, fname) for fname in os.listdir(path_to_literature)])

    os.makedirs(path_to_output, exist_ok=True)

    for book in pbar:

        with open(book) as f:
            _full_content = f.read()

        lines = _full_content.split("\n")
        lines = lines[num_lines_to_remove_beginning:-num_lines_to_remove_end]

        fname = os.path.split(book)[-1]
        with open(os.path.join(path_to_output, fname), "w+") as f:
            f.write("\n".join(lines))

    return None
