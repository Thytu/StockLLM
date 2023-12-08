import os
import unicodedata

from tqdm import tqdm
from typing import List


def main(
    literature_to_filter: List[str],
    path_to_output: str,
) -> None:
    

    pbar = tqdm(literature_to_filter)

    os.makedirs(path_to_output, exist_ok=True)

    for book in pbar:

        with open(book) as f:
            content = f.read()

        content = unicodedata.normalize('NFKC', content)

        fname = os.path.split(book)[-1]
        with open(os.path.join(path_to_output, fname), "w+") as f:
            f.write(content)

    return None
