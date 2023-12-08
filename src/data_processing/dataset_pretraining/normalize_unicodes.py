import os
import json
import unicodedata

from tqdm import tqdm
from typing import List


def main(
    path_to_literature: str,
    path_to_output: str,
) -> None:
    

    pbar = tqdm([os.path.join(path_to_literature, fname) for fname in os.listdir(path_to_literature)])

    os.makedirs(path_to_output, exist_ok=True)

    for book in pbar:

        with open(book) as f:
            content = f.read()

        content = unicodedata.normalize('NFKC', content)

        fname = os.path.split(book)[-1]
        with open(os.path.join(path_to_output, fname), "w+") as f:
            f.write(content)

    return None
