import os

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
        
        for substr in ('\n ', '\t ', '\n\t'):
            content = content.replace(substr, "")

        for char in ('\n', '\t', ' '):
            while char * 2 in content:
                content = content.replace(char * 2, char)

        fname = os.path.split(book)[-1]
        with open(os.path.join(path_to_output, fname), "w+") as f:
            f.write(content)

    return None
