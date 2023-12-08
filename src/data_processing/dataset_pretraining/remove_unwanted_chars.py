import os

from tqdm import tqdm
from typing import List


def main(
    path_to_literature: List[str],
    path_to_output: str,
) -> None:
    

    pbar = tqdm([os.path.join(path_to_literature, fname) for fname in os.listdir(path_to_literature)])

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
