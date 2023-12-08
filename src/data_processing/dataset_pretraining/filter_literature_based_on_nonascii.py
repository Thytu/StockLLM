import os
import shutil

from tqdm import tqdm
from typing import List


def main(
    path_to_literature: List[str],
    max_nonascii_percentage: float,
    path_to_output: str,
) -> None:
    

    pbar = tqdm([os.path.join(path_to_literature, fname) for fname in os.listdir(path_to_literature)])

    _remaining = []

    for book in pbar:

        with open(book) as f:
            _full_content = f.read()
            _cleaned = _full_content.encode().decode("ascii", errors="ignore")
            
        percentage_of_nonascii_chars = 100 - len(_cleaned) * 100 / len(_full_content) 
        
        if percentage_of_nonascii_chars <= max_nonascii_percentage:
            _remaining.append(book)

    os.makedirs(path_to_output, exist_ok=True)
    for book in _remaining:
        fname = os.path.split(book)[-1]
        
        shutil.copyfile(book, os.path.join(path_to_output, fname))

    print(f"Kept {len(_remaining) * 100 / len(os.listdir(path_to_literature))}% of the books")

    return None
