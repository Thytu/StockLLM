import os
import shutil

from tqdm import tqdm
from typing import Dict


def main(
    literature_to_filter: Dict[str, str],
    max_nonascii_percentage: float,
    path_to_output: str,
) -> None:
    

    pbar = tqdm(literature_to_filter)

    _remaining = []

    for book in pbar:

        with open(book) as f:
            _full_content = f.read()
            _cleaned = _full_content.encode().decode("ascii", errors="ignore")
            
        percentage_of_nonascii_chars = 100 - len(_cleaned) * 100 / len(_full_content) 
        
        if percentage_of_nonascii_chars > max_nonascii_percentage:
            print(f"Dropping {book}: {percentage_of_nonascii_chars=:.2f} > {max_nonascii_percentage=}")
        else:
            _remaining.append(book)
    
    print(f"Remainings:\n{_remaining}")
    

    os.makedirs(path_to_output, exist_ok=True)
    for book in _remaining:
        fname = os.path.split(book)[-1]
        
        shutil.copyfile(book, os.path.join(path_to_output, fname))

    print(f"Kept {len(_remaining) * 100 / len(literature_to_filter)}% of the books")

    return None
