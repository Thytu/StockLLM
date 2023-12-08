import os
import json
import tempfile
import multiprocessing

from urllib.request import urlretrieve
from pdfminer.high_level import extract_text



def __txt_handler(url, output):
    urlretrieve(url=url, filename=output)


def __pdf_handler(url, output):

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:

        urlretrieve(url=url, filename=tmp.name)

        text = extract_text(tmp.name)
        
        # TODO: remove all non utf-8 chars

        with open(output, "w+", encoding='utf-8') as f:
            f.write(text)


__format_handlers = {
    "txt": __txt_handler,
    "pdf": __pdf_handler,
}


def __apply_formater(args):

    book, path_to_output = args

    __format_handlers[book["format"]](book["url"], os.path.join(
        path_to_output,
        book["title"].replace("/", "")
    ))
    
    print(f"[x] Finished : {book['title']}")


def main(
    path_to_list_of_literature: str,
    path_to_output: str,
) -> None:

    os.makedirs(path_to_output, exist_ok=True)


    books = json.load(open(path_to_list_of_literature))
    

    with multiprocessing.Pool(os.cpu_count() - 1) as pool:
        pool.map(
            __apply_formater,
            [(b, path_to_output) for b in books],
        )

    return None
