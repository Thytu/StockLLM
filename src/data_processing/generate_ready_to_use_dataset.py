import os

from transformers import BatchEncoding
from model import get_tokenizer, tokenize
from datasets import Dataset, DatasetDict


PROMPT = "<s>[INST]{}[/INST]\n[IN]{}[/IN]\n[OUT]"
LABEL_PROMPT = "\n{}[/OUT]</s>"


def generate_and_tokenize_prompt(sample, tokenizer) -> BatchEncoding:

    result = tokenize(
        tokenizer=tokenizer,
        prompt=PROMPT.format(
            sample["task"],
            "\n".join([f"{k}: {v}" for (k, v) in sample["input"].items()])
        ),
        return_tensors=False,
    )

    result["labels"] = tokenize(
        tokenizer=tokenizer,
        prompt=LABEL_PROMPT.format(
            "\n".join([f"{k}: {v}" for (k, v) in sample["output"].items()])
        ),
        return_tensors=False,
    )

    return result

def main(
    path_to_test_set: str,
    path_to_train_set: str,
    path_to_output_dataset: str,
) -> None:

    tokenizer = get_tokenizer()

    test_set = Dataset.from_parquet(path_to_test_set).map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=os.cpu_count(),
    )

    train_set = Dataset.from_parquet(path_to_train_set).map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=os.cpu_count(),
    )

    DatasetDict({
        "test": test_set,
        "train": train_set,
    }).save_to_disk(path_to_output_dataset)
