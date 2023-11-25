import os
import json
import pandas as pd

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
            "\n".join([f"{k}: {v}" for (k, v) in json.loads(sample["input"]).items()])
        ),
    )

    result["labels"] = tokenize(
        tokenizer=tokenizer,
        prompt=LABEL_PROMPT.format(
            "\n".join([f"{k}: {v}" for (k, v) in json.loads(sample["expected_output"]).items()])
        ),
    )

    return result

def main(
    path_to_test_set: str,
    path_to_train_set: str,
    path_to_output_dataset: str,
) -> None:

    COLUMNS_TO_KEEP = ["input_ids", "labels"]

    tokenizer = get_tokenizer()

    test_set = Dataset.from_csv(path_to_test_set).map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=os.cpu_count(),
    )
    test_set = test_set.remove_columns([c for c in test_set.column_names if c not in COLUMNS_TO_KEEP])

    train_set = Dataset.from_csv(path_to_train_set).map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=os.cpu_count(),
    )
    train_set = train_set.remove_columns([c for c in train_set.column_names if c not in COLUMNS_TO_KEEP])

    DatasetDict({
        "test": test_set,
        "train": train_set,
    }).save_to_disk(path_to_output_dataset)
