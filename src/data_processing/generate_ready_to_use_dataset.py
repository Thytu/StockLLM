import os
import json

from typing import Optional
from transformers import BatchEncoding
from model import get_tokenizer, tokenize
from datasets import Dataset, DatasetDict


PROMPT = "<s>[INST]{}[/INST]\n[IN]{}[/IN]\n[OUT]"
LABEL_PROMPT = "\n{}[/OUT]</s>"


def generate_and_tokenize_prompt(samples, tokenizer, max_length: Optional[int] = None) -> BatchEncoding:

    result = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for idx in range(len(samples["input"])):
        _tokenized_sample = tokenize(
            tokenizer=tokenizer,
            prompt=PROMPT.format(
                samples["task"][idx],
                "\n".join([f"{k}: {v}" for (k, v) in json.loads(samples["input"][idx]).items()])
            ), return_length=True,
            max_length=max_length,
            truncation=False,
        )

        if max_length and _tokenized_sample["length"][0] > max_length:
            continue

        _tokenized_labels = tokenize(
            tokenizer=tokenizer,
            prompt=LABEL_PROMPT.format(
                "\n".join([f"{k}: {v}" for (k, v) in json.loads(samples["expected_output"][idx]).items()])
            ), return_length=True,
            max_length=max_length,
            truncation=False,
        )

        if max_length and _tokenized_labels["length"][0] > max_length:
            continue

        result["input_ids"].append(_tokenized_sample["input_ids"])
        result["attention_mask"].append(_tokenized_sample["attention_mask"])
        result["labels"].append(_tokenized_labels["input_ids"])

    return result


def main(
    path_to_test_set: str,
    path_to_train_set: str,
    path_to_output_dataset: str,
    model_max_length: Optional[int] = None,
) -> None:

    COLUMNS_TO_KEEP = ["input_ids", "labels"]

    tokenizer = get_tokenizer()

    test_set = Dataset.from_csv(path_to_test_set)
    test_set.cleanup_cache_files()
    test_set = test_set.map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": model_max_length,
        },
        num_proc=os.cpu_count(),
        batched=True,
        remove_columns=[c for c in test_set.column_names if c not in COLUMNS_TO_KEEP],
    )

    train_set = Dataset.from_csv(path_to_train_set)
    train_set.cleanup_cache_files()
    train_set = train_set.map(
        generate_and_tokenize_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": model_max_length,
        },
        num_proc=os.cpu_count(),
        batched=True,
        remove_columns=[c for c in train_set.column_names if c not in COLUMNS_TO_KEEP],
    )

    DatasetDict({
        "test": test_set,
        "train": train_set,
    }).save_to_disk(path_to_output_dataset)
