import os
import json

from typing import Optional
from transformers import BatchEncoding
from model import get_tokenizer, tokenize
from datasets import Dataset, DatasetDict
from .formatting_prompts_func import formatting_prompts_func


def generate_and_tokenize_prompt(samples, tokenizer, max_length: Optional[int] = None) -> BatchEncoding:

    result = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    texts = formatting_prompts_func(samples)

    ids_to_remove = []
    for idx, _text in enumerate(texts):
        _tokenized_sample = tokenize(
            tokenizer=tokenizer,
            prompt=_text,
            return_length=True,
            max_length=max_length,
            truncation=False,
        )
        _tokenized_sample_lenght = sum([1 for token in _tokenized_sample["input_ids"] if token != pad_token_id])

        if _tokenized_sample_lenght > max_length:
            ids_to_remove.append(idx)

    for i in reversed(ids_to_remove):
        del texts[i]

    return result


def main(
    path_to_test_set: str,
    path_to_train_set: str,
    path_to_output_dataset: str,
    model_max_length: Optional[int] = None,
) -> None:

    COLUMNS_TO_KEEP = ["text"]

    tokenizer = get_tokenizer()

    test_set = Dataset.from_csv(path_to_test_set)
    test_set.cleanup_cache_files()
    # TODO: filter based on lenght

    train_set = Dataset.from_csv(path_to_train_set)
    train_set.cleanup_cache_files()
    # TODO: filter based on lenght

    DatasetDict({
        "test": test_set,
        "train": train_set,
    }).save_to_disk(path_to_output_dataset)
