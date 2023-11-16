import os

from typing import List
from datasets import load_from_disk, concatenate_datasets


def main(
    paths: List[str],
    test_size: float,
    path_to_output_dataset: str,
):

    training_sets = []
    test_sets = []

    for _path in paths:
        samples = load_from_disk(_path)

        samples = samples.train_test_split(test_size=test_size)
        training_sets.append(samples["train"])
        test_sets.append(samples["test"])

    concatenate_datasets(training_sets).save_to_disk(
        os.path.join(path_to_output_dataset, "train")
    )
    concatenate_datasets(test_sets).save_to_disk(
        os.path.join(path_to_output_dataset, "test")
    )
