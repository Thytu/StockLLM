from typing import Optional
from datasets import Dataset, DatasetDict


def main(
    path_to_test_set: str,
    path_to_train_set: str,
    path_to_output_dataset: str,
    model_max_length: Optional[int] = None,
) -> None:

    test_set = Dataset.from_csv(path_to_test_set)
    test_set.cleanup_cache_files()
    # TODO: filter based on lenght

    train_set = Dataset.from_csv(path_to_train_set)
    train_set.cleanup_cache_files()
    # TODO: filter based on lenght

    DatasetDict({
        "test": test_set.shuffle(),
        "train": train_set.shuffle(),
    }).save_to_disk(path_to_output_dataset)
