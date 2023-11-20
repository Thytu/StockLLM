import pandas as pd

from typing import List

PROMPT = """<s>[INST]{}[/INST][IN]{}[/IN][OUT]{}[/OUT]
"""

def main(
    paths: List[str],
    test_size: float,
    path_to_train_set: str,
    path_to_test_set: str,
) -> None:

    training_sets = []
    test_sets = []

    for _path in paths:
        df = pd.read_parquet(_path)
        df = df.sample(frac=1)

        test_sets.append(df[:int(len(df) * test_size)])
        training_sets.append(df[int(len(df) * test_size):])


    test_sets = pd.concat(test_sets, ignore_index=True)
    training_sets = pd.concat(training_sets, ignore_index=True)

    test_sets.to_parquet(path_to_test_set)
    training_sets.to_parquet(path_to_train_set)
