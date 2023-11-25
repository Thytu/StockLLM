import os
import json
import numpy as np
import pandas as pd

from typing import List


def main(
    paths: List[str],
    test_size: float,
    path_to_train_set: str,
    path_to_test_set: str,
) -> None:

    COLUMNS_TO_KEEP = [
        "task",
        "input",
        "expected_output",
        "KIND",
    ]

    for path in (path_to_train_set, path_to_test_set):
        os.makedirs(
            name=path if not "." in path else os.path.split(path)[0],
            exist_ok=True,
        )

    training_sets = []
    test_sets = []

    for _path in paths:
        df = pd.read_parquet(_path)[COLUMNS_TO_KEEP]
        df = df.sample(frac=1)

        test_sets.append(df[:int(len(df) * test_size)])
        training_sets.append(df[int(len(df) * test_size):])

    def default_json_exporter(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()

    test_sets = pd.concat(test_sets, ignore_index=True)
    test_sets["input"] = test_sets["input"].map(lambda x: json.dumps(x, default=default_json_exporter))
    test_sets["expected_output"] = test_sets["expected_output"].map(lambda x: json.dumps(x, default=default_json_exporter))

    training_sets = pd.concat(training_sets, ignore_index=True)
    training_sets["input"] = training_sets["input"].map(lambda x: json.dumps(x, default=default_json_exporter))
    training_sets["expected_output"] = training_sets["expected_output"].map(lambda x: json.dumps(x, default=default_json_exporter))

    test_sets.to_csv(path_to_test_set, index=False)
    training_sets.to_csv(path_to_train_set, index=False)
