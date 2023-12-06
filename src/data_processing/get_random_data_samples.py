from functools import partial
from datasets import load_dataset, Dataset


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def get_random_data_samples(number_of_samples: int) -> Dataset:

    if number_of_samples < 0:
        raise RuntimeError(f"{number_of_samples=} must be positive.")

    dataset = load_dataset("laion/strategic_game_chess", streaming=True).shuffle()

    dataset = dataset["train"].take(number_of_samples)

    dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, dataset),
        features=dataset.features,
    )
    dataset.cleanup_cache_files()

    return dataset
