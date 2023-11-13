def generate_pgn(moves):
    pairs_of_move = [[]]

    for m in moves:
        if len(pairs_of_move[-1]) < 2:
            pairs_of_move[-1].append(m)
        else:
            pairs_of_move.append([m])

    pgn_notation = ""

    for nb in range(len(pairs_of_move)):
        pgn_notation += f"{nb + 1}. {pairs_of_move[nb][0]} {pairs_of_move[nb][1] if len(pairs_of_move[nb]) > 1 else ''}"

    return pgn_notation


def _add_pgn_notation(example):
    example["PGN"] = generate_pgn(example["Moves"])

    return example


def _gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def main(dataset, training_samples, validation_samples):

    dataset = dataset["train"].take(training_samples + validation_samples)

    dataset = Dataset.from_generator(
        partial(_gen_from_iterable_dataset, dataset),
        features=dataset.features,
    )
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        _add_pgn_notation,
        num_proc=24,
    )

    return dataset


if __name__ == "__main__":
    import os

    from functools import partial
    from datasets import load_dataset, Dataset
    from init_dataset import TRAINING_SAMPLES, VALIDATION_SAMPLES


    OUTPUT_FOLDER = "outputs/PGNs"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    dataset = load_dataset("laion/strategic_game_chess", streaming=True)

    dataset = main(
        dataset=dataset,
        training_samples=TRAINING_SAMPLES,
        validation_samples=VALIDATION_SAMPLES,
    )

    dataset.save_to_disk(OUTPUT_FOLDER)
