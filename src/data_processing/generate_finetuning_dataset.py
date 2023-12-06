from typing import Optional
from datasets import Dataset, DatasetDict
from .get_random_data_samples import get_random_data_samples
from .generate_PGNs import generate_pgn

import io
import chess.pgn

# FIXME: this is not real PGN
def _create_sample(sample):
    
    game = chess.pgn.read_game(io.StringIO(generate_pgn(sample['Moves'])))
    game.headers['Result'] = sample['Result']
    game.headers['Termination'] = sample['Termination']
    
    return {'text': str(game)}

def main(
    number_of_samples: float,
) -> DatasetDict:
    
    train_dataset = get_random_data_samples(number_of_samples)
    train_dataset = train_dataset.map(
        _create_sample
    )
    
    test_dataset = get_random_data_samples(1000)
    test_dataset = test_dataset.map(
        _create_sample
    )

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
