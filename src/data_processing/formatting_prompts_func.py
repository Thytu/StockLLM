import io
import json
import chess.pgn

from typing import List, Dict
from .generate_PGNs import generate_pgn


def causal_formatting_prompts_func(example: Dict[str, str]) -> List[str]:
    output_texts = []
    
    def _create_sample(moves, result, termination):
    
        game = chess.pgn.read_game(io.StringIO(generate_pgn(moves)))
        game.headers['Result'] = result
        game.headers['Termination'] = termination
        
        return str(game)

    for idx in range(len(example['Moves'])):
        output_texts.append(_create_sample(
            example["Moves"][idx],
            example["Result"][idx],
            example["Termination"][idx],
        ))

    return output_texts


def instruct_formatting_prompts_func(example: Dict[str, str]) -> List[str]:
    output_texts = []

    def _format_dict_to_string(data):
        data = json.loads(data)

        return "\n" + "\n".join([f"{k}: {v}" for (k, v) in data.items()])

    for idx in range(len(example['task'])):

        inputs = _format_dict_to_string(example["input"][idx])
        expected_output = _format_dict_to_string(example["expected_output"][idx])

        text = f"<s>[INST]{example['task'][idx]}[/INST]\n[IN]{inputs}[/IN]\n[OUT]{expected_output}[/OUT]</s>"
        output_texts.append(text)

    return output_texts
