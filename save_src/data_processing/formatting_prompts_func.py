import json

from .get_dict_move_in_utf8_to_token import get_dict_move_in_utf8_to_token


def formatting_prompts_func(example):
    output_texts = []

    # TODO: provide model's vocab size as param
    # TODO: instead of doing a if 'move' in data I could to a .replace() :)
    _dict_move_in_utf8_to_token = get_dict_move_in_utf8_to_token()

    def _format_dict_to_string(data):
        data = json.loads(data)

        for column in ("move", "missing move"):
            if column in data:
                data[column] = _dict_move_in_utf8_to_token[data[column][:4]] + (data[column][4] if len(data[column]) > 4 else "")

        for column in ("moves", "missing moves"):
            if column in data:
                try:
                    data[column] = ["?" if move == "?" else _dict_move_in_utf8_to_token[move[:4]] + (move[4] if len(move) > 4 else "") for move in data[column]]
                except Exception as e:
                    print(example["input"][idx])
                    print(example["expected_output"][idx])
                    print(example["KIND"][idx])
                    print(example["task"][idx])
                    print([move for move in data[column]])
                    raise e

        return data

    for idx in range(len(example['task'])):

        inputs = _format_dict_to_string(example["input"][idx])
        expected_output = _format_dict_to_string(example["expected_output"][idx])

        text = f"<s>[INST]{example['task'][idx]}[/INST]\n[IN]{inputs}[/IN]\n[OUT]{expected_output}[/OUT]</s>"
        output_texts.append(text)

    return output_texts
