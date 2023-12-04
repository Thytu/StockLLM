import json

from .get_dict_move_in_utf8_to_token import get_dict_move_in_utf8_to_token


def formatting_prompts_func(example):
    output_texts = []

    # TODO: provide model's vocab size as param
    # TODO: instead of doing a if 'move' in data I could to a .replace() :)
    _dict_move_in_utf8_to_token = get_dict_move_in_utf8_to_token()

    def _format_dict_to_string(data):
        data = json.loads(data)

        return "\n" + "\n".join([f"{k}: {v}" for (k, v) in data.items()])

    for idx in range(len(example['task'])):

        inputs = _format_dict_to_string(example["input"][idx])
        expected_output = _format_dict_to_string(example["expected_output"][idx])

        text = f"<s>[INST]{example['task'][idx]}[/INST]\n[IN]{inputs}[/IN]\n[OUT]{expected_output}[/OUT]</s>"
        output_texts.append(text)

    return output_texts
