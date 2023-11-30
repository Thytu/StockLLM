import json


def formatting_prompts_func(example):
    output_texts = []

    for idx in range(len(example['task'])):

        inputs = "\n".join([f"{k}: {v}" for (k, v) in json.loads(example["input"][idx]).items()])
        expected_output = "\n".join([f"{k}: {v}" for (k, v) in json.loads(example["expected_output"][idx]).items()])

        text = f"<s>[INST]{example['task'][idx]}[/INST]\n[IN]{inputs}[/IN]\n[OUT]{expected_output}[/OUT]</s>"
        output_texts.append(text)

    return output_texts
