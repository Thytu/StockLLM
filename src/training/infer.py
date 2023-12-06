import torch

from peft import PeftModel
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig


base_model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
    model_max_length=2048,
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, "./outputs/poc/checkpoint-1000").to("cuda")
ft_model.eval()

dataset = load_from_disk("outputs/poc-dataset/test").shuffle()

generation_config = GenerationConfig.from_pretrained(base_model_id)
print(f"Previous {generation_config.max_length=}")
generation_config.max_length = 2048

import json


from itertools import combinations


ROWS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
COLUMNS = ('1', '2', '3', '4', '5', '6', '7', '8')


def get_dict_move_in_utf8_to_token(model_vocab_size: int = 32_000):

    positions_in_utf8 = []
    tokenizer = eval_tokenizer

    for row in ROWS:
        for column in COLUMNS:
            positions_in_utf8.append(row + column)

    moves_in_utf8 = []
    
    for (pos_1, pos_2) in list(combinations(positions_in_utf8, 2)):
        moves_in_utf8 += [pos_1 + pos_2, pos_2 + pos_1]

    tokens_start_at = model_vocab_size - len(moves_in_utf8)

    return {
        utf_8: token_id for (utf_8, token_id) in zip(
            moves_in_utf8,
            [tokenizer.decode(i) for i in range(tokens_start_at, tokens_start_at + len(moves_in_utf8))]
        )
    }

_dict_move_in_utf8_to_token = get_dict_move_in_utf8_to_token()

def _format_dict_to_string(data):
    data = json.loads(data)

    for column in ("move", "missing move"):
        if column in data:
            data[column] = _dict_move_in_utf8_to_token[data[column][:4]] + (data[column][4] if len(data[column]) > 4 else "")

    for column in ("moves", "missing moves"):
        if column in data:
            data[column] = ["?" if move == "?" else _dict_move_in_utf8_to_token[move[:4]] + (move[4] if len(move) > 4 else "") for move in data[column]]

    return data


with torch.no_grad():

    for idx in range(10):

        inputs = _format_dict_to_string(dataset[idx]["input"])
        inputs = "\n".join([f"{k}: {v}" for (k, v) in inputs.items()])
        text = f"<s>[INST]{dataset[idx]['task']}[/INST]\n[IN]{inputs}[/IN]\n[OUT]"

        tokens = eval_tokenizer([text], return_tensors="pt").to("cuda")

        model_output = ft_model.generate(
            **tokens,
            generation_config=generation_config,
        )
        model_output = eval_tokenizer.decode(model_output[0], skip_special_tokens=True)
        print(model_output)
