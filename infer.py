import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    # use_auth_token=True
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)


from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-StockLLM/checkpoint-950")

from random import random


def __remove_random_moves(moves, prob_to_remove: float = 0.1):
    remaining_moves = []
    removed_moves = []

    for m in moves:
        if random() <= prob_to_remove:
            removed_moves.append(m)
            remaining_moves.append("?")
        else:
            remaining_moves.append(m)

    return remaining_moves, removed_moves


def generate_MLM_prompt(data_point, return_tensors=None):
    moves, removed_moves = __remove_random_moves(data_point["Moves"])

    full_prompt =f"""task: Given an incomplit set of chess moves, write the missing chess moves.
Missing moves are written with a "?" mark, simpy write the missing ones.

i.e give, "e2e4, e7e6, d2d4, d7d5, b1c3, ?, e4e5, c7c5, d1g4, g8e7, d4c5, b4c3, b2c3" you must write "f8b4".

Moves: {moves}

Result:
""".replace("'", "")

    result = eval_tokenizer(full_prompt, return_tensors=return_tensors)
    result["labels"] = eval_tokenizer(str(removed_moves))["input_ids"]#.copy()

    return result


def generate_regression_prompt(data_point, return_tensors=None):
    full_prompt =f"""task: Given a set of chess moves announce the result of the game

Moves: {data_point["Moves"]}

Result:
""".replace("'", "")

    result = eval_tokenizer(full_prompt, return_tensors=return_tensors)
    result["labels"] = eval_tokenizer(data_point["Result"])["input_ids"]

    return result


from datasets import load_dataset

dataset = load_dataset("laion/strategic_game_chess", streaming=True)

data_sample = next(iter(dataset["train"]))

# model_input = generate_MLM_prompt(data_sample)
model_input = generate_regression_prompt(data_sample, return_tensors="pt")

# with torch.no_grad():
#     print(_tokenizer.decode(model.generate(**model_input, max_new_tokens=128, pad_token_id=2)[0], skip_special_tokens=True))

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=128)[0], skip_special_tokens=True))
