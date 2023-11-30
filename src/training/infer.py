import torch

from peft import PeftModel
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, "./outputs/poc/checkpoint-1000").to("cuda")
ft_model.eval()

dataset = load_from_disk("outputs/poc-dataset/test").shuffle()


import json


def formatting_prompts_func(example):
    output_texts = []

    for idx in range(len(example['task'])):

        inputs = "\n".join([f"{k}: {v}" for (k, v) in json.loads(example["input"][idx]).items()])
        expected_output = "\n".join([f"{k}: {v}" for (k, v) in json.loads(example["expected_output"][idx]).items()])

        text = f"<s>[INST]{example['task'][idx]}[/INST]\n[IN]{inputs}[/IN]\n[OUT]{expected_output}[/OUT]</s>"
        output_texts.append(text)

    return output_texts


with torch.no_grad():
    for idx in range(10):

        inputs = "\n".join([f"{k}: {v}" for (k, v) in json.loads(dataset[idx]["input"]).items()])
        text = f"<s>[INST]{dataset[idx]['task']}[/INST]\n[IN]{inputs}[/IN]\n[OUT]"

        tokens = eval_tokenizer([text], return_tensors="pt")

        model_output = ft_model.generate(**tokens)
        model_output = eval_tokenizer.decode(model_output[0], skip_special_tokens=True)
        print(model_output)
