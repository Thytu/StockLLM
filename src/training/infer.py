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
eval_tokenizer.pad_token = eval_tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, "Mistral-7B-v0.1-StockLLM//checkpoint-200").to("cuda")
ft_model.eval()


dataset = load_from_disk("outputs/dataset/test").shuffle()


with torch.no_grad():
    model_output = ft_model.generate(inputs=torch.tensor(dataset[0]["input_ids"]).unsqueeze(0))
    print(eval_tokenizer.decode(model_output[0], skip_special_tokens=True))

    model_output = ft_model.generate(inputs=torch.tensor(dataset[1]["input_ids"]).unsqueeze(0))
    print(eval_tokenizer.decode(model_output[0], skip_special_tokens=True))

# model_input = generate_MLM_prompt(eval_tokenizer, data_sample, return_tensors="pt")

# with torch.no_grad():
#     print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=128)[0], skip_special_tokens=True))

# model_input = generate_MLMlastonly_prompt(eval_tokenizer, data_sample, return_tensors="pt")

# with torch.no_grad():
#     print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=128)[0], skip_special_tokens=True))
