from datasets import load_dataset

dataset = load_dataset("laion/strategic_game_chess", streaming=True)

from transformers import AutoTokenizer

base_model_id = "mistralai/Mistral-7B-v0.1"

_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=128,
    padding_side="left",
    add_eos_token=True
)

_tokenizer.pad_token = _tokenizer.eos_token

def tokenize(prompt, return_tensors=None):
    return _tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors=return_tensors,
    )

def generate_regression_prompt(data_point, return_tensors=None):
    full_prompt =f"""task: Given a set of chess moves announce the result of the game
Write 1-0 if the white wins, 0-1 of the black wins and 1/2-1/2 in case of draw.

i.e given "[...] g7g8q, h5h4, e5f4, b5e2, g8g5, h4h3, g5g3" you must write "1-0".
i.e given "[...] "e5d4, b4b3, a6a1, b3b4, a1b1, b4a3, d4c5, a3a4, b1b4" you must write "0-1".

Moves: {data_point["Moves"]}

Result:
""".replace("'", "")

    result = tokenize(full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(data_point["Result"])["input_ids"]

    return result

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
Missing chess move are indicated with a "?" mark. Just write the missing moves.

i.e given "[...] e2e4, e7e6, d2d4, d7d5, b1c3, ?, e4e5, c7c5, d1g4, g8e7, d4c5, b4c3, b2c3" you must write "f8b4".
i.e given "[...] d2d4, d7d5, b1c3, f8b4, ?, c7c5, d1g4, g8e7, d4c5, b4c3, b2c3" you must write "e4e5".

Moves: {moves}

Result:
""".replace("'", "")

    result = tokenize(full_prompt, return_tensors=return_tensors)
    result["labels"] = tokenize(str(removed_moves))["input_ids"]#.copy()

    return result

def generate_random_prompt(
    data_sample,
    generate_MLM_prompt_prob: float = 0.8,
):
    if random() <= generate_MLM_prompt_prob:
        return generate_MLM_prompt(data_sample)

    return generate_regression_prompt(data_sample)

import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenized_train_dataset = dataset["train"].take(80_000).map(generate_random_prompt)
tokenized_val_dataset = dataset["train"].skip(80_000).take(100).map(generate_random_prompt)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

import wandb, os

wandb.login()

wandb_project = "StockLLM"
os.environ["WANDB_PROJECT"] = wandb_project

import transformers
from datetime import datetime

project = wandb_project
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

_tokenizer.pad_token = _tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=2,
        max_steps=1000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        # do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(_tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()