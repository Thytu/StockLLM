import os
import wandb
import torch
import transformers
import dvc.api

from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


WANDB_PROJECT = "StockLLM"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
# BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
PROJECT_NAME = BASE_MODEL_ID.split("/")[-1] + "-" + WANDB_PROJECT
RUN_NAME = f"{PROJECT_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"


wandb.login()
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

wandb.init(project=WANDB_PROJECT, name=RUN_NAME)
wandb.run.config["dvc-params"] = dvc.api.params_show()

_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    model_max_length=2048,
    padding_side="left",
    add_eos_token=True
)
_tokenizer.pad_token = _tokenizer.eos_token

train_set = load_from_disk("outputs/dataset/train")
test_set = load_from_disk("outputs/dataset/train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
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
model = accelerator.prepare_model(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    args=transformers.TrainingArguments(
        output_dir=PROJECT_NAME,
        warmup_steps=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=2,
        max_steps=int(50_000 / 4),
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=RUN_NAME,          # Name of the W&B run (optional)
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(_tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()
