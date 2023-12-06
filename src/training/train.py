import os
import wandb
import dvc.api
import importlib
import transformers

from typing import Dict, Any
from trl import SFTTrainer
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, prepare_model_for_kbit_training
from model import get_bitesandbytes_config, get_lora_config


def get_trainer(
    model,
    tokenizer,
    train_set,
    test_set,
    output_dir,
    formatting_func,
    **kwargs
) -> transformers.Trainer:

    default_params = {
        "output_dir": output_dir,

        "max_steps": 10_000,
        "warmup_steps": 5,
        "eval_steps": 100,
        "logging_steps": 1,
        "save_steps": 100,
        # "gradient_accumulation_steps": 2, # TODO: test impact on GPU/CPU

        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2.5e-5, # Want about 10x smaller than the Mistral learning rate
        "bf16": True,
        "optim": "paged_adamw_8bit",

        "logging_dir": os.path.join(output_dir, "logs"),
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "do_eval": True,
        "report_to": "wandb",
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 8,
    }

    default_params.update(**kwargs)

    return SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=test_set,
        max_seq_length=tokenizer.model_max_length,
        args=transformers.TrainingArguments(**default_params),
        # formatting_func=instruct_formatting_prompts_func, # TODO: use param to know wich fn to use
        formatting_func=formatting_func, # TODO: use param to know wich fn to use
        # dataset_text_field="text", # TODO: same as below
        data_collator = DataCollatorForLanguageModeling( # TODO: verify I can to that for the instruct part
            tokenizer=tokenizer,
            mlm=False
        )
    )


def main(
    project_name: str,
    subproject_name: str,
    path_to_model: str,
    model_parameters: Dict[str, Any],
    training_parameters: Dict[str, Any],
    path_to_dataset: str,
    path_to_outputs: str,
    to_log_to_wandb: bool = None,
):
    
    project_name += f'-{subproject_name}'

    bnb_config = get_bitesandbytes_config(**model_parameters["bitesandbytes_parameters"])

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path_to_model,
        use_flash_attention_2=True, # TODO: must not be hardcoded
        quantization_config=bnb_config,
    )
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config=get_lora_config())
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=path_to_model,
        model_max_length=model_parameters["model_max_length"],
        padding_side="right",
        add_eos_token=True,
        
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(path_to_dataset)

    run_name = datetime.now().strftime('%Y-%m-%d-%H-%M')

    formatting_func = getattr(
        importlib.import_module(f"data_processing.formatting_prompts_func"),
        training_parameters.pop("formatting_func"),
    )
    trainer: transformers.Trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_set=dataset["train"],
        test_set=dataset["test"],
        run_name=run_name,
        output_dir=path_to_outputs,
        formatting_func=formatting_func,
        **training_parameters,
    )

    wandb.login()

    with wandb.init(project=project_name, name=run_name) as run:
        run.config["run-params"] = to_log_to_wandb

        trainer.train()


if __name__ == "__main__":

    import dvc.api

    params = dvc.api.params_show()

    path_to_outputs = "outputs/poc/"

    main(
        project_name=params['general']['project-name'],
        model_parameters=params["model"],
        path_to_outputs=path_to_outputs,
        to_log_to_wandb={
            "general": params["general"],
            "model": params["model"],
            "stages": params["stages"],
        },
    )
