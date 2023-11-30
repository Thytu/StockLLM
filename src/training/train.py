import os
import wandb
import dvc.api
import transformers

from datetime import datetime
from datasets import load_from_disk
from peft import get_peft_model, prepare_model_for_kbit_training
from model import get_model, get_bitesandbytes_config, get_lora_config, get_tokenizer
from trl import SFTTrainer


def get_trainer(
    model,
    tokenizer,
    train_set,
    test_set,
    output_dir,
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

    from data_processing.formatting_prompts_func import formatting_prompts_func

    return SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=test_set,
        max_seq_length=tokenizer.model_max_length,
        args=transformers.TrainingArguments(**default_params),
        formatting_func=formatting_prompts_func,
    )


def main(
    project_name,
    model_parameters,
    training_parameters,
    path_to_dataset,
    path_to_outputs: str,
    to_log_to_wandb=None,
):

    wandb.login()
    os.environ["WANDB_PROJECT"] = project_name

    bnb_config = get_bitesandbytes_config(**model_parameters["bitesandbytes_parameters"])

    model = get_model(quantization_config=bnb_config)
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config=get_lora_config())
    model.config.use_cache = False

    tokenizer = get_tokenizer(
        model_max_length=model_parameters["model_max_length"],
    )

    dataset = load_from_disk(path_to_dataset)

    run_name = f"{project_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    trainer: transformers.Trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_set=dataset["train"],
        test_set=dataset["test"],
        run_name=run_name,
        output_dir=path_to_outputs,
        **training_parameters,
    )

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
