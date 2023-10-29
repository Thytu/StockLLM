from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from prompts import generate_random_prompt
from functools import partial


VALIDATION_SAMPLES = 100
TRAINING_SAMPLES = 20_000
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"
GENERATE_MLM_PROMPT_PROB = 1


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def main():
    dataset = load_dataset("laion/strategic_game_chess", streaming=True)

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        model_max_length=128,
        padding_side="left",
        add_eos_token=True
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    tokenized_train_dataset = dataset["train"].take(TRAINING_SAMPLES + VALIDATION_SAMPLES)
    tokenized_train_dataset = Dataset.from_generator(
        partial(gen_from_iterable_dataset, tokenized_train_dataset),
        features=tokenized_train_dataset.features,
    )
    tokenized_train_dataset = tokenized_train_dataset.map(
        generate_random_prompt,
        fn_kwargs={
            "tokenizer": _tokenizer,
            "generate_MLM_prompt_prob": GENERATE_MLM_PROMPT_PROB,
        },
    )
    tokenized_train_dataset.save_to_disk("cached_dataset")


if __name__ == "__main__":
    main()
