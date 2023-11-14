from transformers import AutoTokenizer


MODEL_MAX_LENGTH = 2048
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"


def tokenize(
    tokenizer,
    prompt,
    return_tensors=None,
    max_length: int = MODEL_MAX_LENGTH,
):
    return tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=return_tensors,
    )


def get_tokenizer():
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="left",
        add_eos_token=True
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    return _tokenizer
