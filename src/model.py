from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding


MODEL_MAX_LENGTH = 4096
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"


def tokenize(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    return_tensors: Optional[bool] = None,
    max_length: int = MODEL_MAX_LENGTH,
) -> BatchEncoding:
    return tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=return_tensors,
    )


def decode(tokenizer: PreTrainedTokenizer, token_ids: BatchEncoding) -> str:
    tokenizer.decode(token_ids, skip_special_tokens=True)


def get_tokenizer():
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="left",
        add_eos_token=True
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    return _tokenizer
