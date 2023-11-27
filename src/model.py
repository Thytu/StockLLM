from typing import Optional
from torch import bfloat16
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"

_DEFAULT_MODEL_MAX_LENGTH = 4096


def tokenize(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    return_tensors: Optional[bool] = None,
    max_length: int = _DEFAULT_MODEL_MAX_LENGTH,
    truncation: bool = True,
    **kwargs,
) -> BatchEncoding:

    return tokenizer.__call__(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=truncation,
        return_tensors=return_tensors,
        **kwargs,
    )


def decode(tokenizer: PreTrainedTokenizer, token_ids: BatchEncoding) -> str:
    tokenizer.decode(token_ids, skip_special_tokens=True)


def get_tokenizer(**kwargs):

    default_value = {
        "model_max_length": _DEFAULT_MODEL_MAX_LENGTH,
        "padding_side": "left",
        "add_eos_token": True
    }

    default_value.update(kwargs)

    _tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_ID,
        **default_value,
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    return _tokenizer


# NOTE: 4bits are the responsables for the 25% aten:_copy (GPU -> CPU -> GPU)
def get_bitesandbytes_config(**kwargs):

    default_value = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": bfloat16
    }

    default_value.update(kwargs)

    return BitsAndBytesConfig(default_value)


def get_lora_config(**kwargs):

    default_params = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        "bias": "none",
        "lora_dropout": 0.05,
        "task_type": "CAUSAL_LM",
    }

    default_params.update(kwargs)

    return LoraConfig(default_params)


def get_model(**kwargs):
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
