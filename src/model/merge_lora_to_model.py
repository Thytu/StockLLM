from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    path_to_model: str,
    path_to_adapter: str,
    path_to_output: str,
):
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="outputs/untrained-model-w-vocab/",
        load_in_8bit=False,
        device_map="cpu",
    )
        
    model = PeftModel.from_pretrained(
        model, 
        path_to_adapter, 
        device_map="cpu",
    )

    model = model.merge_and_unload()

    # making a copy to model's dir
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=path_to_model
    )

    model.save_pretrained(path_to_output)
    tokenizer.save_pretrained(path_to_output)
