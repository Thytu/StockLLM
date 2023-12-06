import dvc.api

from model import get_default_model, get_default_tokenizer
from data_processing.get_dict_move_in_utf8_to_token import get_dict_move_in_utf8_to_token


def main(
    model_parameters,
    path_to_output: str,
):
    
    model = get_default_model(
        use_flash_attention_2=False,
    )

    tokenizer = get_default_tokenizer(
        model_max_length=model_parameters["model_max_length"],
    )

    # Adding chess moves to vocabulaty
    _dict_move_in_utf8_to_token = get_dict_move_in_utf8_to_token()
    tokenizer.add_tokens(list(_dict_move_in_utf8_to_token.keys()))
    model.resize_token_embeddings(len(tokenizer))
    
    model.save_pretrained(path_to_output)
    tokenizer.save_pretrained(path_to_output)


if __name__ == "__main__":

    import dvc.api

    params = dvc.api.params_show()

    main(
        **params["stages"]["add-chessmoves-to-model-vocab"]["params"],
    )
